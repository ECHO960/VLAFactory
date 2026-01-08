"""
RLDS dataset utilities for VLAFactory.

OXE RLDS parsing and mixture definitions are adapted from OpenVLA's dataset
pipeline (see OpenVLA repository for original reference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, Union

import os
import numpy as np
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info

from vlafactory.data.rlds.oxe import OXE_NAMED_MIXTURES, OXE_DATASET_CONFIGS


class TokenizerLike(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Any:
        ...


class ImageProcessorLike(Protocol):
    def __call__(self, image: Any, **kwargs: Any) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class SampleProcessingConfig:
    max_length: int = 512
    action_dim: int = 7


@dataclass(frozen=True)
class RLDSStreamConfig:
    data_root_dir: str
    data_mix: Union[str, Sequence[Tuple[str, float]]]
    train: bool = True
    shuffle_files: bool = True
    seed: int = 0
    dataset_versions: Optional[Dict[str, str]] = None
    default_version: str = "0.1.0"


def process_sample(
    sample: Dict[str, Any],
    *,
    tokenizer: Optional[TokenizerLike],
    image_processor: Optional[ImageProcessorLike],
    max_length: int,
    action_dim: int,
) -> Dict[str, Any]:
    instruction = sample.get("instruction", sample.get("text", ""))
    processed: Dict[str, Any] = {}

    if tokenizer is not None and instruction:
        tokenized = tokenizer(
            instruction,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        processed["input_ids"] = tokenized["input_ids"].squeeze(0)
        processed["attention_mask"] = tokenized["attention_mask"].squeeze(0)
    else:
        processed["input_ids"] = torch.zeros(max_length, dtype=torch.long)
        processed["attention_mask"] = torch.ones(max_length, dtype=torch.long)

    image = sample.get("image", sample.get("image_path"))
    if image is not None and image_processor is not None:
        try:
            from PIL import Image

            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(image)
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image.astype("uint8")).convert("RGB")
            else:
                pil_image = image
            processed_image = image_processor(pil_image, return_tensors="pt")
            processed["images"] = processed_image["pixel_values"].squeeze(0)
        except Exception:
            processed["images"] = None
    else:
        processed["images"] = None

    if "action" in sample:
        action = sample["action"]
        if isinstance(action, np.ndarray):
            action_tensor = torch.tensor(action, dtype=torch.float32)
        elif isinstance(action, list):
            action_tensor = torch.tensor(action, dtype=torch.float32)
        else:
            action_tensor = torch.tensor([action], dtype=torch.float32)

        if len(action_tensor) < action_dim:
            padding = torch.zeros(action_dim - len(action_tensor))
            action_tensor = torch.cat([action_tensor, padding])
        elif len(action_tensor) > action_dim:
            action_tensor = action_tensor[:action_dim]
        processed["action_labels"] = action_tensor

    if "response" in sample:
        processed["labels"] = processed["input_ids"].clone()

    if "dataset_name" in sample:
        processed["dataset_name"] = sample["dataset_name"]

    return processed


class OXERLDSDataset(IterableDataset):
    """Iterable dataset that streams OXE RLDS data and standardizes to VLA samples."""

    def __init__(
        self,
        stream: RLDSStreamConfig,
        processing: SampleProcessingConfig,
        *,
        tokenizer: Optional[TokenizerLike] = None,
        image_processor: Optional[ImageProcessorLike] = None,
    ) -> None:
        self._stream = stream
        self._processing = processing
        self._tokenizer = tokenizer
        self._image_processor = image_processor

        self._ensure_tfds()
        self._mixture_spec = self._resolve_mixture(self._stream.data_mix)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        collated: Dict[str, torch.Tensor] = {}

        keys = set()
        for sample in batch:
            keys.update(sample.keys())

        for key in keys:
            values = [sample.get(key) for sample in batch if sample.get(key) is not None]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        return collated

    def _ensure_tfds(self) -> None:
        try:
            import tensorflow as tf  # noqa: F401
            import tensorflow_datasets as tfds  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Loading OXE RLDS datasets requires tensorflow and tensorflow_datasets. "
                "Install them to use OXERLDSDataset."
            ) from exc

    def _resolve_mixture(self, data_mix: Union[str, Sequence[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        if isinstance(data_mix, (list, tuple)):
            return list(data_mix)
        if data_mix in OXE_NAMED_MIXTURES:
            return OXE_NAMED_MIXTURES[data_mix]
        return [(data_mix, 1.0)]

    def _resolve_builder_dir(self, dataset_name: str) -> Optional[str]:
        root_dir = self._stream.data_root_dir
        if not root_dir or not os.path.isdir(root_dir):
            return None
        dataset_dir = os.path.join(root_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            return None
        versions = self._stream.dataset_versions or {}
        if dataset_name in versions:
            version = versions[dataset_name]
        elif dataset_name in ("robo_net", "cmu_playing_with_food"):
            version = "1.0.0"
        elif dataset_name in ("language_table", "aloha_mobile"):
            version = "0.0.1"
        else:
            version = self._stream.default_version
        builder_dir = os.path.join(dataset_dir, version)
        return builder_dir if os.path.isdir(builder_dir) else None

    def _make_tfds_dataset(self, dataset_name: str):
        import tensorflow as tf
        import tensorflow_datasets as tfds

        builder_dir = self._resolve_builder_dir(dataset_name)
        if builder_dir is not None:
            builder = tfds.builder_from_directory(builder_dir=builder_dir)
        else:
            builder = tfds.builder(dataset_name, data_dir=self._stream.data_root_dir)
        if "val" not in builder.info.splits:
            split = "train[:95%]" if self._stream.train else "train[95%:]"
        else:
            split = "train" if self._stream.train else "val"
        tf.config.set_visible_devices([], "GPU")
        return builder.as_dataset(split=split, shuffle_files=self._stream.shuffle_files)

    def _flatten_steps(self, steps: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(steps, dict):
            first_value = next(iter(steps.values()))
            length = len(first_value)
            for i in range(length):
                yield {k: v[i] for k, v in steps.items()}
            return
        if isinstance(steps, (list, tuple)):
            for step in steps:
                yield step
            return
        yield steps

    def _select_image(self, dataset_name: str, observation: Dict[str, Any]) -> Any:
        config = OXE_DATASET_CONFIGS.get(dataset_name, {})
        image_obs_keys = config.get("image_obs_keys", {})
        primary_key = image_obs_keys.get("primary")
        if primary_key and primary_key in observation:
            return observation[primary_key]
        for fallback_key in ("image_primary", "image", "rgb", "rgb_static", "rgb_observation"):
            if fallback_key in observation:
                return observation[fallback_key]
        return None

    def _decode_language(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return ""
            if value.dtype.type is np.bytes_:
                return bytes(value.tolist()).decode("utf-8", errors="ignore")
            return str(value.tolist())
        return str(value)

    def _normalize_action(self, action: Any) -> Optional[np.ndarray]:
        if action is None:
            return None
        if isinstance(action, dict):
            if "rel_actions_world" in action:
                return np.asarray(action["rel_actions_world"], dtype=np.float32)
            if "world_vector" in action and "rotation_delta" in action:
                gripper = None
                for key in ("open_gripper", "gripper_closedness_action", "gripper", "gripper_state"):
                    if key in action:
                        gripper = action[key]
                        break
                if gripper is None:
                    gripper = np.zeros_like(action["world_vector"][..., :1])
                gripper = np.asarray(gripper, dtype=np.float32)
                if gripper.ndim == 1:
                    gripper = gripper[..., None]
                return np.concatenate(
                    [
                        np.asarray(action["world_vector"], dtype=np.float32),
                        np.asarray(action["rotation_delta"], dtype=np.float32),
                        gripper,
                    ],
                    axis=-1,
                )
            if len(action) == 1:
                return np.asarray(next(iter(action.values())), dtype=np.float32)
            return None
        if isinstance(action, (list, tuple, np.ndarray)):
            return np.asarray(action, dtype=np.float32)
        return np.asarray([action], dtype=np.float32)

    def _standardize_step(self, dataset_name: str, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        observation = step.get("observation", step)
        task = step.get("task", {})
        instruction = (
            task.get("language_instruction")
            or step.get("language_instruction")
            or observation.get("language_instruction")
            or observation.get("natural_language_instruction")
        )
        instruction = self._decode_language(instruction)

        image = self._select_image(dataset_name, observation)
        action = self._normalize_action(step.get("action"))
        if action is None:
            return None
        return {
            "instruction": instruction,
            "image": image,
            "action": action,
            "dataset_name": dataset_name,
        }

    def _iter_dataset_steps(self, dataset_name: str) -> Iterable[Dict[str, Any]]:
        import tensorflow_datasets as tfds

        dataset = self._make_tfds_dataset(dataset_name)
        for episode in tfds.as_numpy(dataset):
            steps = episode.get("steps")
            if steps is None:
                standardized = self._standardize_step(dataset_name, episode)
                if standardized is not None:
                    yield standardized
                continue
            for step in self._flatten_steps(steps):
                standardized = self._standardize_step(dataset_name, step)
                if standardized is not None:
                    yield standardized

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = get_worker_info()
        seed = self._stream.seed if worker_info is None else self._stream.seed + worker_info.id
        rng = random.Random(seed)

        dataset_iters = [iter(self._iter_dataset_steps(name)) for name, _ in self._mixture_spec]
        weights = [weight for _, weight in self._mixture_spec]
        active = list(range(len(dataset_iters)))

        while active:
            idx = rng.choices(active, weights=[weights[i] for i in active], k=1)[0]
            try:
                sample = next(dataset_iters[idx])
            except StopIteration:
                active.remove(idx)
                continue
            yield process_sample(
                sample,
                tokenizer=self._tokenizer,
                image_processor=self._image_processor,
                max_length=self._processing.max_length,
                action_dim=self._processing.action_dim,
            )
