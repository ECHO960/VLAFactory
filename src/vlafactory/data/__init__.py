"""Data handling components for VLA training."""

from typing import Any, Dict, List, Tuple
import os

from vlafactory.data.rlds.dataset import OXERLDSDataset, RLDSStreamConfig, SampleProcessingConfig


def load_dataset(
    data_args: Any,
    model_args: Any | None = None,
) -> Dict[str, Any]:
    max_length = getattr(data_args, "max_length", None) or getattr(data_args, "cutoff_len", 512)
    action_dim = getattr(model_args, "action_dim", 7) if model_args is not None else 7

    data_mix = getattr(data_args, "dataset", None)
    eval_mix = getattr(data_args, "eval_dataset", None)

    if data_mix is None:
        raise ValueError("`dataset` must be provided for OXE RLDS training.")

    mixture_spec = _normalize_mixture_spec(data_mix, getattr(data_args, "interleave_probs", None))
    train_stream = RLDSStreamConfig(
        data_root_dir=data_args.dataset_dir,
        data_mix=mixture_spec,
        train=True,
    )
    processing = SampleProcessingConfig(max_length=max_length, action_dim=action_dim)
    train_dataset = OXERLDSDataset(
        train_stream,
        processing,
        tokenizer=None,
    )
    eval_dataset = None
    if eval_mix is not None:
        eval_spec = _normalize_mixture_spec(eval_mix, getattr(data_args, "interleave_probs", None))
        eval_stream = RLDSStreamConfig(
            data_root_dir=data_args.dataset_dir,
            data_mix=eval_spec,
            train=False,
        )
        eval_dataset = OXERLDSDataset(
            eval_stream,
            processing,
            tokenizer=None,
        )
    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "collator": OXERLDSDataset.collate_fn,
    }


def _normalize_mixture_spec(
    dataset_names: Any,
    interleave_probs: Any | None,
) -> List[Tuple[str, float]] | str:
    if isinstance(dataset_names, list):
        names = dataset_names
    elif isinstance(dataset_names, str):
        names = [dataset_names]
    else:
        return dataset_names

    if len(names) == 1:
        return names[0]

    if interleave_probs is None:
        weights = [1.0] * len(names)
    else:
        weights = list(interleave_probs)
        if len(weights) != len(names):
            raise ValueError("interleave_probs must match the length of dataset.")
    return list(zip(names, weights))


__all__ = ["OXERLDSDataset", "load_dataset"]
