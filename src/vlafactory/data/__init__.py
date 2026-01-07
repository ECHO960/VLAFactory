"""Data handling components for VLA training."""

from typing import Any, Dict
import os
import torch

from vlafactory.data.vla_dataset import VLADataset


def load_dataset(
    data_args: Any,
) -> Dict[str, Any]:
    data_path = data_args.data_path

    max_length = data_args.max_length or data_args.cutoff_len
    dataset = VLADataset(
        data_path=data_path,
        tokenizer=None,
        max_length=max_length,
        action_dim=model_args.action_dim,
    )

    train_split = getattr(data_args, "train_split", 0.9)
    train_size = int(len(dataset) * train_split)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    return {
        "train": train_dataset,
        "eval": eval_dataset if eval_size > 0 else None,
        "collator": VLADataset.collate_fn,
    }


__all__ = ["VLADataset", "load_dataset"]
