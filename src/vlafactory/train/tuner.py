"""Minimal training launcher for VLAFactory."""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict

import torch
from transformers import TrainingArguments

from vlafactory.configs import get_default_config, load_config
from vlafactory.data.vla_dataset import VLADataset
from vlafactory.model.vla_model import VLAModel
from vlafactory.train.sft.workflow import train_sft


def _filter_training_args(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    signature = inspect.signature(TrainingArguments.__init__)
    valid_keys = set(signature.parameters.keys())
    valid_keys.discard("self")
    return {key: value for key, value in raw_args.items() if key in valid_keys}


def _build_datasets(config: Dict[str, Any], use_dummy: bool) -> Dict[str, Any]:
    data_path = config["data"]["data_path"]
    if use_dummy:
        output_dir = config["training"].get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        data_path = VLADataset.create_dummy_dataset(
            num_samples=100,
            save_path=os.path.join(output_dir, "dummy_data.json"),
        )
        config["data"]["data_path"] = data_path

    dataset = VLADataset(
        data_path=data_path,
        tokenizer=None,
        max_length=config["data"]["max_length"],
        action_dim=config["model"]["action_dim"],
    )

    train_split = config["data"].get("train_split", 0.9)
    train_size = int(len(dataset) * train_split)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    return {"train": train_dataset, "eval": eval_dataset if eval_size > 0 else None}


def _build_model(config: Dict[str, Any]) -> VLAModel:
    return VLAModel(
        language_model=None,
        vision_encoder=None,
        action_dim=config["model"]["action_dim"],
        hidden_dim=config["model"]["hidden_dim"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLAFactory training launcher")
    parser.add_argument("--stage", type=str, default="sft")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.stage != "sft":
        raise ValueError(f"Unsupported stage: {args.stage}")

    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    if args.data_path:
        config["data"]["data_path"] = args.data_path
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.num_train_epochs is not None:
        config["training"]["num_train_epochs"] = args.num_train_epochs
    if args.per_device_train_batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.action_dim is not None:
        config["model"]["action_dim"] = args.action_dim
    if args.max_length is not None:
        config["data"]["max_length"] = args.max_length

    datasets = _build_datasets(config, args.use_dummy_data)
    model = _build_model(config)

    training_args = _filter_training_args(config["training"])
    train_sft(
        model=model,
        tokenizer=None,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        training_args=training_args,
        data_collator=VLADataset.collate_fn,
        do_train=True,
        do_eval=args.do_eval,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
