"""Lightweight CLI launcher for VLAFactory."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Union

from omegaconf import OmegaConf

from vlafactory.configs import _parse_eval_args, _parse_infer_args, _parse_train_args
from vlafactory.train.sft.workflow import run_sft


ArgsLike = Union[Dict[str, Any], List[str], None]


def read_args(args: ArgsLike = None) -> Union[Dict[str, Any], List[str]]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    if len(sys.argv) < 2:
        return []

    command = sys.argv[1]
    start_idx = 2 if command in {"train", "eval", "inference"} else 1
    if len(sys.argv) > start_idx and sys.argv[start_idx].endswith((".yaml", ".yml", ".json")):
        override_config = OmegaConf.from_cli(sys.argv[start_idx + 1 :])
        dict_config = OmegaConf.load(Path(sys.argv[start_idx]).absolute())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))

    return sys.argv[start_idx:]


def main(args: ArgsLike = None) -> None:
    if args is None:
        if len(sys.argv) < 2:
            raise ValueError("Expected a command or config path.")
        command = sys.argv[1]
    elif isinstance(args, list) and args:
        command = args[0]
    else:
        command = "train"

    parsed = read_args(args)
    if command not in {"train", "eval", "inference"} and isinstance(parsed, dict):
        command = "train"

    if args is not None and isinstance(parsed, list) and command in {"train", "eval", "inference"}:
        parsed = parsed[1:]

    if command == "train":
        model_args, data_args, training_args, finetuning_args, generating_args = _parse_train_args(parsed)
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args)
        return

    if command == "eval":
        _parse_eval_args(parsed)
        return

    if command == "inference":
        _parse_infer_args(parsed)
        return

    raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
