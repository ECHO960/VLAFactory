"""Core model components for VLAFactory."""

from typing import Any

from vlafactory.model.vla_model import VLAModel


def load_model(
    model_args: Any,
    finetuning_args: Any = None,
    training_args: Any = None,
) -> VLAModel:
    return VLAModel(
        language_model=None,
        vision_encoder=None,
        action_dim=model_args.action_dim,
        hidden_dim=model_args.hidden_dim,
    )


__all__ = ["VLAModel", "load_model"]
