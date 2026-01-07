"""Seq2Seq trainer for VLA SFT."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from transformers import Seq2SeqTrainer


class VLASFTTrainer(Seq2SeqTrainer):
    """Minimal Seq2SeqTrainer wrapper for VLA training."""

    def __init__(
        self,
        *args: Any,
        label_names: Optional[Tuple[str, ...]] = None,
        **kwargs: Any,
    ) -> None:
        if label_names is None:
            label_names = ("labels", "action_labels")
        super().__init__(*args, label_names=list(label_names), **kwargs)

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ) -> Any:
        outputs = model(**inputs)
        loss = outputs.get("loss")
        if loss is None:
            raise ValueError("Model output did not include 'loss'.")
        return (loss, outputs) if return_outputs else loss
