"""SFT workflow for VLA models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from transformers import TrainingArguments

from vlafactory.train.sft.trainer import VLASFTTrainer
from vlafactory.data import load_dataset
from vlafactory.model import load_model


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
) -> Tuple[VLASFTTrainer, Dict[str, Any]]:
    """Run a minimal SFT workflow for VLA models."""

    # build dataset and data loader
    model = load_model(model_args, training_args)
    datasets = load_dataset(data_args)

    data_collator = getattr(datasets["train"], "collate_fn", None)
    if data_collator is None:
        data_collator = datasets.get("collator")

    trainer = VLASFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=None,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=None,
    )

    metrics: Dict[str, Any] = {}

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics.update(train_result.metrics)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval and datasets["eval"] is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    return trainer, metrics
