"""SFT workflow for VLA models."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase, TrainingArguments

from vlafactory.train.sft.trainer import VLASFTTrainer


def _build_training_args(
    training_args: Optional[Union[TrainingArguments, Mapping[str, Any]]],
) -> TrainingArguments:
    if isinstance(training_args, TrainingArguments):
        args = training_args
    else:
        args_dict: Dict[str, Any] = dict(training_args or {})
        args_dict.setdefault("output_dir", "./output")
        args_dict.setdefault("remove_unused_columns", False)
        args_dict.setdefault("logging_steps", 10)
        args = TrainingArguments(**args_dict)
    if args.remove_unused_columns:
        args.remove_unused_columns = False
    return args


def train_sft(
    model: Any,
    tokenizer: Optional[PreTrainedTokenizerBase],
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    training_args: Optional[Union[TrainingArguments, Mapping[str, Any]]] = None,
    data_collator: Optional[Any] = None,
    compute_metrics: Optional[Any] = None,
    callbacks: Optional[Any] = None,
    do_train: bool = True,
    do_eval: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> Tuple[VLASFTTrainer, Dict[str, Any]]:
    """Run a minimal SFT training workflow for VLA models."""

    args = _build_training_args(training_args)

    trainer = VLASFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    metrics: Dict[str, Any] = {}

    if do_train:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics.update(train_result.metrics)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if do_eval and eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    return trainer, metrics
