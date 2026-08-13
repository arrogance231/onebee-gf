from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from typing import Any

import yaml
from pydantic import BaseModel


class DPOTrainingConfig(BaseModel):
    base_model: str
    base_model_revision: str = "main"
    train_file: str
    val_file: str
    output_dir: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"
    # DPO typically uses a much lower LR than SFT — standard convention.
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    # DPO typically 1 epoch, not 2 (SFT).
    num_train_epochs: float = 1.0
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    bf16: bool = True
    # DPO's KL-penalty coefficient.
    beta: float = 0.1
    # The standard DPO loss.
    loss_type: str = "sigmoid"
    seed: int = 1337
    report_to: str = "none"
    run_name: str | None = None


def load_dpo_config(path: str) -> DPOTrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DPOTrainingConfig(**raw)


def build_lora_config(config: DPOTrainingConfig):
    from peft import LoraConfig

    # "all-linear" is supported directly by recent peft versions as a single string.
    # Any other value is treated as a comma-separated list of module names.
    if config.lora_target_modules == "all-linear":
        target_modules: str | list[str] = "all-linear"
    else:
        target_modules = [m.strip() for m in config.lora_target_modules.split(",")]

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )


def build_training_arguments(config: DPOTrainingConfig, num_training_examples: int | None = None):
    from trl import DPOConfig

    base_kwargs = dict(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        bf16=config.bf16,
        seed=config.seed,
        report_to=[config.report_to],
        run_name=config.run_name,
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_seq_length,
    )
    try:
        return DPOConfig(warmup_ratio=config.warmup_ratio, **base_kwargs)
    except TypeError:
        # Some trl/transformers versions dropped warmup_ratio from DPOConfig
        # entirely (docs/model_quirks.md) — compute an equivalent warmup_steps
        # count instead.
        warmup_steps = 0
        if num_training_examples is not None:
            effective_batch = (
                config.per_device_train_batch_size * config.gradient_accumulation_steps
            )
            steps_per_epoch = max(1, num_training_examples // effective_batch)
            total_steps = steps_per_epoch * int(config.num_train_epochs)
            warmup_steps = max(0, int(config.warmup_ratio * total_steps))
        return DPOConfig(warmup_steps=warmup_steps, **base_kwargs)


def _load_jsonl(path: str) -> list[dict]:
    data: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def run_dpo(
    config: DPOTrainingConfig,
    dry_run: bool = False,
    model_loader: Callable[[str, str], Any] | None = None,
    tokenizer_loader: Callable[[str, str], Any] | None = None,
    trainer_factory: Callable[..., Any] | None = None,
) -> None:
    if model_loader is None:

        def model_loader(model_name: str, revision: str):
            import torch
            from transformers import AutoModelForCausalLM

            # Multimodal models (e.g. gemma4-e2b) need AutoModelForImageTextToText,
            # not AutoModelForCausalLM (same fix as HFEngine.load(), see
            # docs/model_quirks.md #3) — try it first, fall back for text-only models.
            # `dtype` (not the deprecated `torch_dtype`) per transformers 5.15.0+
            # (docs/model_quirks.md #2).
            try:
                from transformers import AutoModelForImageTextToText

                return AutoModelForImageTextToText.from_pretrained(
                    model_name, revision=revision, dtype=torch.bfloat16
                )
            except Exception:
                return AutoModelForCausalLM.from_pretrained(
                    model_name, revision=revision, dtype=torch.bfloat16
                )

    if tokenizer_loader is None:

        def tokenizer_loader(model_name: str, revision: str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer

    if trainer_factory is None:

        def trainer_factory(**kwargs):
            from datasets import Dataset
            from trl import DPOTrainer

            # trl requires an actual datasets.Dataset, not a plain list of dicts
            # (our own _load_jsonl loader intentionally returns a plain list — see
            # its docstring — so convert only here, at the real-trl boundary).
            # Modern trl wants `processing_class` instead of the old `tokenizer`
            # kwarg. `args` here is already a real trl.DPOConfig (built by
            # build_training_arguments), so no re-construction needed.
            if "tokenizer" in kwargs:
                kwargs["processing_class"] = kwargs.pop("tokenizer")
            if isinstance(kwargs.get("train_dataset"), list):
                kwargs["train_dataset"] = Dataset.from_list(kwargs["train_dataset"])
            if isinstance(kwargs.get("eval_dataset"), list):
                kwargs["eval_dataset"] = Dataset.from_list(kwargs["eval_dataset"])

            return DPOTrainer(**kwargs)

    model = model_loader(config.base_model, config.base_model_revision)
    tokenizer = tokenizer_loader(config.base_model, config.base_model_revision)

    # Note: JSONL loading via stdlib json — a Week-1 simplification vs. using HF
    # datasets for streaming or larger corpora later.
    train_dataset = _load_jsonl(config.train_file)
    val_dataset = _load_jsonl(config.val_file)

    lora_config = build_lora_config(config)
    training_args = build_training_arguments(config, num_training_examples=len(train_dataset))

    trainer = trainer_factory(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    if dry_run:
        return

    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, "adapter"))
    merged_model = trainer.model.merge_and_unload()
    merged_dir = os.path.join(config.output_dir, "merged")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    # For multimodal base models, saving only the tokenizer drops the image
    # processor config (preprocessor_config.json), silently degrading the merged
    # checkpoint to text-only — caught via a real training run. Also save the full
    # AutoProcessor when the base model has one; no-op (caught) for text-only models.
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            config.base_model, revision=config.base_model_revision
        )
        processor.save_pretrained(merged_dir)
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO training")
    parser.add_argument("--config", required=True, help="Path to YAML DPO config file")
    parser.add_argument("--dry-run", action="store_true", help="Build everything but skip training")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from config")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = load_dpo_config(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    run_dpo(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
