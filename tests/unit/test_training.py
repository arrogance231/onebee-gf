from __future__ import annotations

import json
import sys
from unittest import mock

import pytest
import yaml

from onebee.training.sft import (
    SFTConfig,
    _build_parser,
    _load_jsonl,
    build_lora_config,
    build_training_arguments,
    effective_batch_size,
    load_sft_config,
    run_sft,
)

# ---------------------------------------------------------------------------
# Fake / stub helpers
# ---------------------------------------------------------------------------


def _make_fake_peft_module():
    """Return a fake peft module with a stub LoraConfig class."""
    fake_peft = mock.MagicMock()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_peft.LoraConfig = FakeLoraConfig
    return fake_peft


def _make_fake_transformers_module():
    """Return a fake transformers module with a stub TrainingArguments class."""
    fake_tr = mock.MagicMock()

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_tr.TrainingArguments = FakeTrainingArguments
    return fake_tr


# ---------------------------------------------------------------------------
# SFTConfig defaults
# ---------------------------------------------------------------------------


class TestSFTConfigDefaults:
    def test_base_model_required(self):
        with pytest.raises(ValueError):
            SFTConfig()  # type: ignore[call-arg]

    def test_defaults_match_day4_spec(self):
        cfg = SFTConfig(
            base_model="foo",
            train_file="train.jsonl",
            val_file="val.jsonl",
            output_dir="out",
        )
        assert cfg.base_model_revision == "main"
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.lora_target_modules == "all-linear"
        assert cfg.learning_rate == 1e-4
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.warmup_ratio == 0.03
        assert cfg.num_train_epochs == 2.0
        assert cfg.max_seq_length == 2048
        assert cfg.per_device_train_batch_size == 8
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.bf16 is True
        assert cfg.packing is False
        assert cfg.neftune_noise_alpha is None
        assert cfg.seed == 1337
        assert cfg.report_to == "wandb"
        assert cfg.run_name is None


# ---------------------------------------------------------------------------
# load_sft_config
# ---------------------------------------------------------------------------


class TestLoadSFTConfig:
    def test_round_trip(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        content = {
            "base_model": "test/model",
            "train_file": "data/train.jsonl",
            "val_file": "data/val.jsonl",
            "output_dir": "outputs/test",
            "lora_r": 8,
            "run_name": "my-run",
        }
        yaml_path.write_text(yaml.dump(content))
        cfg = load_sft_config(str(yaml_path))
        assert cfg.base_model == "test/model"
        assert cfg.lora_r == 8
        assert cfg.run_name == "my-run"
        assert cfg.lora_alpha == 32  # default

    def test_missing_required_field_raises(self, tmp_path):
        yaml_path = tmp_path / "bad.yaml"
        # missing output_dir (required field)
        content = {
            "base_model": "test/model",
            "train_file": "data/train.jsonl",
            "val_file": "data/val.jsonl",
        }
        yaml_path.write_text(yaml.dump(content))
        with pytest.raises(Exception):
            load_sft_config(str(yaml_path))


# ---------------------------------------------------------------------------
# effective_batch_size
# ---------------------------------------------------------------------------


class TestEffectiveBatchSize:
    def test_single_device(self):
        cfg = SFTConfig(
            base_model="m",
            train_file="t.jsonl",
            val_file="v.jsonl",
            output_dir="o",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
        )
        assert effective_batch_size(cfg) == 32

    def test_multi_device(self):
        cfg = SFTConfig(
            base_model="m",
            train_file="t.jsonl",
            val_file="v.jsonl",
            output_dir="o",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
        )
        assert effective_batch_size(cfg, num_devices=4) == 128

    def test_non_default_values(self):
        cfg = SFTConfig(
            base_model="m",
            train_file="t.jsonl",
            val_file="v.jsonl",
            output_dir="o",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
        )
        assert effective_batch_size(cfg) == 32


# ---------------------------------------------------------------------------
# build_lora_config — isolated via fake peft module
# ---------------------------------------------------------------------------


class TestBuildLoraConfig:
    def test_all_linear_passes_string(self):
        fake_peft = _make_fake_peft_module()
        with mock.patch.dict(sys.modules, {"peft": fake_peft}):
            cfg = SFTConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="o",
                lora_target_modules="all-linear",
            )
            result = build_lora_config(cfg)
        assert result.target_modules == "all-linear"
        assert result.r == 16
        assert result.lora_alpha == 32
        assert result.lora_dropout == 0.05
        assert result.task_type == "CAUSAL_LM"

    def test_comma_separated_becomes_list(self):
        fake_peft = _make_fake_peft_module()
        with mock.patch.dict(sys.modules, {"peft": fake_peft}):
            cfg = SFTConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="o",
                lora_target_modules="q_proj,v_proj,k_proj",
            )
            result = build_lora_config(cfg)
        assert result.target_modules == ["q_proj", "v_proj", "k_proj"]

    def test_comma_separated_with_spaces_gets_stripped(self):
        fake_peft = _make_fake_peft_module()
        with mock.patch.dict(sys.modules, {"peft": fake_peft}):
            cfg = SFTConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="o",
                lora_target_modules=" q_proj , v_proj , k_proj ",
            )
            result = build_lora_config(cfg)
        assert result.target_modules == ["q_proj", "v_proj", "k_proj"]

    def test_single_module_as_non_all_linear(self):
        fake_peft = _make_fake_peft_module()
        with mock.patch.dict(sys.modules, {"peft": fake_peft}):
            cfg = SFTConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="o",
                lora_target_modules="q_proj",
            )
            result = build_lora_config(cfg)
        assert result.target_modules == ["q_proj"]


# ---------------------------------------------------------------------------
# build_training_arguments — isolated via fake transformers module
# ---------------------------------------------------------------------------


class TestBuildTrainingArguments:
    def test_maps_fields_correctly(self):
        fake_tr = _make_fake_transformers_module()
        with mock.patch.dict(sys.modules, {"transformers": fake_tr}):
            cfg = SFTConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="/tmp/out",
                learning_rate=5e-5,
                lr_scheduler_type="linear",
                warmup_ratio=0.1,
                num_train_epochs=3.0,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                bf16=False,
                seed=42,
                report_to="none",
                run_name="test-run",
            )
            result = build_training_arguments(cfg)
        assert result.output_dir == "/tmp/out"
        assert result.learning_rate == 5e-5
        assert result.lr_scheduler_type == "linear"
        assert result.warmup_ratio == 0.1
        assert result.num_train_epochs == 3.0
        assert result.per_device_train_batch_size == 4
        assert result.gradient_accumulation_steps == 8
        assert result.bf16 is False
        assert result.seed == 42
        assert result.report_to == ["none"]
        assert result.run_name == "test-run"

    def test_report_to_is_list(self):
        fake_tr = _make_fake_transformers_module()
        with mock.patch.dict(sys.modules, {"transformers": fake_tr}):
            cfg = SFTConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="o",
                report_to="wandb",
            )
            result = build_training_arguments(cfg)
        assert result.report_to == ["wandb"]


# ---------------------------------------------------------------------------
# run_sft — injectable factories, dry-run, no real imports
# ---------------------------------------------------------------------------


class TestRunSFT:
    def test_dry_run_with_fakes_completes(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
        val_path.write_text(json.dumps({"messages": [{"role": "user", "content": "hello"}]}) + "\n")

        cfg = SFTConfig(
            base_model="test-model",
            train_file=str(train_path),
            val_file=str(val_path),
            output_dir=str(tmp_path / "output"),
        )

        model_calls = []
        tokenizer_calls = []
        trainer_calls = []

        def fake_model_loader(model_name, revision):
            model_calls.append((model_name, revision))
            return mock.MagicMock()

        def fake_tokenizer_loader(model_name, revision):
            tokenizer_calls.append((model_name, revision))
            tok = mock.MagicMock()
            tok.pad_token_id = None
            tok.eos_token_id = 0
            tok.apply_chat_template.return_value = "formatted text"
            return tok

        def fake_trainer_factory(**kwargs):
            trainer_calls.append(kwargs)
            return mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {"peft": _make_fake_peft_module(), "transformers": _make_fake_transformers_module()},
        ):
            run_sft(
                cfg,
                dry_run=True,
                model_loader=fake_model_loader,
                tokenizer_loader=fake_tokenizer_loader,
                trainer_factory=fake_trainer_factory,
            )

        assert len(model_calls) == 1
        assert model_calls[0] == ("test-model", "main")

        assert len(tokenizer_calls) == 1
        assert tokenizer_calls[0] == ("test-model", "main")

        assert len(trainer_calls) == 1
        tc = trainer_calls[0]
        assert tc["max_seq_length"] == 2048
        assert tc["packing"] is False
        assert tc["neftune_noise_alpha"] is None
        assert tc["formatting_func"] is not None

    def test_fake_callables_receive_expected_data(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text(json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n")
        val_path.write_text(json.dumps({"messages": [{"role": "user", "content": "a"}]}) + "\n")

        cfg = SFTConfig(
            base_model="my-model",
            base_model_revision="dev",
            train_file=str(train_path),
            val_file=str(val_path),
            output_dir=str(tmp_path / "output"),
            max_seq_length=1024,
            packing=True,
            neftune_noise_alpha=5.0,
        )

        trainer_kwargs: dict = {}

        def fake_model_loader(model_name, revision):
            return mock.MagicMock()

        def fake_tokenizer_loader(model_name, revision):
            tok = mock.MagicMock()
            tok.pad_token_id = 0
            tok.eos_token_id = 0
            tok.apply_chat_template.return_value = "fmt"
            return tok

        def fake_trainer_factory(**kwargs):
            nonlocal trainer_kwargs
            trainer_kwargs = kwargs
            return mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {"peft": _make_fake_peft_module(), "transformers": _make_fake_transformers_module()},
        ):
            run_sft(
                cfg,
                dry_run=True,
                model_loader=fake_model_loader,
                tokenizer_loader=fake_tokenizer_loader,
                trainer_factory=fake_trainer_factory,
            )

        assert len(trainer_kwargs["train_dataset"]) == 1
        assert trainer_kwargs["train_dataset"][0]["messages"][0]["content"] == "q"
        assert len(trainer_kwargs["eval_dataset"]) == 1
        assert trainer_kwargs["eval_dataset"][0]["messages"][0]["content"] == "a"
        assert trainer_kwargs["max_seq_length"] == 1024
        assert trainer_kwargs["packing"] is True
        assert trainer_kwargs["neftune_noise_alpha"] == 5.0


# ---------------------------------------------------------------------------
# _load_jsonl
# ---------------------------------------------------------------------------


class TestLoadJSONL:
    def test_loads_valid_jsonl(self, tmp_path):
        p = tmp_path / "data.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n\n{"c": 3}\n')
        result = _load_jsonl(str(p))
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = _load_jsonl(str(p))
        assert result == []

    def test_only_blank_lines(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text("\n\n  \n")
        result = _load_jsonl(str(p))
        assert result == []


# ---------------------------------------------------------------------------
# CLI / argparse
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help_does_not_require_torch(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Validate shipped config file
# ---------------------------------------------------------------------------


class TestShippedConfig:
    def test_sft_yaml_loads_and_matches_defaults(self):
        import os as _os

        cfg_path = _os.path.join(
            _os.path.dirname(__file__), "..", "..", "configs", "training", "sft.yaml"
        )
        cfg = load_sft_config(cfg_path)
        assert cfg.base_model == "Qwen/Qwen3-1.7B-Instruct"
        assert cfg.base_model_revision == "main"
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.lora_target_modules == "all-linear"
        assert cfg.learning_rate == 1e-4
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.warmup_ratio == 0.03
        assert cfg.num_train_epochs == 2.0
        assert cfg.max_seq_length == 2048
        assert cfg.per_device_train_batch_size == 8
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.bf16 is True
        assert cfg.packing is False
        assert cfg.neftune_noise_alpha is None
        assert cfg.seed == 1337
        assert cfg.report_to == "wandb"
        assert cfg.run_name is None
