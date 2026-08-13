from __future__ import annotations

import json
import sys
from unittest import mock

import pytest
import yaml

from onebee.training.dpo import (
    DPOTrainingConfig,
    _build_parser,
    _load_jsonl,
    build_lora_config,
    build_training_arguments,
    load_dpo_config,
    main,
    run_dpo,
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


def _make_fake_trl_module():
    """Return a fake trl module with a stub DPOConfig class."""
    fake_trl = mock.MagicMock()

    class FakeDPOConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_trl.DPOConfig = FakeDPOConfig
    return fake_trl


def _make_fake_trl_module_no_warmup_ratio():
    """Mimics trl versions that dropped warmup_ratio from DPOConfig — raises
    TypeError on that kwarg, accepts warmup_steps."""
    fake_trl = mock.MagicMock()

    class FakeDPOConfigNoRatio:
        def __init__(self, **kwargs):
            if "warmup_ratio" in kwargs:
                raise TypeError(
                    "DPOConfig.__init__() got an unexpected keyword argument 'warmup_ratio'"
                )
            self.kwargs = kwargs
            for k, v in kwargs.items():
                setattr(self, k, v)

    fake_trl.DPOConfig = FakeDPOConfigNoRatio
    return fake_trl


# ---------------------------------------------------------------------------
# DPOTrainingConfig defaults
# ---------------------------------------------------------------------------


class TestDPOTrainingConfigDefaults:
    def test_base_model_required(self):
        with pytest.raises(ValueError):
            DPOTrainingConfig()  # type: ignore[call-arg]

    def test_defaults_match_spec(self):
        cfg = DPOTrainingConfig(
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
        assert cfg.learning_rate == 5e-6
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.warmup_ratio == 0.03
        assert cfg.num_train_epochs == 1.0
        assert cfg.max_seq_length == 2048
        assert cfg.per_device_train_batch_size == 4
        assert cfg.gradient_accumulation_steps == 2
        assert cfg.bf16 is True
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"
        assert cfg.seed == 1337
        assert cfg.report_to == "none"
        assert cfg.run_name is None


# ---------------------------------------------------------------------------
# load_dpo_config
# ---------------------------------------------------------------------------


class TestLoadDPOConfig:
    def test_round_trip(self, tmp_path):
        yaml_path = tmp_path / "cfg.yaml"
        content = {
            "base_model": "test/model",
            "train_file": "data/train.jsonl",
            "val_file": "data/val.jsonl",
            "output_dir": "outputs/test",
            "lora_r": 8,
            "beta": 0.05,
            "run_name": "my-run",
        }
        yaml_path.write_text(yaml.dump(content))
        cfg = load_dpo_config(str(yaml_path))
        assert cfg.base_model == "test/model"
        assert cfg.lora_r == 8
        assert cfg.beta == 0.05
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
            load_dpo_config(str(yaml_path))


# ---------------------------------------------------------------------------
# build_lora_config — isolated via fake peft module
# ---------------------------------------------------------------------------


class TestBuildLoraConfig:
    def test_all_linear_passes_string(self):
        fake_peft = _make_fake_peft_module()
        with mock.patch.dict(sys.modules, {"peft": fake_peft}):
            cfg = DPOTrainingConfig(
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
            cfg = DPOTrainingConfig(
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
            cfg = DPOTrainingConfig(
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
            cfg = DPOTrainingConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="o",
                lora_target_modules="q_proj",
            )
            result = build_lora_config(cfg)
        assert result.target_modules == ["q_proj"]


# ---------------------------------------------------------------------------
# build_training_arguments — isolated via fake trl module
# ---------------------------------------------------------------------------


class TestBuildTrainingArguments:
    def test_maps_fields_correctly(self):
        fake_trl = _make_fake_trl_module()
        with mock.patch.dict(sys.modules, {"trl": fake_trl}):
            cfg = DPOTrainingConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="/tmp/out",
                learning_rate=5e-6,
                lr_scheduler_type="linear",
                warmup_ratio=0.1,
                num_train_epochs=1.0,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
                bf16=False,
                beta=0.2,
                loss_type="ipo",
                max_seq_length=4096,
                seed=42,
                report_to="none",
                run_name="test-run",
            )
            result = build_training_arguments(cfg)
        assert result.output_dir == "/tmp/out"
        assert result.learning_rate == 5e-6
        assert result.lr_scheduler_type == "linear"
        assert result.warmup_ratio == 0.1
        assert result.num_train_epochs == 1.0
        assert result.per_device_train_batch_size == 4
        assert result.gradient_accumulation_steps == 2
        assert result.bf16 is False
        assert result.beta == 0.2
        assert result.loss_type == "ipo"
        assert result.max_length == 4096
        assert result.seed == 42
        assert result.report_to == ["none"]
        assert result.run_name == "test-run"

    def test_falls_back_to_warmup_steps_when_ratio_unsupported(self):
        # Regression test: some trl/transformers versions dropped warmup_ratio
        # from DPOConfig entirely — caught before spending real training time.
        fake_trl = _make_fake_trl_module_no_warmup_ratio()
        with mock.patch.dict(sys.modules, {"trl": fake_trl}):
            cfg = DPOTrainingConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="/tmp/out",
                warmup_ratio=0.1,
                num_train_epochs=1.0,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=2,
            )
            # 202 examples, effective batch 8 -> 25 steps/epoch * 1 epoch = 25 steps
            # -> warmup_steps = int(0.1 * 25) = 2
            result = build_training_arguments(cfg, num_training_examples=202)
        assert not hasattr(result, "warmup_ratio") or "warmup_ratio" not in result.kwargs
        assert result.kwargs["warmup_steps"] == 2

    def test_falls_back_with_zero_warmup_steps_when_no_example_count(self):
        fake_trl = _make_fake_trl_module_no_warmup_ratio()
        with mock.patch.dict(sys.modules, {"trl": fake_trl}):
            cfg = DPOTrainingConfig(
                base_model="m", train_file="t.jsonl", val_file="v.jsonl", output_dir="o"
            )
            result = build_training_arguments(cfg)
        assert result.kwargs["warmup_steps"] == 0

    def test_report_to_is_list(self):
        fake_trl = _make_fake_trl_module()
        with mock.patch.dict(sys.modules, {"trl": fake_trl}):
            cfg = DPOTrainingConfig(
                base_model="m",
                train_file="t.jsonl",
                val_file="v.jsonl",
                output_dir="o",
                report_to="wandb",
            )
            result = build_training_arguments(cfg)
        assert result.report_to == ["wandb"]


# ---------------------------------------------------------------------------
# run_dpo — injectable factories, dry-run, no real imports
# ---------------------------------------------------------------------------


class TestRunDPO:
    def test_dry_run_with_fakes_completes(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text(json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n")
        val_path.write_text(json.dumps({"prompt": "x", "chosen": "y", "rejected": "z"}) + "\n")

        cfg = DPOTrainingConfig(
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
            return tok

        def fake_trainer_factory(**kwargs):
            trainer_calls.append(kwargs)
            return mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {"peft": _make_fake_peft_module(), "trl": _make_fake_trl_module()},
        ):
            run_dpo(
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

    def test_fake_callables_receive_expected_data(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text(json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n")
        val_path.write_text(json.dumps({"prompt": "x", "chosen": "y", "rejected": "z"}) + "\n")

        cfg = DPOTrainingConfig(
            base_model="my-model",
            base_model_revision="dev",
            train_file=str(train_path),
            val_file=str(val_path),
            output_dir=str(tmp_path / "output"),
            max_seq_length=1024,
        )

        trainer_kwargs: dict = {}

        def fake_model_loader(model_name, revision):
            return mock.MagicMock()

        def fake_tokenizer_loader(model_name, revision):
            tok = mock.MagicMock()
            tok.pad_token_id = 0
            tok.eos_token_id = 0
            return tok

        def fake_trainer_factory(**kwargs):
            nonlocal trainer_kwargs
            trainer_kwargs = kwargs
            return mock.MagicMock()

        with mock.patch.dict(
            sys.modules,
            {"peft": _make_fake_peft_module(), "trl": _make_fake_trl_module()},
        ):
            run_dpo(
                cfg,
                dry_run=True,
                model_loader=fake_model_loader,
                tokenizer_loader=fake_tokenizer_loader,
                trainer_factory=fake_trainer_factory,
            )

        assert len(trainer_kwargs["train_dataset"]) == 1
        assert trainer_kwargs["train_dataset"][0]["prompt"] == "q"
        assert trainer_kwargs["train_dataset"][0]["chosen"] == "a"
        assert trainer_kwargs["train_dataset"][0]["rejected"] == "b"
        assert len(trainer_kwargs["eval_dataset"]) == 1
        assert trainer_kwargs["eval_dataset"][0]["prompt"] == "x"
        assert trainer_kwargs["eval_dataset"][0]["chosen"] == "y"
        assert trainer_kwargs["eval_dataset"][0]["rejected"] == "z"


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

    def test_main_help_exits_cleanly(self):
        with mock.patch("sys.argv", ["dpo", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Validate shipped config file
# ---------------------------------------------------------------------------


class TestShippedConfig:
    def test_dpo_yaml_loads_and_matches_defaults(self):
        import os as _os

        cfg_path = _os.path.join(
            _os.path.dirname(__file__), "..", "..", "configs", "training", "dpo.yaml"
        )
        cfg = load_dpo_config(cfg_path)
        # Chains DPO on top of the Day-4 SFT checkpoint.
        assert cfg.base_model == "outputs/sft/v0/merged"
        assert cfg.base_model_revision == "main"
        assert cfg.train_file == "data/dpo/v0/train.jsonl"
        assert cfg.val_file == "data/dpo/v0/val.jsonl"
        assert cfg.output_dir == "outputs/dpo/v0"
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.lora_target_modules == "all-linear"
        assert cfg.learning_rate == 5e-6
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.warmup_ratio == 0.03
        assert cfg.num_train_epochs == 1.0
        assert cfg.max_seq_length == 2048
        assert cfg.per_device_train_batch_size == 4
        assert cfg.gradient_accumulation_steps == 2
        assert cfg.bf16 is True
        assert cfg.beta == 0.1
        assert cfg.loss_type == "sigmoid"
        assert cfg.seed == 1337
        assert cfg.report_to == "none"
        assert cfg.run_name is None
