from __future__ import annotations

import json
import time

import pytest

from onebee.inference.bench import BenchResult, run_latency_bench, save_bench_results
from onebee.inference.engine import (
    GenerationConfig,
    GenerationResult,
    Generator,
    HFEngine,
    LlamaCppEngine,
)


class FakeEngine:
    @property
    def name(self) -> str:
        return "fake-engine"

    def load(self) -> None:
        pass

    def apply_chat_template(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def generate(self, messages: list[dict], config: GenerationConfig) -> GenerationResult:
        time.sleep(0.001)
        prompt = self.apply_chat_template(messages)
        prompt_tokens = len(prompt.split())
        completion_tokens = config.max_new_tokens
        return GenerationResult(
            text="fake " * completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=5.0,
            total_ms=50.0,
            tokens_per_sec=completion_tokens / 0.05,
        )


class TestFakeEngine:
    def test_implements_generator_protocol(self):
        engine = FakeEngine()
        assert isinstance(engine, Generator)

    def test_apply_chat_template(self):
        engine = FakeEngine()
        result = engine.apply_chat_template(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        )
        assert "user: hello" in result
        assert "assistant: hi there" in result

    def test_generate_returns_result(self):
        engine = FakeEngine()
        engine.load()
        config = GenerationConfig(max_new_tokens=64)
        result = engine.generate([{"role": "user", "content": "hello"}], config)
        assert isinstance(result, GenerationResult)
        assert result.completion_tokens == 64
        assert result.ttft_ms == 5.0
        assert result.total_ms == 50.0


class TestGenerationConfig:
    def test_defaults(self):
        config = GenerationConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.05
        assert config.max_new_tokens == 256
        assert config.seed is None
        assert config.deterministic is False

    def test_custom_values(self):
        config = GenerationConfig(
            temperature=0.5,
            top_p=0.95,
            max_new_tokens=128,
            seed=42,
            deterministic=True,
        )
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.max_new_tokens == 128
        assert config.seed == 42
        assert config.deterministic is True

    def test_serialization(self):
        config = GenerationConfig(max_new_tokens=100)
        data = config.model_dump()
        loaded = GenerationConfig(**data)
        assert loaded == config

    def test_seed_can_be_none(self):
        config = GenerationConfig(seed=None)
        assert config.seed is None

    def test_seed_can_be_int(self):
        config = GenerationConfig(seed=12345)
        assert config.seed == 12345


class TestGenerationResult:
    def test_create_result(self):
        result = GenerationResult(
            text="hello world",
            prompt_tokens=10,
            completion_tokens=5,
            ttft_ms=100.0,
            total_ms=500.0,
            tokens_per_sec=10.0,
        )
        assert result.text == "hello world"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.tokens_per_sec == 10.0


class TestHFEngineBasic:
    def test_can_import_and_instantiate_without_torch(self):
        engine = HFEngine(model_name="gpt2")
        assert engine.name == "hf:gpt2"
        assert engine.model_name == "gpt2"
        assert engine.revision == "main"
        assert engine._loaded is False

    def test_apply_chat_template_without_tokenizer(self):
        engine = HFEngine(model_name="gpt2")
        result = engine.apply_chat_template(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        assert "user: hello" in result
        assert "assistant: hi" in result

    def test_generate_raises_without_load(self):
        engine = HFEngine(model_name="gpt2")
        with pytest.raises(RuntimeError, match=r"load\(\) must be called before generate"):
            engine.generate(
                [{"role": "user", "content": "hello"}],
                GenerationConfig(max_new_tokens=1),
            )


class TestLlamaCppEngineBasic:
    def test_can_import_and_instantiate_without_llama_cpp(self):
        engine = LlamaCppEngine(model_path="/nonexistent/model.gguf")
        assert engine.name == "llama-cpp:/nonexistent/model.gguf"
        assert engine._loaded is False

    def test_apply_chat_template(self):
        engine = LlamaCppEngine(model_path="/nonexistent/model.gguf")
        result = engine.apply_chat_template(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        assert "user: hello" in result
        assert "assistant: hi" in result

    def test_generate_raises_without_load(self):
        engine = LlamaCppEngine(model_path="/nonexistent/model.gguf")
        with pytest.raises(RuntimeError, match=r"load\(\) must be called before generate"):
            engine.generate(
                [{"role": "user", "content": "hello"}],
                GenerationConfig(max_new_tokens=1),
            )


class TestRunLatencyBench:
    def test_with_fake_engine(self):
        engine = FakeEngine()
        engine.load()
        results = run_latency_bench(
            engine,
            context_lengths=[32, 64],
            batch_size=1,
            n_repeats=2,
        )
        assert len(results) == 2
        assert all(isinstance(r, BenchResult) for r in results)
        assert results[0].context_length == 32
        assert results[1].context_length == 64
        for r in results:
            assert r.batch_size == 1
            assert r.ttft_ms > 0
            assert r.decode_tok_s > 0
            assert r.peak_vram_mb is None

    def test_with_prompt_source(self):
        engine = FakeEngine()
        engine.load()
        source = "hello world " * 50
        results = run_latency_bench(
            engine,
            context_lengths=[16],
            batch_size=1,
            n_repeats=1,
            prompt_source=source,
        )
        assert len(results) == 1


class TestSaveBenchResults:
    def test_writes_valid_json(self, tmp_path):
        results = [
            BenchResult(
                context_length=512,
                batch_size=1,
                ttft_ms=100.0,
                prefill_tok_s=5000.0,
                decode_tok_s=50.0,
                peak_vram_mb=1024.0,
            ),
            BenchResult(
                context_length=1024,
                batch_size=1,
                ttft_ms=200.0,
                prefill_tok_s=4000.0,
                decode_tok_s=45.0,
                peak_vram_mb=None,
            ),
        ]
        path = tmp_path / "subdir" / "results.json"
        save_bench_results(results, str(path), engine_name="test-engine")

        assert path.exists()
        with open(path) as f:
            data = json.load(f)

        assert data["engine"] == "test-engine"
        assert len(data["results"]) == 2
        assert data["results"][0]["context_length"] == 512
        assert data["results"][1]["context_length"] == 1024
        assert data["results"][1]["peak_vram_mb"] is None

    def test_creates_parent_dirs(self, tmp_path):
        results = [
            BenchResult(
                context_length=512,
                batch_size=1,
                ttft_ms=100.0,
                prefill_tok_s=5000.0,
                decode_tok_s=50.0,
                peak_vram_mb=None,
            ),
        ]
        path = tmp_path / "a" / "b" / "c" / "results.json"
        save_bench_results(results, str(path), engine_name="test-engine")
        assert path.exists()
