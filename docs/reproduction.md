# Reproduction

## Install

```bash
uv sync                # base install: CPU-only, sufficient for tests and CI
uv sync --extra gpu     # + torch/transformers/vllm/trl/peft/etc. for training and inference
```

## Run the test suite

```bash
pytest
```

## Build the PMB benchmark

```bash
python scripts/build_pmb.py --out-dir data/benchmarks/pmb_v0 \
  --n-personas 8 --sessions-per-persona 6 --turns-per-session 14 --facts-per-persona 40
```

The repo ships a small fixture-scale run (2 personas) under `data/benchmarks/pmb_v0/` for
offline smoke-testing; the command above regenerates it at full v0 scale. A live teacher
endpoint (`--teacher`) is required for non-fixture generation — not yet wired up (see
`scripts/build_pmb.py --help`).

## Run the evaluation harness

```python
from onebee.evaluation.harness import run_harness, save_harness_result
# see tests/integration/test_smoke.py for a minimal end-to-end example
```

## Populate a memory store

```python
from onebee.memory.store import MemoryStore
store = MemoryStore("data/stores/example.db")
```

## Check for train/eval contamination

```bash
python scripts/check_contamination.py \
  --train-glob "data/**/train*.jsonl" --eval-glob "data/benchmarks/**/probes*.jsonl"
```

## Train the SFT adapter

Requires the `gpu` extra and a GPU workstation — not run in CI.

```bash
python -m onebee.training.sft --config configs/training/sft.yaml
```

## Regenerate figures

```bash
make figures
```

Every command above should be run from a clean clone, in order, to earn the reproducibility
claim. If a step breaks, that is a bug in this file or the code — file it.
