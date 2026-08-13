# Hardware

Measured throughput numbers go here, not spec sheets — this file is populated by
`inference/bench.py` runs (see `make baseline`), not written by hand.

## Training / inference workstation

Remote GPU rental box, provisioned 2026-08-13.

| Field | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 97,887 MiB (~96 GB GDDR7) |
| GPU UUID | `GPU-afc5b5be-a181-3c3a-45d5-55f9263f000e` |
| Board part number | 900-2G153-0000-000 |
| VBIOS version | 98.02.AF.00.01 |
| Driver version | 580.173.02 |
| CUDA version (driver-reported) | 13.0 |
| CUDA toolkit (`nvcc`) | not installed — not needed yet; PyTorch ships its own CUDA runtime |
| MIG mode | Disabled |
| Virtualization mode | Pass-Through |
| Persistence mode | Enabled |
| CPU | AMD EPYC 9555, 30 vCPUs on-line, 1 thread/core |
| System RAM | 88 GiB total, 87 GiB free at provisioning |
| Disk | 145 GB total, 114 GB free at provisioning |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.8.0-137-generic |
| PyTorch build | `torch==2.13.0+cu130` |
| `torch.cuda.is_available()` | `True` |
| `torch.cuda.get_device_capability(0)` | `(12, 0)` — sm_120, confirms Blackwell support (the doc's flagged likely Day-1 blocker did not occur) |
| Measured TFLOPs (30s matmul bench) | not run — the Day-1 baseline used the real end-to-end latency bench below instead, which is the number that actually matters for the project |

Notes: on first connect, `nvidia-smi` failed with a driver/library version mismatch (stale
kernel modules vs. a newer installed driver package, plus a pending kernel upgrade); resolved
by rebooting the box. `nvidia-smi` confirmed working post-reboot with the values above.
`uv sync --extra gpu` installed the full training/inference stack (torch 2.13.0+cu130,
transformers 5.15.0, trl 1.10.0, vllm 0.27.1, sentence-transformers 5.7.0, wandb 0.28.2) without
issue. See `docs/model_quirks.md` for model-specific environment quirks discovered on this box
(cuDNN/Blackwell incompatibility, missing deps, chat-template bugs) and `docs/adr/
0001-model-selection.md` for the Day-1 base-model decision.

### Baseline latency — `gemma4-e2b` (`google/gemma-4-E2B-it`), 2026-08-13

Real, unoptimized HF `transformers` eager-mode generation (`run_latency_bench`, batch=1,
3 repeats per context length, `max_new_tokens=64`), full results in
`results/v0.0/latency_baseline.json`:

| Context length | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak VRAM (MB) |
|---|---|---|---|---|
| 512 | 215.3 | 17,562 | 50.5 | 9,842 |
| 1024 | 28.4 | 23,833 | 50.3 | 9,915 |
| 2048 | 45.0 | 28,494 | 49.6 | 10,067 |
| 4096 | 100.2 | 26,386 | 48.1 | 10,359 |

This is an unoptimized eager-mode baseline (no flash-attention, no KV-cache-aware batching,
no quantization) — ~50 tok/s decode is a floor to improve on, not a target. GGUF quantization
and a proper inference runtime (llama.cpp/vLLM/MLC) are deferred to a later pass per the
project's roadmap; these numbers exist to have *something real* committed for the Day-1 exit
criteria, not to be the final efficiency answer.

## Mobile test device

| Field | Value |
|---|---|
| Model | TBD |
| OS version | TBD |
| Thermal protocol | ≥20 min idle, ≥50% battery, airplane mode, per the project's controlled variables |

Run `python -m scripts.run_bench` to populate `results/v0.0/latency_baseline.json` and update
this file with the resulting numbers.
