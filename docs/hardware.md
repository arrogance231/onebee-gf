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
| Measured TFLOPs (30s matmul bench) | TBD — pending `inference/bench.py` run |

Notes: on first connect, `nvidia-smi` failed with a driver/library version mismatch (stale
kernel modules vs. a newer installed driver package, plus a pending kernel upgrade); resolved
by rebooting the box. `nvidia-smi` confirmed working post-reboot with the values above.
`uv sync --extra gpu` installed the full training/inference stack (torch 2.13.0+cu130,
transformers 5.15.0, trl 1.10.0, vllm 0.27.1, sentence-transformers 5.7.0, wandb 0.28.2) without
issue.

## Mobile test device

| Field | Value |
|---|---|
| Model | TBD |
| OS version | TBD |
| Thermal protocol | ≥20 min idle, ≥50% battery, airplane mode, per the project's controlled variables |

Run `python -m scripts.run_bench` to populate `results/v0.0/latency_baseline.json` and update
this file with the resulting numbers.
