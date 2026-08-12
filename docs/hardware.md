# Hardware

Measured throughput numbers go here, not spec sheets — this file is populated by
`inference/bench.py` runs (see `make baseline`), not written by hand.

## Training / inference workstation

| Field | Value |
|---|---|
| GPU | RTX PRO 6000 Blackwell (96 GB GDDR7) |
| CUDA | TBD — record `nvidia-smi` output and driver/CUDA version on first bench run |
| Measured TFLOPs (30s matmul bench) | TBD |
| PyTorch build | TBD |

## Mobile test device

| Field | Value |
|---|---|
| Model | TBD |
| OS version | TBD |
| Thermal protocol | ≥20 min idle, ≥50% battery, airplane mode, per the project's controlled variables |

Run `python -m scripts.run_bench` to populate `results/v0.0/latency_baseline.json` and update
this file with the resulting numbers.
