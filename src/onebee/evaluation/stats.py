from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = float(np.mean(arr[indices]))
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def paired_bootstrap_diff(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    if len(values_a) != len(values_b):
        raise ValueError(
            f"Length mismatch: values_a has {len(values_a)}, values_b has {len(values_b)}"
        )
    arr_a = np.array(values_a, dtype=np.float64)
    arr_b = np.array(values_b, dtype=np.float64)
    diffs = arr_a - arr_b
    return bootstrap_ci(diffs.tolist(), n_resamples=n_resamples, ci=ci, seed=seed)


def holm_bonferroni(
    p_values: dict[str, float], alpha: float = 0.05
) -> dict[str, bool]:
    items = sorted(p_values.items(), key=lambda x: x[1])
    n = len(items)
    result: dict[str, bool] = {name: False for name in p_values}
    for rank, (name, p) in enumerate(items, start=1):
        adjusted_alpha = alpha / (n - rank + 1)
        if p < adjusted_alpha:
            result[name] = True
        else:
            break
    return result


def cohens_h(p1: float, p2: float) -> float:
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def minimum_detectable_effect(
    n: int,
    baseline_p: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Approximate MDE for a two-proportion test using a normal approximation.

    Uses fixed z-quantiles for the standard ``alpha=0.05`` (two-tailed, z=1.96)
    and ``power=0.80`` (z=0.842).  Arbitrary ``alpha`` and ``power`` values
    fall back to these constants — this function does not depend on SciPy.
    """
    z_alpha = 1.96
    z_beta = 0.842
    p = baseline_p
    return float((z_alpha + z_beta) * math.sqrt(2 * p * (1 - p) / n))
