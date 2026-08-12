from __future__ import annotations

from pathlib import Path

import pytest

from onebee.evaluation.harness import run_harness
from onebee.evaluation.metrics.personalized import Probe


@pytest.mark.smoke
def test_smoke_pmb_v0_harness() -> None:
    benchmarks_dir = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "pmb_v0"
    probes_path = benchmarks_dir / "probes.jsonl"

    assert probes_path.exists(), f"probes.jsonl not found at {probes_path}"

    probes: list[Probe] = []
    with open(probes_path) as f:
        for line in f:
            probes.append(Probe.model_validate_json(line))

    assert len(probes) > 0, "No probes loaded"

    def always_abstain(probe: Probe) -> tuple[str, list[str]]:
        return "I don't know.", []

    result = run_harness(
        probes=probes,
        response_fn=always_abstain,
        system_name="smoke-test-abstain",
    )

    assert result.n_probes == len(probes)
    assert isinstance(result.n_probes, int)
    assert result.n_probes > 0

    categories_found = set(result.per_category.keys())
    assert len(categories_found) > 0
