"""End-to-end guard: shipped models scored against a recorded baseline.

Tracks each fold-0 validation sequence with every shipped checkpoint and compares the CTC
scores, wall clock and peak memory to `assets/regression_e2e.json`. Catches changes that
leave the unit tests green but move real tracking output.

Run the spores units with `pixi run -e test test_checkpoints` and the marked full-size
ones with `pixi run -e test test_checkpoints_large`. A unit is skipped when its model or
its data is missing. Cost is asserted only on the machine that recorded the baseline,
scores everywhere. Refresh with `python tests/_regression_support.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _regression_support import (
    BASELINE_PATH,
    CACHE_MODE,
    LARGE_DATASETS,
    expand_units,
    is_available,
    machine,
    measure_unit,
    unit_id,
)

# scoring is deterministic, so this absorbs float-summation noise only
METRIC_TOL = 1e-6
# loose on purpose: these catch an order-of-magnitude regression, not a few percent
COST_HEADROOM = {"wall_s": 2.0, "peak_tree_rss_gib": 1.25, "peak_vram_gib": 1.25}
# a short unit is dominated by fixed cost, so a busy machine alone can double it
_WALL_FLOOR_S = 20.0

BASE = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
UNITS = [
    pytest.param(
        unit,
        id=unit_id(unit),
        marks=pytest.mark.large if unit["dataset"] in LARGE_DATASETS else (),
    )
    for unit in BASE.get("units", [])
]


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory) -> Path:
    """Feature cache shared by every unit, since the models repeat the same sequences."""
    return tmp_path_factory.mktemp("regression_cache")


@pytest.mark.checkpoints
@pytest.mark.parametrize("unit", UNITS)
def test_shipped_model_matches_baseline(
    unit: dict, cache_dir: Path, tmp_path: Path
) -> None:
    """A shipped model on a real sequence reproduces its recorded scores and cost."""
    unavailable = is_available(unit)
    if unavailable:
        pytest.skip(unavailable)

    result, stderr = measure_unit(unit, cache_dir, tmp_path / "out")
    assert result is not None, f"tracking {unit_id(unit)} failed:\n{stderr[-4000:]}"

    mismatches = []
    for name, expected in unit["metrics"].items():
        actual = result["metrics"].get(name)
        if actual is None:
            mismatches.append(f"{name}: missing from the run")
        elif abs(actual - expected) > METRIC_TOL:
            mismatches.append(f"{name}: {expected:.6f} -> {actual:.6f}")
    assert not mismatches, f"{unit_id(unit)} scores moved:\n  " + "\n  ".join(mismatches)

    if machine() != BASE.get("machine") or CACHE_MODE != BASE.get("cache_mode"):
        pytest.skip("cost recorded under different conditions; scores checked")

    for name, headroom in COST_HEADROOM.items():
        expected, actual = unit.get(name), result.get(name)
        if expected is None or actual is None:
            continue
        budget = expected * headroom
        if name == "wall_s":
            budget = max(budget, expected + _WALL_FLOOR_S)
        assert actual <= budget, (
            f"{unit_id(unit)} {name} grew past {budget:.2f}: "
            f"{expected:.2f} -> {actual:.2f}"
        )


@pytest.mark.checkpoints
def test_baseline_covers_every_available_unit() -> None:
    """A refresh that dropped a unit cannot shrink the guard unnoticed."""
    recorded = {unit_id(u) for u in BASE.get("units", [])}
    missing = sorted(
        unit_id(u)
        for u in expand_units()
        if is_available(u) is None and unit_id(u) not in recorded
    )
    assert not missing, f"runnable but not in the baseline: {missing}"
