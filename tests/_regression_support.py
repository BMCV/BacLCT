"""Definitions and the recorder for the checkpoint regression guard.

The test and the recorder go through the same steps, so a unit is measured exactly the
way it is asserted. Run this module as a script to refresh the baseline. Not named
`test_*`, so pytest does not collect it.

Usage::

    python tests/_regression_support.py
    python tests/_regression_support.py --large
    python tests/_regression_support.py --units baclct_toiam_pc-toiam-03
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "tests" / "assets" / "regression_e2e.json"
WORKER = REPO_ROOT / "tests" / "_regression_worker.py"

# spores carries the guard: every sequence is scorable in the time one toiam sequence
# takes. `radius: None` keeps the model's trained search radius, 55 is what the paper's
# toiam benchmark used to stay under 30 GiB of host memory.
SPECS = [
    {"model": "baclct_spore_classification_bf", "dataset": "spores", "radius": None},
    {"model": "baclct_track", "dataset": "spores", "radius": None},
    {"model": "baclct_toiam_pc", "dataset": "spores", "radius": None},
    {"model": "baclct_toiam_pc", "dataset": "toiam", "radius": 55},
]

# full-size sequences cost tens of GiB and tens of minutes, so their units carry the
# `large` mark and stay out of the default checkpoint run
LARGE_DATASETS = {"toiam"}

# validation only, since the baseline is committed. Every sequence rather than a
# representative one: a graph or tracker change can move some sequences and not others.
FOLD = 0
PHASE = "val"

# every unit runs twice and reports the second run, so the numbers describe prediction
# and tracking rather than the caches the first run builds
CACHE_MODE = "warm"

METRICS = ("TRA", "LNK", "CHOTA", "BC(0)", "BC(3)")


def data_root() -> Path:
    """Directory holding the datasets."""
    return Path(os.environ.get("BACLCT_DATA_ROOT", REPO_ROOT / "data")).resolve()


def model_dir(model: str) -> Path:
    """Directory of one shipped model."""
    root = Path(os.environ.get("BACLCT_MODEL_DIR", REPO_ROOT / "shipped_models"))
    return (root / model).resolve()


def machine() -> dict:
    """Identify the host well enough to know whether timings are comparable."""
    info = {"cpu_count": os.cpu_count(), "gpu": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # torch missing or CUDA broken: scores still apply
        pass
    return info


def unit_id(unit: dict) -> str:
    """Stable identifier used as the pytest parameter id and the baseline key."""
    return f"{unit['model']}-{unit['dataset']}-{unit['seq']}"


def expand_units() -> list[dict]:
    """One unit per spec and validation sequence, taken from the split file."""
    from baclct.io import get_sequences_from_split

    units = []
    for spec in SPECS:
        split_file = data_root() / spec["dataset"] / "splits.yaml"
        if not split_file.exists():
            continue
        for seq in get_sequences_from_split(split_file, FOLD, phase=PHASE):
            units.append({**spec, "seq": seq})
    return units


def is_available(unit: dict) -> str | None:
    """Reason the unit cannot run, or `None` when it can."""
    if not (model_dir(unit["model"]) / ".hydra" / "config.yaml").exists():
        return f"model {unit['model']} not available"
    if not (data_root() / unit["dataset"] / unit["seq"]).is_dir():
        return f"data for {unit['dataset']}/{unit['seq']} not available"
    return None


def run_unit(unit: dict, cache_dir: Path, output_dir: Path) -> tuple[dict | None, str]:
    """Track one sequence in a subprocess. Returns its result and the worker's stderr."""
    cmd = [
        sys.executable,
        str(WORKER),
        "--model-dir",
        str(model_dir(unit["model"])),
        "--data-root",
        str(data_root()),
        "--dataset",
        unit["dataset"],
        "--seq",
        unit["seq"],
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(cache_dir),
    ]
    if unit.get("radius") is not None:
        cmd += ["--radius", str(unit["radius"])]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if proc.returncode != 0:
        return None, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def measure_unit(
    unit: dict, cache_dir: Path, output_dir: Path
) -> tuple[dict | None, str]:
    """Run one unit twice and report the second, so its caches are warm when measured."""
    run_unit(unit, cache_dir, output_dir / "warmup")
    return run_unit(unit, cache_dir, output_dir / "measured")


def _baseline() -> dict:
    """Recorded baseline, or an empty one before the first recording."""
    return json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}


def _merge(recorded: list[dict]) -> list[dict]:
    """Baseline units with the freshly recorded ones in their place.

    A partial refresh keeps the units it did not measure.
    """
    fresh = {unit_id(u): u for u in recorded}
    kept = [fresh.pop(unit_id(u), u) for u in _baseline().get("units", [])]
    return kept + list(fresh.values())


def main() -> int:
    """Measure every requested unit and write the baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", nargs="+", help="Restrict to these unit ids.")
    parser.add_argument(
        "--large", action="store_true", help="Include the full-size units."
    )
    args = parser.parse_args()

    units = [u for u in expand_units() if not args.units or unit_id(u) in args.units]
    if not (args.units or args.large):
        units = [u for u in units if u["dataset"] not in LARGE_DATASETS]
    if not units:
        print("no units to record", file=sys.stderr)
        return 1
    if args.units and _baseline().get("machine", machine()) != machine():
        print("refusing to merge into a baseline from another machine", file=sys.stderr)
        return 1

    recorded, failed = [], []
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        cache_dir.mkdir()

        for unit in units:
            uid = unit_id(unit)
            unavailable = is_available(unit)
            if unavailable:
                print(f"[skip ] {uid}: {unavailable}")
                continue

            print(f"[run  ] {uid} ...", flush=True)
            result, stderr = measure_unit(unit, cache_dir, Path(tmp) / "out" / uid)
            if result is None:
                print(f"[FAIL ] {uid}:\n{stderr[-3000:]}", file=sys.stderr)
                failed.append(uid)
                continue

            recorded.append(
                {
                    **unit,
                    "metrics": {
                        m: result["metrics"][m] for m in METRICS if m in result["metrics"]
                    },
                    "wall_s": round(result["wall_s"], 1),
                    "peak_tree_rss_gib": round(result["peak_tree_rss_gib"], 2),
                    "peak_vram_gib": (
                        round(result["peak_vram_gib"], 2)
                        if result["peak_vram_gib"] is not None
                        else None
                    ),
                }
            )
            print(f"[ok   ] {uid}: {recorded[-1]['metrics']}")

    if failed:
        print(f"nothing written, {len(failed)} unit(s) failed: {failed}", file=sys.stderr)
        return 1
    if not recorded:
        print("nothing recorded: no model/data pair was available", file=sys.stderr)
        return 1

    BASELINE_PATH.write_text(
        json.dumps(
            {"machine": machine(), "cache_mode": CACHE_MODE, "units": _merge(recorded)},
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {BASELINE_PATH} ({len(recorded)} units measured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
