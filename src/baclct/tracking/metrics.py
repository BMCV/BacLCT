"""Compute CTC tracking metrics for a predicted sequence."""

from __future__ import annotations

from pathlib import Path

from ctc_metrics import evaluate_sequence, validate_sequence
from ctc_metrics.metrics import ALL_METRICS

from baclct.utils.logger import get_pylogger

logger = get_pylogger(__name__)


def compute_ctcmetrics(
    gt_dir: Path,
    pred_dir: Path,
    validation_only: bool = False,
    metrics: list[str] | None = None,
) -> dict | bool:
    """Compute CTC metrics and CHOTA using py_ctcmetrics."""
    gt_dir_str = str(gt_dir.resolve()) if isinstance(gt_dir, Path) else gt_dir
    pred_dir_str = str(pred_dir.resolve()) if isinstance(pred_dir, Path) else pred_dir

    if validation_only:
        return bool(validate_sequence(pred_dir_str)["Valid"])

    try:
        result: dict = evaluate_sequence(pred_dir_str, gt_dir_str, metrics=metrics)  # type: ignore
    except IndexError:
        # ctc_metrics raises while computing CHOTA when the gt has empty frames
        logger.warning("CHOTA could not be computed due to empty frames in GT")
        reduced = [m for m in (metrics or ALL_METRICS) if m != "CHOTA"]
        result = evaluate_sequence(pred_dir_str, gt_dir_str, metrics=reduced)
        result["CHOTA"] = None

    return result
