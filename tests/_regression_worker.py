"""Track one sequence and report its metrics, wall clock and peak memory as JSON.

Run as a subprocess so that peak RSS covers dataloader and joblib worker processes, which
a self-only measurement misses, and so an out-of-memory kill costs one sequence rather
than the test session. Not named `test_*`, so pytest does not collect it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import psutil


def _tree_rss_kb(root: psutil.Process) -> int:
    """Summed RSS of `root` and everything under it, in KiB."""
    total = 0
    for proc in (root, *root.children(recursive=True)):
        try:
            total += proc.memory_info().rss
        except psutil.Error:  # a worker that exited while the tree was walked
            continue
    return total // 1024


class _TreeSampler(threading.Thread):
    """Sample the summed RSS of this process tree until stopped."""

    def __init__(self, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_kb = 0
        # NOT _stop: that name is `threading.Thread`'s own, and join() calls it
        self._done = threading.Event()

    def run(self) -> None:
        root = psutil.Process()
        while not self._done.is_set():
            self.peak_kb = max(self.peak_kb, _tree_rss_kb(root))
            self._done.wait(self.interval)

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=2)


def main() -> int:
    """Track the requested sequence and print one JSON object to stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--radius", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch

    from baclct.api import BacLCT
    from baclct.io import load_images_and_masks
    from baclct.tracking.metrics import compute_ctcmetrics

    data_dir = (args.data_root / args.dataset).resolve()
    images, masks = load_images_and_masks(
        data_dir, args.seq, data_format="ctc", lazy=True, segmentation_name="GT"
    )

    radius: int | str | None = args.radius
    if radius is not None:
        try:
            radius = int(radius)
        except ValueError:
            pass  # a relative radius such as '2.5x'

    overrides = {
        # every shipped model holds exactly one checkpoint
        "checkpoint": "best",
        # ground-truth masks need no correction
        "tracker": {"segmentation_correction": None},
    }

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    sampler = _TreeSampler()
    sampler.start()
    start = time.perf_counter()
    try:
        BacLCT(str(args.model_dir), config_overrides=overrides).track(
            images=images,
            masks=masks,
            graph_search_radius=radius,
            output_dir=args.output_dir,
            sequence_id=args.seq,
            export_format="ctc",
            cache_dir=args.cache_dir,
        )
    finally:
        wall = time.perf_counter() - start
        sampler.stop()

    metrics = compute_ctcmetrics(
        data_dir / f"{args.seq}_GT",
        args.output_dir / args.seq,
        metrics=["TRA", "LNK", "CHOTA", "BC"],
    )
    assert isinstance(metrics, dict)

    print(
        json.dumps(
            {
                "metrics": {
                    k: v for k, v in metrics.items() if isinstance(v, (int, float))
                },
                "wall_s": wall,
                "peak_tree_rss_gib": sampler.peak_kb / 1024**2,
                "peak_vram_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3 if cuda else None
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
