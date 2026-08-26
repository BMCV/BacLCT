"""On-disk layout helpers for the partitioned edge store.

The store is a directory holding one `dt={k}/part.parquet` partition per temporal stride
plus a `meta.json` describing how it was built. These helpers are pure path and IO
utilities; the edge-direction and validation logic lives in `EdgeFinder`.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from baclct.utils.data import collect


def partition_file(cache_dir: Path, dt: int) -> Path:
    """Path to the partition holding edges of temporal stride `dt`."""
    return cache_dir / f"dt={dt}" / "part.parquet"


def available_partitions(cache_dir: Path) -> set[int]:
    """Temporal strides `dt` already present in the store."""
    if not cache_dir.exists():
        return set()
    return {
        int(p.name.split("=")[1])
        for p in cache_dir.glob("dt=*")
        if (p / "part.parquet").exists()
    }


def read_meta(cache_dir: Path) -> dict | None:
    """Store metadata, or `None` if the store does not exist."""
    meta_file = cache_dir / "meta.json"
    return json.loads(meta_file.read_text()) if meta_file.exists() else None


def write_meta(cache_dir: Path, meta: dict):
    """Write store metadata."""
    (cache_dir / "meta.json").write_text(json.dumps(meta))


def write_partitions(edge_data: pl.DataFrame | pl.LazyFrame, cache_dir: Path):
    """Write edges to per-stride `dt={k}/part.parquet` partitions, sorted by src, dst."""
    df = collect(edge_data) if isinstance(edge_data, pl.LazyFrame) else edge_data
    df = df.with_columns(pl.col("dist_temp").abs().alias("_dt"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for (dt,), group in df.group_by(["_dt"]):
        part_file = partition_file(cache_dir, int(dt))
        part_file.parent.mkdir(parents=True, exist_ok=True)
        group.drop("_dt").sort("src", "dst").write_parquet(part_file)


def load_partitions(cache_dir: Path, dts: list[int]) -> pl.LazyFrame | None:
    """Scan the requested strides, or `None` if none of them exist."""
    files = [
        str(partition_file(cache_dir, dt))
        for dt in sorted(dts)
        if partition_file(cache_dir, dt).exists()
    ]
    return pl.scan_parquet(files) if files else None
