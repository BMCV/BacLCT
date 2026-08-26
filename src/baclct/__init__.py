"""BacLCT CLI Entrypoints."""

from __future__ import annotations

import os

# cap number of polars threads to prevent oversubscription during training.
# performance-wise, a low amount (tested 2 vs. 8, etc.) is better during training (if
# graphs are not created on-the-fly; ~10% if only using 12 cores with 8 workers). for
# feature extraction, a higher number of workers is crucial. increasing beyond 8 has only
# small benefits (but also only tested with 12 cores).
_cores = os.cpu_count() or 8
_threads = str(8 if _cores >= 8 else max(1, _cores // 2))
os.environ.setdefault("POLARS_MAX_THREADS", _threads)

# abort instead of idling upon deadlock (timeout set in ddp.yaml)
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

from omegaconf import OmegaConf  # noqa
from .api import BacLCT  # noqa

try:
    from ._version import __version__
except ImportError:
    # fallback if _version.py doesn't exist
    __version__ = "0.0.0.dev0"

__all__ = ["BacLCT", "__version__"]

# globally register math resolvers so all entry points (train, predict, tests) can
# dynamically scale the GNN configurations. replace=True is crucial here to prevent errors
# if __init__ is evaluated multiple times.
OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y), replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: int(x / y), replace=True)


# set default cosine similarity. "deep" if cosine similarity is requested via edge feats
# not computed if no deep image features are available or cosine similarity not requested
OmegaConf.register_new_resolver(
    "default_similarity_input",
    lambda names: "deep" if "cosine_similarity" in (names or []) else None,
    replace=True,
)


def _auto_precision() -> str:
    """Set available mixed precision."""
    import torch  # local import to not mess with ENV vars/contexts

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"  # e.g., RTX3090
    return "16-mixed"  # e.g., V100


OmegaConf.register_new_resolver("auto_precision", _auto_precision, replace=True)
