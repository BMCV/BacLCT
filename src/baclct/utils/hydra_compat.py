"""Compatibility shims applied before Hydra builds its CLI parser.

Import this module for its side effects from `baclct.train`, the only `@hydra.main` entry
point.
"""

from __future__ import annotations

import argparse
import sys

# hydra passes a lazy object as the help of --shell-completion. python 3.14 added
# ArgumentParser._check_help, which does `'%' not in help` and raises on a non-str.
# hydra applies the same patch itself from 1.4.0, so this can go once that is the floor.
if sys.version_info >= (3, 14):  # pragma: no cover - version-gated
    _original_check_help = argparse.ArgumentParser._check_help  # type: ignore[attr-defined]

    def _check_help(self, action) -> None:
        if isinstance(getattr(action, "help", None), str):
            _original_check_help(self, action)

    argparse.ArgumentParser._check_help = _check_help  # type: ignore[attr-defined]
