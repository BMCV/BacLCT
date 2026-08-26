"""Logger compatible with Lightning's multiple ranks.

Adapted from https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/pylogger.py.
"""

from __future__ import annotations

import logging
import sys

from lightning.pytorch.utilities import rank_zero_only

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)


class LightningLogFilter(logging.Filter):
    """Filter for lightning logs."""

    def filter(self, record):
        """Define filters."""
        msg = record.getMessage()
        if "TPU available" in msg:
            return False
        if "Trainer already configured with model summary callbacks" in msg:
            return False
        if "Using 16bit Automatic Mixed Precision (AMP)" in msg:
            return False
        return True


def configure_lightning_loggers():
    """Silences some lightning logs."""
    logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.WARNING)
    rank_zero_logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
    # add filter to rank_zero_logger
    rank_zero_logger.addFilter(LightningLogFilter())


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Initializes a multi-GPU-friendly python command line logger.

    Adapted from github.com/ashleve/lightning-hydra-template.

    Args:
        name: The name of the logger.

    Returns:
        A logger object.
    """
    logger = logging.getLogger(name)
    # logger.setLevel(logging.INFO); handled by hydra

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
