"""Test configuration for the napari plugin."""

from __future__ import annotations

import os

# these tests need a Qt platform; default to offscreen so a headless run works
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
