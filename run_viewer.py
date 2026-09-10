"""Convenience launcher for FAIRaman Viewer.

Usage:
    python run_viewer.py
    python run_viewer.py path/to/file.h5

This launcher works from a source checkout without installing the package in
editable mode, provided the runtime dependencies are installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fairaman_viewer.cli import main

raise SystemExit(main())
