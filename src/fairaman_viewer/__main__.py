"""Allow both ``python -m fairaman_viewer`` and direct execution.

Direct execution of a package's ``__main__.py`` normally breaks relative imports.
The fallback below adds the package ``src`` directory to ``sys.path`` so the
launcher is also friendly to users running this file from an IDE or by path.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from fairaman_viewer.cli import main
else:
    from .cli import main

raise SystemExit(main())
