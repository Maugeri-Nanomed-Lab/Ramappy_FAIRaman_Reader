"""Dependency diagnostics for user-friendly startup checks."""

from importlib.util import find_spec

REQUIRED = {
    "h5py": "h5py",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "ramappy": "ramappy @ git+https://github.com/ramappy/ramappy.git",
    "flet": "flet",
    "flet_charts": "flet-charts",
}

def missing_required() -> dict[str, str]:
    return {module: pip for module, pip in REQUIRED.items() if find_spec(module) is None}
