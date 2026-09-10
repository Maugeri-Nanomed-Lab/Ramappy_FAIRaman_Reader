"""Application-wide configuration constants.

Keep only stable display/configuration values here. Scientific defaults belong
next to the algorithm that owns them.
"""

APP_NAME = "FAIRaman Viewer"
APP_VERSION = "0.3.0"

COLORMAPS = ["viridis", "magma", "inferno", "cividis", "gray", "hot", "RdBu_r"]
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#ff7f0e", "#8c564b", "#17becf", "#e377c2",
]
SEVERITY_COLOR = {"error": "#b00020", "warning": "#b06000", "info": "#4a4a4a"}
