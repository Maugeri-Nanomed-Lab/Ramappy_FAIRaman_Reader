"""Advanced-analysis plugin API."""

from .models import AnalysisContext, AnalysisPlugin, AnalysisResult
from .registry import ANALYSIS_PLUGINS, get_plugin, iter_plugins, register_plugin

__all__ = [
    "AnalysisContext", "AnalysisPlugin", "AnalysisResult",
    "ANALYSIS_PLUGINS", "get_plugin", "iter_plugins", "register_plugin",
]
