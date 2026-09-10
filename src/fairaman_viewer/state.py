"""Mutable application state, independent from Flet controls."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis.models import AnalysisResult
from .io.fairaman import FairamanFile
from .validation.schema import Issue, validate

@dataclass
class Trace:
    """Spettro selezionato dall'utente e mostrato nel pannello spettrale."""

    y: np.ndarray
    label: str
    marker: tuple[float, float] | None = None


@dataclass
class ViewerState:
    """Unica sorgente di verita' dello stato del viewer."""

    ff: FairamanFile | None = None
    image_data: np.ndarray | None = None
    image_label: str = ""
    traces: list[Trace] = field(default_factory=list)
    pca_cache: dict[int, np.ndarray] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    analysis_results: dict[str, AnalysisResult] = field(default_factory=dict)

    def reset_for_new_file(self, ff: FairamanFile) -> None:
        self.ff = ff
        self.image_data = None
        self.image_label = ""
        self.traces.clear()
        self.pca_cache.clear()
        self.history.clear()
        self.issues = validate(ff)
        self.analysis_results.clear()

    def invalidate_after_processing(self) -> None:
        """Invalida cache e risultati derivati quando cambiano i dati."""
        self.image_data = None
        self.pca_cache.clear()
        self.traces.clear()
        self.analysis_results.clear()

