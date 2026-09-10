"""Common interfaces for advanced-analysis plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

import numpy as np

from ..io.fairaman import FairamanFile

@dataclass
class AnalysisResult:
    """Risultato standardizzato prodotto da un plugin di analisi.

    Separare risultati spettrali e immagini rende il viewer indipendente
    dall'algoritmo specifico. N-FINDR, MCR-ALS, NMF, clustering e metodi futuri
    possono quindi essere visualizzati ed esportati dalla stessa infrastruttura.
    """

    key: str
    title: str
    spectra: dict[str, np.ndarray] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    markers: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class AnalysisContext:
    """Contesto passato ai plugin senza accoppiarli alla GUI Flet."""

    ff: FairamanFile
    previous_results: dict[str, AnalysisResult]


@dataclass
class AnalysisPlugin:
    """Descrizione dichiarativa di un'analisi avanzata."""

    key: str
    label: str
    description: str
    runner: Callable[[AnalysisContext, dict[str, Any]], AnalysisResult]
    optional_packages: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return all(find_spec(name) is not None for name in self.optional_packages)

    @property
    def missing_packages(self) -> list[str]:
        return [name for name in self.optional_packages if find_spec(name) is None]


