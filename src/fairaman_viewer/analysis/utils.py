"""Shared helpers for advanced analysis algorithms."""

import numpy as np

from ..io.fairaman import FairamanFile

def _analysis_matrix(ff: FairamanFile) -> np.ndarray:
    data = np.asarray(ff.obj.data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        data = data.reshape(-1, data.shape[-1])
    if not np.isfinite(data).all():
        raise ValueError("L'analisi richiede dati finiti: sono presenti NaN o inf.")
    return data

