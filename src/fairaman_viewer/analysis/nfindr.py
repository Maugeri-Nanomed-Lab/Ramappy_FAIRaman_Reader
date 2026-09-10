"""N-FINDR endmember extraction plugin."""

from typing import Any

import numpy as np

from .models import AnalysisContext, AnalysisResult
from .utils import _analysis_matrix

def run_nfindr(context: AnalysisContext, params: dict[str, Any]) -> AnalysisResult:
    """Estrae endmember con N-FINDR tramite pyspc-unmix.

    N-FINDR identifica gli indici degli endmember; non interpretiamo le
    abbondanze come output intrinseco dell'algoritmo. Questo evita di mischiare
    endmember extraction e decomposition in un unico passaggio opaco.
    """
    from sklearn.decomposition import PCA
    from pyspc_unmix import NFINDR

    ff = context.ff
    data = _analysis_matrix(ff)
    n_components = int(params.get("n_components", 4))
    random_state = int(params.get("random_state", 21))

    if data.shape[0] < n_components:
        raise ValueError(
            f"Servono almeno {n_components} spettri/pixel, ma il file ne contiene {data.shape[0]}."
        )
    if n_components < 2:
        raise ValueError("N-FINDR richiede almeno 2 endmember.")

    # Per un simplex con p vertici bastano p-1 dimensioni. Limitiamo comunque
    # il numero di PC alla dimensionalita' disponibile.
    pca_dims = min(n_components - 1, data.shape[0] - 1, data.shape[1])
    if pca_dims < 1:
        raise ValueError("Dati insufficienti per la riduzione PCA richiesta da N-FINDR.")

    scores = PCA(n_components=pca_dims, random_state=random_state).fit_transform(data)
    # L'API estimator rende espliciti numero di endmember e random state.
    # Usiamo solo gli indici estratti; l'eventuale decomposition viene lasciata
    # a un passaggio distinto (MCR-ALS/FCLS/NNLS).
    model = NFINDR(n_endmembers=n_components, random_state=random_state)
    model.fit(scores)
    indices = [int(i) for i in model.endmember_indices_]
    if len(indices) != n_components:
        raise RuntimeError(
            f"N-FINDR ha restituito {len(indices)} endmember invece dei {n_components} richiesti."
        )

    spectra = {f"EM{i + 1}": data[idx].copy() for i, idx in enumerate(indices)}
    markers: list[tuple[float, float]] = []
    if ff.coordinate_mode == "point_coordinates" and ff.point_coords is not None:
        markers = [tuple(map(float, ff.point_coords[idx])) for idx in indices]
    elif ff.is_map:
        ny, nx = ff.map_shape()
        extent = ff.spatial_extent()
        for idx in indices:
            row, col = divmod(idx, nx)
            if extent:
                left, right, bottom, top = extent
                x = left + (col + 0.5) / nx * (right - left)
                y_row = row if ff.y_ascending() else ny - 1 - row
                y = bottom + (y_row + 0.5) / ny * (top - bottom)
                markers.append((float(x), float(y)))
            else:
                markers.append((float(col), float(row)))

    return AnalysisResult(
        key="nfindr",
        title=f"N-FINDR · {n_components} endmember",
        spectra=spectra,
        metadata={
            "n_components": n_components,
            "pca_dimensions": pca_dims,
            "endmember_indices": indices,
            "random_state": random_state,
        },
        markers=markers,
    )

