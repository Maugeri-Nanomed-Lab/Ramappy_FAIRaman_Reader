"""MCR-ALS decomposition plugin."""

from typing import Any

import numpy as np

from .models import AnalysisContext, AnalysisResult
from .utils import _analysis_matrix

def run_mcr_als(context: AnalysisContext, params: dict[str, Any]) -> AnalysisResult:
    """MCR-ALS non-negativa inizializzata con il risultato N-FINDR corrente."""
    from pymcr.mcr import McrAR

    ff = context.ff
    data = _analysis_matrix(ff)
    n_components = int(params.get("n_components", 4))
    max_iter = int(params.get("max_iter", 100))

    if np.nanmin(data) < 0:
        raise ValueError(
            "MCR-ALS e' configurata con vincoli di non-negativita', ma i dati contengono "
            "valori negativi. Il viewer non applica offset nascosti: prepara intenzionalmente "
            "dati non-negativi oppure usa in futuro un plugin MCR con vincoli diversi."
        )

    init = context.previous_results.get("nfindr")
    if init is None:
        raise ValueError(
            "MCR-ALS richiede un'inizializzazione esplicita. Esegui prima N-FINDR "
            "con lo stesso numero di componenti."
        )
    if len(init.spectra) != n_components:
        raise ValueError(
            f"L'ultimo N-FINDR contiene {len(init.spectra)} endmember; "
            f"MCR-ALS ne richiede {n_components}. Riesegui N-FINDR."
        )

    st0 = np.vstack([init.spectra[f"EM{i + 1}"] for i in range(n_components)])
    model = McrAR(c_regr="NNLS", st_regr="NNLS", max_iter=max_iter)
    model.fit(data, ST=st0, verbose=False)

    st = np.asarray(model.ST_opt_, float)
    c = np.asarray(model.C_opt_, float)
    spectra = {f"MCR{i + 1}": st[i].copy() for i in range(st.shape[0])}

    images: dict[str, np.ndarray] = {}
    if ff.is_map:
        shape = ff.map_shape()
        for i in range(c.shape[1]):
            images[f"MCR{i + 1} abundance"] = c[:, i].reshape(shape)
    elif ff.coordinate_mode == "point_coordinates":
        # Per coordinate sparse conserviamo il vettore; la GUI potra' renderlo
        # come scatter quando verra' richiesto come immagine di analisi.
        for i in range(c.shape[1]):
            images[f"MCR{i + 1} abundance"] = c[:, i].copy()

    return AnalysisResult(
        key="mcr_als",
        title=f"MCR-ALS · {n_components} componenti",
        spectra=spectra,
        images=images,
        metadata={
            "n_components": n_components,
            "max_iter": max_iter,
            "iterations": int(getattr(model, "n_iter_opt", getattr(model, "n_iter", 0))),
            "initialization": "N-FINDR endmembers",
            "c_regressor": "NNLS",
            "st_regressor": "NNLS",
        },
    )

