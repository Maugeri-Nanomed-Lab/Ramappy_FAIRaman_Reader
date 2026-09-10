"""Baseline correction algorithms and execution helpers."""


import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ramappy import Spectrum, processing
from ramappy.core import SpectralMap

# blocco di spettri elaborato per volta: sotto _PARALLEL_MIN_ITEMS di ramappy
# (256) per restare sul percorso seriale, ma abbastanza grande da non pagare
# troppo overhead per blocco.
CHUNK = 250


# ---------------------------------------------------------------- parametri --
@dataclass
class Param:
    """Un parametro di un metodo, con quanto serve alla GUI per disegnarlo."""

    key: str
    label: str
    kind: str  # "float" | "logfloat" | "int" | "choice" | "bool"
    default: Any = None
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""
    optional: bool = False  # se True, vuoto significa "lascia decidere al metodo"


@dataclass
class BaselineMethod:
    key: str
    label: str
    params: list[Param] = field(default_factory=list)
    speed: str = "veloce"  # "veloce" | "media" | "lenta"
    note: str = ""


_KIND = Param(
    "kind", "Interpolazione", "choice", "slinear",
    choices=("slinear", "linear"),
    help="slinear raccorda i vertici dell'inviluppo convesso con spline di primo grado.",
)
_MAXPTS = Param(
    "max_points", "Max punti", "int", None, lo=3, hi=500, optional=True,
    help="Limita i vertici dell'inviluppo. Vuoto = tutti.",
)
# lambda_ NON e' opzionale: ramappy inoltra lambda_=None a pybaselines come
# lam=None, che produce una baseline interamente NaN senza sollevare errori.
# Vedi PLS_METHODS/_LAM_FALLBACK piu' sotto.
_LAM = Param(
    "lambda_", "Lambda (log10)", "logfloat", 5.0, lo=1.0, hi=10.0,
    help="Rigidita' della baseline: piu' alto = piu' liscia. Tipico 10^4-10^7.",
)
_ETA = Param(
    "eta", "Eta", "float", 0.5, lo=0.0, hi=1.0,
    help="Peso della componente di derivata prima (solo drpls).",
)
_ORDER = Param(
    "poly_order", "Grado polinomio", "int", 5, lo=1, hi=12,
    help="Gradi alti seguono meglio le curvature ma rischiano di mangiare i picchi.",
)
_HALFWIN = Param(
    "max_half_window", "Semi-finestra", "int", None, lo=2, hi=400, optional=True,
    help="Ampiezza massima del filtro SNIP in punti. Vuoto = stimata dai dati.",
)

METHODS: dict[str, BaselineMethod] = {
    m.key: m
    for m in [
        BaselineMethod(
            "rubberband", "Rubberband (inviluppo convesso)",
            [_KIND, _MAXPTS], "veloce",
            "Nessun parametro di rigidita': la baseline e' l'inviluppo convesso "
            "dal basso. Robusto, ma sottostima il fondo sotto bande larghe.",
        ),
        BaselineMethod(
            "arpls", "arPLS (asimmetrica riponderata)",
            [_LAM], "veloce",
            "Buon compromesso generale su spettri Raman biologici.",
        ),
        BaselineMethod(
            "iarpls", "iarPLS (arPLS migliorata)", [_LAM], "veloce",
            "Variante di arPLS meno incline a tagliare i picchi deboli.",
        ),
        BaselineMethod(
            "aspls", "asPLS (adattiva)", [_LAM], "media",
            "Adatta la penalita' punto per punto; utile con fondo molto variabile.",
        ),
        BaselineMethod(
            "drpls", "drPLS (doppiamente riponderata)", [_LAM, _ETA], "media",
        ),
        BaselineMethod(
            "airpls", "airPLS", [_LAM], "media",
            "Puo' fallire (LinAlgError) su spettri con fondo molto ripido o "
            "valori non finiti: in quel caso prova arPLS.",
        ),
        BaselineMethod(
            "snip", "SNIP (clipping iterativo)", [_HALFWIN], "lenta",
            "Circa 30 s ogni 10.000 spettri. Su mappe grandi conviene tararlo "
            "in anteprima e poi lanciarlo una volta sola.",
        ),
        BaselineMethod(
            "poly", "Polinomiale", [_ORDER], "veloce",
            "Adatta un polinomio a tutto lo spettro, picchi compresi: "
            "quasi sempre imodpoly e' preferibile.",
        ),
        BaselineMethod(
            "imodpoly", "imodPoly (polinomiale modificata)", [_ORDER], "media",
            "Polinomiale iterativa che esclude progressivamente i picchi.",
        ),
        BaselineMethod(
            "goldindec", "Goldindec", [_ORDER], "lenta",
        ),
        BaselineMethod(
            "beads", "BEADS", [], "proibitiva",
            "Circa 0,7 s per spettro: su una mappa da 9.000 spettri sono quasi "
            "due ore. Utilizzabile in anteprima o su spettri singoli.",
        ),
    ]
}

METHOD_KEYS = list(METHODS)

# Metodi penalized least squares: richiedono sempre un lam esplicito.
PLS_METHODS = {"arpls", "iarpls", "aspls", "drpls", "airpls"}
_LAM_FALLBACK = 1e5  # default di pybaselines

SPEED_HINT = {
    "veloce": "sotto il secondo su 10.000 spettri",
    "media": "qualche secondo su 10.000 spettri",
    "lenta": "decine di secondi su 10.000 spettri",
    "proibitiva": "ore su una mappa: usare in anteprima",
}


def default_params(method: str) -> dict[str, Any]:
    return {p.key: p.default for p in METHODS[method].params}


def to_kwargs(method: str, values: dict[str, Any]) -> dict[str, Any]:
    """Converte i valori della GUI negli argomenti attesi da ramappy."""
    out: dict[str, Any] = {}
    for param in METHODS[method].params:
        value = values.get(param.key, param.default)
        if value is None or value == "":
            continue
        if param.kind == "logfloat":
            out[param.key] = float(10.0 ** float(value))
        elif param.kind == "int":
            out[param.key] = int(value)
        elif param.kind == "float":
            out[param.key] = float(value)
        else:
            out[param.key] = value

    # rete di sicurezza: senza lam esplicito i metodi PLS restituiscono NaN
    if method in PLS_METHODS and "lambda_" not in out:
        out["lambda_"] = _LAM_FALLBACK
    return out


def describe(method: str, values: dict[str, Any]) -> str:
    """Riga leggibile con metodo e parametri effettivi, per la history."""
    kwargs = to_kwargs(method, values)
    if not kwargs:
        return method
    bits = []
    for key, value in kwargs.items():
        bits.append(f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}")
    return f"{method}({', '.join(bits)})"


# ---------------------------------------------------------------- anteprima --
def estimate_baseline(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    values: dict[str, Any] | None = None,
    roi: tuple[float, float] | None = None,
) -> np.ndarray:
    """Baseline stimata per un singolo spettro, senza toccare i dati originali.

    Passa dallo stesso `correct_baseline` usato sull'intera mappa, cosi'
    l'anteprima non puo' divergere dal risultato applicato.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    spec = Spectrum(x=x.copy(), data=y.astype(np.float32).copy())
    kwargs = to_kwargs(method, values or {})
    roi_x = [[float(roi[0]), float(roi[1])]] if roi else None
    processing.correct_baseline(spec, method=method, n_jobs=1, roi_x=roi_x, **kwargs)
    corrected = np.asarray(spec.data, float).reshape(-1)

    baseline = y - corrected
    if roi is not None:  # fuori dalla ROI non e' stata stimata alcuna baseline
        outside = (x < min(roi)) | (x > max(roi))
        baseline[outside] = np.nan
    return baseline


# ------------------------------------------------------------ applicazione --
class Cancelled(Exception):
    """Sollevata quando l'utente annulla l'elaborazione."""


def estimate_duration(method: str, n_spectra: int) -> float:
    """Stima grossolana dei secondi necessari, per avvertire prima di partire."""
    # secondi ogni 1000 spettri, misurati su spettri da 1015 punti
    per_1k = {
        "rubberband": 0.03, "poly": 0.03, "arpls": 1.1, "iarpls": 1.1,
        "airpls": 0.6, "drpls": 1.2, "aspls": 2.7, "imodpoly": 0.6,
        "snip": 4.2, "goldindec": 13.0, "beads": 710.0,
    }.get(method, 1.0)
    return per_1k * n_spectra / 1000.0


def apply_baseline(
    obj: SpectralMap | Spectrum,
    method: str,
    values: dict[str, Any] | None = None,
    roi: tuple[float, float] | None = None,
    force_nonnegative: bool = False,
    progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
    n_jobs: int = 1,
) -> dict:
    """Applica la correzione in blocchi, riportando l'avanzamento.

    Ogni spettro e' trattato indipendentemente da tutti i metodi disponibili,
    quindi elaborare a blocchi da' lo stesso risultato di una chiamata unica.
    `force_nonnegative` fa eccezione: e' uno scostamento globale e va applicato
    alla fine, dopo che tutti i blocchi sono stati corretti.
    """
    kwargs = to_kwargs(method, values or {})
    roi_x = [[float(roi[0]), float(roi[1])]] if roi else None
    data = obj.data
    single = data.ndim == 1
    matrix = data.reshape(1, -1) if single else data
    n = matrix.shape[0]
    x = np.asarray(obj.x, float)

    started = time.perf_counter()
    done = 0
    for start in range(0, n, CHUNK):
        if cancel is not None and cancel.is_set():
            raise Cancelled(f"annullato dopo {done}/{n} spettri")
        stop = min(start + CHUNK, n)
        block = np.asarray(matrix[start:stop], dtype=np.float32)

        holder = Spectrum(x=x.copy(), data=block.copy(), ignore_sort=True)
        processing.correct_baseline(
            holder, method=method, n_jobs=n_jobs, roi_x=roi_x, **kwargs
        )
        matrix[start:stop] = np.asarray(holder.data, dtype=matrix.dtype).reshape(
            stop - start, -1
        )

        done = stop
        if progress is not None:
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (n - done) if done else 0.0
            progress(done / n, f"{done}/{n} spettri · {eta:.0f}s rimanenti")

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"La correzione con {describe(method, values or {})} ha prodotto valori non "
            "finiti. Con i metodi PLS accade se lambda non viene passato; con airPLS "
            "anche su fondi molto ripidi. Prova arPLS con lambda 10^5."
        )

    if force_nonnegative:
        global_min = float(matrix.min())
        if global_min < 0:
            matrix -= global_min

    return {
        "method": describe(method, values or {}),
        "n_spectra": n,
        "seconds": time.perf_counter() - started,
        "roi": roi,
        "force_nonnegative": force_nonnegative,
    }
