"""Algoritmi di normalizzazione dell'intensità e helper di esecuzione.

La maggior parte dei metodi normalizza ogni spettro in modo indipendente
(row-wise). MSC e PQN sono invece "a due passate": stimano prima un riferimento
globale sull'intero dataset, poi trasformano pixel per pixel. Vedi
``docs/NORMALIZATION.md``.

Il modulo non importa Flet ne' ramappy: opera su array NumPy e su un oggetto
spettrale di cui usa solo ``.data`` e ``.x`` (stessa convenzione di
``processing/baseline.py``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CHUNK = 4000  # righe per blocco: la normalizzazione e' O(n) e leggera
_EPS = 1e-12


class Cancelled(Exception):
    """Sollevata quando l'utente annulla la normalizzazione."""


# ---------------------------------------------------------------- parametri --
@dataclass
class Param:
    """Un parametro di un metodo, con quanto serve alla GUI per disegnarlo."""

    key: str
    label: str
    kind: str  # "float" | "int" | "choice" | "bool"
    default: Any = None
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""
    optional: bool = False


@dataclass
class NormalizationMethod:
    key: str
    label: str
    params: list[Param] = field(default_factory=list)
    note: str = ""
    scope_note: str = ""  # come il metodo usa la ROI
    two_pass: bool = False  # richiede un riferimento globale sul dataset


_CENTER = Param(
    "center", "Centro banda (cm⁻¹)", "float", None, lo=0.0, hi=1e5,
    help="Posizione nominale della banda di riferimento (es. 1003 fenilalanina, 520.7 Si).",
)
_MODE = Param(
    "mode", "Misura", "choice", "height", choices=("height", "area"),
    help="Altezza al picco oppure area integrata sulla finestra.",
)
_WINDOW = Param(
    "window", "Semi-finestra (cm⁻¹)", "float", 8.0, lo=0.5, hi=200.0,
    help="Semi-ampiezza della finestra di integrazione (mode=area).",
)
_SEARCH = Param(
    "search", "Ricerca max (± cm⁻¹)", "float", 4.0, lo=0.0, hi=100.0, optional=True,
    help="Cerca il massimo entro ± questo intervallo dal centro (mode=height). 0 = centro esatto.",
)
_LOCAL_BL = Param(
    "local_baseline", "Baseline locale", "bool", False,
    help="Sottrae una retta tra gli estremi della finestra prima di misurare la banda. "
    "Utile se lo spettro non e' ancora corretto in linea di base.",
)
_PQN_PRIOR = Param(
    "prior", "Pre-normalizzazione", "choice", "area", choices=("area", "l2", "none"),
    help="Normalizzazione grezza applicata prima di stimare i quozienti PQN.",
)


METHODS: dict[str, NormalizationMethod] = {
    m.key: m
    for m in [
        NormalizationMethod(
            "l2", "Vettoriale L2 (euclidea)",
            note="‖s‖₂ = 1. Non cambia la forma, solo la scala.",
            scope_note="Con ROI: divide per la norma calcolata sulla ROI.",
        ),
        NormalizationMethod(
            "l1", "L1 (somma unitaria)",
            note="Σ|s| = 1.",
            scope_note="Con ROI: usa la somma sulla ROI.",
        ),
        NormalizationMethod(
            "max", "Massimo",
            note="max|s| = 1. Sensibile agli spike: applicare dopo il despike.",
            scope_note="Con ROI: massimo sulla ROI.",
        ),
        NormalizationMethod(
            "area", "Area (AUC, integrazione reale)",
            note="Integrale = 1 sull'asse spettrale reale. Applicare dopo la baseline.",
            scope_note="Con ROI: integrale sulla ROI.",
        ),
        NormalizationMethod(
            "rms", "RMS",
            note="√⟨s²⟩ = 1.",
            scope_note="Con ROI: RMS sulla ROI.",
        ),
        NormalizationMethod(
            "snv", "SNV",
            note="(s − media) / deviazione, per spettro. Rimuove offset e scala.",
            scope_note="Con ROI: media e deviazione dalla ROI, applicate a tutto lo spettro.",
        ),
        NormalizationMethod(
            "snv_robust", "SNV robusto (mediana / MAD)",
            note="Variante SNV che resiste a bande intense e outlier.",
            scope_note="Con ROI: mediana e MAD dalla ROI.",
        ),
        NormalizationMethod(
            "minmax", "Min–max [0, 1]",
            note="Utile per la visualizzazione, non per l'analisi multivariata.",
            scope_note="Con ROI: minimo e massimo dalla ROI.",
        ),
        NormalizationMethod(
            "reference_band", "Banda di riferimento",
            params=[_CENTER, _MODE, _WINDOW, _SEARCH, _LOCAL_BL],
            note="Divide l'intero spettro per l'intensita' di una banda scelta (es. 1003 "
            "fenilalanina, 520.7 Si). Riscala soltanto: non sottrae il fondo, quindi va "
            "usata dopo la baseline. La ROI e' ignorata: conta la finestra attorno al centro.",
        ),
        NormalizationMethod(
            "msc", "MSC (multiplicative scatter correction)",
            two_pass=True,
            note="Regredisce ogni spettro sul riferimento (media del dataset) e corregge "
            "pendenza e offset. Il riferimento e' registrato nella history.",
            scope_note="Con ROI: la regressione usa solo la ROI, la correzione tutto lo spettro.",
        ),
        NormalizationMethod(
            "pqn", "PQN (probabilistic quotient normalization)",
            params=[_PQN_PRIOR], two_pass=True,
            note="Fattore = mediana del rapporto spettro / riferimento (mediana del dataset). "
            "Robusto quando poche bande dominano lo spettro.",
        ),
    ]
}

METHOD_KEYS = list(METHODS)


def default_params(method: str) -> dict[str, Any]:
    return {p.key: p.default for p in METHODS[method].params}


def to_kwargs(method: str, values: dict[str, Any]) -> dict[str, Any]:
    """Converte i valori grezzi della GUI nei tipi attesi dai kernel."""
    out: dict[str, Any] = {}
    for param in METHODS[method].params:
        value = values.get(param.key, param.default)
        if value is None or value == "":
            if param.optional or param.default is None:
                continue
            value = param.default
        if param.kind == "int":
            out[param.key] = int(value)
        elif param.kind == "float":
            out[param.key] = float(value)
        elif param.kind == "bool":
            out[param.key] = value if isinstance(value, bool) else str(value).strip().lower() in {
                "1", "true", "vero", "si", "sì", "yes", "on",
            }
        else:
            out[param.key] = value
    return out


def describe(method: str, values: dict[str, Any]) -> str:
    """Riga leggibile con metodo e parametri effettivi, per la history."""
    kwargs = to_kwargs(method, values)
    if not kwargs:
        return method
    bits = [
        f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in kwargs.items()
    ]
    return f"{method}({', '.join(bits)})"


# ------------------------------------------------------------------- kernel --
def _roi_mask(x: np.ndarray, roi: tuple[float, float] | None) -> np.ndarray:
    if roi is None:
        return np.ones(x.shape, dtype=bool)
    lo, hi = min(roi), max(roi)
    mask = (x >= lo) & (x <= hi)
    if not mask.any():
        raise ValueError(f"La ROI {lo:.0f}–{hi:.0f} cm⁻¹ non contiene punti spettrali.")
    return mask


def _apply_divisor(block: np.ndarray, divisor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide ogni riga per il suo scalare, lasciando invariate le righe degeneri."""
    divisor = np.asarray(divisor, float)
    bad = ~np.isfinite(divisor) | (np.abs(divisor) < _EPS)
    safe = np.where(bad, 1.0, divisor)
    out = block / safe[:, None]
    out[bad] = block[bad]
    return out, bad


def _k_l2(block, x, mask):
    return _apply_divisor(block, np.sqrt(np.nansum(block[:, mask] ** 2, axis=1)))


def _k_l1(block, x, mask):
    return _apply_divisor(block, np.nansum(np.abs(block[:, mask]), axis=1))


def _k_max(block, x, mask):
    return _apply_divisor(block, np.nanmax(np.abs(block[:, mask]), axis=1))


def _k_rms(block, x, mask):
    return _apply_divisor(block, np.sqrt(np.nanmean(block[:, mask] ** 2, axis=1)))


def _k_area(block, x, mask):
    xm = x[mask]
    order = np.argsort(xm)
    integral = np.trapezoid(block[:, mask][:, order], x=xm[order], axis=1)
    return _apply_divisor(block, np.abs(integral))


def _k_minmax(block, x, mask):
    lo = np.nanmin(block[:, mask], axis=1)
    hi = np.nanmax(block[:, mask], axis=1)
    span = hi - lo
    bad = ~np.isfinite(span) | (np.abs(span) < _EPS)
    safe = np.where(bad, 1.0, span)
    out = (block - lo[:, None]) / safe[:, None]
    out[bad] = block[bad]
    return out, bad


def _k_snv(block, x, mask):
    mu = np.nanmean(block[:, mask], axis=1)
    sd = np.nanstd(block[:, mask], axis=1)
    bad = ~np.isfinite(sd) | (sd < _EPS)
    safe = np.where(bad, 1.0, sd)
    out = (block - mu[:, None]) / safe[:, None]
    out[bad] = block[bad]
    return out, bad


def _k_snv_robust(block, x, mask):
    med = np.nanmedian(block[:, mask], axis=1)
    mad = np.nanmedian(np.abs(block[:, mask] - med[:, None]), axis=1) * 1.4826
    bad = ~np.isfinite(mad) | (mad < _EPS)
    safe = np.where(bad, 1.0, mad)
    out = (block - med[:, None]) / safe[:, None]
    out[bad] = block[bad]
    return out, bad


def _k_reference_band(block, x, params):
    center = float(params["center"])
    window = float(params.get("window", 8.0) or 8.0)
    search = float(params.get("search", 0.0) or 0.0)
    mode = params.get("mode", "height")
    local_baseline = bool(params.get("local_baseline", True))

    sel = (x >= center - window) & (x <= center + window)
    if not sel.any():
        raise ValueError(
            f"La finestra attorno a {center:.0f} cm⁻¹ (± {window:.0f}) non contiene punti spettrali."
        )
    xs = x[sel]
    seg = block[:, sel].astype(float, copy=True)

    if local_baseline and xs.size >= 2:
        x0, x1 = float(xs[0]), float(xs[-1])
        y0, y1 = seg[:, 0], seg[:, -1]
        slope = (y1 - y0) / (x1 - x0 if abs(x1 - x0) > _EPS else 1.0)
        seg = seg - (y0[:, None] + slope[:, None] * (xs[None, :] - x0))

    if mode == "area":
        order = np.argsort(xs)
        divisor = np.abs(np.trapezoid(seg[:, order], x=xs[order], axis=1))
    else:
        if search > 0:
            near = (xs >= center - search) & (xs <= center + search)
            region = seg[:, near] if near.any() else seg
        else:
            region = seg[:, [int(np.argmin(np.abs(xs - center)))]]
        divisor = np.nanmax(region, axis=1)

    return _apply_divisor(block, divisor)


def _prenormalize(matrix: np.ndarray, x: np.ndarray, prior: str) -> np.ndarray:
    mask = np.ones(x.shape, dtype=bool)
    if prior == "area":
        return _k_area(matrix, x, mask)[0]
    if prior == "l2":
        return _k_l2(matrix, x, mask)[0]
    return matrix


def _k_msc(block, x, mask, reference):
    ref = np.asarray(reference["reference"], float)[mask]
    design = np.column_stack([np.ones_like(ref), ref])  # colonna offset + colonna pendenza
    coef, *_ = np.linalg.lstsq(design, block[:, mask].T, rcond=None)  # (2, n)
    offset, slope = coef[0], coef[1]
    bad = ~np.isfinite(slope) | (np.abs(slope) < _EPS)
    safe = np.where(bad, 1.0, slope)
    out = (block - offset[:, None]) / safe[:, None]
    out[bad] = block[bad]
    return out, bad


def _k_pqn(block, x, mask, reference):
    ref = np.asarray(reference["reference"], float)
    prior = reference.get("prior", "area")
    pre = _prenormalize(block, x, prior)
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = pre[:, mask] / ref[None, mask]
    quotient[~np.isfinite(quotient)] = np.nan
    factor = np.nanmedian(quotient, axis=1)
    return _apply_divisor(pre, factor)


_KERNELS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "l2": _k_l2, "l1": _k_l1, "max": _k_max, "area": _k_area, "rms": _k_rms,
    "minmax": _k_minmax, "snv": _k_snv, "snv_robust": _k_snv_robust,
}


# ------------------------------------------------------------- orchestrazione --
def fit_reference(
    matrix: np.ndarray,
    x: np.ndarray,
    method: str,
    values: dict[str, Any] | None = None,
    roi: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """Stima il riferimento globale per i metodi a due passate (MSC, PQN)."""
    if method not in METHODS or not METHODS[method].two_pass:
        return None
    data = np.asarray(matrix, float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    finite = np.isfinite(data).all(axis=1)
    source = data[finite] if finite.any() else data
    if method == "msc":
        return {"kind": "msc", "reference": np.nanmean(source, axis=0)}
    if method == "pqn":
        prior = to_kwargs("pqn", values or {}).get("prior", "area")
        pre = _prenormalize(source, np.asarray(x, float), prior)
        return {"kind": "pqn", "prior": prior, "reference": np.nanmedian(pre, axis=0)}
    return None


def transform(
    block: np.ndarray,
    x: np.ndarray,
    method: str,
    values: dict[str, Any] | None = None,
    roi: tuple[float, float] | None = None,
    reference: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalizza un blocco (n, p). Ritorna ``(normalizzato, maschera_degeneri)``."""
    data = np.asarray(block, float)
    x = np.asarray(x, float)
    if method == "reference_band":
        return _k_reference_band(data, x, to_kwargs("reference_band", values or {}))
    if method in ("msc", "pqn"):
        if reference is None:
            raise ValueError(f"{method} richiede un riferimento gia' stimato.")
        kernel = _k_msc if method == "msc" else _k_pqn
        return kernel(data, x, _roi_mask(x, roi), reference)
    if method not in _KERNELS:
        raise ValueError(f"Metodo di normalizzazione sconosciuto: {method}")
    return _KERNELS[method](data, x, _roi_mask(x, roi))


def estimate_normalization(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    values: dict[str, Any] | None = None,
    roi: tuple[float, float] | None = None,
    reference: dict[str, Any] | None = None,
) -> np.ndarray:
    """Spettro normalizzato per l'anteprima, senza toccare i dati originali."""
    out, _ = transform(np.asarray(y, float).reshape(1, -1), x, method, values, roi, reference)
    return out.reshape(-1)


def apply_normalization(
    obj: Any,
    method: str,
    values: dict[str, Any] | None = None,
    roi: tuple[float, float] | None = None,
    reference: dict[str, Any] | None = None,
    progress: Callable[[float, str], None] | None = None,
    cancel: Any | None = None,
) -> dict[str, Any]:
    """Applica la normalizzazione in blocchi, riportando l'avanzamento.

    Muta ``obj.data`` in place. I metodi per-spettro sono indipendenti tra
    blocchi; MSC e PQN stimano prima il riferimento sull'intero dataset.
    """
    if method not in METHODS:
        raise ValueError(f"Metodo di normalizzazione sconosciuto: {method}")
    values = values or {}
    x = np.asarray(obj.x, float)
    data = obj.data
    matrix = data.reshape(1, -1) if data.ndim == 1 else data
    n = matrix.shape[0]

    started = time.perf_counter()
    if METHODS[method].two_pass and reference is None:
        if progress is not None:
            progress(0.0, "stima del riferimento sul dataset…")
        reference = fit_reference(matrix, x, method, values, roi)
        if reference is None:
            raise ValueError(f"Impossibile stimare il riferimento per {method}.")

    degenerate = 0
    done = 0
    for start in range(0, n, CHUNK):
        if cancel is not None and cancel.is_set():
            raise Cancelled(f"annullato dopo {done}/{n} spettri")
        stop = min(start + CHUNK, n)
        out, bad = transform(np.asarray(matrix[start:stop], float), x, method, values, roi, reference)
        matrix[start:stop] = out.astype(matrix.dtype, copy=False)
        degenerate += int(np.count_nonzero(bad))
        done = stop
        if progress is not None:
            progress(done / n, f"{done}/{n} spettri")

    if degenerate >= n:
        raise ValueError(
            f"Nessuno spettro e' normalizzabile con {describe(method, values)}: "
            "verifica il metodo, la ROI o la banda di riferimento."
        )

    info: dict[str, Any] = {
        "method": describe(method, values),
        "n_spectra": n,
        "seconds": time.perf_counter() - started,
        "roi": roi,
        "degenerate": degenerate,
        "two_pass": METHODS[method].two_pass,
    }
    if reference is not None:
        info["reference"] = reference.get("kind")
        if "prior" in reference:
            info["reference_prior"] = reference["prior"]
    return info
