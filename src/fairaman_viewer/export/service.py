"""Pure export functions with no Flet dependency."""

import csv
import io

import numpy as np
from matplotlib.figure import Figure

from ..config import APP_NAME, APP_VERSION
from ..io.fairaman import FairamanFile, flatten_metadata
from ..state import Trace, ViewerState
from ..validation.schema import summary

def figure_bytes(fig: Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    """Serializza una Figure mantenendo l'export indipendente da Flet."""
    buf = io.BytesIO()
    kwargs = {"format": fmt, "bbox_inches": "tight"}
    if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
        kwargs["dpi"] = dpi
    fig.savefig(buf, **kwargs)
    return buf.getvalue()


def spectra_csv_bytes(ff: FairamanFile, traces: list[Trace], include_mean: bool) -> bytes:
    axis = ff.spectral_axis
    columns: list[tuple[str, np.ndarray]] = [("raman_shift", axis)]
    data = np.asarray(ff.obj.data, float)
    if data.ndim == 1:
        columns.append((ff.title, data))
    else:
        if include_mean:
            columns.append(("mean_all", data.mean(0)))
        columns.extend((tr.label, tr.y) for tr in traces)

    if len(columns) == 1:
        raise ValueError("Nessuno spettro selezionato per l'export.")

    sio = io.StringIO(newline="")
    writer = csv.writer(sio)
    writer.writerow([name for name, _ in columns])
    for row in zip(*[values for _, values in columns]):
        writer.writerow([f"{float(v):.8g}" for v in row])
    return sio.getvalue().encode("utf-8")


def map_csv_bytes(image: np.ndarray, label: str) -> bytes:
    arr = np.asarray(image, float)
    sio = io.StringIO(newline="")
    sio.write(f"# FAIRaman map export: {label}\n")
    if arr.ndim == 1:
        writer = csv.writer(sio)
        writer.writerow(["point_index", "value"])
        writer.writerows((i, f"{v:.8g}") for i, v in enumerate(arr))
    elif arr.ndim == 2:
        writer = csv.writer(sio)
        writer.writerow([f"col_{i}" for i in range(arr.shape[1])])
        for row in arr:
            writer.writerow([f"{float(v):.8g}" for v in row])
    else:
        raise ValueError(f"Forma mappa non supportata per CSV: {arr.shape}")
    return sio.getvalue().encode("utf-8")


def report_text(state: ViewerState) -> str:
    ff = state.ff
    if ff is None:
        raise ValueError("Nessun file caricato.")

    lines = [
        f"{APP_NAME} - report",
        f"viewer_version: {APP_VERSION}",
        f"file: {ff.path}",
        f"fairaman_version: {ff.fairaman_version}",
        f"source_format: {ff.source_format}",
        f"coordinate_mode: {ff.coordinate_mode}",
        f"spectra: {ff.n_spectra}",
        "",
        "== METADATI ==",
    ]
    for label, block in (("PROJECT", ff.project), ("SAMPLE", ff.sample), ("ENTRY", ff.entry)):
        lines.append(f"\n[{label}]")
        for key, value in flatten_metadata(block):
            lines.append(f"  {key} = {value}")

    lines += ["", "== MODELLO DATI =="]
    lines += [f"  @{key} = {value}" for key, value in sorted(ff.data_model.items())]

    lines += ["", "== ELABORAZIONI APPLICATE =="]
    lines += ([f"  {i}. {step}" for i, step in enumerate(state.history, 1)]
              or ["  nessuna: dati come letti dal file"])

    lines += ["", "== ANALISI AVANZATE =="]
    if state.analysis_results:
        for result in state.analysis_results.values():
            lines.append(f"  {result.title}")
            for key, value in result.metadata.items():
                lines.append(f"    {key}: {value}")
    else:
        lines.append("  nessuna")

    lines += ["", f"== VALIDAZIONE ==  ({summary(state.issues)})"]
    lines += [f"  {issue}" for issue in state.issues]
    return "\n".join(lines)

