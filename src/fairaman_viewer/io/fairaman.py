"""Read FAIRaman HDF5 files and expose a small domain model."""


import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ramappy import Spectrum
from ramappy.core import SpectralMap
from ramappy.io import read_file

# Attributi di ENTRY/data che descrivono l'oggetto spettrale.
DATA_MODEL_ATTRS = (
    "coordinate_mode",
    "coordinate_source",
    "coordinate_units",
    "coordinate_validated",
    "geometry_warning",
    "reshape_applied",
    "reshape_source",
    "nx",
    "ny",
    "n_points",
    "n_wavenumbers",
    "spectral_count",
    "signal",
    "axes",
)

MAX_INLINE_ARRAY = 128  # elementi oltre i quali un dataset non finisce nei metadati


# --------------------------------------------------------------------------
# helper di decodifica
# --------------------------------------------------------------------------
def decode(value: Any) -> Any:
    """Riporta un valore h5py a un tipo Python leggibile."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, np.ndarray)):
        return [decode(v) for v in np.asarray(value).ravel().tolist()]
    return value


def _dataset_value(ds: h5py.Dataset) -> Any | None:
    arr = np.asarray(ds)
    if arr.shape == () or arr.size == 1:
        return decode(arr.reshape(-1)[0] if arr.size == 1 else arr.item())
    if arr.size <= MAX_INLINE_ARRAY:
        return decode(arr)
    return None


def _group_to_dict(group: h5py.Group | None, skip: set[str] | None = None) -> dict:
    """Legge ricorsivamente un gruppo HDF5 in un dizionario annidato."""
    if group is None:
        return {}
    skip = skip or set()
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        if key == "NX_class":
            continue
        out[f"@{key}"] = decode(value)
    for key, item in group.items():
        if key in skip:
            continue
        if isinstance(item, h5py.Group):
            nested = _group_to_dict(item)
            if nested:
                out[key] = nested
        elif isinstance(item, h5py.Dataset):
            parsed = _dataset_value(item)
            if parsed is not None:
                out[key] = parsed
    return out


# --------------------------------------------------------------------------
# contenitore
# --------------------------------------------------------------------------
@dataclass
class AuxImage:
    """Immagine ausiliaria con la sua calibrazione spaziale, se presente."""

    name: str
    array: np.ndarray
    # extent nella convenzione matplotlib (left, right, bottom, top), in micrometri,
    # con l'array gia' orientato per imshow(origin="lower")
    extent: tuple[float, float, float, float] | None = None
    attrs: dict = field(default_factory=dict)


@dataclass
class FairamanFile:
    path: Path
    obj: SpectralMap | Spectrum
    project: dict
    sample: dict
    entry: dict
    data_model: dict
    aux_images: dict[str, AuxImage]
    coordinate_mode: str
    point_coords: np.ndarray | None = None  # (n_points, 2) solo in point_coordinates
    fairaman_version: str | None = None
    source_format: str | None = None
    warnings: list[str] = field(default_factory=list)

    # -- comodita' ---------------------------------------------------------
    @property
    def is_map(self) -> bool:
        return isinstance(self.obj, SpectralMap) and self.obj.n_pixels > 1

    @property
    def n_spectra(self) -> int:
        return int(self.obj.n_pixels)

    @property
    def spectral_axis(self) -> np.ndarray:
        return np.asarray(self.obj.spectral_axis)

    @property
    def title(self) -> str:
        return str(self.entry.get("title") or self.path.name)

    def map_shape(self) -> tuple[int, int] | None:
        if isinstance(self.obj, SpectralMap):
            return tuple(self.obj.map_shape)
        return None

    def spatial_extent(self) -> tuple[float, float, float, float] | None:
        """Extent (left, right, bottom, top) in micrometri della mappa."""
        gx, gy = self._grid_coords
        if gx is None or gy is None or gx.size < 2 or gy.size < 2:
            return None
        dx = float(np.median(np.abs(np.diff(gx)))) / 2.0
        dy = float(np.median(np.abs(np.diff(gy)))) / 2.0
        return (
            float(gx.min()) - dx,
            float(gx.max()) + dx,
            float(gy.min()) - dy,
            float(gy.max()) + dy,
        )

    def y_ascending(self) -> bool:
        """True se la riga 0 del cubo corrisponde alla y minore."""
        _, gy = self._grid_coords
        if gy is None or gy.size < 2:
            return True
        return bool(gy[0] <= gy[-1])

    _grid_coords: tuple[np.ndarray | None, np.ndarray | None] = (None, None)


# --------------------------------------------------------------------------
# loader
# --------------------------------------------------------------------------
def load_fairaman(path: str | Path) -> FairamanFile:
    """Carica un file FAIRaman .h5 e restituisce un FairamanFile."""
    path = Path(path)
    msgs: list[str] = []

    with h5py.File(path, "r") as f:
        if "ENTRY/data/intensity" not in f:
            raise ValueError(
                "Non sembra un file FAIRaman: manca ENTRY/data/intensity."
            )

        root_attrs = {k: decode(v) for k, v in f.attrs.items()}
        data_grp = f["ENTRY/data"]

        data_model = {
            key: decode(data_grp.attrs[key])
            for key in DATA_MODEL_ATTRS
            if key in data_grp.attrs
        }
        for key in ("spectral_count", "n_points"):
            if key in data_grp and key not in data_model:
                data_model[key] = _dataset_value(data_grp[key])

        mode = str(data_model.get("coordinate_mode", "")).strip() or None
        intensity_shape = data_grp["intensity"].shape
        raman = np.asarray(data_grp["raman_shift"]).squeeze()

        grid_x = np.asarray(data_grp["x"]).squeeze() if "x" in data_grp else None
        grid_y = np.asarray(data_grp["y"]).squeeze() if "y" in data_grp else None

        project = _group_to_dict(f.get("PROJECT"))
        sample = _group_to_dict(f.get("SAMPLE"))
        entry = _group_to_dict(f.get("ENTRY"), skip={"data", "auxiliary"})

        aux_images = _read_aux_images(f)

        # modalita' dedotta dalla forma se l'attributo manca o e' incoerente
        if intensity_shape and len(intensity_shape) == 2:
            inferred = "point_coordinates"
        elif len(intensity_shape) == 3:
            inferred = "regular_grid"
        else:
            inferred = "single"
        if mode and mode != inferred and not (mode == "regular_grid" and inferred == "single"):
            msgs.append(
                f"coordinate_mode dichiarato '{mode}' ma la forma di intensity "
                f"{intensity_shape} corrisponde a '{inferred}'. Uso '{inferred}'."
            )
        mode = inferred

        point_coords = None
        if mode == "point_coordinates":
            n_pts = intensity_shape[0]
            px = grid_x if grid_x is not None and grid_x.size == n_pts else np.arange(n_pts)
            py = grid_y if grid_y is not None and grid_y.size == n_pts else np.zeros(n_pts)
            point_coords = np.column_stack([np.asarray(px, float), np.asarray(py, float)])

        intensity = np.asarray(data_grp["intensity"])

    # --- costruzione dell'oggetto ramappy ---------------------------------
    if mode == "regular_grid" and len(intensity_shape) == 3:
        # riuso del reader nativo: gestisce la permutazione degli assi via @axes,
        # la white-light e la SpatialGrid.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = read_file(str(path), format="hdf5_nexus", as_hsi=True)
    elif mode == "point_coordinates":
        obj = Spectrum(
            x=raman,
            data=intensity.astype(np.float32),
            name=str(entry.get("title") or path.stem),
        )
        msgs.append(
            f"Modalita' point_coordinates: {intensity.shape[0]} spettri non su griglia, "
            "visualizzati come scatter."
        )
    else:
        obj = Spectrum(
            x=raman,
            data=np.asarray(intensity, dtype=np.float32).reshape(-1),
            name=str(entry.get("title") or path.stem),
        )

    if raman.size > 1 and raman[0] > raman[-1]:
        msgs.append(
            "raman_shift memorizzato in ordine decrescente: l'asse e' stato "
            "riordinato in senso crescente per la visualizzazione."
        )

    ff = FairamanFile(
        path=path,
        obj=obj,
        project=project,
        sample=sample,
        entry=entry,
        data_model=data_model,
        aux_images=aux_images,
        coordinate_mode=mode,
        point_coords=point_coords,
        fairaman_version=root_attrs.get("fairaman_version"),
        source_format=root_attrs.get("source_format"),
        warnings=msgs,
    )
    ff._grid_coords = (grid_x, grid_y)
    return ff


def _read_aux_images(f: h5py.File) -> dict[str, AuxImage]:
    """Legge ENTRY/auxiliary/* recuperando la calibrazione spaziale."""
    out: dict[str, AuxImage] = {}
    aux = f.get("ENTRY/auxiliary")
    if aux is None:
        return out

    for name, grp in aux.items():
        if not isinstance(grp, h5py.Group) or "image" not in grp:
            continue
        ds = grp["image"]
        arr = np.asarray(ds)
        attrs = {k: decode(v) for k, v in ds.attrs.items()}

        extent = None
        row0_at_top = str(attrs.get("origin", "")).lower() == "upper"

        # 1) coordinate esplicite nel gruppo
        if "x" in grp and "y" in grp:
            gx = np.asarray(grp["x"]).squeeze()
            gy = np.asarray(grp["y"]).squeeze()
            if gx.ndim == 1 and gy.ndim == 1 and gx.size > 1 and gy.size > 1:
                extent = (
                    float(gx.min()),
                    float(gx.max()),
                    float(gy.min()),
                    float(gy.max()),
                )
                # se le y decrescono, la riga 0 sta in alto
                row0_at_top = bool(gy[0] > gy[-1]) or row0_at_top

        # 2) fallback su origine + passo per pixel
        if extent is None and {"x0_um", "y0_um", "dx_um_per_px", "dy_um_per_px"} <= attrs.keys():
            h, w = arr.shape[0], arr.shape[1]
            x0 = float(attrs["x0_um"])
            y0 = float(attrs["y0_um"])
            extent = (
                x0,
                x0 + w * float(attrs["dx_um_per_px"]),
                y0,
                y0 + h * float(attrs["dy_um_per_px"]),
            )

        # orientamento uniforme: imshow(origin="lower")
        if row0_at_top:
            arr = np.flipud(arr)

        out[name] = AuxImage(name=name, array=arr, extent=extent, attrs=attrs)
    return out


def flatten_metadata(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    """Appiattisce un dizionario annidato in coppie (percorso, valore)."""
    rows: list[tuple[str, Any]] = []
    for key, value in d.items():
        full = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten_metadata(value, full))
        else:
            rows.append((full, value))
    return rows
