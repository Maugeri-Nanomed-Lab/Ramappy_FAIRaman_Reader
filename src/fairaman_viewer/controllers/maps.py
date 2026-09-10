"""Map computation, rendering and spatial interaction."""

import numpy as np
from matplotlib.widgets import RectangleSelector
from ramappy import analysis
from ramappy.core import SpectralMap

from ..state import Trace

class MapControllerMixin:
    """Behavior mixin for FairamanViewerApp."""

    def _compute_image(self) -> tuple[np.ndarray | None, str]:
            ff = self.state.ff
            if ff is None:
                return None, "nessun file"
            obj = ff.obj
            mode = str(self.image_mode.value or "Intensità totale")

            if mode.startswith("Analisi: "):
                key = mode.removeprefix("Analisi: ")
                for result in self.state.analysis_results.values():
                    if key in result.images:
                        return np.asarray(result.images[key], float), key
                return None, f"risultato '{key}' non disponibile"

            if mode == "Banda spettrale":
                try:
                    lo, hi = float(self.band_lo.value or ""), float(self.band_hi.value or "")
                except ValueError:
                    return None, "intervallo di banda non valido"
                lo, hi = min(lo, hi), max(lo, hi)
                axis = ff.spectral_axis
                sel = (axis >= lo) & (axis <= hi)
                if not sel.any():
                    return None, f"nessun punto tra {lo:.0f} e {hi:.0f} cm⁻¹"
                if isinstance(obj, SpectralMap):
                    img = np.asarray(obj.get_band_intensity(roi_x=[[lo, hi]]))
                else:
                    img = np.asarray(obj.data)[..., sel].mean(-1)
                return np.asarray(img, float), f"banda {lo:.0f}–{hi:.0f} cm⁻¹"

            if mode == "PCA score":
                if not isinstance(obj, SpectralMap) or obj.n_pixels < 4:
                    return None, "PCA disponibile solo su mappe"
                try:
                    k = int(self.pc_index.value or 1)
                except ValueError:
                    return None, "indice PC non valido"
                if k not in self.state.pca_cache:
                    self.set_status(f"Calcolo PCA (PC1–PC{max(k, 3)})…")
                    analysis.principal_components(obj, pca_n_components=max(k, 3))
                    for image in (obj.images or {}).values():
                        name = str(getattr(image, "name", ""))
                        if name.upper().startswith("PC"):
                            try:
                                self.state.pca_cache[int(name[2:])] = np.asarray(image.data, float)
                            except ValueError:
                                pass
                if k not in self.state.pca_cache:
                    return None, f"PC{k} non disponibile"
                return self.state.pca_cache[k], f"PCA score PC{k}"

            data = np.asarray(obj.data, float)
            total = data.sum(-1)
            if isinstance(obj, SpectralMap):
                total = total.reshape(obj.map_shape)
            return total, "intensità totale"

    def refresh_map(self) -> None:
            ff = self.state.ff
            if ff is None:
                return
            ax = self.map_ax
            ax.clear()
            if self._colorbar is not None:
                try:
                    self._colorbar.remove()
                except Exception:
                    pass
                self._colorbar = None
            if self._selector is not None:
                try:
                    self._selector.set_active(False)
                    self._selector.disconnect_events()
                except Exception:
                    pass
                self._selector = None

            image, label = self._compute_image()
            if image is None:
                ax.set_axis_off()
                ax.text(0.5, 0.5, label, ha="center", va="center", color="#b00020", transform=ax.transAxes)
                self.state.image_data = None
                self.safe_update(self.map_chart)
                return

            self.state.image_data = np.asarray(image, float)
            self.state.image_label = label

            if ff.coordinate_mode == "point_coordinates" and ff.point_coords is not None:
                values = np.asarray(image).ravel()
                sc = ax.scatter(
                    ff.point_coords[:, 0], ff.point_coords[:, 1], c=values,
                    cmap=str(self.cmap.value), s=48, edgecolor="k", linewidth=0.3,
                )
                self._colorbar = self.map_fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
                ax.set_xlabel("x (µm)")
                ax.set_ylabel("y (µm)")
                ax.set_title(f"{label} · {ff.n_spectra} punti", fontsize=10)
                ax.set_aspect("equal", adjustable="datalim")
                self._draw_analysis_markers(ax)
                self.safe_update(self.map_chart)
                return

            if not ff.is_map:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "File a spettro singolo\nnessuna mappa da visualizzare", ha="center", va="center", fontsize=11, color="#666", transform=ax.transAxes)
                self.state.image_data = None
                self.safe_update(self.map_chart)
                return

            extent = ff.spatial_extent()
            img = image if ff.y_ascending() else np.flipud(image)
            wl = ff.aux_images.get("white_light")
            if self.show_wl.value and wl is not None and wl.extent is not None and extent:
                ax.imshow(wl.array, extent=wl.extent, origin="lower", interpolation="bilinear")
                alpha = float(self.wl_alpha.value or 0.55)
            else:
                alpha = 1.0

            finite = image[np.isfinite(image)]
            vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
            im = ax.imshow(
                img,
                extent=extent,
                origin="lower",
                cmap=str(self.cmap.value),
                alpha=alpha,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            self._colorbar = self.map_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_title(f"{label} · {image.shape[0]}×{image.shape[1]} px", fontsize=10)
            if extent:
                ax.set_xlabel("x (µm)")
                ax.set_ylabel("y (µm)")
            else:
                ax.set_axis_off()

            self._draw_analysis_markers(ax)
            # MatplotlibChart inoltra gli eventi del canvas. Manteniamo RectangleSelector
            # come comportamento equivalente al viewer Tkinter; se un backend Flet futuro
            # cambiasse la semantica del tasto destro, questa e' l'unica parte da adattare.
            try:
                self._selector = RectangleSelector(
                    ax,
                    self._on_rectangle,
                    useblit=False,
                    button=[3],
                    minspanx=2,
                    minspany=2,
                    spancoords="pixels",
                    interactive=False,
                )
            except Exception:
                self._selector = None
            self.safe_update(self.map_chart)

    def _draw_analysis_markers(self, ax) -> None:
            result = self.state.analysis_results.get("nfindr")
            if result and result.markers:
                for i, (x, y) in enumerate(result.markers):
                    ax.plot(x, y, marker="o", ms=8, mfc="none", mec="white", mew=1.6)
                    ax.text(x, y, f" EM{i+1}", color="white", fontsize=7, weight="bold")

    def _pixel_from_event(self, event) -> tuple[int, int] | None:
            ff = self.state.ff
            if ff is None or not ff.is_map or event.xdata is None or event.ydata is None:
                return None
            ny, nx = ff.map_shape()
            extent = ff.spatial_extent()
            if extent:
                left, right, bottom, top = extent
                col = int((event.xdata - left) / (right - left) * nx)
                row = int((event.ydata - bottom) / (top - bottom) * ny)
                if not ff.y_ascending():
                    row = ny - 1 - row
            else:
                col, row = int(round(event.xdata)), int(round(event.ydata))
            return (row, col) if 0 <= row < ny and 0 <= col < nx else None

    def _on_map_click(self, event) -> None:
            ff = self.state.ff
            if ff is None or event.inaxes is not self.map_ax or event.button != 1:
                return
            if ff.coordinate_mode == "point_coordinates" and ff.point_coords is not None:
                if event.xdata is None or event.ydata is None:
                    return
                d = np.hypot(ff.point_coords[:, 0] - event.xdata, ff.point_coords[:, 1] - event.ydata)
                idx = int(np.argmin(d))
                self.add_trace(np.asarray(ff.obj.data)[idx], f"punto {idx}", tuple(ff.point_coords[idx]))
                return
            pixel = self._pixel_from_event(event)
            if pixel is None:
                return
            row, col = pixel
            self.add_trace(np.asarray(ff.obj.cube)[row, col], f"px ({col}, {row})")

    def _on_rectangle(self, click, release) -> None:
            ff = self.state.ff
            if ff is None or not ff.is_map:
                return
            p1, p2 = self._pixel_from_event(click), self._pixel_from_event(release)
            if p1 is None or p2 is None:
                return
            r0, r1 = sorted((p1[0], p2[0]))
            c0, c1 = sorted((p1[1], p2[1]))
            block = np.asarray(ff.obj.cube)[r0:r1 + 1, c0:c1 + 1].reshape(-1, ff.spectral_axis.size)
            if not block.size:
                return
            agg = str(self.agg_mode.value or "mean")
            spec = {"mean": np.mean, "median": np.median, "max": np.max}[agg](block, axis=0)
            self.add_trace(spec, f"{agg} di {block.shape[0]} px [{c0}:{c1}, {r0}:{r1}]")

    def add_trace(self, values, label: str, marker=None) -> None:
            self.state.traces.append(Trace(np.asarray(values, float), label, marker))
            self.state.traces = self.state.traces[-8:]
            self.refresh_spectra()
            self.set_status(f"Spettro aggiunto: {label}")

    def clear_traces(self) -> None:
            self.state.traces.clear()
            self.refresh_spectra()
