"""Map computation, rendering and QuPath-style spatial interaction.

This controller keeps scientific image computation separate from UI composition,
while providing direct map navigation and ROI tools:

- mouse wheel: zoom around cursor
- Pan tool: left-button drag
- Pixel tool: left click adds the selected spectrum
- Rectangle / polygon / freehand ROI tools
- persistent ROI overlays and ROI-aggregated spectra
"""

from __future__ import annotations

import time

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import LassoSelector, PolygonSelector, RectangleSelector
from ramappy import analysis
from ramappy.core import SpectralMap

from ..state import Trace


class MapControllerMixin:
    """Behavior mixin for FairamanViewerApp."""

    # ------------------------------------------------------------------
    # Image computation / rendering
    # ------------------------------------------------------------------

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
                lo = float(self.band_lo.value or "")
                hi = float(self.band_hi.value or "")
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

        # ROIs belong to one acquisition.  Opening another file automatically
        # starts with a clean ROI layer.
        token = id(ff)
        if getattr(self, "_roi_file_token", None) != token:
            self._roi_file_token = token
            self._rois = []
            self._roi_counter = 0
            self._map_home_view = None
            self._pan_active = False
            self._update_roi_ui()

        ax = self.map_ax
        ax.clear()

        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None

        self._deactivate_map_selector()

        image, label = self._compute_image()
        if image is None:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                color="#b00020",
                transform=ax.transAxes,
            )
            self.state.image_data = None
            self.render_figure(self.map_fig, self.map_chart)
            return

        self.state.image_data = np.asarray(image, float)
        self.state.image_label = label

        if ff.coordinate_mode == "point_coordinates" and ff.point_coords is not None:
            values = np.asarray(image).ravel()
            sc = ax.scatter(
                ff.point_coords[:, 0],
                ff.point_coords[:, 1],
                c=values,
                cmap=str(self.cmap.value),
                s=48,
                edgecolor="k",
                linewidth=0.3,
            )
            self._colorbar = self.map_fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.set_title(f"{label} · {ff.n_spectra} punti", fontsize=10)
            ax.set_aspect("equal", adjustable="datalim")
            self._draw_analysis_markers(ax)
            self._draw_saved_rois(ax)
            self._remember_home_view()
            self._activate_selector_for_current_tool()
            self.render_figure(self.map_fig, self.map_chart)
            return

        if not ff.is_map:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "File a spettro singolo\nnessuna mappa da visualizzare",
                ha="center",
                va="center",
                fontsize=11,
                color="#666",
                transform=ax.transAxes,
            )
            self.state.image_data = None
            self.render_figure(self.map_fig, self.map_chart)
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
        self._draw_saved_rois(ax)
        self._remember_home_view()
        self._activate_selector_for_current_tool()
        self.render_figure(self.map_fig, self.map_chart)

    # ------------------------------------------------------------------
    # Existing analysis markers / pixel conversion
    # ------------------------------------------------------------------

    def _draw_analysis_markers(self, ax) -> None:
        result = self.state.analysis_results.get("nfindr")
        if result and result.markers:
            for i, (x, y) in enumerate(result.markers):
                ax.plot(x, y, marker="o", ms=8, mfc="none", mec="white", mew=1.6)
                ax.text(x, y, f" EM{i + 1}", color="white", fontsize=7, weight="bold")

    def _pixel_from_event(self, event) -> tuple[int, int] | None:
        ff = self.state.ff
        if ff is None or not ff.is_map or event.xdata is None or event.ydata is None:
            return None

        ny, nx = ff.map_shape()
        extent = ff.spatial_extent()
        if extent:
            left, right, bottom, top = extent
            col = int((event.xdata - left) / (right - left) * nx)
            display_row = int((event.ydata - bottom) / (top - bottom) * ny)
            row = display_row if ff.y_ascending() else ny - 1 - display_row
        else:
            col = int(round(event.xdata))
            display_row = int(round(event.ydata))
            row = display_row if ff.y_ascending() else ny - 1 - display_row

        return (row, col) if 0 <= row < ny and 0 <= col < nx else None

    # ------------------------------------------------------------------
    # QuPath-style tool modes
    # ------------------------------------------------------------------

    def set_map_tool(self, mode: str) -> None:
        """Select direct interaction mode for the Raman map."""
        valid = {"pixel", "pan", "rectangle", "polygon", "freehand"}
        if mode not in valid:
            return

        self._map_tool = mode
        self._pan_active = False
        self._deactivate_map_selector()
        self._activate_selector_for_current_tool()

        names = {
            "pixel": "Pixel",
            "pan": "Pan",
            "rectangle": "ROI rettangolo",
            "polygon": "ROI poligono",
            "freehand": "ROI libera",
        }
        if hasattr(self, "map_tool_label"):
            self.map_tool_label.value = names[mode]
            self.safe_update(self.map_tool_label)
        self.set_status(f"Strumento mappa: {names[mode]}")

    def _deactivate_map_selector(self) -> None:
        selector = getattr(self, "_selector", None)
        if selector is None:
            return
        try:
            selector.set_active(False)
        except Exception:
            pass
        try:
            selector.disconnect_events()
        except Exception:
            pass
        self._selector = None

    def _activate_selector_for_current_tool(self) -> None:
        ff = self.state.ff
        if ff is None or self.state.image_data is None:
            return

        mode = getattr(self, "_map_tool", "pixel")
        if mode not in {"rectangle", "polygon", "freehand"}:
            return

        # Selection previews use a vivid cyan edge that remains visible over
        # both H&E and false-colour Raman maps.
        props = {"color": "#00A7D8", "linewidth": 1.6, "alpha": 0.95}

        try:
            if mode == "rectangle":
                self._selector = RectangleSelector(
                    self.map_ax,
                    self._on_rectangle_roi,
                    useblit=False,
                    button=[1],
                    minspanx=3,
                    minspany=3,
                    spancoords="pixels",
                    interactive=False,
                    props={
                        "facecolor": "#00A7D8",
                        "edgecolor": "#00A7D8",
                        "alpha": 0.14,
                        "linewidth": 1.6,
                    },
                )
            elif mode == "polygon":
                self._selector = PolygonSelector(
                    self.map_ax,
                    self._on_polygon_roi,
                    useblit=False,
                    props=props,
                )
            elif mode == "freehand":
                self._selector = LassoSelector(
                    self.map_ax,
                    self._on_lasso_roi,
                    useblit=False,
                    button=[1],
                    props=props,
                )
        except Exception as exc:
            self._selector = None
            self.set_status(f"Impossibile attivare lo strumento ROI: {exc}")

    # ------------------------------------------------------------------
    # Pixel picking
    # ------------------------------------------------------------------

    def _on_map_click(self, event) -> None:
        """Add the spectrum at the clicked pixel / irregular point."""
        ff = self.state.ff
        if ff is None or event.inaxes is not self.map_ax or event.button != 1:
            return

        if ff.coordinate_mode == "point_coordinates" and ff.point_coords is not None:
            if event.xdata is None or event.ydata is None:
                return
            d = np.hypot(
                ff.point_coords[:, 0] - event.xdata,
                ff.point_coords[:, 1] - event.ydata,
            )
            idx = int(np.argmin(d))
            self.add_trace(
                np.asarray(ff.obj.data)[idx],
                f"punto {idx}",
                tuple(ff.point_coords[idx]),
            )
            return

        pixel = self._pixel_from_event(event)
        if pixel is None:
            return
        row, col = pixel
        self.add_trace(np.asarray(ff.obj.cube)[row, col], f"px ({col}, {row})")

    # ------------------------------------------------------------------
    # Direct pan / zoom
    # ------------------------------------------------------------------

    def _remember_home_view(self) -> None:
        self._map_home_view = (self.map_ax.get_xlim(), self.map_ax.get_ylim())

    @staticmethod
    def _clamp_interval(
        interval: tuple[float, float], home: tuple[float, float]
    ) -> tuple[float, float]:
        """Keep a panned/zoomed interval inside the full-map extent."""
        a, b = interval
        h0, h1 = home
        reverse = b < a
        low, high = sorted((a, b))
        hlow, hhigh = sorted((h0, h1))
        span = high - low
        hspan = hhigh - hlow

        if hspan <= 0 or span >= hspan:
            return home
        if low < hlow:
            high += hlow - low
            low = hlow
        if high > hhigh:
            low -= high - hhigh
            high = hhigh

        return (high, low) if reverse else (low, high)

    def _on_map_scroll(self, event) -> None:
        """Zoom around the mouse cursor using the wheel."""
        if event.inaxes is not self.map_ax or event.xdata is None or event.ydata is None:
            return

        home = getattr(self, "_map_home_view", None)
        if home is None:
            self._remember_home_view()
            home = self._map_home_view

        zoom_base = 1.20
        if event.button == "up":
            scale = 1.0 / zoom_base
        elif event.button == "down":
            scale = zoom_base
        else:
            return

        x0, x1 = self.map_ax.get_xlim()
        y0, y1 = self.map_ax.get_ylim()
        x = float(event.xdata)
        y = float(event.ydata)

        # Avoid zooming to an unusably tiny field of view.
        home_xspan = abs(home[0][1] - home[0][0])
        home_yspan = abs(home[1][1] - home[1][0])
        if (
            scale < 1
            and (abs(x1 - x0) <= home_xspan / 500 or abs(y1 - y0) <= home_yspan / 500)
        ):
            return

        new_xlim = (
            x - (x - x0) * scale,
            x + (x1 - x) * scale,
        )
        new_ylim = (
            y - (y - y0) * scale,
            y + (y1 - y) * scale,
        )

        self.map_ax.set_xlim(self._clamp_interval(new_xlim, home[0]))
        self.map_ax.set_ylim(self._clamp_interval(new_ylim, home[1]))
        self.render_figure(self.map_fig, self.map_chart)

    def _on_map_mouse_press(self, event) -> None:
        if event.inaxes is not self.map_ax or event.button != 1:
            return

        mode = getattr(self, "_map_tool", "pixel")
        if mode == "pixel":
            self._on_map_click(event)
            return
        if mode != "pan" or event.xdata is None or event.ydata is None:
            return

        self._pan_active = True
        # Use display pixels as the drag reference.  xdata/ydata change when
        # the axes move, while screen coordinates remain stable throughout
        # the gesture and therefore avoid cumulative pan drift.
        self._pan_start_px = (float(event.x), float(event.y))
        self._pan_xlim = self.map_ax.get_xlim()
        self._pan_ylim = self.map_ax.get_ylim()
        self._last_pan_draw = 0.0

    def _on_map_mouse_move(self, event) -> None:
        if not getattr(self, "_pan_active", False):
            return
        if event.inaxes is not self.map_ax or event.xdata is None or event.ydata is None:
            return

        start = getattr(self, "_pan_start_px", None)
        xlim = getattr(self, "_pan_xlim", None)
        ylim = getattr(self, "_pan_ylim", None)
        if start is None or xlim is None or ylim is None:
            return

        bbox = self.map_ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx_px = float(event.x) - start[0]
        dy_px = float(event.y) - start[1]
        dx = dx_px / bbox.width * (xlim[1] - xlim[0])
        dy = dy_px / bbox.height * (ylim[1] - ylim[0])
        new_xlim = (xlim[0] - dx, xlim[1] - dx)
        new_ylim = (ylim[0] - dy, ylim[1] - dy)

        home = getattr(self, "_map_home_view", None)
        if home is not None:
            new_xlim = self._clamp_interval(new_xlim, home[0])
            new_ylim = self._clamp_interval(new_ylim, home[1])

        self.map_ax.set_xlim(new_xlim)
        self.map_ax.set_ylim(new_ylim)

        # Flet/WebAgg redraws can be expensive during mouse movement.  Roughly
        # 30 fps feels smooth without flooding the UI thread.
        now = time.monotonic()
        if now - getattr(self, "_last_pan_draw", 0.0) >= 0.033:
            self._last_pan_draw = now
            self.render_figure(self.map_fig, self.map_chart)

    def _on_map_mouse_release(self, event) -> None:
        if not getattr(self, "_pan_active", False):
            return
        self._pan_active = False
        self.render_figure(self.map_fig, self.map_chart)

    def reset_map_view(self, _e=None) -> None:
        home = getattr(self, "_map_home_view", None)
        if home is None:
            return
        self.map_ax.set_xlim(home[0])
        self.map_ax.set_ylim(home[1])
        self.render_figure(self.map_fig, self.map_chart)
        self.set_status("Vista mappa ripristinata.")

    # ------------------------------------------------------------------
    # ROI creation / rendering / spectra
    # ------------------------------------------------------------------

    def _on_rectangle_roi(self, click, release) -> None:
        if (
            click.xdata is None
            or click.ydata is None
            or release.xdata is None
            or release.ydata is None
        ):
            return
        x0, y0 = float(click.xdata), float(click.ydata)
        x1, y1 = float(release.xdata), float(release.ydata)
        vertices = np.asarray(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            dtype=float,
        )
        self._create_roi(vertices, "rectangle")

    def _on_polygon_roi(self, vertices) -> None:
        self._create_roi(np.asarray(vertices, dtype=float), "polygon")

    def _on_lasso_roi(self, vertices) -> None:
        self._create_roi(np.asarray(vertices, dtype=float), "freehand")

    def _create_roi(self, vertices: np.ndarray, kind: str) -> None:
        ff = self.state.ff
        if ff is None or vertices.ndim != 2 or vertices.shape[0] < 3:
            return

        # Remove repeated/non-finite points that occasionally occur at the end
        # of selector gestures.
        vertices = vertices[np.isfinite(vertices).all(axis=1)]
        if vertices.shape[0] < 3:
            return

        block, count_label = self._spectra_in_roi(vertices)
        if block is None or block.size == 0:
            self.notify("La ROI non contiene spettri.")
            self._restart_current_selector()
            return

        agg = str(self.agg_mode.value or "mean")
        reducer = {"mean": np.mean, "median": np.median, "max": np.max}.get(agg, np.mean)
        spectrum = reducer(block, axis=0)

        self._roi_counter = int(getattr(self, "_roi_counter", 0)) + 1
        name = f"ROI {self._roi_counter}"
        centroid = tuple(np.nanmean(vertices, axis=0))
        roi = {
            "name": name,
            "kind": kind,
            "vertices": np.asarray(vertices, float),
            "n_spectra": int(block.shape[0]),
            "aggregate": agg,
        }
        if not hasattr(self, "_rois"):
            self._rois = []
        self._rois.append(roi)

        self._draw_roi_artist(self.map_ax, len(self._rois) - 1, roi)
        self.add_trace(
            spectrum,
            f"{name} · {agg} ({block.shape[0]} {count_label})",
            centroid,
        )
        self._update_roi_ui()
        self.render_figure(self.map_fig, self.map_chart)
        self.set_status(
            f"{name} creata: {block.shape[0]} {count_label}; spettro {agg} aggiunto."
        )
        self._restart_current_selector()

    def _spectra_in_roi(self, vertices: np.ndarray) -> tuple[np.ndarray | None, str]:
        """Return spectra whose spatial positions lie inside an ROI polygon."""
        ff = self.state.ff
        if ff is None:
            return None, "spettri"

        path = MplPath(vertices)

        if ff.coordinate_mode == "point_coordinates" and ff.point_coords is not None:
            coords = np.asarray(ff.point_coords, float)
            mask = path.contains_points(coords, radius=1e-9)
            data = np.asarray(ff.obj.data, float)
            return data[mask], "punti"

        if not ff.is_map:
            return None, "spettri"

        ny, nx = ff.map_shape()
        extent = ff.spatial_extent()

        if extent:
            left, right, bottom, top = map(float, extent)
            x = left + (np.arange(nx) + 0.5) * (right - left) / nx
            display_rows = np.arange(ny) if ff.y_ascending() else (ny - 1 - np.arange(ny))
            y = bottom + (display_rows + 0.5) * (top - bottom) / ny
        else:
            x = np.arange(nx, dtype=float)
            y = np.arange(ny, dtype=float)
            if not ff.y_ascending():
                y = y[::-1]

        xx, yy = np.meshgrid(x, y)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = path.contains_points(points, radius=1e-9).reshape(ny, nx)
        cube = np.asarray(ff.obj.cube, float)
        return cube[mask], "px"

    def _roi_color(self, index: int) -> str:
        colors = [
            "#00A7D8",
            "#FF7A00",
            "#7A5AF8",
            "#12B76A",
            "#E31B54",
            "#F79009",
        ]
        return colors[index % len(colors)]

    def _draw_saved_rois(self, ax) -> None:
        for i, roi in enumerate(getattr(self, "_rois", [])):
            self._draw_roi_artist(ax, i, roi)

    def _draw_roi_artist(self, ax, index: int, roi: dict) -> None:
        vertices = np.asarray(roi["vertices"], float)
        if vertices.shape[0] < 3:
            return
        color = self._roi_color(index)
        patch = MplPolygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1.7,
            alpha=0.17,
            zorder=8,
        )
        ax.add_patch(patch)
        centroid = np.nanmean(vertices, axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            str(roi["name"]),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            weight="bold",
            zorder=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": color,
                "edgecolor": "white",
                "linewidth": 0.7,
                "alpha": 0.92,
            },
        )

    def _restart_current_selector(self) -> None:
        if getattr(self, "_map_tool", "pixel") not in {"rectangle", "polygon", "freehand"}:
            return
        self._deactivate_map_selector()
        self._activate_selector_for_current_tool()

    def delete_last_roi(self, _e=None) -> None:
        rois = getattr(self, "_rois", [])
        if not rois:
            self.notify("Nessuna ROI da eliminare.")
            return
        removed = rois.pop()
        self._redraw_map_without_recompute()
        self._update_roi_ui()
        self.set_status(f"{removed['name']} eliminata.")

    def clear_rois(self, _e=None) -> None:
        if not getattr(self, "_rois", []):
            return
        self._rois.clear()
        self._roi_counter = 0
        self._redraw_map_without_recompute()
        self._update_roi_ui()
        self.set_status("ROI eliminate.")

    def _redraw_map_without_recompute(self) -> None:
        """Redraw ROI overlays while preserving the current zoom/pan view."""
        current_xlim = self.map_ax.get_xlim()
        current_ylim = self.map_ax.get_ylim()
        self.refresh_map()
        home = getattr(self, "_map_home_view", None)
        if home is not None:
            self.map_ax.set_xlim(self._clamp_interval(current_xlim, home[0]))
            self.map_ax.set_ylim(self._clamp_interval(current_ylim, home[1]))
            self.render_figure(self.map_fig, self.map_chart)

    def _update_roi_ui(self) -> None:
        if hasattr(self, "roi_count_label"):
            n = len(getattr(self, "_rois", []))
            self.roi_count_label.value = f"{n} ROI"
            self.safe_update(self.roi_count_label)

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def add_trace(self, values, label: str, marker=None) -> None:
        self.state.traces.append(Trace(np.asarray(values, float), label, marker))
        self.state.traces = self.state.traces[-8:]
        self.refresh_spectra()
        self.set_status(f"Spettro aggiunto: {label}")

    def clear_traces(self) -> None:
        self.state.traces.clear()
        self.refresh_spectra()
