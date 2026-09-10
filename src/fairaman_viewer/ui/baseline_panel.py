"""Flet panel for baseline preview and application."""

from __future__ import annotations

import threading
from typing import Any, TYPE_CHECKING

import flet as ft
import numpy as np

from ..processing import baseline as bl
from .components import _dropdown, _num_field

if TYPE_CHECKING:
    from ..app import FairamanViewerApp

class BaselinePanel:
    """Pannello Flet per la baseline, guidato dal registro ``METHODS``.

    Il pannello non conosce ramappy oltre alle funzioni del namespace ``bl``.
    Questo permette in futuro di spostarlo in un modulo UI senza toccare il core.
    """

    def __init__(self, app: "FairamanViewerApp"):
        self.app = app
        self.cancel_event: threading.Event | None = None
        self.param_controls: dict[str, ft.Control] = {}

        self.method = _dropdown(
            "rubberband",
            bl.METHOD_KEYS,
            label="Metodo",
            width=230,
            on_select=self._on_method_change,
        )
        self.method_label = ft.Text(size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.note = ft.Text(size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.params = ft.Row(wrap=True, spacing=8, run_spacing=6)

        self.use_roi = ft.Checkbox(label="Solo intervallo", value=False, on_change=self._preview_change)
        self.roi_lo = _num_field(label="Da cm⁻¹", on_change=self._preview_change)
        self.roi_hi = _num_field(label="A cm⁻¹", on_change=self._preview_change)
        self.force_nonnegative = ft.Checkbox(label="Forza non negativo", value=False)
        self.preview_on = ft.Checkbox(label="Anteprima", value=True, on_change=self._preview_change)

        self.apply_btn = ft.Button(
            content="Applica a tutti gli spettri",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self.apply,
        )
        self.cancel_btn = ft.OutlinedButton(
            content="Annulla",
            icon=ft.Icons.CANCEL,
            disabled=True,
            on_click=self.cancel,
        )
        self.progress = ft.ProgressBar(value=0, expand=True)
        self.progress_label = ft.Text(size=11)

        self.control = ft.Column(
            controls=[
                ft.Row([self.method, self.method_label], wrap=True),
                self.note,
                self.params,
                ft.Row(
                    [self.use_roi, self.roi_lo, self.roi_hi, self.force_nonnegative],
                    wrap=True,
                ),
                ft.Row([self.preview_on, self.apply_btn, self.cancel_btn], wrap=True),
                ft.Row([self.progress, self.progress_label]),
            ],
            spacing=6,
        )
        self._rebuild_params()

    @property
    def method_key(self) -> str:
        return str(self.method.value or "rubberband")

    def _on_method_change(self, _e=None) -> None:
        self._rebuild_params()
        self.app.refresh_spectra()

    def _preview_change(self, _e=None) -> None:
        self.app.refresh_spectra()

    def _rebuild_params(self) -> None:
        self.param_controls.clear()
        controls: list[ft.Control] = []
        method = bl.METHODS[self.method_key]
        self.method_label.value = f"{method.label} · {method.speed} · {bl.SPEED_HINT[method.speed]}"
        self.note.value = method.note or ""

        for param in method.params:
            if param.kind == "choice":
                ctrl = _dropdown(
                    str(param.default), list(param.choices), label=param.label, width=180,
                    on_select=self._preview_change,
                )
            elif param.kind == "logfloat":
                readout = ft.Text(f"10^{float(param.default):.1f}", width=64)

                def changed(e, r=readout):
                    r.value = f"10^{float(e.control.value):.1f}"
                    r.update()
                    self.app.refresh_spectra()

                ctrl = ft.Slider(
                    value=float(param.default),
                    min=float(param.lo or 1),
                    max=float(param.hi or 10),
                    divisions=int(((param.hi or 10) - (param.lo or 1)) * 10),
                    label="{value}",
                    width=180,
                    on_change=changed,
                )
                controls.append(ft.Column([ft.Text(param.label, size=11), ft.Row([ctrl, readout])], spacing=0))
                self.param_controls[param.key] = ctrl
                continue
            else:
                default = "" if param.default is None else str(param.default)
                ctrl = _num_field(default, label=param.label, width=130, on_change=self._preview_change)
                ctrl.tooltip = param.help or None

            ctrl.tooltip = param.help or None
            controls.append(ctrl)
            self.param_controls[param.key] = ctrl

        if not controls:
            controls = [ft.Text("Nessun parametro regolabile.", size=11)]
        self.params.controls = controls
        self.app.safe_update(self.method_label, self.note, self.params)

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, ctrl in self.param_controls.items():
            values[key] = getattr(ctrl, "value", None)
        return values

    def roi(self) -> tuple[float, float] | None:
        if not self.use_roi.value:
            return None
        try:
            lo, hi = float(self.roi_lo.value or ""), float(self.roi_hi.value or "")
        except ValueError:
            return None
        return min(lo, hi), max(lo, hi)

    def preview(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        if not self.preview_on.value:
            return None
        try:
            return bl.estimate_baseline(x, y, self.method_key, self.values(), self.roi())
        except Exception as exc:
            self.progress_label.value = f"anteprima non riuscita: {exc}"[:90]
            self.app.safe_update(self.progress_label)
            return None

    def apply(self, _e=None) -> None:
        ff = self.app.state.ff
        if ff is None:
            return
        method = self.method_key
        eta = bl.estimate_duration(method, ff.n_spectra)
        if eta > 20:
            self.app.confirm(
                "Elaborazione lunga",
                f"{bl.METHODS[method].label} su {ff.n_spectra} spettri richiede circa "
                f"{eta:.0f} s. Puoi annullare durante l'esecuzione.",
                lambda: self._start_apply(method),
            )
        else:
            self._start_apply(method)

    def _start_apply(self, method: str) -> None:
        self.cancel_event = threading.Event()
        self.apply_btn.disabled = True
        self.cancel_btn.disabled = False
        self.progress.value = 0
        self.progress_label.value = "avvio…"
        self.app.safe_update(self.apply_btn, self.cancel_btn, self.progress, self.progress_label)
        values, roi, force = self.values(), self.roi(), bool(self.force_nonnegative.value)
        self.app.page.run_thread(self._worker, method, values, roi, force)

    def _worker(self, method: str, values: dict, roi, force: bool) -> None:
        ff = self.app.state.ff
        if ff is None:
            return
        try:
            info = bl.apply_baseline(
                ff.obj,
                method,
                values,
                roi=roi,
                force_nonnegative=force,
                progress=self.set_progress,
                cancel=self.cancel_event,
                n_jobs=1,
            )
            self.app.on_baseline_ok(info)
        except bl.Cancelled as exc:
            self.app.on_baseline_cancelled(str(exc))
        except Exception as exc:
            self.app.on_baseline_error(f"{type(exc).__name__}: {exc}")

    def cancel(self, _e=None) -> None:
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.progress_label.value = "annullamento…"
            self.app.safe_update(self.progress_label)

    def set_progress(self, fraction: float, text: str) -> None:
        self.progress.value = fraction
        self.progress_label.value = text
        self.app.safe_update(self.progress, self.progress_label)

    def finish(self, text: str) -> None:
        self.apply_btn.disabled = False
        self.cancel_btn.disabled = True
        self.progress.value = 0
        self.progress_label.value = text
        self.cancel_event = None
        self.app.safe_update(self.apply_btn, self.cancel_btn, self.progress, self.progress_label)


