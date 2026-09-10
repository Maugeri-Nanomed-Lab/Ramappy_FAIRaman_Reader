"""Pannello Flet per l'anteprima e l'applicazione della normalizzazione."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import flet as ft
import numpy as np

from ..processing import normalization as nz
from .components import _dropdown, _num_field

if TYPE_CHECKING:
    from ..app import FairamanViewerApp


class NormalizationPanel:
    """Pannello guidato dal registro ``nz.METHODS``.

    Come ``BaselinePanel``, non conosce ramappy: dialoga solo con le funzioni
    del namespace ``nz``. I metodi a due passate (MSC, PQN) hanno bisogno di un
    riferimento globale, che il pannello stima una volta e mette in cache.
    """

    def __init__(self, app: FairamanViewerApp):
        self.app = app
        self.cancel_event: threading.Event | None = None
        self.param_controls: dict[str, ft.Control] = {}
        self._reference: dict[str, Any] | None = None
        self._reference_key: tuple | None = None

        self.method = _dropdown(
            "l2", nz.METHOD_KEYS, label="Metodo", width=260, on_select=self._on_method_change
        )
        self.method_label = ft.Text(size=12, color=ft.Colors.ON_SURFACE_VARIANT)
        self.note = ft.Text(size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.params = ft.Row(wrap=True, spacing=8, run_spacing=6)

        self.use_roi = ft.Checkbox(label="Solo intervallo", value=False, on_change=self._preview_change)
        self.roi_lo = _num_field(label="Da cm⁻¹", on_change=self._preview_change)
        self.roi_hi = _num_field(label="A cm⁻¹", on_change=self._preview_change)
        self.preview_on = ft.Checkbox(label="Anteprima", value=True, on_change=self._preview_change)

        self.apply_btn = ft.Button(
            content="Applica a tutti gli spettri", icon=ft.Icons.PLAY_ARROW, on_click=self.apply
        )
        self.cancel_btn = ft.OutlinedButton(
            content="Annulla", icon=ft.Icons.CANCEL, disabled=True, on_click=self.cancel
        )
        self.progress = ft.ProgressBar(value=0, expand=True)
        self.progress_label = ft.Text(size=11)

        self.control = ft.Column(
            controls=[
                ft.Row([self.method, self.method_label], wrap=True),
                self.note,
                self.params,
                ft.Row([self.use_roi, self.roi_lo, self.roi_hi], wrap=True),
                ft.Row([self.preview_on, self.apply_btn, self.cancel_btn], wrap=True),
                ft.Row([self.progress, self.progress_label]),
            ],
            spacing=6,
        )
        self._rebuild_params()

    @property
    def method_key(self) -> str:
        return str(self.method.value or "l2")

    # ------------------------------------------------------------- reattivita' --
    def _on_method_change(self, _e=None) -> None:
        self._invalidate_reference()
        self._rebuild_params()
        self.app.refresh_spectra()

    def _preview_change(self, _e=None) -> None:
        self._invalidate_reference()
        self.app.refresh_spectra()

    def _rebuild_params(self) -> None:
        self.param_controls.clear()
        method = nz.METHODS[self.method_key]
        self.method_label.value = method.label + (" · due passate" if method.two_pass else "")
        self.note.value = "  ".join(part for part in (method.note, method.scope_note) if part)

        controls: list[ft.Control] = []
        for param in method.params:
            if param.kind == "choice":
                ctrl: ft.Control = _dropdown(
                    str(param.default), list(param.choices), label=param.label, width=150,
                    on_select=self._preview_change,
                )
            elif param.kind == "bool":
                ctrl = ft.Checkbox(
                    label=param.label, value=bool(param.default), on_change=self._preview_change
                )
            else:
                default = "" if param.default is None else str(param.default)
                ctrl = _num_field(default, label=param.label, width=150, on_change=self._preview_change)
            ctrl.tooltip = param.help or None
            controls.append(ctrl)
            self.param_controls[param.key] = ctrl

        if not controls:
            controls = [ft.Text("Nessun parametro regolabile.", size=11)]
        self.params.controls = controls
        self.app.safe_update(self.method_label, self.note, self.params)

    def values(self) -> dict[str, Any]:
        return {key: getattr(ctrl, "value", None) for key, ctrl in self.param_controls.items()}

    def roi(self) -> tuple[float, float] | None:
        if not self.use_roi.value:
            return None
        try:
            lo, hi = float(self.roi_lo.value or ""), float(self.roi_hi.value or "")
        except ValueError:
            return None
        return min(lo, hi), max(lo, hi)

    # -------------------------------------------------------------- riferimento --
    def _invalidate_reference(self) -> None:
        self._reference = None
        self._reference_key = None

    def _reference_signature(self) -> tuple:
        ff = self.app.state.ff
        values = tuple(sorted((k, str(v)) for k, v in self.values().items()))
        return (self.method_key, values, self.roi(), id(getattr(ff, "obj", None)))

    def _ensure_reference(self) -> dict[str, Any] | None:
        if not nz.METHODS[self.method_key].two_pass:
            return None
        ff = self.app.state.ff
        if ff is None:
            return None
        signature = self._reference_signature()
        if self._reference is None or self._reference_key != signature:
            self._reference = nz.fit_reference(
                np.asarray(ff.obj.data, float),
                np.asarray(ff.obj.x, float),
                self.method_key,
                self.values(),
                self.roi(),
            )
            self._reference_key = signature
        return self._reference

    # ---------------------------------------------------------------- anteprima --
    def preview(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        if not self.preview_on.value:
            return None
        try:
            reference = self._ensure_reference()
            return nz.estimate_normalization(
                x, y, self.method_key, self.values(), self.roi(), reference
            )
        except Exception as exc:
            self.progress_label.value = f"anteprima non riuscita: {exc}"[:90]
            self.app.safe_update(self.progress_label)
            return None

    # ------------------------------------------------------------ applicazione --
    def apply(self, _e=None) -> None:
        ff = self.app.state.ff
        if ff is None:
            return
        try:
            reference = self._ensure_reference()
        except Exception as exc:
            self.app.alert("Riferimento non stimabile", f"{type(exc).__name__}: {exc}")
            return

        self.cancel_event = threading.Event()
        self.apply_btn.disabled = True
        self.cancel_btn.disabled = False
        self.progress.value = 0
        self.progress_label.value = "avvio…"
        self.app.safe_update(self.apply_btn, self.cancel_btn, self.progress, self.progress_label)
        self.app.page.run_thread(
            self._worker, self.method_key, self.values(), self.roi(), reference
        )

    def _worker(self, method: str, values: dict, roi, reference) -> None:
        ff = self.app.state.ff
        if ff is None:
            return
        try:
            info = nz.apply_normalization(
                ff.obj, method, values, roi=roi, reference=reference,
                progress=self.set_progress, cancel=self.cancel_event,
            )
            self.app.on_normalization_ok(info)
        except nz.Cancelled as exc:
            self.app.on_normalization_cancelled(str(exc))
        except Exception as exc:
            self.app.on_normalization_error(f"{type(exc).__name__}: {exc}")

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
        self._invalidate_reference()
        self.app.safe_update(self.apply_btn, self.cancel_btn, self.progress, self.progress_label)
