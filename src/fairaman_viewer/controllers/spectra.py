"""Spectral plotting and processing-pipeline behavior."""

import traceback

import numpy as np
from ramappy import processing
from ramappy.core import SpectralMap

from ..config import PALETTE

class SpectraControllerMixin:
    """Behavior mixin for FairamanViewerApp."""

    def reference_spectrum(self) -> tuple[np.ndarray, str] | None:
            ff = self.state.ff
            if ff is None:
                return None
            if self.state.traces:
                tr = self.state.traces[-1]
                return tr.y, tr.label
            data = np.asarray(ff.obj.data, float)
            if data.ndim == 1:
                return data, ff.title[:40]
            return data.mean(0), f"media ({ff.n_spectra} spettri)"

    def refresh_spectra(self) -> None:
            ff = self.state.ff
            if ff is None:
                return
            axis = ff.spectral_axis
            ax = self.spec_ax
            ax.clear()
            data = np.asarray(ff.obj.data, float)
            if data.ndim == 1:
                ax.plot(axis, data, color=PALETTE[0], lw=1.2, label=ff.title[:40])
            else:
                if self.show_mean.value:
                    ax.plot(axis, data.mean(0), color="#444", lw=1.4, alpha=0.85, label=f"media ({ff.n_spectra} spettri)")
                for i, tr in enumerate(self.state.traces):
                    ax.plot(axis, tr.y, lw=1.1, color=PALETTE[i % len(PALETTE)], label=tr.label)

            ref = self.reference_spectrum()
            if ref is not None:
                values, label = ref
                estimate = self.baseline_panel.preview(axis, values)
                if estimate is not None:
                    ax.plot(axis, estimate, color="#d62728", lw=1.4, ls="--", label=f"baseline stimata su «{label}»")
                    corrected = values - estimate
                    span = float(np.nanmax(values) - np.nanmin(values))
                    offset = float(np.nanmin(corrected)) - 0.12 * span
                    ax.plot(axis, corrected + offset - float(np.nanmin(corrected)), color="#2ca02c", lw=1.0, alpha=0.9, label="corretto (traslato)")
                    ax.axhline(offset, color="#2ca02c", lw=0.6, ls=":", alpha=0.6)
                roi = self.baseline_panel.roi()
                if roi:
                    ax.axvspan(roi[0], roi[1], color="#1f77b4", alpha=0.07)

            self._render_normalization_preview(axis)

            ax.set_xlabel("Raman shift (cm⁻¹)")
            ax.set_ylabel("Intensità (a.u.)")
            handles, labels = ax.get_legend_handles_labels()
            if self._spec_ax2 is not None and self._spec_ax2.get_visible():
                h2, l2 = self._spec_ax2.get_legend_handles_labels()
                handles, labels = handles + h2, labels + l2
            if handles:
                ax.legend(handles, labels, fontsize=7, loc="best", framealpha=0.85)
            ax.grid(alpha=0.25, lw=0.5)
            self.render_figure(self.spec_fig, self.spec_chart)

    def _render_normalization_preview(self, axis) -> None:
            """Anteprima della normalizzazione su un asse y secondario."""
            ax2 = self._spec_ax2
            if ax2 is not None:
                ax2.clear()
                ax2.set_visible(False)
            ref = self.reference_spectrum()
            if ref is None:
                return
            values, label = ref
            preview = self.normalization_panel.preview(axis, values)
            if preview is None:
                return
            if self._spec_ax2 is None:
                self._spec_ax2 = self.spec_ax.twinx()
            ax2 = self._spec_ax2
            ax2.set_visible(True)
            ax2.plot(axis, preview, color="#9467bd", lw=1.1, ls="-.", label=f"norm. anteprima «{label}»")
            ax2.set_ylabel("intensità normalizzata", color="#9467bd", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="#9467bd", labelsize=8)

    def apply_pipeline(self, _e=None) -> None:
            ff = self.state.ff
            if ff is None:
                return
            steps: list[str] = []
            if self.use_despike.value:
                steps.append("despike")
            if self.use_smooth.value:
                steps.append("savgol")
            if not steps:
                self.notify("Nessun passaggio selezionato.")
                return
            self.set_status("Elaborazione in corso…")
            self.page.run_thread(self._run_pipeline, steps)

    def _run_pipeline(self, steps: list[str]) -> None:
            ff = self.state.ff
            if ff is None:
                return
            obj = ff.obj
            try:
                for step in steps:
                    if step == "despike" and isinstance(obj, SpectralMap):
                        processing.despike(obj, correct_spikes=True, n_jobs=1)
                    elif step == "savgol":
                        window = int(self.savgol_window.value or 9)
                        if window % 2 == 0:
                            window += 1
                        processing.smooth_spectral(
                            obj,
                            method="savgol",
                            window_length=window,
                            polyorder=int(self.savgol_order.value or 3),
                        )
                descr = " → ".join(steps)
                self.state.history.append(descr)
                self.after_processing()
                self.set_status(f"Applicato: {descr}")
            except Exception:
                self.alert("Errore nell'elaborazione", traceback.format_exc(limit=3))
                self.set_status("Elaborazione fallita.")

    def on_baseline_ok(self, info: dict) -> None:
            self.state.history.append(
                f"baseline {info['method']}"
                + (f" su {info['roi'][0]:.0f}–{info['roi'][1]:.0f} cm⁻¹" if info["roi"] else "")
                + (" + non negativo" if info["force_nonnegative"] else "")
            )
            self.baseline_panel.finish(f"fatto in {info['seconds']:.1f} s")
            self.after_processing()
            self.set_status(
                f"Linea di base corretta su {info['n_spectra']} spettri con {info['method']} "
                f"in {info['seconds']:.1f} s."
            )

    def on_baseline_cancelled(self, message: str) -> None:
            self.baseline_panel.finish("annullato")
            self.after_processing()
            self.alert(
                "Elaborazione annullata",
                f"{message}\n\nGli spettri gia' elaborati sono stati modificati. "
                "Usa «Ripristina grezzi» per tornare al file originale.",
            )

    def on_baseline_error(self, message: str) -> None:
            self.baseline_panel.finish("errore")
            self.alert("Errore nella correzione", message)
            self.set_status("Correzione fallita.")

    def on_normalization_ok(self, info: dict) -> None:
            note = f"normalizzazione {info['method']}"
            if info.get("roi"):
                note += f" su {info['roi'][0]:.0f}–{info['roi'][1]:.0f} cm⁻¹"
            if info.get("reference"):
                note += f" [rif. {info['reference']}"
                note += f"/{info['reference_prior']}]" if info.get("reference_prior") else "]"
            self.state.history.append(note)
            tail = ""
            if info.get("degenerate"):
                tail = f"  {info['degenerate']} spettri non normalizzabili lasciati invariati."
            self.normalization_panel.finish(f"fatto in {info['seconds']:.1f} s")
            self.after_processing()
            self.set_status(
                f"Normalizzazione «{info['method']}» applicata a {info['n_spectra']} spettri "
                f"in {info['seconds']:.1f} s.{tail}"
            )

    def on_normalization_cancelled(self, message: str) -> None:
            self.normalization_panel.finish("annullato")
            self.after_processing()
            self.alert(
                "Normalizzazione annullata",
                f"{message}\n\nGli spettri gia' elaborati sono stati modificati. "
                "Usa «Ripristina grezzi» per tornare al file originale.",
            )

    def on_normalization_error(self, message: str) -> None:
            self.normalization_panel.finish("errore")
            self.alert("Errore nella normalizzazione", message)
            self.set_status("Normalizzazione fallita.")

    def after_processing(self) -> None:
            self.state.invalidate_after_processing()
            self.refresh_map()
            self.refresh_spectra()
            self.fill_data_model()
            self.refresh_analysis_controls()
