"""Advanced-analysis execution and result presentation."""

from typing import Any

import flet as ft

from ..analysis.models import AnalysisContext, AnalysisPlugin, AnalysisResult
from ..analysis.registry import ANALYSIS_PLUGINS
from ..config import PALETTE

class AnalysisControllerMixin:
    """Behavior mixin for FairamanViewerApp."""

    def update_analysis_description(self) -> None:
            plugin = ANALYSIS_PLUGINS[str(self.analysis_method.value or "nfindr")]
            availability = "Disponibile" if plugin.available else f"Dipendenze mancanti: {', '.join(plugin.missing_packages)}"
            self.analysis_description.value = f"{plugin.description}\n{availability}"
            self.safe_update(self.analysis_description)

    def run_advanced_analysis(self, _e=None) -> None:
            ff = self.state.ff
            if ff is None:
                return
            plugin = ANALYSIS_PLUGINS[str(self.analysis_method.value or "nfindr")]
            if not plugin.available:
                self.alert(
                    "Dipendenze opzionali mancanti",
                    f"Per {plugin.label} installa: {', '.join(plugin.missing_packages)}",
                )
                return
            try:
                params = {
                    "n_components": int(self.analysis_n_components.value or 4),
                    "max_iter": int(self.analysis_max_iter.value or 100),
                    "random_state": int(self.analysis_random_state.value or 21),
                }
            except ValueError:
                self.alert("Parametri non validi", "Componenti, max iter e random state devono essere interi.")
                return
            self.analysis_status.value = f"Esecuzione {plugin.label}…"
            self.safe_update(self.analysis_status)
            self.page.run_thread(self._analysis_worker, plugin, params)

    def _analysis_worker(self, plugin: AnalysisPlugin, params: dict[str, Any]) -> None:
            ff = self.state.ff
            if ff is None:
                return
            try:
                result = plugin.runner(AnalysisContext(ff, self.state.analysis_results.copy()), params)
                self.state.analysis_results[plugin.key] = result
                self.state.history.append(f"analysis {result.title}")
                self.analysis_status.value = f"Completata: {result.title}"
                self.refresh_analysis_result_figure(result)
                self.refresh_analysis_controls()
                self.refresh_map()
                self.fill_data_model()
                self.set_status(f"Analisi completata: {result.title}")
            except Exception as exc:
                self.analysis_status.value = f"Errore: {type(exc).__name__}: {exc}"
                self.safe_update(self.analysis_status)
                self.alert("Errore analisi", f"{type(exc).__name__}: {exc}")

    def refresh_analysis_result_figure(self, result: AnalysisResult) -> None:
            ax = self.analysis_ax
            ax.clear()
            ff = self.state.ff
            if ff is None:
                return
            for i, (label, spectrum) in enumerate(result.spectra.items()):
                ax.plot(ff.spectral_axis, spectrum, lw=1.1, color=PALETTE[i % len(PALETTE)], label=label)
            if result.spectra:
                ax.set_xlabel("Raman shift (cm⁻¹)")
                ax.set_ylabel("Intensità / componente")
                ax.legend(fontsize=7, loc="best")
                ax.grid(alpha=0.25, lw=0.5)
                ax.set_title(result.title, fontsize=10)
            else:
                ax.text(0.5, 0.5, "Nessun profilo spettrale", ha="center", va="center", transform=ax.transAxes)
            self.safe_update(self.analysis_chart)

    def refresh_analysis_controls(self) -> None:
            image_names = [name for result in self.state.analysis_results.values() for name in result.images]
            self.analysis_result_selector.options = [ft.DropdownOption(key=name, text=name) for name in image_names]
            if image_names and self.analysis_result_selector.value not in image_names:
                self.analysis_result_selector.value = image_names[0]

            base = ["Intensità totale", "Banda spettrale", "PCA score"]
            advanced = [f"Analisi: {name}" for name in image_names]
            self.image_mode.options = [ft.DropdownOption(key=name, text=name) for name in base + advanced]
            if self.image_mode.value not in base + advanced:
                self.image_mode.value = "Intensità totale"
            self.safe_update(self.analysis_result_selector, self.image_mode)

    def show_analysis_map(self, _e=None) -> None:
            name = str(self.analysis_result_selector.value or "")
            if not name:
                self.notify("Nessuna mappa di analisi disponibile.")
                return
            self.image_mode.value = f"Analisi: {name}"
            self.safe_update(self.image_mode)
            self.refresh_map()
