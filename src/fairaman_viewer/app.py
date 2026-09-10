"""Main Flet application composition root.

`FairamanViewerApp` deliberately contains only application setup and layout.
Behavior is split into controller mixins by responsibility, while scientific
logic remains outside the UI/controller layer entirely.
"""

from __future__ import annotations

from pathlib import Path

import flet as ft
import flet_charts as fch
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from ramappy import Spectrum

from .analysis.registry import ANALYSIS_PLUGINS
from .config import APP_NAME, COLORMAPS, NORMS
from .controllers import (
    AnalysisControllerMixin,
    ExportControllerMixin,
    FileControllerMixin,
    MapControllerMixin,
    SpectraControllerMixin,
)
from .state import ViewerState
from .ui.baseline_panel import BaselinePanel
from .ui.components import _card, _dropdown, _num_field
from .validation.schema import summary


class FairamanViewerApp(
    FileControllerMixin,
    MapControllerMixin,
    SpectraControllerMixin,
    AnalysisControllerMixin,
    ExportControllerMixin,
):
    """Compose the FAIRaman Viewer application from focused behaviors."""

    def __init__(self, page: ft.Page, initial_file: str | None = None):
            self.page = page
            self.initial_file = initial_file
            self.state = ViewerState()
            self._temporary_inputs: list[Path] = []
            self._selector: RectangleSelector | None = None
            self._colorbar = None

            self._configure_page()
            self._build_controls()
            self._build_layout()
            self._wire_matplotlib_events()

            if initial_file:
                self.open_file(initial_file)

    def _configure_page(self) -> None:
            self.page.title = APP_NAME
            self.page.padding = 0
            self.page.spacing = 0
            self.page.theme_mode = ft.ThemeMode.SYSTEM

    def safe_update(self, *controls: ft.Control) -> None:
            """Aggiornamento tollerante durante thread di elaborazione."""
            for control in controls:
                try:
                    control.update()
                except Exception:
                    pass
            try:
                self.page.schedule_update()
            except Exception:
                pass

    def render_figure(self, figure, *controls: ft.Control) -> None:
            """Rirasterizza una figura Matplotlib e sincronizza i controlli Flet.

            `flet-charts` invia un nuovo fotogramma al widget solo quando il backend
            WebAgg esegue `canvas.draw()`. Ricostruire gli assi e chiamare solo
            `control.update()` non basta: senza questo `draw_idle()` la chart resta
            ferma all'ultimo fotogramma disegnato.
            """
            try:
                figure.canvas.draw_idle()
            except Exception:
                pass
            self.safe_update(*controls)

    def set_status(self, text: str) -> None:
            self.status.value = text
            self.safe_update(self.status)

    def notify(self, text: str) -> None:
            self.page.show_dialog(ft.SnackBar(content=ft.Text(text)))

    def alert(self, title: str, message: str) -> None:
            dialog = ft.AlertDialog(
                title=ft.Text(title),
                content=ft.Text(message, selectable=True),
                actions=[ft.TextButton("OK", on_click=lambda e: self.page.pop_dialog())],
            )
            self.page.show_dialog(dialog)

    def confirm(self, title: str, message: str, action: Callable[[], None]) -> None:
            def yes(_e):
                self.page.pop_dialog()
                action()

            dialog = ft.AlertDialog(
                title=ft.Text(title),
                content=ft.Text(message),
                actions=[
                    ft.TextButton("Annulla", on_click=lambda e: self.page.pop_dialog()),
                    ft.FilledButton("Procedi", on_click=yes),
                ],
            )
            self.page.show_dialog(dialog)

    def _build_controls(self) -> None:
            # Header / file actions
            self.header = ft.Text("Nessun file caricato", weight=ft.FontWeight.BOLD, size=14)
            self.status = ft.Text("Pronto.", size=11, expand=True)

            # Metadata
            self.metadata_filter = ft.TextField(
                label="Filtro metadati",
                dense=True,
                on_change=lambda e: self.fill_metadata(),
            )
            self.metadata_list = ft.ListView(expand=True, spacing=2)
            self.validation_summary = ft.Text("—", weight=ft.FontWeight.BOLD)
            self.validation_list = ft.ListView(expand=True, spacing=6)
            self.data_model_text = ft.Text(selectable=True, size=11)

            # Map controls
            self.image_mode = _dropdown(
                "Intensità totale",
                ["Intensità totale", "Banda spettrale", "PCA score"],
                label="Immagine",
                width=190,
                on_select=lambda e: self.refresh_map(),
            )
            self.band_lo = _num_field(label="Da cm⁻¹", on_change=lambda e: None)
            self.band_hi = _num_field(label="A cm⁻¹", on_change=lambda e: None)
            self.pc_index = _num_field("1", label="PC", width=70)
            self.cmap = _dropdown(
                "viridis", COLORMAPS, label="Colormap", width=150,
                on_select=lambda e: self.refresh_map(),
            )
            self.show_wl = ft.Checkbox(label="White light", value=False, on_change=lambda e: self.refresh_map())
            self.wl_alpha = ft.Slider(
                value=0.55, min=0, max=1, divisions=20, width=130,
                on_change=lambda e: self.refresh_map(),
            )
            self.agg_mode = _dropdown("mean", ["mean", "median", "max"], label="ROI", width=110)

            self.map_fig, self.map_ax = plt.subplots(figsize=(6.0, 5.2), dpi=100, layout="constrained")
            self.map_ax.set_axis_off()
            self.map_chart = fch.MatplotlibChartWithToolbar(figure=self.map_fig, expand=True)

            # Spectrum + processing
            self.show_mean = ft.Checkbox(label="Media globale", value=True, on_change=lambda e: self.refresh_spectra())
            self.use_despike = ft.Checkbox(label="Despike", value=False)
            self.use_smooth = ft.Checkbox(label="SavGol", value=False)
            self.savgol_window = _num_field("9", label="Finestra", width=92)
            self.savgol_order = _num_field("3", label="Grado", width=82)
            self.use_norm = ft.Checkbox(label="Normalizza", value=False)
            self.norm_kind = _dropdown("minmax_scale", NORMS, label="Norma", width=170)

            self.baseline_panel = BaselinePanel(self)

            self.spec_fig, self.spec_ax = plt.subplots(figsize=(6.0, 5.2), dpi=100, layout="constrained")
            self.spec_chart = fch.MatplotlibChartWithToolbar(figure=self.spec_fig, expand=True)

            # Advanced analysis controls
            self.analysis_method = _dropdown(
                "nfindr",
                list(ANALYSIS_PLUGINS),
                label="Analisi",
                width=210,
                on_select=lambda e: self.update_analysis_description(),
            )
            self.analysis_n_components = _num_field("4", label="Componenti", width=110)
            self.analysis_max_iter = _num_field("100", label="Max iter", width=100)
            self.analysis_random_state = _num_field("21", label="Random state", width=115)
            self.analysis_description = ft.Text(size=11, color=ft.Colors.ON_SURFACE_VARIANT)
            self.analysis_status = ft.Text(size=11)
            self.analysis_result_selector = _dropdown("", [], label="Mappa risultato", width=240)

            self.analysis_fig, self.analysis_ax = plt.subplots(figsize=(6.0, 3.8), dpi=100, layout="constrained")
            self.analysis_chart = fch.MatplotlibChartWithToolbar(figure=self.analysis_fig, expand=True)
            self.update_analysis_description()

    def _build_layout(self) -> None:
            topbar = ft.Container(
                padding=8,
                content=ft.Row(
                    controls=[
                        ft.FilledButton("Apri .h5", icon=ft.Icons.FOLDER_OPEN, on_click=self.choose_file),
                        ft.OutlinedButton("Ricarica", icon=ft.Icons.REFRESH, on_click=lambda e: self.reload()),
                        ft.VerticalDivider(width=8),
                        ft.Button("Spettri CSV", icon=ft.Icons.TABLE_VIEW, on_click=self.export_spectra),
                        ft.Button("Report", icon=ft.Icons.DESCRIPTION, on_click=self.export_report),
                        ft.VerticalDivider(width=8),
                        self.header,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )

            metadata_tabs = ft.Tabs(
                length=3,
                expand=True,
                content=ft.Column(
                    expand=True,
                    controls=[
                        ft.TabBar(tabs=[ft.Tab(label="Metadati"), ft.Tab(label="Validazione"), ft.Tab(label="Modello dati")]),
                        ft.TabBarView(
                            expand=True,
                            controls=[
                                ft.Column([self.metadata_filter, self.metadata_list], expand=True),
                                ft.Column([self.validation_summary, self.validation_list], expand=True),
                                ft.ListView(controls=[self.data_model_text], expand=True),
                            ],
                        ),
                    ],
                ),
            )

            left = ft.Container(width=365, padding=8, content=metadata_tabs)

            map_controls = ft.Column(
                controls=[
                    ft.Row(
                        [
                            self.image_mode, self.band_lo, self.band_hi, self.pc_index,
                            ft.Button("Aggiorna", icon=ft.Icons.REFRESH, on_click=lambda e: self.refresh_map()),
                        ],
                        wrap=True,
                    ),
                    ft.Row([self.cmap, self.show_wl, self.wl_alpha, self.agg_mode], wrap=True),
                    ft.Row(
                        [
                            ft.OutlinedButton("Mappa PNG", icon=ft.Icons.IMAGE, on_click=lambda e: self.export_figure("map", "png")),
                            ft.OutlinedButton("Mappa SVG", icon=ft.Icons.DRAW, on_click=lambda e: self.export_figure("map", "svg")),
                            ft.OutlinedButton("Dati mappa CSV", icon=ft.Icons.DATA_OBJECT, on_click=self.export_map_data),
                        ],
                        wrap=True,
                    ),
                    ft.Text("Click sinistro: spettro pixel · trascinamento destro: ROI", size=10, color=ft.Colors.ON_SURFACE_VARIANT),
                ],
                spacing=5,
            )
            center = ft.Container(
                expand=1,
                padding=8,
                content=ft.Column(
                    expand=True,
                    controls=[_card("Mappa", map_controls), self.map_chart],
                ),
            )

            baseline_tab = ft.Container(padding=6, content=self.baseline_panel.control)
            pipeline_tab = ft.Container(
                padding=6,
                content=ft.Column(
                    controls=[
                        ft.Row(
                            [self.use_despike, self.use_smooth, self.savgol_window, self.savgol_order],
                            wrap=True,
                        ),
                        ft.Row([self.use_norm, self.norm_kind], wrap=True),
                        ft.Row(
                            [
                                ft.FilledButton("Applica pipeline", icon=ft.Icons.PLAY_ARROW, on_click=self.apply_pipeline),
                                ft.OutlinedButton("Ripristina grezzi", icon=ft.Icons.RESTORE, on_click=lambda e: self.reload()),
                                ft.OutlinedButton("Pulisci spettri", icon=ft.Icons.CLEAR_ALL, on_click=lambda e: self.clear_traces()),
                                self.show_mean,
                            ],
                            wrap=True,
                        ),
                        ft.Row(
                            [
                                ft.OutlinedButton("Spettri PNG", icon=ft.Icons.IMAGE, on_click=lambda e: self.export_figure("spectra", "png")),
                                ft.OutlinedButton("Spettri SVG", icon=ft.Icons.DRAW, on_click=lambda e: self.export_figure("spectra", "svg")),
                            ],
                            wrap=True,
                        ),
                    ],
                ),
            )
            advanced_tab = ft.Container(
                padding=6,
                content=ft.Column(
                    controls=[
                        ft.Row(
                            [
                                self.analysis_method,
                                self.analysis_n_components,
                                self.analysis_max_iter,
                                self.analysis_random_state,
                            ],
                            wrap=True,
                        ),
                        self.analysis_description,
                        ft.Row(
                            [
                                ft.FilledButton("Esegui analisi", icon=ft.Icons.SCIENCE, on_click=self.run_advanced_analysis),
                                self.analysis_result_selector,
                                ft.OutlinedButton("Mostra sulla mappa", icon=ft.Icons.MAP, on_click=self.show_analysis_map),
                                ft.OutlinedButton("Figura PNG", icon=ft.Icons.IMAGE, on_click=lambda e: self.export_figure("analysis", "png")),
                            ],
                            wrap=True,
                        ),
                        self.analysis_status,
                        ft.Container(height=260, content=self.analysis_chart),
                    ],
                    scroll=ft.ScrollMode.AUTO,
                ),
            )
            processing_tabs = ft.Tabs(
                length=3,
                content=ft.Column(
                    controls=[
                        ft.TabBar(tabs=[ft.Tab(label="Baseline"), ft.Tab(label="Pipeline"), ft.Tab(label="Analisi avanzate")]),
                        ft.Container(
                            height=320,
                            content=ft.TabBarView(
                                controls=[baseline_tab, pipeline_tab, advanced_tab]
                            ),
                        ),
                    ],
                ),
            )

            right = ft.Container(
                expand=1,
                padding=8,
                content=ft.Column(
                    expand=True,
                    controls=[processing_tabs, self.spec_chart],
                ),
            )

            body = ft.Row(
                expand=True,
                spacing=0,
                controls=[left, ft.VerticalDivider(width=1), center, ft.VerticalDivider(width=1), right],
            )
            statusbar = ft.Container(padding=ft.Padding.symmetric(horizontal=8, vertical=4), content=ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, size=14), self.status]))

            self.page.add(ft.Column(expand=True, spacing=0, controls=[topbar, ft.Divider(height=1), body, ft.Divider(height=1), statusbar]))

    def _wire_matplotlib_events(self) -> None:
            self.map_fig.canvas.mpl_connect("button_press_event", self._on_map_click)
