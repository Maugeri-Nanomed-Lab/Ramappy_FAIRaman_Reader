"""Modern Flet composition root for FAIRaman Viewer.

Version 3 keeps the existing controllers/scientific logic untouched and
reorganises the UI around the two primary visual tasks:

1. detailed spectral inspection (large, top-centre)
2. Raman image / tissue inspection (centre, below spectra)

Metadata remain on the left and preprocessing/analysis tools on the right,
following the visual hierarchy of modern digital-pathology viewers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import flet as ft
import flet_charts as fch
import matplotlib.pyplot as plt
from ramappy import Spectrum

from .analysis.registry import ANALYSIS_PLUGINS
from .config import APP_NAME, COLORMAPS
from .controllers import (
    AnalysisControllerMixin,
    ExportControllerMixin,
    FileControllerMixin,
    MapControllerMixin,
    SpectraControllerMixin,
)
from .state import ViewerState
from .ui.baseline_panel import BaselinePanel
from .ui.components import _dropdown, _num_field
from .ui.normalization_panel import NormalizationPanel
from .validation.schema import summary


class FairamanViewerApp(
    FileControllerMixin,
    MapControllerMixin,
    SpectraControllerMixin,
    AnalysisControllerMixin,
    ExportControllerMixin,
):
    """Compose the FAIRaman Viewer application from focused behaviours."""

    def __init__(self, page: ft.Page, initial_file: str | None = None):
        self.page = page
        self.initial_file = initial_file
        self.state = ViewerState()
        self._temporary_inputs: list[Path] = []
        self._selector = None
        self._colorbar = None
        self._spec_ax2 = None

        # Direct map interaction state (QuPath-like navigation / ROI tools).
        self._map_tool = "pixel"
        self._pan_active = False
        self._pan_start_px = None
        self._pan_xlim = None
        self._pan_ylim = None
        self._map_home_view = None
        self._last_pan_draw = 0.0
        self._rois: list[dict] = []
        self._roi_counter = 0
        self._roi_file_token = None

        self._configure_page()
        self._build_controls()
        self._build_layout()
        self._wire_matplotlib_events()

        if initial_file:
            self.open_file(initial_file)

    # ------------------------------------------------------------------
    # Application shell
    # ------------------------------------------------------------------

    def _configure_page(self) -> None:
        self.page.title = APP_NAME
        self.page.padding = 0
        self.page.spacing = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.bgcolor = "#F5F7FA"

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
        """Rirasterizza una figura Matplotlib e sincronizza i controlli Flet."""
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

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        # Header / file actions
        self.header = ft.Text(
            "Nessun file caricato",
            weight=ft.FontWeight.W_600,
            size=13,
            color="#344054",
        )
        self.status = ft.Text("Pronto.", size=11, expand=True, color="#475467")

        # Metadata
        self.metadata_filter = ft.TextField(
            label="Cerca nei metadati",
            dense=True,
            prefix_icon=ft.Icons.SEARCH,
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
            width=175,
            on_select=lambda e: self.refresh_map(),
        )
        self.band_lo = _num_field(label="Da cm⁻¹", width=100, on_change=lambda e: None)
        self.band_hi = _num_field(label="A cm⁻¹", width=100, on_change=lambda e: None)
        self.pc_index = _num_field("1", label="PC", width=65)
        self.cmap = _dropdown(
            "viridis",
            COLORMAPS,
            label="Colormap",
            width=135,
            on_select=lambda e: self.refresh_map(),
        )
        self.show_wl = ft.Checkbox(
            label="White light",
            value=False,
            on_change=lambda e: self.refresh_map(),
        )
        self.wl_alpha = ft.Slider(
            value=0.55,
            min=0,
            max=1,
            divisions=20,
            width=120,
            on_change=lambda e: self.refresh_map(),
        )
        self.agg_mode = _dropdown("mean", ["mean", "median", "max"], label="ROI", width=105)

        # QuPath-like interaction tools.  The scientific ROI calculations live
        # in MapControllerMixin; these controls only select the current gesture.
        self.map_tool_label = ft.Text(
            "Pixel", size=10, weight=ft.FontWeight.W_600, color="#344054"
        )
        self.roi_count_label = ft.Text("0 ROI", size=10, color="#667085")
        self.map_pixel_button = ft.IconButton(
            icon=ft.Icons.TOUCH_APP,
            tooltip="Pixel: click per aggiungere lo spettro",
            on_click=lambda e: self.set_map_tool("pixel"),
        )
        self.map_pan_button = ft.IconButton(
            icon=ft.Icons.PAN_TOOL,
            tooltip="Pan: trascina la mappa con il tasto sinistro",
            on_click=lambda e: self.set_map_tool("pan"),
        )
        self.map_rectangle_button = ft.IconButton(
            icon=ft.Icons.CROP_SQUARE,
            tooltip="ROI rettangolare: trascina per disegnare",
            on_click=lambda e: self.set_map_tool("rectangle"),
        )
        self.map_polygon_button = ft.IconButton(
            icon=ft.Icons.POLYLINE,
            tooltip="ROI poligonale: clicca i vertici e chiudi sul primo punto",
            on_click=lambda e: self.set_map_tool("polygon"),
        )
        self.map_freehand_button = ft.IconButton(
            icon=ft.Icons.GESTURE,
            tooltip="ROI libera: disegna tenendo premuto il tasto sinistro",
            on_click=lambda e: self.set_map_tool("freehand"),
        )
        self.map_home_button = ft.IconButton(
            icon=ft.Icons.HOME,
            tooltip="Mostra tutta la mappa",
            on_click=self.reset_map_view,
        )
        self.map_delete_roi_button = ft.IconButton(
            icon=ft.Icons.UNDO,
            tooltip="Elimina l'ultima ROI",
            on_click=self.delete_last_roi,
        )
        self.map_clear_roi_button = ft.IconButton(
            icon=ft.Icons.DELETE_OUTLINE,
            tooltip="Elimina tutte le ROI",
            on_click=self.clear_rois,
        )

        # Figure dimensions are deliberately wider than before. The Flet
        # containers still control the actual responsive size.
        self.map_fig, self.map_ax = plt.subplots(
            figsize=(8.5, 3.2), dpi=100, layout="constrained"
        )
        self.map_ax.set_axis_off()
        self.map_chart = fch.MatplotlibChartWithToolbar(
            figure=self.map_fig,
            expand=True,
        )

        # Spectrum + processing
        self.show_mean = ft.Checkbox(
            label="Media globale",
            value=True,
            on_change=lambda e: self.refresh_spectra(),
        )
        self.use_despike = ft.Checkbox(label="Despike", value=False)
        self.use_smooth = ft.Checkbox(label="SavGol", value=False)
        self.savgol_window = _num_field("9", label="Finestra", width=88)
        self.savgol_order = _num_field("3", label="Grado", width=78)
        self.baseline_panel = BaselinePanel(self)
        self.normalization_panel = NormalizationPanel(self)

        # The spectrum is the primary analytical view: give Matplotlib a wide
        # canvas and let it occupy the largest central panel.
        self.spec_fig, self.spec_ax = plt.subplots(
            figsize=(11.5, 4.0), dpi=100, layout="constrained"
        )
        self.spec_chart = fch.MatplotlibChartWithToolbar(
            figure=self.spec_fig,
            expand=True,
        )

        # Advanced analysis controls
        self.analysis_method = _dropdown(
            "nfindr",
            list(ANALYSIS_PLUGINS),
            label="Analisi",
            width=205,
            on_select=lambda e: self.update_analysis_description(),
        )
        self.analysis_n_components = _num_field("4", label="Componenti", width=105)
        self.analysis_max_iter = _num_field("100", label="Max iter", width=95)
        self.analysis_random_state = _num_field("21", label="Random state", width=110)
        self.analysis_description = ft.Text(size=11, color=ft.Colors.ON_SURFACE_VARIANT)
        self.analysis_status = ft.Text(size=11)
        self.analysis_result_selector = _dropdown("", [], label="Mappa risultato", width=230)

        self.analysis_fig, self.analysis_ax = plt.subplots(
            figsize=(6.0, 3.8), dpi=100, layout="constrained"
        )
        self.analysis_chart = fch.MatplotlibChartWithToolbar(
            figure=self.analysis_fig,
            expand=True,
        )
        self.update_analysis_description()

    # ------------------------------------------------------------------
    # Modern layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        def panel(
            title: str,
            content: ft.Control,
            *,
            expand: int | bool | None = None,
            subtitle: str | None = None,
            trailing: list[ft.Control] | None = None,
            padding: int = 10,
        ) -> ft.Container:
            """Small visual wrapper used throughout the viewer."""
            title_row_controls: list[ft.Control] = [
                ft.Column(
                    spacing=0,
                    controls=[
                        ft.Text(title, size=14, weight=ft.FontWeight.W_600, color="#101828"),
                        ft.Text(subtitle, size=10, color="#667085") if subtitle else ft.Container(),
                    ],
                    expand=True,
                )
            ]
            if trailing:
                title_row_controls.extend(trailing)

            return ft.Container(
                expand=expand,
                bgcolor="#FFFFFF",
                border_radius=12,
                padding=padding,
                content=ft.Column(
                    expand=True,
                    spacing=8,
                    controls=[
                        ft.Row(
                            controls=title_row_controls,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Divider(height=1, color="#EAECF0"),
                        ft.Container(expand=True, content=content),
                    ],
                ),
            )

        # --------------------------------------------------------------
        # Top application bar
        # --------------------------------------------------------------
        brand = ft.Row(
            spacing=8,
            controls=[
                ft.Container(
                    width=34,
                    height=34,
                    border_radius=9,
                    bgcolor="#EEF4FF",
                    alignment=ft.Alignment.CENTER,
                    content=ft.Icon(ft.Icons.SHOW_CHART, color="#3B82F6", size=21),
                ),
                ft.Column(
                    spacing=0,
                    controls=[
                        ft.Text("FAIRaman Viewer", size=17, weight=ft.FontWeight.BOLD, color="#101828"),
                        ft.Text(
                            "Findable · Accessible · Interoperable · Reusable",
                            size=9,
                            color="#667085",
                        ),
                    ],
                ),
            ],
        )

        topbar = ft.Container(
            height=58,
            bgcolor="#FFFFFF",
            padding=ft.Padding.symmetric(horizontal=12, vertical=8),
            content=ft.Row(
                controls=[
                    brand,
                    ft.VerticalDivider(width=16, color="#EAECF0"),
                    ft.FilledButton(
                        "Apri .h5",
                        icon=ft.Icons.FOLDER_OPEN,
                        on_click=self.choose_file,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.REFRESH,
                        tooltip="Ricarica file",
                        on_click=lambda e: self.reload(),
                    ),
                    ft.IconButton(
                        icon=ft.Icons.TABLE_VIEW,
                        tooltip="Esporta spettri CSV",
                        on_click=self.export_spectra,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DESCRIPTION,
                        tooltip="Esporta report",
                        on_click=self.export_report,
                    ),
                    ft.Container(expand=True),
                    self.header,
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        # --------------------------------------------------------------
        # Left: data / metadata browser
        # --------------------------------------------------------------
        metadata_tabs = ft.Tabs(
            length=3,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label="Metadati"),
                            ft.Tab(label="Validazione"),
                            ft.Tab(label="Modello dati"),
                        ]
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            ft.Column(
                                [self.metadata_filter, self.metadata_list],
                                expand=True,
                                spacing=8,
                            ),
                            ft.Column(
                                [self.validation_summary, self.validation_list],
                                expand=True,
                                spacing=8,
                            ),
                            ft.ListView(
                                controls=[self.data_model_text],
                                expand=True,
                            ),
                        ],
                    ),
                ],
            ),
        )

        left = ft.Container(
            width=300,
            padding=8,
            content=panel(
                "Data browser",
                metadata_tabs,
                subtitle="Campione, metadati e conformità FAIR",
                expand=True,
            ),
        )

        # --------------------------------------------------------------
        # Spectrum: primary visual area, top-centre
        # --------------------------------------------------------------
        spectrum_actions = ft.Row(
            spacing=4,
            wrap=True,
            controls=[
                self.show_mean,
                ft.OutlinedButton(
                    "PNG",
                    icon=ft.Icons.IMAGE,
                    on_click=lambda e: self.export_figure("spectra", "png"),
                ),
                ft.OutlinedButton(
                    "SVG",
                    icon=ft.Icons.DRAW,
                    on_click=lambda e: self.export_figure("spectra", "svg"),
                ),
                ft.OutlinedButton(
                    "CSV",
                    icon=ft.Icons.TABLE_VIEW,
                    on_click=self.export_spectra,
                ),
                ft.IconButton(
                    icon=ft.Icons.CLEAR_ALL,
                    tooltip="Pulisci spettri selezionati",
                    on_click=lambda e: self.clear_traces(),
                ),
            ],
        )

        spectrum_panel = panel(
            "Raman spectrum",
            self.spec_chart,
            subtitle="Ispezione dettagliata degli spettri e confronto tra pixel / ROI",
            trailing=[spectrum_actions],
            expand=True,
            padding=9,
        )

        # --------------------------------------------------------------
        # Raman map: secondary visual area below spectra
        # --------------------------------------------------------------
        # Keep map controls compact.  In the previous version the second Row
        # used both wrap=True and an expanding spacer; on narrower central
        # widths this could turn into two/three rows and push the map below
        # the visible viewport.  Two fixed single-line groups avoid that.
        map_primary_controls = ft.Row(
            wrap=False,
            spacing=6,
            controls=[
                self.image_mode,
                self.band_lo,
                self.band_hi,
                self.pc_index,
                self.cmap,
                ft.FilledButton(
                    "Aggiorna",
                    icon=ft.Icons.REFRESH,
                    on_click=lambda e: self.refresh_map(),
                ),
            ],
        )

        map_view_controls = ft.Row(
            wrap=False,
            spacing=6,
            controls=[
                self.show_wl,
                ft.Text("Overlay", size=11, color="#667085"),
                self.wl_alpha,
                self.agg_mode,
            ],
        )

        map_export_controls = ft.Row(
            wrap=False,
            spacing=0,
            controls=[
                ft.IconButton(
                    icon=ft.Icons.IMAGE,
                    tooltip="Esporta mappa PNG",
                    on_click=lambda e: self.export_figure("map", "png"),
                ),
                ft.IconButton(
                    icon=ft.Icons.DRAW,
                    tooltip="Esporta mappa SVG",
                    on_click=lambda e: self.export_figure("map", "svg"),
                ),
                ft.IconButton(
                    icon=ft.Icons.DATA_OBJECT,
                    tooltip="Esporta dati mappa CSV",
                    on_click=self.export_map_data,
                ),
            ],
        )

        # Compact image-viewer toolbar in the panel header.  It does not consume
        # vertical map space, which is important on ordinary laptop displays.
        map_interaction_tools = ft.Row(
            wrap=False,
            spacing=0,
            controls=[
                self.map_pixel_button,
                self.map_pan_button,
                self.map_rectangle_button,
                self.map_polygon_button,
                self.map_freehand_button,
                ft.VerticalDivider(width=8, color="#EAECF0"),
                self.map_home_button,
                self.map_delete_roi_button,
                self.map_clear_roi_button,
            ],
        )

        map_toolbar = ft.Column(
            spacing=2,
            controls=[
                map_primary_controls,
                ft.Row(
                    wrap=False,
                    spacing=6,
                    controls=[
                        map_view_controls,
                        ft.Container(width=6),
                        ft.Text("Tool:", size=10, color="#667085"),
                        self.map_tool_label,
                        ft.Text("·", size=10, color="#98A2B3"),
                        self.roi_count_label,
                        ft.Container(expand=True),
                        map_export_controls,
                    ],
                ),
            ],
        )

        map_content = ft.Column(
            expand=True,
            spacing=3,
            controls=[
                map_toolbar,
                # A minimum visible map area is more useful than another text
                # row: point picking and ROI selection happen directly here.
                ft.Container(expand=True, content=self.map_chart),
            ],
        )

        map_panel = panel(
            "Raman image & tissue view",
            map_content,
            subtitle="Rotella: zoom · Pan: trascina · Pixel/ROI: selezione spettrale",
            trailing=[map_interaction_tools],
            padding=9,
        )

        # The map is an interaction surface, not merely a secondary preview.
        # Reserve a real, always-visible area for it; the spectrum receives all
        # remaining vertical space.  This avoids a tall Matplotlib spectrum or
        # wrapped controls pushing the map below the window.
        map_area = ft.Container(
            height=370,
            content=map_panel,
        )

        centre = ft.Container(
            expand=3,
            padding=8,
            content=ft.Column(
                expand=True,
                spacing=6,
                controls=[spectrum_panel, map_area],
            ),
        )

        # --------------------------------------------------------------
        # Right: processing + advanced analysis
        # --------------------------------------------------------------
        baseline_tab = ft.Container(
            padding=6,
            content=ft.Column(
                controls=[self.baseline_panel.control],
                scroll=ft.ScrollMode.AUTO,
            ),
        )

        normalization_tab = ft.Container(
            padding=6,
            content=ft.Column(
                controls=[self.normalization_panel.control],
                scroll=ft.ScrollMode.AUTO,
            ),
        )

        pipeline_tab = ft.Container(
            padding=6,
            content=ft.Column(
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                controls=[
                    ft.Text("Pre-processing recipe", size=12, weight=ft.FontWeight.W_600),
                    ft.Row(
                        [self.use_despike, self.use_smooth],
                        wrap=True,
                    ),
                    ft.Row(
                        [self.savgol_window, self.savgol_order],
                        wrap=True,
                    ),
                    ft.FilledButton(
                        "Applica pipeline",
                        icon=ft.Icons.PLAY_ARROW,
                        on_click=self.apply_pipeline,
                    ),
                    ft.OutlinedButton(
                        "Ripristina dati grezzi",
                        icon=ft.Icons.RESTORE,
                        on_click=lambda e: self.reload(),
                    ),
                    ft.OutlinedButton(
                        "Pulisci spettri",
                        icon=ft.Icons.CLEAR_ALL,
                        on_click=lambda e: self.clear_traces(),
                    ),
                ],
            ),
        )

        advanced_tab = ft.Container(
            padding=6,
            content=ft.Column(
                scroll=ft.ScrollMode.AUTO,
                spacing=8,
                controls=[
                    self.analysis_method,
                    ft.Row(
                        [
                            self.analysis_n_components,
                            self.analysis_max_iter,
                        ],
                        wrap=True,
                    ),
                    self.analysis_random_state,
                    self.analysis_description,
                    ft.FilledButton(
                        "Esegui analisi",
                        icon=ft.Icons.SCIENCE,
                        on_click=self.run_advanced_analysis,
                    ),
                    self.analysis_result_selector,
                    ft.OutlinedButton(
                        "Mostra sulla mappa",
                        icon=ft.Icons.MAP,
                        on_click=self.show_analysis_map,
                    ),
                    ft.OutlinedButton(
                        "Esporta figura",
                        icon=ft.Icons.IMAGE,
                        on_click=lambda e: self.export_figure("analysis", "png"),
                    ),
                    self.analysis_status,
                    ft.Container(height=230, content=self.analysis_chart),
                ],
            ),
        )

        processing_tabs = ft.Tabs(
            length=4,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(
                        tabs=[
                            ft.Tab(label="Baseline"),
                            ft.Tab(label="Norm."),
                            ft.Tab(label="Pipeline"),
                            ft.Tab(label="Analysis"),
                        ]
                    ),
                    ft.TabBarView(
                        expand=True,
                        controls=[
                            baseline_tab,
                            normalization_tab,
                            pipeline_tab,
                            advanced_tab,
                        ],
                    ),
                ],
            ),
        )

        right = ft.Container(
            width=330,
            padding=8,
            content=panel(
                "Processing",
                processing_tabs,
                subtitle="Pre-processing, normalizzazione e analisi avanzate",
                expand=True,
            ),
        )

        body = ft.Row(
            expand=True,
            spacing=0,
            controls=[
                left,
                ft.VerticalDivider(width=1, color="#EAECF0"),
                centre,
                ft.VerticalDivider(width=1, color="#EAECF0"),
                right,
            ],
        )

        statusbar = ft.Container(
            height=30,
            bgcolor="#FFFFFF",
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.CHECK_CIRCLE_OUTLINE, size=14, color="#12B76A"),
                    self.status,
                    ft.Text("HDF5", size=10, color="#667085"),
                    ft.Text("FAIRaman", size=10, color="#667085"),
                ]
            ),
        )

        self.page.add(
            ft.Column(
                expand=True,
                spacing=0,
                controls=[
                    topbar,
                    ft.Divider(height=1, color="#EAECF0"),
                    body,
                    ft.Divider(height=1, color="#EAECF0"),
                    statusbar,
                ],
            )
        )

    def _wire_matplotlib_events(self) -> None:
        # One direct interaction layer for the map.  MapControllerMixin routes
        # these gestures according to the selected QuPath-like tool.
        self.map_fig.canvas.mpl_connect("button_press_event", self._on_map_mouse_press)
        self.map_fig.canvas.mpl_connect("motion_notify_event", self._on_map_mouse_move)
        self.map_fig.canvas.mpl_connect("button_release_event", self._on_map_mouse_release)
        self.map_fig.canvas.mpl_connect("scroll_event", self._on_map_scroll)
