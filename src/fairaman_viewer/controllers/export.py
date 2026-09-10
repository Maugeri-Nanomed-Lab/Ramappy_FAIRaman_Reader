"""Save/export behavior for spectra, maps, figures and reports."""

from pathlib import Path

import flet as ft

from ..export.service import figure_bytes, map_csv_bytes, report_text, spectra_csv_bytes

class ExportControllerMixin:
    """Behavior mixin for FairamanViewerApp."""

    async def _save_bytes(self, data: bytes, file_name: str, extensions: list[str]) -> None:
            path = await ft.FilePicker().save_file(
                dialog_title="Esporta",
                file_name=file_name,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=extensions,
                src_bytes=data,
            )
            if self.page.web:
                self.set_status(f"Esportato: {file_name}")
            elif path:
                self.set_status(f"Salvato: {Path(path).name}")

    async def export_spectra(self, _e=None) -> None:
            ff = self.state.ff
            if ff is None:
                return
            try:
                data = spectra_csv_bytes(ff, self.state.traces, bool(self.show_mean.value))
            except ValueError as exc:
                self.notify(str(exc))
                return
            await self._save_bytes(data, f"{ff.path.stem}_spettri.csv", ["csv"])

    async def export_report(self, _e=None) -> None:
            ff = self.state.ff
            if ff is None:
                return
            data = report_text(self.state).encode("utf-8")
            await self._save_bytes(data, f"{ff.path.stem}_report.txt", ["txt"])

    async def export_figure(self, which: str, fmt: str) -> None:
            ff = self.state.ff
            if ff is None:
                return
            fig = {"map": self.map_fig, "spectra": self.spec_fig, "analysis": self.analysis_fig}[which]
            suffix = {"map": "mappa", "spectra": "spettri", "analysis": "analisi"}[which]
            data = figure_bytes(fig, fmt=fmt, dpi=300)
            await self._save_bytes(data, f"{ff.path.stem}_{suffix}.{fmt}", [fmt])

    async def export_map_data(self, _e=None) -> None:
            ff = self.state.ff
            if ff is None or self.state.image_data is None:
                self.notify("Nessuna mappa visualizzata da esportare.")
                return
            data = map_csv_bytes(self.state.image_data, self.state.image_label)
            await self._save_bytes(data, f"{ff.path.stem}_mappa.csv", ["csv"])
