"""File loading, metadata, validation and data-model presentation."""

import tempfile
from pathlib import Path

import flet as ft
import numpy as np

from ..config import SEVERITY_COLOR
from ..io.fairaman import flatten_metadata, load_fairaman
from ..validation.schema import summary, validate

class FileControllerMixin:
    """Behavior mixin for FairamanViewerApp."""

    async def choose_file(self, _e=None) -> None:
            picked = await ft.FilePicker().pick_files(
                dialog_title="Apri un file FAIRaman",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["h5", "hdf5"],
                allow_multiple=False,
                with_data=self.page.web,
            )
            if not picked:
                return
            selected = picked[0]
            if selected.path:
                self.open_file(selected.path)
                return
            if selected.bytes:
                tmp = Path(tempfile.gettempdir()) / f"fairaman_{selected.name}"
                tmp.write_bytes(selected.bytes)
                self._temporary_inputs.append(tmp)
                self.open_file(tmp)
                return
            self.alert("File non accessibile", "Il selettore non ha restituito ne' un percorso ne' i byte del file.")

    def reload(self) -> None:
            if self.state.ff is not None:
                self.open_file(self.state.ff.path)

    def open_file(self, path: str | Path) -> None:
            path = Path(path)
            self.set_status(f"Lettura di {path.name}…")
            try:
                ff = load_fairaman(path)
            except Exception as exc:
                self.alert("Errore di lettura", f"{type(exc).__name__}: {exc}")
                self.set_status("Lettura fallita.")
                return

            self.state.reset_for_new_file(ff)
            shape = ff.map_shape()
            descr = f"{shape[0]}×{shape[1]} px" if shape and ff.is_map else f"{ff.n_spectra} spettri"
            self.header.value = (
                f"{ff.title} · {descr} · {ff.coordinate_mode} · FAIRaman v{ff.fairaman_version or '?'}"
            )

            axis = ff.spectral_axis
            if not self.band_lo.value:
                span = float(axis.max() - axis.min())
                self.band_lo.value = f"{axis.min() + 0.35 * span:.0f}"
                self.band_hi.value = f"{axis.min() + 0.45 * span:.0f}"

            self.fill_metadata()
            self.fill_validation()
            self.fill_data_model()
            self.refresh_map()
            self.refresh_spectra()
            self.refresh_analysis_controls()

            msg = f"Caricato {path.name}."
            if ff.warnings:
                msg += "  " + ff.warnings[0]
            self.set_status(msg)
            self.safe_update(self.header, self.band_lo, self.band_hi)

    def fill_metadata(self) -> None:
            ff = self.state.ff
            self.metadata_list.controls = []
            if ff is None:
                self.safe_update(self.metadata_list)
                return
            needle = (self.metadata_filter.value or "").strip().lower()
            for label, block in (("PROJECT", ff.project), ("SAMPLE", ff.sample), ("ENTRY", ff.entry)):
                rows = [
                    (path, value)
                    for path, value in flatten_metadata(block)
                    if not needle or needle in path.lower() or needle in str(value).lower()
                ]
                if not rows:
                    continue
                controls = [
                    ft.ListTile(
                        title=ft.Text(path, size=11),
                        subtitle=ft.Text("" if value is None else str(value), size=10, selectable=True),
                        dense=True,
                    )
                    for path, value in rows
                ]
                self.metadata_list.controls.append(ft.ExpansionTile(title=ft.Text(label), controls=controls, expanded=True))
            self.safe_update(self.metadata_list)

    def fill_validation(self) -> None:
            self.validation_list.controls = []
            if self.state.ff is None:
                return
            self.state.issues = validate(self.state.ff)
            self.validation_summary.value = summary(self.state.issues)
            for issue in self.state.issues:
                tag = {"error": "ERRORE", "warning": "AVVISO", "info": "NOTA"}[issue.severity]
                self.validation_list.controls.append(
                    ft.Container(
                        padding=6,
                        content=ft.Column(
                            [
                                ft.Text(f"{tag} · {issue.where}", weight=ft.FontWeight.BOLD, color=SEVERITY_COLOR[issue.severity], size=11),
                                ft.Text(issue.message, color=SEVERITY_COLOR[issue.severity], size=10),
                            ],
                            spacing=2,
                        ),
                    )
                )
            self.safe_update(self.validation_summary, self.validation_list)

    def fill_data_model(self) -> None:
            ff = self.state.ff
            if ff is None:
                self.data_model_text.value = ""
                self.safe_update(self.data_model_text)
                return
            axis = ff.spectral_axis
            lines = [
                f"file              {ff.path.name}",
                f"fairaman_version  {ff.fairaman_version}",
                f"source_format     {ff.source_format}",
                "",
                "ENTRY/data (canonical spectral object)",
            ]
            lines += [f"  @{key:<22} {value}" for key, value in sorted(ff.data_model.items())]
            lines += [
                "",
                "asse spettrale",
                f"  n punti                 {axis.size}",
                f"  intervallo              {axis.min():.2f} – {axis.max():.2f} cm⁻¹",
                f"  passo mediano           {np.median(np.diff(axis)):.3f} cm⁻¹" if axis.size > 1 else "",
                "",
                "oggetto ramappy",
                f"  tipo                    {type(ff.obj).__name__}",
                f"  n spettri               {ff.n_spectra}",
                f"  forma dati              {tuple(np.asarray(ff.obj.data).shape)}",
            ]
            extent = ff.spatial_extent()
            if extent:
                lines.append(
                    f"  extent (µm)             x {extent[0]:.1f}…{extent[1]:.1f}  "
                    f"y {extent[2]:.1f}…{extent[3]:.1f}"
                )
            if ff.aux_images:
                lines += ["", "immagini ausiliarie"]
                for name, img in ff.aux_images.items():
                    lines.append(f"  {name:<22} {img.array.shape}  {'calibrata' if img.extent else 'NON calibrata'}")
            lines += ["", "elaborazioni applicate"]
            lines += ([f"  {i}. {step}" for i, step in enumerate(self.state.history, 1)]
                      or ["  nessuna: dati come letti dal file"])
            if self.state.analysis_results:
                lines += ["", "analisi derivate (invalidate se cambia il preprocessing)"]
                lines += [f"  {r.title}" for r in self.state.analysis_results.values()]
            self.data_model_text.value = "\n".join(lines)
            self.safe_update(self.data_model_text)
