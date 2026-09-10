# FAIRaman Viewer

Desktop viewer and analysis application for FAIRaman Raman spectroscopy files, built with **Flet**, **Matplotlib**, and **ramappy**.

The repository intentionally separates scientific logic from the GUI so that new processing and analysis methods can be added without growing a monolithic application file.

## Architecture

```text
src/fairaman_viewer/
├── app.py                  # composition root and application layout
├── cli.py                  # command-line entry point
├── config.py               # application constants
├── state.py                # GUI-independent application state
├── controllers/            # focused application behavior mixins
│   ├── files.py            # loading, metadata and validation
│   ├── maps.py             # map rendering and spatial selection
│   ├── spectra.py          # spectral plotting and processing
│   ├── analysis.py         # advanced-analysis orchestration
│   └── export.py           # save/export actions
├── io/
│   └── fairaman.py         # HDF5/FAIRaman loading and domain model
├── validation/
│   └── schema.py           # FAIRaman validation rules
├── processing/
│   └── baseline.py         # baseline correction
├── analysis/
│   ├── models.py           # plugin contracts and result model
│   ├── registry.py         # analysis registry
│   ├── nfindr.py           # N-FINDR
│   ├── mcr_als.py          # MCR-ALS
│   └── utils.py
├── export/
│   └── service.py          # CSV, report and figure export
└── ui/
    ├── components.py       # reusable Flet controls
    └── baseline_panel.py   # baseline UI
```

## Install for development

```bash
git clone <repository-url>
cd fairaman-viewer
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[advanced,dev]"
```

Run with:

```bash
fairaman-viewer
# or
python -m fairaman_viewer path/to/file.h5
```

## Adding an advanced analysis

1. Create a module under `src/fairaman_viewer/analysis/`, e.g. `fcls.py`.
2. Implement a runner with the contract:

```python
def run_fcls(context: AnalysisContext, params: dict[str, Any]) -> AnalysisResult:
    ...
```

3. Register it in `analysis/registry.py` with an `AnalysisPlugin`.
4. Return spectra, images/maps, metadata and markers through `AnalysisResult`. The UI and export layer can then consume the result without knowing algorithm internals.

This is the intended extension path for FCLS, NMF, clustering, SAM, spectral entropy, quality metrics and future workflows.

## Export

The export service is GUI-independent. Current support includes:

- spectrum CSV
- map-value CSV
- text report
- Matplotlib figure export (PNG/SVG through the application)

Keeping export functions outside Flet makes it straightforward to add TIFF, NumPy/Zarr, HDF5-derived analysis products, or reproducible analysis bundles later.

## License

GPL-3.0-or-later.

## Avvio rapido da sorgente

Se preferisci non installare il package in editable mode, dalla cartella del repository puoi usare:

```bash
python run_viewer.py
```

Il launcher aggiunge automaticamente `src/` al Python path. È comunque necessario avere installato le dipendenze runtime.

Sono supportati anche:

```bash
python -m fairaman_viewer
fairaman-viewer
```

Dalla versione 0.3.1 anche l'esecuzione diretta di `src/fairaman_viewer/__main__.py` è tollerata, anche se `python -m fairaman_viewer` resta il metodo raccomandato.

## Version 0.3.2

Fixes Matplotlib integration with Flet by creating managed figures through `matplotlib.pyplot.subplots()`. This is required by `flet-charts`, which expects each figure to have an active canvas manager.
