# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with optional + dev extras)
pip install -e ".[advanced,dev]"

# Run the app
fairaman-viewer                       # installed entry point
python -m fairaman_viewer path/to/file.h5
python run_viewer.py                  # from a source checkout, no install needed

# Pre-commit checks (from CONTRIBUTING.md)
ruff check src tests
pytest
python -m compileall -q src

# Single test
pytest tests/test_registry.py::test_builtin_plugins_are_registered
```

Requires Python >= 3.11. `ramappy` and `pyspc-unmix` install from git; the `advanced`
extra (`scikit-learn`, `pymcr`, `pyspc-unmix`) is what enables the N-FINDR / MCR-ALS
plugins — the app runs without it but those analyses report missing dependencies.

## Architecture

The hard rule of this codebase: **dependencies point inward toward scientific/domain
code, never outward toward Flet.** `docs/ARCHITECTURE.md` and `CONTRIBUTING.md` state
the design rules; enforce them in any change.

Layers (`src/fairaman_viewer/`):

- `io/fairaman.py` — reads FAIRaman HDF5 (`ENTRY/data/intensity` etc.), infers
  `coordinate_mode` (`regular_grid` / `point_coordinates` / `single`) from array shape,
  and wraps the data in a `ramappy` `SpectralMap` or `Spectrum`. Exposes the
  `FairamanFile` domain model. No Flet.
- `validation/schema.py` — pure schema/data-quality checks returning `list[Issue]`
  (error/warning/info). No Flet.
- `processing/baseline.py` — baseline-correction registry (`METHODS`), preview
  (`estimate_baseline`) and chunked cancellable application (`apply_baseline`). Wraps
  `ramappy.processing`. No Flet.
- `processing/normalization.py` — normalization registry (`METHODS`), `transform` /
  `estimate_normalization` (preview) / `apply_normalization` (chunked, cancellable).
  Pure NumPy, no ramappy/Flet. MSC and PQN are two-pass: `fit_reference` then
  chunked `transform`. See `docs/NORMALIZATION.md`. Paired with `ui/normalization_panel.py`.
- `analysis/` — advanced analyses as plugins. `models.py` defines
  `AnalysisContext` / `AnalysisResult` / `AnalysisPlugin`; `registry.py` holds
  `ANALYSIS_PLUGINS` and `register_plugin`. Runners (`nfindr.py`, `mcr_als.py`) take
  `(AnalysisContext, params: dict)` and return an `AnalysisResult` (spectra / images /
  metadata / markers). They must not draw controls or import Flet; optional deps are
  imported lazily inside the runner.
- `export/service.py` — pure functions producing `bytes` / `str` from models or
  Matplotlib figures (`spectra_csv_bytes`, `map_csv_bytes`, `report_text`,
  `figure_bytes`). No Flet controls as inputs — keeps headless/batch export possible.
- `state.py` — `ViewerState`, the single source of truth. `reset_for_new_file` and
  `invalidate_after_processing` clear derived caches (`pca_cache`, `analysis_results`,
  `traces`) — any code that mutates the working spectral data must go through the
  latter.
- `controllers/` — mixin classes (`FileControllerMixin`, `MapControllerMixin`,
  `SpectraControllerMixin`, `AnalysisControllerMixin`, `ExportControllerMixin`)
  composed into `FairamanViewerApp`. This is the only layer that may know Flet *and*
  call the services. Long-running work runs via `self.page.run_thread(...)` and reports
  through callbacks; UI updates go through `self.safe_update(...)` (tolerant of
  background-thread races).
- `ui/` — reusable Flet controls (`components.py`) and `baseline_panel.py`.
- `app.py` — composition root: builds controls, lays them out, wires Matplotlib
  canvas events. Setup and layout only, no behavior.

Charts are `flet_charts.MatplotlibChartWithToolbar` over figures created with
`plt.subplots()` (managed pyplot figures are required by `flet-charts`, per README
0.3.2 notes).

### Processing vs. analysis

Processing (`processing/`) mutates the working dataset in place and therefore
**invalidates** PCA caches and analysis results (`state.invalidate_after_processing`).
Analysis (`analysis/`) is derived from the current dataset and must not mutate it.
Do not silently transform scientific data to satisfy an algorithm's constraints
(e.g. MCR-ALS non-negativity) — surface the requirement to the user instead.

### Adding an advanced analysis

1. New module in `analysis/`, e.g. `fcls.py`, with `def run_fcls(context, params) -> AnalysisResult`.
2. Register an `AnalysisPlugin` in `analysis/registry.py` (set `optional_packages` for lazy deps).
3. Return results only through `AnalysisResult`; the UI and export layers consume it
   without knowing algorithm internals.

## Conventions

- Ruff: `line-length = 100`, rules `E,F,I,UP,B` (`E501` ignored), target py311.
- Docstrings and user-facing strings are in Italian; keep that consistent.
- `*.h5` / `*.hdf5` and `exports/` are gitignored — never commit sample data.

Note: `config.APP_VERSION` (`0.3.0`) currently lags `pyproject.toml` version (`0.3.2`);
`test_config.py` only checks it is truthy. Keep them in sync when bumping.
