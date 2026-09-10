# Architecture

FAIRaman Viewer follows a layered structure. Dependencies should point inward toward scientific/domain code, never from scientific code toward Flet.

```text
Flet UI / app composition
        │
        ▼
controller mixins ───────► export service
        │                     │
        ├──────────────► analysis registry/plugins
        │                     │
        ├──────────────► processing
        │
        └──────────────► state
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
              I/O model                validation
                 │
                 ▼
          ramappy / HDF5 / NumPy
```

## Responsibilities

- `io/`: parse FAIRaman files and expose file/domain objects. No Flet.
- `validation/`: schema and data-quality checks. No Flet.
- `processing/`: transformations that modify spectra (baseline correction, normalization). No Flet. See [NORMALIZATION.md](NORMALIZATION.md).
- `analysis/`: derived scientific analyses. Algorithms return `AnalysisResult`; they do not draw controls.
- `export/`: serialize figures/data/reports. No Flet.
- `state.py`: central mutable viewer state and derived-result invalidation.
- `controllers/`: coordinate user actions and call services. They may know Flet.
- `ui/`: reusable Flet controls/panels.
- `app.py`: composition root and layout only.

## Advanced-analysis contract

An analysis runner receives an `AnalysisContext` and plain parameter dictionary, then returns an `AnalysisResult` containing any combination of:

- `spectra`: named 1-D spectral profiles;
- `images`: named map-like arrays or point-wise vectors;
- `metadata`: reproducibility/provenance parameters;
- `markers`: spatial locations associated with extracted spectra.

This makes N-FINDR, MCR-ALS, FCLS, NMF and future algorithms interchangeable from the UI's point of view.

## Processing vs analysis

Processing changes the working spectral dataset and therefore invalidates PCA caches and advanced-analysis results. Analysis is derived from the current working dataset and should not silently mutate it.

Most processing is per-spectrum and runs in independent chunks. Normalization methods that need a dataset-global reference (MSC, PQN) are two-pass: `fit_reference` then chunked `transform`; the reference is cached on the panel and its identity is recorded in `state.history`.

## Export direction

Export functions accept data/models or Matplotlib figures, not Flet controls. This keeps future batch/headless export possible and makes unit testing much easier.
