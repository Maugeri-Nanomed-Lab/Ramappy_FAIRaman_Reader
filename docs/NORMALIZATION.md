# Normalization

**Status: implemented (first pass).** `processing/normalization.py` +
`ui/normalization_panel.py` provide a dedicated "Normalizzazione" tab, mirroring
the baseline panel. The old ad-hoc `l2 / l1 / max / …` checkboxes in the
"Pipeline" tab (and `config.NORMS`) have been removed to avoid a second code
path.

Implemented methods: `l2`, `l1`, `max`, `area` (real-axis integration), `rms`,
`snv`, `snv_robust`, `minmax`, `reference_band` (height/area + search window +
optional local baseline), `msc`, `pqn`. Not yet done: EMSC; dataset-level
scaling as an analysis-time option (family D below).

This document records the design and the parts still open. It follows the
dependency rule of [ARCHITECTURE.md](ARCHITECTURE.md): scientific code stays
free of Flet.

## Why normalize

On a Raman map / hyperspectral cube the absolute intensity of a pixel carries
non-chemical, largely **multiplicative** variation:

- laser-power drift and effective integration-time variation;
- topography / focus changes across the field (sample–objective distance);
- material thickness, density and packing under the focal volume;
- elastic scattering and self-absorption, which vary pixel-to-pixel in imaging;
- non-uniform detector response / vignetting.

Normalization makes pixels comparable — a precondition for PCA, MCR-ALS,
N-FINDR, clustering and band-ratio maps. It is **destructive and
non-invertible**, so it is handled like baseline correction: recorded in
`state.history`, triggers `state.invalidate_after_processing()`, undone only by
reloading the file ("Ripristina grezzi").

## Method taxonomy

Four families. They are not interchangeable.

### A. Per-spectrum intensity normalization (row-wise)

| Method | Effect | GUI parameters | Notes |
|---|---|---|---|
| Vector L2 (Euclidean) | ‖s‖₂ = 1 | ROI optional | the Raman "vector normalization"; de-facto default |
| L1 | Σ\|s\| = 1 | ROI optional | equals discrete-sum area only on a non-negative, evenly-spaced spectrum |
| Max / peak | max(s) = 1 | ROI optional, window | sensitive to spikes and to a single band |
| Area / AUC | integral = 1 | ROI optional | **integrate on the real x-axis** (trapezoid/Simpson), not a sample sum |
| SNV | (s − μ) / σ per spectrum | — | removes offset + scale; heavily used in HSI |
| Robust SNV | median / MAD | — | resists strong peaks and outliers |
| RMS | √⟨s²⟩ = 1 | ROI optional | common in biological Raman |
| Min–max | rescale to [0, 1] | ROI optional | for display, not for chemometrics |

`ramappy`'s `norm="scale", by_pixel=True` **is SNV** (per-row zero mean, unit
variance). "Vector normalization" in the Raman literature is plain **L2**
(`norm="l2"`), without SNV's centering step. `frobenius` in `ramappy` normalizes
the whole map by one scalar, not per pixel — not the per-pixel vector norm.

### B. Reference-based normalization

- **Reference band** — divide the whole spectrum by the intensity of one chosen
  band:
  - peak **height** at the position, or **integrated area** over a window;
  - **local baseline** under the band (2-point / trapezoid) so background is
    not counted;
  - **peak-search window (± cm⁻¹)** around the nominal position, to absorb
    calibration drift between pixels;
  - useful anchors: phenylalanine 1003 cm⁻¹, amide I ~1650, CH₂ ~1450;
    substrate bands CaF₂ ~321, Si 520.7, sapphire, diamond 1332.
- **Band ratio** — force band A / band B = 1 (when a trustworthy internal
  standard exists).

`ramappy` only offers single-point `spectral_position` (one wavenumber, noisy)
and `area` restricted to a ROI (which normalizes *that ROI*, not the whole
spectrum by the band). Whole-spectrum reference-band normalization is in-repo
work.

### C. Scatter correction (matters for HSI)

- **MSC** (multiplicative scatter correction) — regress each spectrum against a
  reference spectrum (dataset mean), correct slope and offset.
- **EMSC** (extended MSC) — adds polynomial terms and, optionally, known
  interferent spectra; close to a standard in Raman/IR imaging. Conceptually
  overlaps with baseline correction — decide whether it lives here or in the
  baseline module.
- **PQN** (probabilistic quotient normalization) — factor = median of the
  ratio spectrum / reference; robust when a few bands dominate; needs a
  reference (median spectrum) and usually a coarse prior normalization
  (typically area).

These need a dataset-global reference, so they are **two-pass** (see below).

### D. Dataset-level scaling — not preprocessing here

Mean-centering, auto-scaling (unit variance), Pareto, Poisson / √ scaling for
photon-counting data. These are column-wise (across pixels), model-specific and
must be recomputed per analysis. They belong in `AnalysisContext` / the runner,
**not** in the destructive preprocessing chain.

## ramappy coverage

Available via `normalize_intensities`: `l1`, `l2`, `max`, `area` (Simpson, real
axis), `minmax_scale`, `frobenius`, `scale`, `spectral_position`. Supports
`roi_x`, `separate_regions`, `by_pixel`, `mask`; it is a `pipeline_step` with
`supports_preview=True`.

Implemented in-repo (pure NumPy, no Flet, `processing/normalization.py`): robust
SNV (median / MAD), RMS, whole-spectrum reference-band (height / area + search
window + optional local baseline), MSC, PQN. The per-spectrum scalar methods
(`l2`, `l1`, `max`, `area`, `minmax`) are also reimplemented there rather than
delegated, so preview, ROI semantics and degenerate-row handling are uniform.

Still in-repo work: EMSC.

## Correctness constraints

Per [CONTRIBUTING.md](../CONTRIBUTING.md) — *do not silently transform
scientific data to satisfy algorithm constraints; surface the requirement*:

- **Ordering.** Baseline before area / SNV / L2 / RMS — a residual offset
  corrupts them. Spike removal before max / area. If normalization is requested
  before baseline, warn explicitly; do not silently block or reorder.
- **Area integration** on the true (possibly non-uniform) x-axis; handle
  negative regions left by baseline.
- **Reference band.** In pixels where the band is absent, below noise, or ~0
  the division blows up. Detect, count the affected pixels, surface them, offer
  to mask/flag — never emit inf/NaN silently.
- **SNV / RMS** on a short ROI or near-flat spectrum → tiny σ → blow-up. Offer
  the robust variant; report.
- **Dead / masked / all-NaN pixels** — skipped and counted in the summary.
- **`by_pixel`** defaults to per-spectrum (imaging). Cross-pixel stays
  available but warns.

## Two-pass reference pattern

Baseline correction is purely per-spectrum, so it processes in independent
chunks (`processing/baseline.py`). MSC and PQN are not: they first estimate a
**global reference** over the whole selected dataset, then transform pixel by
pixel. `processing/normalization.py` implements this as:

- `fit_reference(matrix, x, method, values, roi)` → a reference dict, then
  `transform(block, …, reference=…)` per chunk in `apply_normalization`;
- the reference is cached on `NormalizationPanel` (keyed by method + params +
  ROI + `id(ff.obj)`), recomputed when any of those change or the file
  reloads — the panel, not `state`, so `state` stays UI-agnostic;
- `on_normalization_ok` records the reference kind (and PQN prior) in
  `state.history` — recomputing after further processing changes the result.

## Provenance

Every applied normalization records a descriptor in `state.history` (method +
parameters + reference band / ratio; for MSC/PQN also the reference source).
`export/service.report_text` already serializes `state.history`, so the
descriptor must be complete enough to reproduce the step.

## Open architectural question

`ramappy` ships a full pipeline framework (`ramappy.pipeline`: `Pipeline`,
`HistoryLog`, `replay`, YAML serialization, step registry, previews) that the
app currently does not use — baseline and the Pipeline-tab checkboxes are
hand-rolled. Before adding a third hand-rolled panel, decide: keep the current
pattern (simpler, consistent with what exists), or adopt `ramappy.pipeline` for
orchestration, provenance and replay across baseline + normalization + despike.
This choice shapes `state.history` and the report format.

## References

- Barnes, Dhanoa, Lister (1989) — standard normal variate transformation and
  de-trending. *Applied Spectroscopy* 43(5).
- Geladi, MacDougall, Martens (1985) — multiplicative scatter correction.
  *Applied Spectroscopy* 39(3).
- Afseth, Kohler (2012) — extended multiplicative signal correction for Raman.
  *Chemometrics and Intelligent Laboratory Systems* 117.
- Dieterle, Ross, Schlotterbeck, Senn (2006) — probabilistic quotient
  normalization. *Analytical Chemistry* 78(13).
- Lasch (2012) — spectral pre-processing for biomedical vibrational
  spectroscopy and microspectroscopic imaging. *Chemometrics and Intelligent
  Laboratory Systems* 117.
- Bocklitz, Walter, Hartmann, Rösch, Popp (2011) — how to pre-process Raman
  spectra for reliable and stable models. *Analytica Chimica Acta* 704.
- Guo, Bocklitz, Popp — common preprocessing of Raman spectra (review).
- Ryabchykov et al. — automatization of Raman spectra preprocessing.
