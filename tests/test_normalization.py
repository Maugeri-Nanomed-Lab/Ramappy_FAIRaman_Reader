import numpy as np

from fairaman_viewer.processing import normalization as nz


def _spectrum(n=256, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(400.0, 1800.0, n)
    y = 50.0 + 20.0 * np.exp(-((x - 1003.0) ** 2) / (2 * 15.0 ** 2)) + rng.normal(0, 0.4, n)
    return x, y


def test_registry_exposes_core_methods():
    for key in ("l2", "l1", "max", "area", "rms", "snv", "snv_robust", "minmax",
                "reference_band", "msc", "pqn"):
        assert key in nz.METHODS
    assert nz.METHODS["msc"].two_pass and nz.METHODS["pqn"].two_pass
    assert not nz.METHODS["l2"].two_pass


def test_l2_gives_unit_euclidean_norm():
    x, y = _spectrum()
    out = nz.estimate_normalization(x, y, "l2")
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_area_gives_unit_integral_on_real_axis():
    x, y = _spectrum()
    out = nz.estimate_normalization(x, y, "area")
    assert np.isclose(abs(np.trapezoid(out, x)), 1.0)


def test_snv_has_zero_mean_and_unit_std():
    x, y = _spectrum()
    out = nz.estimate_normalization(x, y, "snv")
    assert abs(float(out.mean())) < 1e-9
    assert np.isclose(float(out.std()), 1.0)


def test_robust_snv_has_zero_median():
    x, y = _spectrum()
    out = nz.estimate_normalization(x, y, "snv_robust")
    assert abs(float(np.median(out))) < 1e-9


def test_roi_l2_uses_only_the_region():
    x, y = _spectrum()
    out = nz.estimate_normalization(x, y, "l2", roi=(950.0, 1050.0))
    sel = (x >= 950.0) & (x <= 1050.0)
    assert np.isclose(np.linalg.norm(out[sel]), 1.0)


def test_reference_band_height_sets_peak_to_one():
    x, y = _spectrum()
    out = nz.estimate_normalization(
        x, y, "reference_band",
        {"center": 1003.0, "mode": "height", "search": 10.0, "local_baseline": False},
    )
    assert np.isclose(float(out.max()), 1.0, atol=1e-6)


def test_reference_band_zero_band_leaves_spectrum_unchanged():
    x = np.linspace(400.0, 1800.0, 256)
    y = np.zeros_like(x)
    out = nz.estimate_normalization(
        x, y, "reference_band", {"center": 1003.0, "mode": "area", "window": 10.0}
    )
    assert np.array_equal(out, y)


def test_msc_aligns_scaled_and_shifted_copies():
    x, y = _spectrum(seed=1)
    matrix = np.vstack([y, 3.0 * y + 5.0, 0.5 * y - 2.0])
    reference = nz.fit_reference(matrix, x, "msc")
    out, bad = nz.transform(matrix, x, "msc", {}, None, reference)
    assert not bad.any()
    assert np.allclose(out[0], out[1], atol=1e-6)
    assert np.allclose(out[0], out[2], atol=1e-6)


def test_pqn_is_finite_on_diluted_copies():
    x, y = _spectrum(seed=2)
    matrix = np.vstack([y, 2.0 * y, 0.3 * y])
    reference = nz.fit_reference(matrix, x, "pqn", {"prior": "area"})
    out, bad = nz.transform(matrix, x, "pqn", {"prior": "area"}, None, reference)
    assert np.isfinite(out).all()
    assert not bad.any()


def test_nan_spectrum_is_marked_degenerate_and_left_alone():
    x, y = _spectrum()
    dead = np.full_like(y, np.nan)
    out, bad = nz.transform(np.vstack([y, dead]), x, "l2", {}, None, None)
    assert bad[1] and not bad[0]
    assert np.allclose(out[0] * np.linalg.norm(y), y)


def test_apply_normalization_mutates_in_place_and_reports():
    x, y = _spectrum()
    matrix = np.vstack([y, 2.0 * y, 3.0 * y]).astype(np.float32)

    class _Obj:
        def __init__(self, data, xx):
            self.data = data
            self.x = xx

    obj = _Obj(matrix, x)
    info = nz.apply_normalization(obj, "l2")
    assert info["n_spectra"] == 3
    assert info["degenerate"] == 0
    for row in obj.data:
        assert np.isclose(np.linalg.norm(row), 1.0, atol=1e-5)
