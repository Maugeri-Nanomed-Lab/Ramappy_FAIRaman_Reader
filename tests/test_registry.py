from fairaman_viewer.analysis.registry import ANALYSIS_PLUGINS


def test_builtin_plugins_are_registered():
    assert "nfindr" in ANALYSIS_PLUGINS
    assert "mcr_als" in ANALYSIS_PLUGINS
