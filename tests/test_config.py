from fairaman_viewer.config import APP_NAME, APP_VERSION


def test_app_metadata():
    assert APP_NAME == "FAIRaman Viewer"
    assert APP_VERSION
