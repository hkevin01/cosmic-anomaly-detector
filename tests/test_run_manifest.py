from cosmic_anomaly_detector.utils.config import get_config
from cosmic_anomaly_detector.utils.run_manifest import build_manifest


def test_run_manifest_structure() -> None:
    cfg = get_config()
    manifest = build_manifest(cfg)
    assert "timestamp" in manifest
    assert "config" in manifest
    assert "image_processing" in manifest["config"]
