from typing import Any

from cosmic_anomaly_detector.core.detector import AnomalyDetector


def test_gravity_adapter_invocation(monkeypatch: Any) -> None:
    calls = {}

    class FakeResult:
        overall_anomaly_score = 0.6
        physical_explanation = "Deviation"

    def fake_analyze(_objects, _image, _wcs):  # type: ignore[no-untyped-def]
        calls["called"] = True
        return [FakeResult()]

    det = AnomalyDetector()
    monkeypatch.setattr(
        det.gravitational_analyzer, "analyze_physics", fake_analyze
    )
    # Provide minimal processed image by patching image processor

    class FakeProc:
        def process(self, _path):  # type: ignore[no-untyped-def]
            import numpy as np
            return {
                "detected_objects": [],
                "image_array": np.zeros((8, 8)),
                "metadata": {},
            }

    det.image_processor = FakeProc()  # type: ignore[attr-defined]
    res = det.analyze_image("dummy.fits")
    assert calls.get("called") is True
    assert any(a["type"] == "gravitational" for a in res.anomalies)
