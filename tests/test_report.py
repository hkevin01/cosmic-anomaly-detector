from typing import Any

from cosmic_anomaly_detector.reporting.report import ReportGenerator


def test_report_generation(tmp_path: Any) -> None:
    run_dir = tmp_path
    results = [
        {
            "file": "example.fits",
            "anomalies": [
                {
                    "type": "gravitational",
                    "severity": 0.9,
                    "description": "Test",
                }
            ],
        }
    ]
    rep = ReportGenerator()
    path = rep.generate_markdown_report(run_dir, results, cfg=object())
    text = path.read_text(encoding="utf-8")
    assert "Cosmic Anomaly Detector Report" in text
    assert "example.fits" in text
    assert "gravitational" in text
    assert "gravitational" in text
