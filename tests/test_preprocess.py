from cosmic_anomaly_detector.processing import preprocess as pp


def test_full_preprocess_synthetic(tmp_path):
    fake_path = tmp_path / "synthetic.fits"
    fake_path.write_text("")
    data = pp.full_preprocess(str(fake_path))
    assert "image" in data and data["image"].shape == (256, 256)
    assert "sources" in data and isinstance(data["sources"], list)
