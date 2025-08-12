import numpy as np

from cosmic_anomaly_detector.core.baseline import BaselineAnomalyScorer


def test_baseline_scorer_detects_injected_point():
    img = np.random.normal(0, 1, (64, 64)).astype(float)
    img[32, 32] += 20  # strong anomaly
    scorer = BaselineAnomalyScorer(sigma=3.0)
    res = scorer.score(img)
    total_val = res.get("total_candidates", 0)
    if not isinstance(total_val, int):  # fallback if unexpected type
        total_val = 0
    assert total_val >= 1
    coords = {(c["y"], c["x"]) for c in res["candidates"]}
    assert (32, 32) in coords
