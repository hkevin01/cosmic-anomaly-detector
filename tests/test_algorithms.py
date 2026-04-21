"""
Tests for processing/algorithms.py — Cosmic Anomaly Detector

Covers all four detection algorithm classes and the run_all_algorithms()
convenience function.
"""

import numpy as np
import pytest

from cosmic_anomaly_detector.processing.algorithms import (
    AlgorithmCandidate,
    IRExcessDetector,
    MatchedFilterDetector,
    MicrolensingAnomalyDetector,
    WaveletSourceDetector,
    run_all_algorithms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blank(size: int = 128) -> np.ndarray:
    """Return a flat, near-zero image (noise floor only)."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((size, size)) * 0.01).clip(0, 1).astype(np.float32)


def make_star(size: int = 128, cx: int = 64, cy: int = 64,
              flux: float = 0.9, sigma: float = 2.5) -> np.ndarray:
    """Return a synthetic image with a single Gaussian point source."""
    img = make_blank(size)
    y, x = np.ogrid[:size, :size]
    img += flux * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_ds_scene(size: int = 128) -> tuple:
    """
    Return (image, detected_objects) with two sources:
      obj0 — compact normal star (no warm halo)
      obj1 — DS candidate (core + broad warm halo)
    """
    rng = np.random.default_rng(42)
    img = (rng.standard_normal((size, size)) * 0.02).clip(0, None).astype(np.float64)
    y, x = np.ogrid[:size, :size]

    # Normal star at (40, 40)
    img += 0.8 * np.exp(-((x - 40) ** 2 + (y - 40) ** 2) / (2 * 2 ** 2))

    # DS candidate at (90, 90)
    img += 0.9 * np.exp(-((x - 90) ** 2 + (y - 90) ** 2) / (2 * 2 ** 2))
    img += 0.35 * np.exp(-((x - 90) ** 2 + (y - 90) ** 2) / (2 * 15 ** 2))

    img = np.clip(img / img.max(), 0.0, 1.0).astype(np.float32)
    objs = [
        {'id': 'o0', 'coordinates': [40.0, 40.0],
         'bounding_box': [32, 32, 49, 49], 'brightness': 0.8, 'score': 0.5},
        {'id': 'o1', 'coordinates': [90.0, 90.0],
         'bounding_box': [70, 70, 111, 111], 'brightness': 0.9, 'score': 0.6},
    ]
    return img, objs


# ---------------------------------------------------------------------------
# AlgorithmCandidate
# ---------------------------------------------------------------------------

class TestAlgorithmCandidate:
    def test_to_dict_keys(self):
        cand = AlgorithmCandidate(
            id='test_01', algorithm='wavelet_starlet',
            anomaly_type='point_source', coordinates=[10.0, 20.0],
            bounding_box=[5, 15, 16, 26], brightness=0.7, score=0.5,
        )
        d = cand.to_dict()
        for key in ('id', 'algorithm', 'anomaly_type', 'coordinates',
                    'bounding_box', 'brightness', 'score',
                    'estimated_mass', 'luminosity'):
            assert key in d, f"Missing key: {key}"

    def test_score_bounds(self):
        cand = AlgorithmCandidate(
            id='t', algorithm='x', anomaly_type='y',
            coordinates=[0.0, 0.0], bounding_box=[0, 0, 1, 1],
            brightness=0.5, score=0.9,
        )
        d = cand.to_dict()
        assert 0.0 <= d['score'] <= 1.0


# ---------------------------------------------------------------------------
# WaveletSourceDetector
# ---------------------------------------------------------------------------

class TestWaveletSourceDetector:
    def test_returns_list(self):
        det = WaveletSourceDetector()
        result = det.detect(make_blank())
        assert isinstance(result, list)

    def test_detects_bright_star(self):
        img = make_star(flux=0.9)
        det = WaveletSourceDetector(n_scales=3, sigma_threshold=3.0)
        candidates = det.detect(img)
        assert len(candidates) >= 1

    def test_candidate_fields(self):
        img = make_star(flux=0.9)
        det = WaveletSourceDetector(n_scales=3, sigma_threshold=3.0)
        candidates = det.detect(img)
        for c in candidates:
            assert 0.0 <= c.score <= 1.0
            assert len(c.coordinates) == 2
            assert len(c.bounding_box) == 4

    def test_empty_image_returns_empty(self):
        img = np.zeros((64, 64), dtype=np.float32)
        det = WaveletSourceDetector()
        assert det.detect(img) == []

    def test_no_false_detections_on_uniform(self):
        """Uniform image should produce no detections above threshold."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        det = WaveletSourceDetector(sigma_threshold=3.0)
        assert det.detect(img) == []

    def test_sorted_by_score_descending(self):
        img = make_star(size=200, cx=100, cy=100, flux=0.95)
        det = WaveletSourceDetector(n_scales=4, sigma_threshold=3.0)
        candidates = det.detect(img)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_algorithm_tag(self):
        img = make_star(flux=0.9)
        det = WaveletSourceDetector(n_scales=3, sigma_threshold=3.0)
        candidates = det.detect(img)
        if candidates:
            assert candidates[0].algorithm == 'wavelet_starlet'


# ---------------------------------------------------------------------------
# MatchedFilterDetector
# ---------------------------------------------------------------------------

class TestMatchedFilterDetector:
    def test_returns_list(self):
        det = MatchedFilterDetector()
        result = det.detect(make_blank())
        assert isinstance(result, list)

    def test_detects_bright_star(self):
        img = make_star(flux=0.95, sigma=2.5)
        det = MatchedFilterDetector(detection_threshold=4.0)
        candidates = det.detect(img)
        assert len(candidates) >= 1

    def test_candidate_schema(self):
        img = make_star(flux=0.95)
        det = MatchedFilterDetector(detection_threshold=4.0)
        candidates = det.detect(img)
        for c in candidates:
            assert 0.0 <= c.score <= 1.0
            assert 'snr' in c.metadata
            assert 'fwhm_pixels' in c.metadata

    def test_algorithm_tag(self):
        img = make_star(flux=0.95)
        det = MatchedFilterDetector(detection_threshold=4.0)
        candidates = det.detect(img)
        if candidates:
            assert candidates[0].algorithm == 'matched_filter'

    def test_empty_image(self):
        img = np.zeros((64, 64), dtype=np.float32)
        det = MatchedFilterDetector()
        assert det.detect(img) == []


# ---------------------------------------------------------------------------
# IRExcessDetector
# ---------------------------------------------------------------------------

class TestIRExcessDetector:
    def test_returns_list(self):
        img, objs = make_ds_scene()
        det = IRExcessDetector()
        result = det.detect(img, objs)
        assert isinstance(result, list)

    def test_flags_ds_candidate(self):
        img, objs = make_ds_scene()
        det = IRExcessDetector(rmse_threshold=0.20, min_gamma=0.05)
        candidates = det.detect(img, objs)
        # At least the DS candidate (obj1) should be flagged
        assert len(candidates) >= 1

    def test_ds_candidate_higher_score(self):
        """DS source (obj1) should have higher score than normal star (obj0)."""
        img, objs = make_ds_scene()
        det = IRExcessDetector(rmse_threshold=0.20, min_gamma=0.05)
        candidates = det.detect(img, objs)
        if len(candidates) >= 2:
            # Find the one corresponding to DS source
            ds_cands = [c for c in candidates
                        if abs(c.coordinates[0] - 90) < 5]
            star_cands = [c for c in candidates
                          if abs(c.coordinates[0] - 40) < 5]
            if ds_cands and star_cands:
                assert ds_cands[0].score >= star_cands[0].score

    def test_metadata_fields(self):
        img, objs = make_ds_scene()
        det = IRExcessDetector(rmse_threshold=0.20, min_gamma=0.05)
        candidates = det.detect(img, objs)
        for c in candidates:
            assert 'covering_factor_gamma' in c.metadata
            assert 'ds_temperature_k' in c.metadata
            assert 'sed_rmse' in c.metadata
            assert 0.0 < c.metadata['covering_factor_gamma'] <= 1.0
            assert c.metadata['ds_temperature_k'] >= 100.0

    def test_no_objects_returns_empty(self):
        img, _ = make_ds_scene()
        det = IRExcessDetector()
        assert det.detect(img, []) == []

    def test_anomaly_type(self):
        img, objs = make_ds_scene()
        det = IRExcessDetector(rmse_threshold=0.20, min_gamma=0.05)
        candidates = det.detect(img, objs)
        for c in candidates:
            assert c.anomaly_type == 'ir_excess_dyson_sphere'

    def test_planck_ratio_physical(self):
        """Verify Planck ratio is higher at shorter wavelength for hot stars."""
        det = IRExcessDetector()
        hot = det._planck_ratio(2.2, 5778.0)
        ir = det._planck_ratio(12.0, 5778.0)
        assert hot > ir, "Hot star emits more at 2.2µm than 12µm"

    def test_planck_ratio_zero_for_very_cold(self):
        """Very cold object barely emits at optical wavelengths."""
        det = IRExcessDetector()
        val = det._planck_ratio(0.5, 10.0)
        assert val < 1e-100 or val == 0.0


# ---------------------------------------------------------------------------
# MicrolensingAnomalyDetector
# ---------------------------------------------------------------------------

class TestMicrolensingAnomalyDetector:
    def test_returns_list(self):
        img, objs = make_ds_scene()
        det = MicrolensingAnomalyDetector()
        result = det.detect(img, objs)
        assert isinstance(result, list)

    def test_no_objects_returns_empty(self):
        img = make_star()
        det = MicrolensingAnomalyDetector()
        assert det.detect(img, []) == []

    def test_paczynski_at_u1(self):
        """A(u=1) = 3/√5 ≈ 1.3416."""
        det = MicrolensingAnomalyDetector()
        a = det._paczynski_magnification(1.0)
        assert abs(a - (3.0 / np.sqrt(5.0))) < 1e-9

    def test_paczynski_decreases_with_u(self):
        det = MicrolensingAnomalyDetector()
        a1 = det._paczynski_magnification(0.5)
        a2 = det._paczynski_magnification(2.0)
        assert a1 > a2

    def test_metadata_fields_if_flagged(self):
        img, objs = make_ds_scene()
        det = MicrolensingAnomalyDetector(magnification_threshold=1.05)
        candidates = det.detect(img, objs)
        for c in candidates:
            assert 'a_observed' in c.metadata
            assert 'a_expected_paczynski' in c.metadata
            assert 'anomaly_ratio' in c.metadata

    def test_score_bounds(self):
        img, objs = make_ds_scene()
        det = MicrolensingAnomalyDetector(magnification_threshold=1.05)
        candidates = det.detect(img, objs)
        for c in candidates:
            assert 0.0 <= c.score <= 1.0


# ---------------------------------------------------------------------------
# run_all_algorithms
# ---------------------------------------------------------------------------

class TestRunAllAlgorithms:
    def test_returns_list_of_dicts(self):
        img = make_star(flux=0.9)
        result = run_all_algorithms(img)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_schema_keys(self):
        img = make_star(flux=0.9)
        result = run_all_algorithms(img)
        required = {'id', 'algorithm', 'anomaly_type', 'coordinates',
                    'bounding_box', 'brightness', 'score'}
        for item in result:
            assert required.issubset(item.keys())

    def test_accepts_3d_rgb_image(self):
        """run_all_algorithms must handle (H, W, 3) colour arrays."""
        img_rgb = np.random.rand(64, 64, 3).astype(np.float32)
        result = run_all_algorithms(img_rgb)
        assert isinstance(result, list)

    def test_accepts_predetected_objects(self):
        img, objs = make_ds_scene()
        result = run_all_algorithms(img, detected_objects=objs)
        assert isinstance(result, list)

    def test_algorithm_tags_present(self):
        img = make_star(flux=0.95)
        result = run_all_algorithms(img)
        tags = {r['algorithm'] for r in result}
        # At least the spatial detectors should produce hits on a bright star
        assert 'wavelet_starlet' in tags or 'matched_filter' in tags
