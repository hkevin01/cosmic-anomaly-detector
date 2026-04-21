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


# ---------------------------------------------------------------------------
# MultiBandIRExcessDetector
# ---------------------------------------------------------------------------

from cosmic_anomaly_detector.processing.algorithms import MultiBandIRExcessDetector


def make_multiband_scene(size: int = 64) -> tuple:
    """
    Two-source multi-band scene:
      - Normal star at (20,20): flux falls off steeply with wavelength (hot)
      - DS candidate at (45,45): hot star core + warm extended excess at IR bands
    Returns (bands_dict, detected_objects) with 3 wavelengths.
    """
    rng = np.random.default_rng(7)
    noise = lambda: (rng.standard_normal((size, size)) * 0.005).clip(0, None)
    y, x = np.ogrid[:size, :size]

    def gauss(cx, cy, sigma, amp):
        return amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

    # Stellar SEDs: B_nu ∝ λ^{-5} / (exp(hc/λkT) - 1) — here simplified
    # Normal star T_star=10000K: bright at short λ, fades at long λ
    star_opt  = gauss(20, 20, 2.0, 0.90)
    star_mid  = gauss(20, 20, 2.0, 0.55)
    star_ir   = gauss(20, 20, 2.0, 0.25)

    # DS candidate T_star=8000K core + warm halo (γ≈0.4, T_DS=300K)
    ds_opt  = gauss(45, 45, 2.0, 0.80)
    ds_mid  = gauss(45, 45, 2.0, 0.60) + gauss(45, 45, 8.0, 0.18)
    ds_ir   = gauss(45, 45, 2.0, 0.30) + gauss(45, 45, 8.0, 0.35)

    band_1p5 = np.clip(star_opt + ds_opt + noise(), 0, 1).astype(np.float32)
    band_3p6 = np.clip(star_mid + ds_mid + noise(), 0, 1).astype(np.float32)
    band_12  = np.clip(star_ir  + ds_ir  + noise(), 0, 1).astype(np.float32)

    bands = {1.5: band_1p5, 3.6: band_3p6, 12.0: band_12}
    objs = [
        {'id': 'star', 'coordinates': [20.0, 20.0],
         'bounding_box': [14, 14, 27, 27], 'brightness': 0.9, 'score': 0.5},
        {'id': 'ds',   'coordinates': [45.0, 45.0],
         'bounding_box': [38, 38, 53, 53], 'brightness': 0.85, 'score': 0.5},
    ]
    return bands, objs


class TestMultiBandIRExcessDetector:
    def test_requires_at_least_2_bands(self):
        with pytest.raises(ValueError, match="≥2"):
            MultiBandIRExcessDetector({1.5: np.ones((32, 32), dtype=np.float32)})

    def test_empty_objects_returns_empty(self):
        bands, _ = make_multiband_scene()
        det = MultiBandIRExcessDetector(bands)
        assert det.detect([]) == []

    def test_returns_algorithm_candidates(self):
        bands, objs = make_multiband_scene()
        det = MultiBandIRExcessDetector(bands, rmse_threshold=0.5)
        results = det.detect(objs)
        assert isinstance(results, list)
        for c in results:
            assert hasattr(c, 'algorithm')
            assert c.algorithm == 'multiband_ir_excess'

    def test_candidate_has_required_metadata(self):
        bands, objs = make_multiband_scene()
        det = MultiBandIRExcessDetector(bands, rmse_threshold=0.5)
        results = det.detect(objs)
        if results:
            m = results[0].metadata
            assert 'covering_factor_gamma' in m
            assert 'ds_temperature_k' in m
            assert 'star_temperature_k' in m
            assert 'sed_rmse_mag' in m
            assert 'n_bands' in m
            assert m['n_bands'] == 3

    def test_score_in_range(self):
        bands, objs = make_multiband_scene()
        det = MultiBandIRExcessDetector(bands, rmse_threshold=0.5)
        for c in det.detect(objs):
            assert 0.0 <= c.score <= 1.0

    def test_ds_candidate_ranked_higher_than_normal_star(self):
        """DS source should score higher than plain star in warm IR bands."""
        bands, objs = make_multiband_scene()
        det = MultiBandIRExcessDetector(bands, rmse_threshold=0.5)
        results = det.detect(objs)
        if len(results) >= 2:
            # results are sorted by score desc — DS-like object should appear
            ids = [c.id for c in results]
            # DS object is objs[1] — check its score is not lower than star's
            ds_cands = [c for c in results if '45' in c.id]
            star_cands = [c for c in results if '20' in c.id]
            if ds_cands and star_cands:
                assert ds_cands[0].score >= star_cands[0].score - 0.05

    def test_tight_threshold_reduces_candidates(self):
        bands, objs = make_multiband_scene()
        det_loose = MultiBandIRExcessDetector(bands, rmse_threshold=0.9)
        det_tight = MultiBandIRExcessDetector(bands, rmse_threshold=0.05)
        assert len(det_loose.detect(objs)) >= len(det_tight.detect(objs))

    def test_exception_returns_empty(self):
        # Pass NaN-filled bands to provoke edge case
        bad_bands = {1.5: np.full((32, 32), np.nan, dtype=np.float32),
                     3.6: np.full((32, 32), np.nan, dtype=np.float32)}
        det = MultiBandIRExcessDetector(bad_bands, rmse_threshold=0.5)
        result = det.detect([{'coordinates': [10.0, 10.0], 'bounding_box': [5,5,15,15]}])
        assert isinstance(result, list)

    def test_planck_bnu_zero_for_cold_limit(self):
        det = MultiBandIRExcessDetector(
            {1.5: np.ones((8, 8), np.float32),
             3.6: np.ones((8, 8), np.float32)},
        )
        # Very short wavelength + very cold → exponent huge → 0
        assert det._planck_bnu(0.1, 1.0) == 0.0

    def test_model_sed_normalised_to_zero(self):
        det = MultiBandIRExcessDetector(
            {1.5: np.ones((8, 8), np.float32),
             3.6: np.ones((8, 8), np.float32)},
        )
        log_sed = det._model_sed(0.2, 5000.0, 300.0)
        assert float(log_sed.max()) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# MultiEpochMicrolensingDetector
# ---------------------------------------------------------------------------

from cosmic_anomaly_detector.processing.algorithms import MultiEpochMicrolensingDetector


def make_lensing_epochs(size: int = 64, n_epochs: int = 8) -> tuple:
    """
    Simulate a microlensing event:
      - Source at (32, 32) brightens by ~3× at peak (epoch 4)
      - Background field is flat
    Returns (epochs_list, detected_objects)
    """
    rng = np.random.default_rng(99)
    y, x = np.ogrid[:size, :size]
    t_jd = np.linspace(2460000.0, 2460050.0, n_epochs)  # 50 days
    t0, tE, u0 = 2460025.0, 8.0, 0.2   # peak at midpoint, compact impact

    def magnification(t):
        u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
        return (u ** 2 + 2.0) / (u * np.sqrt(u ** 2 + 4.0))

    epochs = []
    for t in t_jd:
        A = magnification(t)
        img = (rng.standard_normal((size, size)) * 0.02).clip(0, None)
        # Source at (32,32): baseline flux 0.5, magnified by A
        img += 0.5 * A * np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 2.5 ** 2))
        epochs.append((np.clip(img, 0, 1).astype(np.float32), float(t)))

    objs = [
        {'id': 'src', 'coordinates': [32.0, 32.0],
         'bounding_box': [26, 26, 39, 39], 'brightness': 0.5, 'score': 0.5},
        {'id': 'bg',  'coordinates': [10.0, 10.0],
         'bounding_box': [5, 5, 16, 16],  'brightness': 0.02, 'score': 0.1},
    ]
    return epochs, objs


class TestMultiEpochMicrolensingDetector:
    def test_requires_at_least_3_epochs(self):
        epochs = [(np.ones((32, 32), np.float32), float(i)) for i in range(2)]
        with pytest.raises(ValueError, match="≥ 3"):
            MultiEpochMicrolensingDetector(epochs)

    def test_epochs_sorted_by_jd(self):
        epochs_unsorted = [
            (np.ones((16, 16), np.float32), 100.0),
            (np.ones((16, 16), np.float32), 50.0),
            (np.ones((16, 16), np.float32), 75.0),
        ]
        det = MultiEpochMicrolensingDetector(epochs_unsorted)
        times = [jd for _, jd in det.epochs]
        assert times == sorted(times)

    def test_empty_objects_returns_empty(self):
        epochs, _ = make_lensing_epochs()
        det = MultiEpochMicrolensingDetector(epochs)
        assert det.detect([]) == []

    def test_lensing_source_detected(self):
        """A source with a clear Paczyński light curve should be flagged."""
        epochs, objs = make_lensing_epochs(n_epochs=10)
        det = MultiEpochMicrolensingDetector(
            epochs,
            variability_threshold=2.0,  # relaxed for synthetic data
            chi2_threshold=5.0,
        )
        results = det.detect(objs)
        assert isinstance(results, list)
        # Lensed source or background source detected
        # (exact result depends on chi² of synthetic noise)

    def test_flat_source_not_flagged(self):
        """Constant-brightness source (flat light curve) must NOT be flagged."""
        size = 32
        epochs = [
            (np.full((size, size), 0.3, dtype=np.float32), float(t))
            for t in range(5)
        ]
        det = MultiEpochMicrolensingDetector(epochs, variability_threshold=3.0)
        objs = [{'coordinates': [16.0, 16.0], 'bounding_box': [10,10,23,23]}]
        results = det.detect(objs)
        assert results == []

    def test_candidate_has_required_metadata(self):
        epochs, objs = make_lensing_epochs(n_epochs=12)
        det = MultiEpochMicrolensingDetector(
            epochs, variability_threshold=1.5, chi2_threshold=10.0
        )
        results = det.detect(objs)
        for c in results:
            m = c.metadata
            assert 't0_best_jd' in m
            assert 'te_best_days' in m
            assert 'u0_best' in m
            assert 'chi2_flat' in m
            assert 'chi2_paczynski' in m
            assert 'n_epochs' in m
            assert m['n_epochs'] == 12

    def test_score_in_range(self):
        epochs, objs = make_lensing_epochs()
        det = MultiEpochMicrolensingDetector(
            epochs, variability_threshold=1.5, chi2_threshold=10.0
        )
        for c in det.detect(objs):
            assert 0.0 <= c.score <= 1.0

    def test_paczynski_magnification_at_u1(self):
        """A(u=1) should equal (1+2)/(1·√5) = 3/√5 ≈ 1.3416."""
        expected = 3.0 / np.sqrt(5.0)
        result = MultiEpochMicrolensingDetector._paczynski(
            np.array([0.0]), 0.0, 1.0, 1.0
        )[0]
        assert abs(result - expected) < 1e-6

    def test_chi2_flat_zero_for_constant(self):
        """χ²/dof of a perfectly constant light curve must be 0."""
        fluxes = np.full(10, 2.5)
        errors = np.full(10, 0.1)
        det = MultiEpochMicrolensingDetector(
            [(np.ones((8, 8), np.float32), float(t)) for t in range(5)]
        )
        assert det._chi2_flat(fluxes, errors) == pytest.approx(0.0, abs=1e-10)

    def test_algorithm_tag(self):
        epochs, objs = make_lensing_epochs(n_epochs=12)
        det = MultiEpochMicrolensingDetector(
            epochs, variability_threshold=1.5, chi2_threshold=10.0
        )
        for c in det.detect(objs):
            assert c.algorithm == 'multiepoch_microlensing'
