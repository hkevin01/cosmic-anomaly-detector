"""
Tests for BatchPipeline and BatchConfig — Cosmic Anomaly Detector
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fits(path: str, shape=(64, 64)) -> None:
    """Create a minimal FITS file at *path*."""
    from astropy.io import fits
    data = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    data[30:35, 30:35] += 10.0
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data, name='SCI')])
    hdul[0].header['INSTRUME'] = 'MIRI'
    hdul[0].header['FILTER'] = 'F770W'
    hdul[0].header['TARGNAME'] = 'TEST'
    hdul[0].header['EFFEXPTM'] = 500.0
    hdul.writeto(path, overwrite=True)


# ---------------------------------------------------------------------------
# BatchConfig
# ---------------------------------------------------------------------------

class TestBatchConfig:
    def test_default_n_workers(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import BatchConfig
        cfg = BatchConfig()
        assert cfg.n_workers >= 1
        assert cfg.n_workers <= (os.cpu_count() or 4)

    def test_run_id_auto_generated(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import BatchConfig
        cfg = BatchConfig()
        assert cfg.run_id is not None
        assert len(cfg.run_id) > 4

    def test_two_instances_have_different_run_ids(self):
        import time
        from cosmic_anomaly_detector.processing.parallel_pipeline import BatchConfig
        a = BatchConfig()
        time.sleep(0.01)
        b = BatchConfig()
        assert a.run_id != b.run_id

    def test_custom_run_id_preserved(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import BatchConfig
        cfg = BatchConfig(run_id='my-run-123')
        assert cfg.run_id == 'my-run-123'

    def test_score_threshold_default(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import BatchConfig
        cfg = BatchConfig()
        assert 0.0 < cfg.score_threshold < 1.0


# ---------------------------------------------------------------------------
# FileResult / BatchResult data classes
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_file_result_defaults(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import FileResult
        fr = FileResult(file_path='/some/file.fits', status='ok')
        assert fr.n_sources == 0
        assert fr.candidates == []

    def test_batch_result_defaults(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import BatchResult
        br = BatchResult(run_id='x', total_files=10)
        assert br.n_ok == 0
        assert br.top_detections == []


# ---------------------------------------------------------------------------
# _load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_fits_returns_normalised_array(self, tmp_path):
        from cosmic_anomaly_detector.processing.parallel_pipeline import _load_image
        from cosmic_anomaly_detector.processing.gpu_accelerator import get_accelerator
        fits_path = str(tmp_path / 'test.fits')
        _make_fits(fits_path)
        acc = get_accelerator('cpu')
        img, meta = _load_image(fits_path, acc)
        assert img is not None
        assert img.ndim == 2
        assert float(img.min()) >= -1e-4
        assert float(img.max()) <= 1.0 + 1e-4

    def test_invalid_path_returns_none(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import _load_image
        from cosmic_anomaly_detector.processing.gpu_accelerator import get_accelerator
        acc = get_accelerator('cpu')
        img, meta = _load_image('/nonexistent/file.fits', acc)
        assert img is None

    def test_fits_metadata_populated(self, tmp_path):
        from cosmic_anomaly_detector.processing.parallel_pipeline import _load_image
        from cosmic_anomaly_detector.processing.gpu_accelerator import get_accelerator
        fits_path = str(tmp_path / 'test.fits')
        _make_fits(fits_path)
        acc = get_accelerator('cpu')
        _, meta = _load_image(fits_path, acc)
        assert 'shape' in meta


# ---------------------------------------------------------------------------
# _extract_sources
# ---------------------------------------------------------------------------

class TestExtractSources:
    def test_returns_list(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import _extract_sources
        img = np.zeros((128, 128), dtype=np.float32)
        img[64, 64] = 5.0
        sources = _extract_sources(img, sigma=3.0)
        assert isinstance(sources, list)

    def test_bright_source_detected(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import _extract_sources
        img = np.zeros((64, 64), dtype=np.float32)
        img[32, 32] = 20.0
        sources = _extract_sources(img, sigma=2.0)
        assert len(sources) >= 1

    def test_source_has_required_keys(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import _extract_sources
        img = np.zeros((64, 64), dtype=np.float32)
        img[32, 32] = 20.0
        sources = _extract_sources(img, sigma=2.0)
        if sources:
            assert 'coordinates' in sources[0]
            assert 'brightness' in sources[0]

    def test_flat_image_no_sources(self):
        from cosmic_anomaly_detector.processing.parallel_pipeline import _extract_sources
        img = np.ones((64, 64), dtype=np.float32)
        sources = _extract_sources(img, sigma=10.0)
        assert len(sources) == 0


# ---------------------------------------------------------------------------
# BatchPipeline.run
# ---------------------------------------------------------------------------

class TestBatchPipelineRun:
    def test_empty_list_returns_batch_result(self, tmp_path):
        from cosmic_anomaly_detector.processing.parallel_pipeline import (
            BatchConfig, BatchPipeline,
        )
        cfg = BatchConfig(output_dir=str(tmp_path / 'out'), n_workers=1, run_id='test-empty')
        pipeline = BatchPipeline(cfg)
        result = pipeline.run([])
        assert result.total_files == 0
        assert result.n_ok == 0

    def test_single_valid_fits(self, tmp_path):
        from cosmic_anomaly_detector.processing.parallel_pipeline import (
            BatchConfig, BatchPipeline,
        )
        fits_path = str(tmp_path / 'good.fits')
        _make_fits(fits_path)
        cfg = BatchConfig(
            output_dir=str(tmp_path / 'out'),
            n_workers=1,
            run_id='test-single',
            score_threshold=0.0,   # accept all candidates
        )
        pipeline = BatchPipeline(cfg)
        result = pipeline.run([fits_path])
        assert result.total_files == 1
        assert result.n_ok == 1
        assert result.n_errors == 0

    def test_invalid_path_produces_error_not_exception(self, tmp_path):
        from cosmic_anomaly_detector.processing.parallel_pipeline import (
            BatchConfig, BatchPipeline,
        )
        cfg = BatchConfig(
            output_dir=str(tmp_path / 'out'),
            n_workers=1,
            run_id='test-error',
        )
        pipeline = BatchPipeline(cfg)
        result = pipeline.run(['/no/such/file.fits'])
        assert result.total_files == 1
        # Should be 'skipped' or 'error', never raises
        assert result.file_results[0].status in ('error', 'skipped')

    def test_summary_json_written(self, tmp_path):
        from cosmic_anomaly_detector.processing.parallel_pipeline import (
            BatchConfig, BatchPipeline,
        )
        fits_path = str(tmp_path / 'img.fits')
        _make_fits(fits_path)
        cfg = BatchConfig(
            output_dir=str(tmp_path / 'out'),
            n_workers=1,
            run_id='test-summary',
        )
        pipeline = BatchPipeline(cfg)
        result = pipeline.run([fits_path])
        summary_path = Path(tmp_path) / 'out' / 'test-summary' / 'summary.json'
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert 'run_id' in data
        assert 'total_files' in data

    def test_results_jsonl_written(self, tmp_path):
        from cosmic_anomaly_detector.processing.parallel_pipeline import (
            BatchConfig, BatchPipeline,
        )
        fits_path = str(tmp_path / 'img2.fits')
        _make_fits(fits_path)
        cfg = BatchConfig(
            output_dir=str(tmp_path / 'out'),
            n_workers=1,
            run_id='test-jsonl',
        )
        pipeline = BatchPipeline(cfg)
        result = pipeline.run([fits_path])
        jsonl_path = Path(tmp_path) / 'out' / 'test-jsonl' / 'results.jsonl'
        assert jsonl_path.exists()
        lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert 'file_path' in entry
        assert 'status' in entry
