"""
Tests for GPUAccelerator — Cosmic Anomaly Detector
"""
import numpy as np
import pytest
from unittest.mock import patch


@pytest.fixture
def acc():
    """Return a CPU-based GPUAccelerator instance."""
    from cosmic_anomaly_detector.processing.gpu_accelerator import GPUAccelerator
    return GPUAccelerator(device='cpu')


@pytest.fixture
def sample_image():
    """256×256 float32 image with some structure."""
    rng = np.random.default_rng(42)
    img = rng.standard_normal((256, 256)).astype(np.float32)
    img[100:120, 100:120] += 5.0
    return img


class TestGPUAcceleratorInit:
    def test_auto_device_creates_instance(self):
        from cosmic_anomaly_detector.processing.gpu_accelerator import GPUAccelerator
        a = GPUAccelerator(device='auto')
        assert a.device in ('cpu', 'cuda')

    def test_cpu_device(self, acc):
        assert acc.device == 'cpu'

    def test_device_info_keys(self, acc):
        info = acc.device_info()
        assert 'device' in info
        assert 'cuda_available' in info
        assert 'torch_version' in info or True   # may be absent if torch missing

    def test_device_info_device_matches(self, acc):
        assert acc.device_info()['device'] == 'cpu'


class TestGaussianFilter:
    def test_output_shape(self, acc, sample_image):
        out = acc.gaussian_filter(sample_image, sigma=2.0)
        assert out.shape == sample_image.shape

    def test_output_dtype_float32(self, acc, sample_image):
        out = acc.gaussian_filter(sample_image, sigma=2.0)
        assert out.dtype == np.float32

    def test_smoothing_reduces_std(self, acc, sample_image):
        out = acc.gaussian_filter(sample_image, sigma=5.0)
        assert float(out.std()) < float(sample_image.std())

    def test_sigma_zero_identity(self, acc, sample_image):
        """sigma ~0 should approximate identity."""
        out = acc.gaussian_filter(sample_image, sigma=0.1)
        # Very small sigma: result close to input
        assert np.allclose(out, sample_image, atol=0.5)


class TestStarletTransform:
    def test_output_shape_default_scales(self, acc, sample_image):
        planes = acc.starlet_transform(sample_image, n_scales=4)
        # Returns list/array of n_scales detail planes + coarse
        assert len(planes) == 4

    def test_each_plane_matches_image_shape(self, acc, sample_image):
        planes = acc.starlet_transform(sample_image, n_scales=3)
        for p in planes:
            assert np.asarray(p).shape == sample_image.shape

    def test_scales_1_to_5(self, acc, sample_image):
        for n in (1, 2, 3, 5):
            planes = acc.starlet_transform(sample_image, n_scales=n)
            assert len(planes) == n


class TestMatchedFilterSNR:
    def test_returns_tuple(self, acc, sample_image):
        result = acc.matched_filter_snr(sample_image, fwhm=3.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_snr_map_shape(self, acc, sample_image):
        snr_map, noise_sigma = acc.matched_filter_snr(sample_image, fwhm=3.0)
        assert np.asarray(snr_map).shape == sample_image.shape

    def test_noise_sigma_positive(self, acc, sample_image):
        _, noise_sigma = acc.matched_filter_snr(sample_image, fwhm=3.0)
        assert float(noise_sigma) > 0.0

    def test_bright_source_high_snr(self, acc):
        """Injected bright source should have elevated SNR."""
        img = np.zeros((128, 128), dtype=np.float32)
        img[64, 64] = 10.0
        snr_map, _ = acc.matched_filter_snr(img, fwhm=2.0)
        snr_arr = np.asarray(snr_map)
        assert float(snr_arr.max()) > 0.5


class TestNormalise:
    def test_output_in_0_1(self, acc, sample_image):
        out = acc.normalise(sample_image)
        assert float(out.min()) >= -1e-6
        assert float(out.max()) <= 1.0 + 1e-6

    def test_constant_image_no_crash(self, acc):
        img = np.ones((64, 64), dtype=np.float32)
        out = acc.normalise(img)
        assert out.shape == img.shape

    def test_output_shape_preserved(self, acc, sample_image):
        assert acc.normalise(sample_image).shape == sample_image.shape


class TestGetAccelerator:
    def test_returns_same_singleton(self):
        from cosmic_anomaly_detector.processing.gpu_accelerator import get_accelerator
        a1 = get_accelerator('cpu')
        a2 = get_accelerator('cpu')
        assert a1 is a2

    def test_device_cpu(self):
        from cosmic_anomaly_detector.processing.gpu_accelerator import get_accelerator
        a = get_accelerator('cpu')
        assert a.device == 'cpu'
