"""
Test Configuration for Cosmic Anomaly Detector

Provides test fixtures, utilities, and configuration for unit testing.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, Mock

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from cosmic_anomaly_detector.utils.config import SystemConfig
from cosmic_anomaly_detector.utils.logging import setup_logging


class TestDataGenerator:
    """Generates synthetic test data for cosmic anomaly detection"""
    
    @staticmethod
    def create_mock_jwst_fits(
        width: int = 1024,
        height: int = 1024,
        num_sources: int = 10,
        add_anomaly: bool = False
    ) -> fits.HDUList:
        """Create a mock JWST FITS file for testing"""
        
        # Create primary header
        header = fits.Header()
        header['TELESCOP'] = 'JWST'
        header['INSTRUME'] = 'NIRCAM'
        header['FILTER'] = 'F200W'
        header['EXPTIME'] = 1000.0
        header['DATE-OBS'] = '2024-01-01T12:00:00'
        
        # Add WCS information
        header['CRPIX1'] = width / 2
        header['CRPIX2'] = height / 2
        header['CRVAL1'] = 150.0  # RA in degrees
        header['CRVAL2'] = 2.0    # Dec in degrees
        header['CDELT1'] = -0.000027778  # Pixel scale in degrees
        header['CDELT2'] = 0.000027778
        header['CTYPE1'] = 'RA---TAN'
        header['CTYPE2'] = 'DEC--TAN'
        header['CUNIT1'] = 'deg'
        header['CUNIT2'] = 'deg'
        
        # Create image data with noise
        np.random.seed(42)  # For reproducible tests
        image_data = np.random.normal(100, 10, (height, width)).astype(np.float32)
        
        # Add point sources
        for i in range(num_sources):
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            flux = np.random.uniform(500, 2000)
            
            # Create Gaussian PSF
            y_indices, x_indices = np.ogrid[:height, :width]
            sigma = 2.0
            source = flux * np.exp(-((x_indices - x)**2 + (y_indices - y)**2) / (2 * sigma**2))
            image_data += source
            
        # Add anomalous structure if requested
        if add_anomaly:
            # Create a perfectly circular structure (unnatural)
            center_x, center_y = width // 2, height // 2
            radius = 30
            y_indices, x_indices = np.ogrid[:height, :width]
            mask = ((x_indices - center_x)**2 + (y_indices - center_y)**2) <= radius**2
            image_data[mask] += 1000
            
        # Create HDU list
        primary_hdu = fits.PrimaryHDU(data=image_data, header=header)
        hdul = fits.HDUList([primary_hdu])
        
        return hdul
    
    @staticmethod
    def create_test_coordinates() -> SkyCoord:
        """Create test astronomical coordinates"""
        return SkyCoord(
            ra=150.0 * u.degree,
            dec=2.0 * u.degree,
            frame='icrs'
        )
    
    @staticmethod
    def create_mock_detection_result() -> Dict[str, Any]:
        """Create a mock detection result for testing"""
        return {
            'detection_id': 'TEST_001',
            'coordinates': {
                'ra': 150.0,
                'dec': 2.0,
                'pixel_x': 512,
                'pixel_y': 512
            },
            'confidence_scores': {
                'visual_anomaly': 0.85,
                'gravitational_anomaly': 0.75,
                'artificial_structure': 0.90
            },
            'properties': {
                'size_pixels': 100,
                'brightness': 1500.0,
                'shape_regularity': 0.95,
                'spectral_features': ['emission_line_1', 'absorption_feature_2']
            },
            'metadata': {
                'processing_time': 45.2,
                'image_quality': 'good',
                'instrument': 'NIRCAM',
                'filter': 'F200W'
            }
        }


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests"""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_directory: Path) -> SystemConfig:
    """Provide a test configuration"""
    config = SystemConfig()
    config.data_root_path = str(temp_directory / "data")
    config.output_path = str(temp_directory / "output")
    config.temp_path = str(temp_directory / "temp")
    config.log_level = "DEBUG"
    config.max_workers = 2
    config.batch_size = 4
    config.reproducible_random_seed = 42
    return config


@pytest.fixture
def test_logging_setup(temp_directory: Path) -> None:
    """Setup test logging configuration"""
    log_dir = temp_directory / "logs"
    setup_logging(
        log_level="DEBUG",
        log_dir=str(log_dir),
        console_output=False,  # Disable console output during tests
        file_output=True
    )


@pytest.fixture
def mock_jwst_fits(temp_directory: Path) -> Path:
    """Provide a mock JWST FITS file"""
    fits_path = temp_directory / "test_image.fits"
    hdul = TestDataGenerator.create_mock_jwst_fits(add_anomaly=True)
    hdul.writeto(fits_path)
    return fits_path


@pytest.fixture
def mock_jwst_fits_no_anomaly(temp_directory: Path) -> Path:
    """Provide a mock JWST FITS file without anomalies"""
    fits_path = temp_directory / "test_image_normal.fits"
    hdul = TestDataGenerator.create_mock_jwst_fits(add_anomaly=False)
    hdul.writeto(fits_path)
    return fits_path


@pytest.fixture
def mock_image_processor():
    """Provide a mock image processor for testing"""
    processor = Mock()
    processor.process_image.return_value = Mock(
        image_array=np.random.random((1024, 1024)),
        detected_objects=[{'x': 512, 'y': 512, 'confidence': 0.9}],
        metadata={'processing_time': 10.5},
        processing_steps=['noise_reduction', 'object_detection']
    )
    return processor


@pytest.fixture
def mock_gravitational_analyzer():
    """Provide a mock gravitational analyzer for testing"""
    analyzer = Mock()
    analyzer.analyze_physics.return_value = {
        'gravitational_anomaly_score': 0.75,
        'kepler_compliance': 0.95,
        'mass_estimate': 1.5e30,  # Solar masses
        'orbital_anomalies': [],
        'lensing_detected': False
    }
    return analyzer


@pytest.fixture
def mock_classifier():
    """Provide a mock classifier for testing"""
    classifier = Mock()
    classifier.classify.return_value = {
        'artificial_probability': 0.85,
        'confidence': 0.90,
        'features': {
            'geometric_regularity': 0.95,
            'spectral_anomaly': 0.70,
            'size_consistency': 0.88
        }
    }
    return classifier


@pytest.fixture
def sample_detection_results() -> list:
    """Provide sample detection results for testing"""
    return [
        TestDataGenerator.create_mock_detection_result(),
        {
            **TestDataGenerator.create_mock_detection_result(),
            'detection_id': 'TEST_002',
            'confidence_scores': {
                'visual_anomaly': 0.45,
                'gravitational_anomaly': 0.30,
                'artificial_structure': 0.40
            }
        }
    ]


class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_detection_valid(detection: Dict[str, Any]) -> None:
        """Assert that a detection result has the expected structure"""
        required_keys = [
            'detection_id', 'coordinates', 'confidence_scores', 
            'properties', 'metadata'
        ]
        for key in required_keys:
            assert key in detection, f"Missing required key: {key}"
        
        # Check coordinate structure
        coord_keys = ['ra', 'dec', 'pixel_x', 'pixel_y']
        for key in coord_keys:
            assert key in detection['coordinates'], f"Missing coordinate: {key}"
        
        # Check confidence scores are in valid range
        for score in detection['confidence_scores'].values():
            assert 0.0 <= score <= 1.0, f"Invalid confidence score: {score}"
    
    @staticmethod
    def create_test_pipeline_config() -> Dict[str, Any]:
        """Create a test pipeline configuration"""
        return {
            'image_processing': {
                'noise_reduction_enabled': True,
                'object_detection_threshold': 0.5
            },
            'gravitational_analysis': {
                'strict_validation': False,
                'tolerance': 0.1
            },
            'classification': {
                'model_type': 'test',
                'confidence_threshold': 0.8
            }
        }
    
    @staticmethod
    def mock_fits_header() -> fits.Header:
        """Create a mock FITS header for testing"""
        header = fits.Header()
        header['TELESCOP'] = 'JWST'
        header['INSTRUME'] = 'NIRCAM'
        header['FILTER'] = 'F200W'
        header['EXPTIME'] = 1000.0
        header['DATE-OBS'] = '2024-01-01T12:00:00'
        return header


# Custom assertions for scientific data
def assert_coordinates_valid(coords: Dict[str, float]) -> None:
    """Assert that coordinates are within valid ranges"""
    assert -180 <= coords['ra'] <= 360, f"Invalid RA: {coords['ra']}"
    assert -90 <= coords['dec'] <= 90, f"Invalid Dec: {coords['dec']}"
    assert coords['pixel_x'] >= 0, f"Invalid pixel_x: {coords['pixel_x']}"
    assert coords['pixel_y'] >= 0, f"Invalid pixel_y: {coords['pixel_y']}"


def assert_confidence_scores_valid(scores: Dict[str, float]) -> None:
    """Assert that confidence scores are in valid range"""
    for name, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Invalid confidence score {name}: {score}"


def assert_scientific_metadata_present(metadata: Dict[str, Any]) -> None:
    """Assert that required scientific metadata is present"""
    required_fields = ['instrument', 'filter', 'processing_time']
    for field in required_fields:
        assert field in metadata, f"Missing metadata field: {field}"
