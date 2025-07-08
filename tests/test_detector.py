"""
Unit Tests for Cosmic Anomaly Detector

Tests for the core detection functionality with comprehensive coverage.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cosmic_anomaly_detector.core.detector import AnomalyDetector, DetectionResult
from cosmic_anomaly_detector.utils.config import SystemConfig
from tests.conftest import (
    TestUtilities,
    assert_confidence_scores_valid,
    assert_coordinates_valid,
    assert_detection_valid,
)


class TestAnomalyDetector:
    """Test cases for AnomalyDetector"""

    def setup_method(self):
        """Set up test fixtures"""
        self.detector = AnomalyDetector()

    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector is not None
        assert self.detector.config is not None
        assert hasattr(self.detector, 'image_processor')
        assert hasattr(self.detector, 'gravitational_analyzer')
        assert hasattr(self.detector, 'classifier')

    def test_default_config(self):
        """Test default configuration"""
        config = self.detector._get_default_config()
        assert 'image_processing' in config
        assert 'gravity' in config
        assert 'classification' in config

    @patch('cosmic_anomaly_detector.core.detector.ImageProcessor')
    @patch('cosmic_anomaly_detector.core.detector.GravitationalAnalyzer')
    @patch('cosmic_anomaly_detector.core.detector.ArtificialStructureClassifier')
    def test_analyze_image(self, mock_classifier, mock_analyzer, mock_processor):
        """Test image analysis workflow"""
        # Mock the processors
        mock_processor.return_value.process.return_value = {
            'detected_objects': [{'id': 'test_obj'}],
            'metadata': {'test': 'data'}
        }
        mock_analyzer.return_value.analyze.return_value = {
            'anomalies': [{'type': 'gravitational', 'severity': 0.8}]
        }
        mock_classifier.return_value.classify.return_value = {
            'artificial_candidates': [{'classification': 'dyson_sphere', 'confidence': 0.9}]
        }

        # Create new detector with mocked components
        detector = AnomalyDetector()
        
        # Test analysis
        result = detector.analyze_image("test_image.fits")
        
        # Verify result structure
        assert isinstance(result, DetectionResult)
        assert hasattr(result, 'anomalies')
        assert hasattr(result, 'confidence_scores')
        assert hasattr(result, 'gravitational_analysis')
        assert hasattr(result, 'classification_results')

    def test_detection_result(self):
        """Test DetectionResult functionality"""
        anomalies = [{'type': 'test', 'data': 'example'}]
        confidence_scores = [0.8]
        
        result = DetectionResult(
            anomalies=anomalies,
            confidence_scores=confidence_scores,
            gravitational_analysis={},
            classification_results={},
            processing_metadata={}
        )
        
        assert result.has_anomalies()
        assert len(result.get_high_confidence_anomalies(0.7)) == 1
        assert len(result.get_high_confidence_anomalies(0.9)) == 0

    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        anomalies = [
            {'type': 'gravitational', 'severity': 0.8},
            {'type': 'artificial_structure', 'severity': 0.9},
            {'type': 'unknown', 'severity': 0.7}
        ]
        
        scores = self.detector._calculate_confidence_scores(anomalies)
        
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores)
        # Gravitational should be weighted higher
        assert scores[0] >= 0.8  # Gravitational anomaly
        assert scores[1] == 0.9  # Artificial structure (direct)

    def test_batch_analyze(self):
        """Test batch analysis functionality"""
        with patch.object(self.detector, 'analyze_image') as mock_analyze:
            mock_analyze.return_value = DetectionResult(
                anomalies=[],
                confidence_scores=[],
                gravitational_analysis={},
                classification_results={},
                processing_metadata={}
            )
            
            image_paths = ["image1.fits", "image2.fits", "image3.fits"]
            results = self.detector.batch_analyze(image_paths)
            
            assert len(results) == 3
            assert mock_analyze.call_count == 3

    def test_detection_statistics(self):
        """Test detection statistics calculation"""
        # Create mock results
        results = []
        for i in range(3):
            result = DetectionResult(
                anomalies=[{'id': f'anomaly_{i}'}] * (i + 1),
                confidence_scores=[0.8] * (i + 1),
                gravitational_analysis={},
                classification_results={},
                processing_metadata={}
            )
            results.append(result)
        
        stats = self.detector.get_detection_statistics(results)
        
        assert stats['total_images_analyzed'] == 3
        assert stats['total_anomalies_detected'] == 6  # 1 + 2 + 3
        assert stats['average_anomalies_per_image'] == 2.0


if __name__ == '__main__':
    pytest.main([__file__])
