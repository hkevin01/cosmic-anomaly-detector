"""
Main Anomaly Detector class for analyzing space images
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models.anomaly_models import AnomalyResult
from ..processing.image_processor import ImageProcessor
from .analyzer import GravitationalAnalyzer
from .classifier import ArtificialStructureClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Results from anomaly detection analysis"""
    anomalies: List[Dict]
    confidence_scores: List[float]
    gravitational_analysis: Dict
    classification_results: Dict
    processing_metadata: Dict

    def has_anomalies(self) -> bool:
        """Check if any anomalies were detected"""
        return len(self.anomalies) > 0

    def get_high_confidence_anomalies(self, threshold: float = 0.8) -> List[Dict]:
        """Return anomalies above confidence threshold"""
        return [
            anomaly for anomaly, score in zip(self.anomalies, self.confidence_scores)
            if score >= threshold
        ]


class AnomalyDetector:
    """
    Main class for detecting anomalous structures in space telescope images
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the anomaly detector
        
        Args:
            config: Configuration dictionary for detector settings
        """
        self.config = config or self._get_default_config()
        self.image_processor = ImageProcessor(self.config.get('image_processing', {}))
        self.gravitational_analyzer = GravitationalAnalyzer(self.config.get('gravity', {}))
        self.classifier = ArtificialStructureClassifier(self.config.get('classification', {}))
        
        logger.info("Cosmic Anomaly Detector initialized")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'image_processing': {
                'noise_reduction': True,
                'contrast_enhancement': True,
                'resolution_threshold': (512, 512)
            },
            'gravity': {
                'mass_estimation_method': 'luminosity',
                'orbit_analysis': True,
                'deviation_threshold': 0.1
            },
            'classification': {
                'model_type': 'ensemble',
                'confidence_threshold': 0.7,
                'use_pretrained': True
            }
        }

    def analyze_image(self, image_path: str) -> DetectionResult:
        """
        Analyze a space telescope image for anomalous structures
        
        Args:
            image_path: Path to the image file (FITS, PNG, JPG, etc.)
            
        Returns:
            DetectionResult containing analysis results
        """
        logger.info(f"Starting analysis of image: {image_path}")
        
        # Step 1: Process the image
        processed_data = self.image_processor.process(image_path)
        
        # Step 2: Analyze gravitational properties
        gravity_results = self.gravitational_analyzer.analyze(processed_data)
        
        # Step 3: Classify structures
        classification_results = self.classifier.classify(processed_data)
        
        # Step 4: Identify anomalies
        anomalies = self._identify_anomalies(
            processed_data, gravity_results, classification_results
        )
        
        # Step 5: Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(anomalies)
        
        result = DetectionResult(
            anomalies=anomalies,
            confidence_scores=confidence_scores,
            gravitational_analysis=gravity_results,
            classification_results=classification_results,
            processing_metadata=processed_data.get('metadata', {})
        )
        
        logger.info(f"Analysis complete. Found {len(anomalies)} potential anomalies")
        return result

    def _identify_anomalies(self, processed_data: Dict, gravity_results: Dict, 
                          classification_results: Dict) -> List[Dict]:
        """
        Identify anomalies based on gravitational and classification analysis
        """
        anomalies = []
        
        # Check for gravitational anomalies
        gravity_anomalies = gravity_results.get('anomalies', [])
        for anomaly in gravity_anomalies:
            anomalies.append({
                'type': 'gravitational',
                'description': anomaly.get('description', ''),
                'location': anomaly.get('coordinates', []),
                'severity': anomaly.get('deviation_magnitude', 0),
                'source': 'gravitational_analysis'
            })
        
        # Check for artificial structures
        artificial_structures = classification_results.get('artificial_candidates', [])
        for structure in artificial_structures:
            anomalies.append({
                'type': 'artificial_structure',
                'description': structure.get('classification', ''),
                'location': structure.get('bounding_box', []),
                'severity': structure.get('confidence', 0),
                'source': 'structure_classification'
            })
        
        return anomalies

    def _calculate_confidence_scores(self, anomalies: List[Dict]) -> List[float]:
        """
        Calculate confidence scores for detected anomalies
        """
        scores = []
        for anomaly in anomalies:
            base_score = anomaly.get('severity', 0)
            
            # Adjust based on anomaly type
            if anomaly['type'] == 'gravitational':
                # Higher weight for significant gravitational deviations
                score = min(base_score * 1.2, 1.0)
            elif anomaly['type'] == 'artificial_structure':
                # Use classification confidence directly
                score = base_score
            else:
                score = base_score * 0.8
            
            scores.append(max(0.0, min(1.0, score)))
        
        return scores

    def batch_analyze(self, image_paths: List[str]) -> List[DetectionResult]:
        """
        Analyze multiple images in batch
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of DetectionResult objects
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {image_path}: {str(e)}")
                # Continue with other images
        
        return results

    def get_detection_statistics(self, results: List[DetectionResult]) -> Dict:
        """
        Get statistics from multiple detection results
        """
        total_images = len(results)
        total_anomalies = sum(len(r.anomalies) for r in results)
        high_confidence_anomalies = sum(
            len(r.get_high_confidence_anomalies()) for r in results
        )
        
        return {
            'total_images_analyzed': total_images,
            'total_anomalies_detected': total_anomalies,
            'high_confidence_anomalies': high_confidence_anomalies,
            'average_anomalies_per_image': total_anomalies / max(total_images, 1),
            'anomaly_detection_rate': total_anomalies / max(total_images, 1)
        }
