"""
Cosmic Anomaly Detector

An AI-powered system for analyzing James Webb Space Telescope images 
to identify artificial structures and anomalous objects in space.
"""

__version__ = "0.1.0"
__author__ = "Cosmic Anomaly Detection Team"
__email__ = "contact@cosmic-anomaly-detector.org"

from .core.analyzer import GravitationalAnalyzer
from .core.classifier import ArtificialStructureClassifier
from .core.detector import AnomalyDetector

__all__ = [
    "AnomalyDetector", 
    "GravitationalAnalyzer", 
    "ArtificialStructureClassifier"
]
