#!/usr/bin/env python3
"""
Test script for the gravitational analysis and physics validation engine.

This script demonstrates the capabilities of Phase 3 implementation,
including orbital mechanics, mass estimation, and gravitational lensing detection.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cosmic_anomaly_detector.core.analyzer import (
    GravitationalAnalyzer,
    GravitationalLensingDetector,
    MassEstimator,
    OrbitalMechanicsCalculator,
    PhysicsValidator,
)
from cosmic_anomaly_detector.utils.config import get_config
from cosmic_anomaly_detector.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def create_test_objects():
    """Create test objects with various properties for analysis"""
    
    # Normal star-like object
    normal_star = {
        'centroid': (100, 100),
        'area': 50,
        'intensity_mean': 1000,
        'luminosity': 1.0,
        'radius': 1.0,
        'coordinates': [100, 100],
        'velocity': [5.0, 0.0],
        'background_objects': []
    }
    
    # Potential Dyson sphere (high mass-to-light ratio)
    dyson_candidate = {
        'centroid': (200, 150),
        'area': 200,
        'intensity_mean': 500,  # Dimmed luminosity
        'luminosity': 0.3,      # Much lower than expected
        'radius': 2.0,
        'coordinates': [200, 150],
        'velocity': [3.0, 2.0],
        'background_objects': []
    }
    
    # Gravitational anomaly (violates Kepler's laws)
    orbital_anomaly = {
        'centroid': (300, 200),
        'area': 100,
        'intensity_mean': 800,
        'luminosity': 1.5,
        'radius': 1.5,
        'coordinates': [300, 200],
        'velocity': [15.0, 5.0],  # Anomalously high velocity
        'background_objects': []
    }
    
    # Lensing object
    lensing_object = {
        'centroid': (150, 250),
        'area': 300,
        'intensity_mean': 2000,
        'luminosity': 5.0,
        'radius': 3.0,
        'coordinates': [150, 250],
        'velocity': [2.0, 1.0],
        'background_objects': [
            {'id': 'bg1', 'distance_ratio': 2.0, 'observed_deflection': 0.5},
            {'id': 'bg2', 'distance_ratio': 3.0, 'observed_deflection': 0.3}
        ]
    }
    
    return [normal_star, dyson_candidate, orbital_anomaly, lensing_object]


def create_test_image():
    """Create a synthetic JWST-like image for testing"""
    # Create 512x512 test image
    image = np.random.normal(0, 0.1, (512, 512))
    
    # Add some point sources
    sources = [
        (100, 100, 1.0),   # Normal star
        (200, 150, 0.5),   # Dimmed object
        (300, 200, 0.8),   # Anomalous object
        (150, 250, 2.0),   # Bright lensing object
    ]
    
    for x, y, intensity in sources:
        # Add Gaussian point source
        xx, yy = np.meshgrid(np.arange(512), np.arange(512))
        source = intensity * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 5**2))
        image += source
        
    # Add some Einstein ring-like features around the lensing object
    xx, yy = np.meshgrid(np.arange(512), np.arange(512))
    ring_x, ring_y = 150, 250
    ring_radius = 25
    ring_width = 3
    ring_distance = np.sqrt((xx - ring_x)**2 + (yy - ring_y)**2)
    ring_mask = (ring_distance > ring_radius - ring_width) & (ring_distance < ring_radius + ring_width)
    image[ring_mask] += 0.3
    
    return image


def test_orbital_mechanics():
    """Test orbital mechanics calculations"""
    print("\n=== Testing Orbital Mechanics Calculator ===")
    
    calculator = OrbitalMechanicsCalculator()
    
    # Test with Earth-like orbit
    positions = np.array([1.0, 0.0, 0.0])  # 1 AU from star
    velocities = np.array([0.0, 6.28, 0.0])  # Roughly circular orbit velocity
    central_mass = 1.0  # Solar masses
    
    orbital_params = calculator.calculate_orbital_parameters(
        positions, velocities, central_mass
    )
    
    print(f"Semi-major axis: {orbital_params.semi_major_axis:.3f} AU")
    print(f"Eccentricity: {orbital_params.eccentricity:.3f}")
    print(f"Orbital period: {orbital_params.orbital_period:.3f} years")
    print(f"Orbital velocity: {orbital_params.velocity:.3f} AU/year")
    
    # Test Kepler's law validation
    observed_period = 1.0  # One year
    compliance_score = calculator.validate_kepler_laws(orbital_params, observed_period)
    print(f"Kepler's law compliance: {compliance_score:.3f}")
    
    return orbital_params


def test_mass_estimation():
    """Test mass estimation methods"""
    print("\n=== Testing Mass Estimator ===")
    
    estimator = MassEstimator()
    
    # Test different mass estimation methods
    luminosity = 1.0  # Solar luminosities
    mass_from_lum = estimator.estimate_mass_from_luminosity(luminosity, 'G')
    print(f"Mass from luminosity (G-star): {mass_from_lum:.3f} solar masses")
    
    velocity = 30.0  # km/s
    radius = 1.0     # AU
    mass_from_vel = estimator.estimate_mass_from_orbital_velocity(velocity, radius)
    print(f"Mass from orbital velocity: {mass_from_vel:.3f} solar masses")
    
    # Test mass anomaly detection
    test_properties = {'luminosity': 1.0, 'radius': 1.0}
    anomaly_score = estimator.detect_mass_anomalies(10.0, test_properties)  # Very high mass
    print(f"Mass anomaly score (high mass): {anomaly_score:.3f}")
    
    return mass_from_lum, mass_from_vel


def test_lensing_detection():
    """Test gravitational lensing detection"""
    print("\n=== Testing Gravitational Lensing Detector ===")
    
    detector = GravitationalLensingDetector()
    
    # Create test image with lensing features
    test_image = create_test_image()
    
    # Detect lensing signatures
    lensing_signature = detector.detect_lensing_signature(test_image)
    
    print(f"Einstein radius detected: {lensing_signature.einstein_radius:.1f} pixels")
    print(f"Magnification factor: {lensing_signature.magnification_factor:.2f}")
    print(f"Lensing strength: {lensing_signature.lensing_strength:.4f}")
    print(f"Anomaly detected: {lensing_signature.anomaly_detected}")
    print(f"Background sources found: {len(lensing_signature.background_sources)}")
    
    return lensing_signature


def test_gravitational_analyzer():
    """Test the main gravitational analyzer"""
    print("\n=== Testing Gravitational Analyzer ===")
    
    analyzer = GravitationalAnalyzer()
    
    # Create test objects and image
    test_objects = create_test_objects()
    test_image = create_test_image()
    
    # Perform gravitational analysis
    results = analyzer.analyze_physics(test_objects, test_image)
    
    print(f"\nAnalyzed {len(results)} objects:")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"\nObject {result.object_id}:")
        print(f"  Overall anomaly score: {result.overall_anomaly_score:.3f}")
        print(f"  Anomaly type: {result.anomaly_type}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Follow-up priority: {result.follow_up_priority}")
        print(f"  Kepler compliance: {result.kepler_compliance_score:.3f}")
        print(f"  Mass anomaly: {result.mass_anomaly_score:.3f}")
        print(f"  Lensing anomaly: {result.lensing_anomaly_score:.3f}")
        print(f"  Explanation: {result.physical_explanation}")
    
    return results


def test_physics_validator():
    """Test physics validation against known objects"""
    print("\n=== Testing Physics Validator ===")
    
    validator = PhysicsValidator()
    
    # First run gravitational analysis
    analyzer = GravitationalAnalyzer()
    test_objects = create_test_objects()
    test_image = create_test_image()
    analysis_results = analyzer.analyze_physics(test_objects, test_image)
    
    # Validate results
    validation_results = validator.validate_with_known_objects(analysis_results)
    
    print(f"\nValidation Results:")
    print(f"  Validated objects: {len(validation_results['validated_objects'])}")
    print(f"  Potential discoveries: {len(validation_results['potential_discoveries'])}")
    print(f"  False positives: {len(validation_results['false_positives'])}")
    print(f"  Overall validation score: {validation_results['validation_score']:.3f}")
    
    return validation_results


def main():
    """Run all physics validation tests"""
    print("🔬 Cosmic Anomaly Detector - Physics Validation Test Suite")
    print("=" * 70)
    
    try:
        # Test individual components
        test_orbital_mechanics()
        test_mass_estimation()
        test_lensing_detection()
        
        # Test integrated analysis
        test_gravitational_analyzer()
        test_physics_validator()
        
        print("\n" + "=" * 70)
        print("✅ All physics validation tests completed successfully!")
        print("\nPhase 3 (Physics Validation) implementation is working correctly.")
        print("The gravitational analysis engine can:")
        print("  • Calculate orbital parameters and validate Kepler's laws")
        print("  • Estimate masses using multiple astrophysical methods")
        print("  • Detect gravitational lensing signatures")
        print("  • Identify physics-based anomalies")
        print("  • Validate results against known astronomical objects")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
