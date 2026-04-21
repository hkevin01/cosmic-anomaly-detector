#!/usr/bin/env python3
"""
Simplified test for gravitational analysis - Phase 3 demonstration.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cosmic_anomaly_detector.core.analyzer import (
    GravitationalAnalyzer,
    MassEstimator,
    OrbitalMechanicsCalculator,
)
from cosmic_anomaly_detector.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def test_basic_physics():
    """Test basic physics calculations"""
    print("🔬 Testing Basic Physics Calculations")
    print("=" * 50)
    
    # Test orbital mechanics
    calc = OrbitalMechanicsCalculator()
    print("\n1. Orbital Mechanics Test:")
    
    positions = np.array([1.0, 0.0, 0.0])  # Earth-like orbit
    velocities = np.array([0.0, 6.28, 0.0])
    central_mass = 1.0
    
    orbital_params = calc.calculate_orbital_parameters(positions, velocities, central_mass)
    print(f"   ✓ Semi-major axis: {orbital_params.semi_major_axis:.3f} AU")
    print(f"   ✓ Orbital period: {orbital_params.orbital_period:.3f} years")
    print(f"   ✓ Eccentricity: {orbital_params.eccentricity:.3f}")
    
    # Test mass estimation
    estimator = MassEstimator()
    print("\n2. Mass Estimation Test:")
    
    mass_from_lum = estimator.estimate_mass_from_luminosity(1.0, 'G')
    mass_from_vel = estimator.estimate_mass_from_orbital_velocity(30.0, 1.0)
    print(f"   ✓ Mass from luminosity: {mass_from_lum:.3f} solar masses")
    print(f"   ✓ Mass from velocity: {mass_from_vel:.3f} solar masses")
    
    # Test anomaly detection
    anomaly_score = estimator.detect_mass_anomalies(
        100.0,  # Very high mass
        {'luminosity': 1.0, 'radius': 1.0}
    )
    print(f"   ✓ Mass anomaly score: {anomaly_score:.3f}")
    
    return True


def test_gravitational_analysis():
    """Test full gravitational analysis"""
    print("\n🌌 Testing Gravitational Analysis")
    print("=" * 50)
    
    analyzer = GravitationalAnalyzer()
    
    # Create test objects
    test_objects = [
        {  # Normal star
            'centroid': (100, 100),
            'area': 50,
            'intensity_mean': 1000,
        },
        {  # Potential anomaly (high mass-to-light ratio)
            'centroid': (200, 150),
            'area': 200,
            'intensity_mean': 100,  # Very dim for its size
        },
        {  # Another anomaly
            'centroid': (300, 200),
            'area': 300,
            'intensity_mean': 3000,  # Very bright
        }
    ]
    
    # Create simple test image
    test_image = np.random.normal(0, 0.1, (512, 512))
    
    # Add point sources to image
    for obj in test_objects:
        x, y = obj['centroid']
        intensity = obj['intensity_mean'] / 1000.0
        xx, yy = np.meshgrid(np.arange(512), np.arange(512))
        source = intensity * np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * 5**2))
        test_image += source
    
    # Perform analysis
    results = analyzer.analyze_physics(test_objects, test_image)
    
    print(f"\nAnalyzed {len(results)} objects:")
    print("-" * 40)
    
    for result in results:
        print(f"\n{result.object_id}:")
        print(f"  Anomaly Score: {result.overall_anomaly_score:.3f}")
        print(f"  Type: {result.anomaly_type}")
        print(f"  Priority: {result.follow_up_priority}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Explanation: {result.physical_explanation}")
    
    # Count high-priority objects
    high_priority = [r for r in results if r.follow_up_priority in ['HIGH', 'CRITICAL']]
    print(f"\n📊 Summary:")
    print(f"   Total objects: {len(results)}")
    print(f"   High priority: {len(high_priority)}")
    
    return results


def main():
    """Run simplified physics validation tests"""
    print("🔬 Cosmic Anomaly Detector - Phase 3 Physics Validation")
    print("=" * 70)
    
    try:
        # Test basic components
        test_basic_physics()
        
        # Test integrated analysis
        results = test_gravitational_analysis()
        
        print("\n" + "=" * 70)
        print("✅ Phase 3 Physics Validation Tests Completed!")
        print("\n🎯 Key Capabilities Demonstrated:")
        print("   • Orbital mechanics calculations (Kepler's laws)")
        print("   • Mass estimation from multiple methods")
        print("   • Physics-based anomaly detection")
        print("   • Confidence scoring and prioritization")
        print("   • Scientific explanation generation")
        
        print(f"\n📈 Test Results:")
        print(f"   Objects analyzed: {len(results)}")
        anomalous = [r for r in results if r.overall_anomaly_score > 0.5]
        print(f"   Anomalous objects: {len(anomalous)}")
        
        print("\n🚀 Ready for Phase 4: Machine Learning Integration!")
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
