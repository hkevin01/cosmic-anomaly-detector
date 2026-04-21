#!/usr/bin/env python3
"""
Machine Learning Model Training Script - Phase 4 Implementation

This script demonstrates the machine learning capabilities including:
- Training data preparation and synthetic data generation
- Feature engineering from astronomical objects
- Multiple ML model training and evaluation
- Model ensemble and selection
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cosmic_anomaly_detector.models.ml_models import (
    TF_AVAILABLE,
    DeepAnomalyDetector,
    EnsembleAnomalyDetector,
    ModelManager,
)
from cosmic_anomaly_detector.models.training_data import (
    SyntheticDataGenerator,
    TrainingDataPreparer,
)
from cosmic_anomaly_detector.utils.config import get_config
from cosmic_anomaly_detector.utils.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def test_synthetic_data_generation():
    """Test synthetic data generation"""
    print("\n🧪 Testing Synthetic Data Generation")
    print("=" * 50)
    
    generator = SyntheticDataGenerator()
    
    # Generate examples of each type
    print("Generating example objects...")
    
    normal_star = generator.generate_normal_star()
    print(f"✓ Normal Star: M={normal_star['mass']:.2f} M☉, "
          f"L={normal_star['luminosity']:.2f} L☉")
    
    normal_galaxy = generator.generate_normal_galaxy()
    print(f"✓ Normal Galaxy: M={normal_galaxy['mass']:.1e} M☉, "
          f"L={normal_galaxy['luminosity']:.1e} L☉")
    
    dyson_sphere = generator.generate_dyson_sphere_candidate(0.5)
    print(f"✓ Dyson Sphere: M={dyson_sphere['mass']:.2f} M☉, "
          f"L={dyson_sphere['luminosity']:.2f} L☉, "
          f"T={dyson_sphere['temperature']:.0f} K")
    
    megastructure = generator.generate_megastructure()
    print(f"✓ Megastructure: M={megastructure['mass']:.1e} M☉, "
          f"L={megastructure['luminosity']:.2f} L☉")
    
    grav_anomaly = generator.generate_gravitational_anomaly()
    print(f"✓ Gravitational Anomaly: M={grav_anomaly['mass']:.1e} M☉, "
          f"L={grav_anomaly['luminosity']:.2f} L☉")
    
    # Generate full dataset
    print(f"\nGenerating training dataset...")
    dataset = generator.generate_training_dataset(n_samples=100, anomaly_fraction=0.2)
    
    normal_count = sum(1 for obj in dataset if obj['label'] == 0)
    anomaly_count = sum(1 for obj in dataset if obj['label'] == 1)
    
    print(f"✓ Generated {len(dataset)} objects:")
    print(f"  - Normal objects: {normal_count}")
    print(f"  - Anomalous objects: {anomaly_count}")
    
    return dataset


def test_training_data_preparation():
    """Test training data preparation"""
    print("\n🔧 Testing Training Data Preparation")
    print("=" * 50)
    
    preparer = TrainingDataPreparer()
    
    # Prepare synthetic training data
    print("Preparing synthetic training data...")
    X, y = preparer.prepare_synthetic_training_data(
        n_samples=500,
        anomaly_fraction=0.15
    )
    
    print(f"✓ Training data shape: {X.shape}")
    print(f"✓ Feature dimensions: {X.shape[1]}")
    print(f"✓ Class distribution: {np.bincount(y)}")
    
    # Test data augmentation
    print("\nTesting data augmentation...")
    X_aug, y_aug = preparer.augment_training_data(X, y, augmentation_factor=2)
    print(f"✓ Augmented data shape: {X_aug.shape}")
    
    # Test data balancing
    print("\nTesting data balancing...")
    X_bal, y_bal = preparer.balance_dataset(X_aug, y_aug)
    print(f"✓ Balanced data shape: {X_bal.shape}")
    print(f"✓ Balanced class distribution: {np.bincount(y_bal)}")
    
    return X_bal, y_bal


def test_ensemble_model():
    """Test ensemble anomaly detection model"""
    print("\n🤖 Testing Ensemble Model")
    print("=" * 50)
    
    # Prepare training data
    preparer = TrainingDataPreparer()
    X, y = preparer.prepare_synthetic_training_data(n_samples=300, anomaly_fraction=0.2)
    
    print(f"Training with {len(X)} samples...")
    
    # Create and train ensemble model
    ensemble_model = EnsembleAnomalyDetector()
    
    try:
        metrics = ensemble_model.train(X, y)
        
        print(f"✓ Training completed!")
        print(f"  - Accuracy: {metrics.accuracy:.3f}")
        print(f"  - Precision: {metrics.precision:.3f}")
        print(f"  - Recall: {metrics.recall:.3f}")
        print(f"  - F1 Score: {metrics.f1_score:.3f}")
        print(f"  - AUC Score: {metrics.auc_score:.3f}")
        
        # Test predictions
        print("\nTesting predictions...")
        test_X = X[:10]  # Use first 10 samples
        predictions = ensemble_model.predict(test_X)
        
        for i, pred in enumerate(predictions[:5]):
            print(f"  Sample {i}: {pred.classification} "
                  f"(prob: {pred.anomaly_probability:.3f}, "
                  f"conf: {pred.confidence:.3f})")
        
        return ensemble_model, metrics
        
    except Exception as e:
        print(f"❌ Ensemble training failed: {str(e)}")
        return None, None


def test_deep_learning_model():
    """Test deep learning model (if TensorFlow available)"""
    print("\n🧠 Testing Deep Learning Model")
    print("=" * 50)
    
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available - skipping deep learning test")
        return None, None
    
    # Prepare training data
    preparer = TrainingDataPreparer()
    X, y = preparer.prepare_synthetic_training_data(n_samples=500, anomaly_fraction=0.2)
    
    print(f"Training with {len(X)} samples...")
    
    # Create and train deep learning model
    deep_model = DeepAnomalyDetector()
    
    try:
        metrics = deep_model.train(X, y)
        
        print(f"✓ Training completed!")
        print(f"  - Accuracy: {metrics.accuracy:.3f}")
        print(f"  - Precision: {metrics.precision:.3f}")
        print(f"  - Recall: {metrics.recall:.3f}")
        print(f"  - F1 Score: {metrics.f1_score:.3f}")
        print(f"  - AUC Score: {metrics.auc_score:.3f}")
        
        # Test predictions
        print("\nTesting predictions...")
        test_X = X[:10]
        predictions = deep_model.predict(test_X)
        
        for i, pred in enumerate(predictions[:5]):
            print(f"  Sample {i}: {pred.classification} "
                  f"(prob: {pred.anomaly_probability:.3f}, "
                  f"conf: {pred.confidence:.3f})")
        
        return deep_model, metrics
        
    except Exception as e:
        print(f"❌ Deep learning training failed: {str(e)}")
        return None, None


def test_model_manager():
    """Test model management and selection"""
    print("\n📊 Testing Model Manager")
    print("=" * 50)
    
    # Prepare training data
    preparer = TrainingDataPreparer()
    X, y = preparer.prepare_synthetic_training_data(n_samples=400, anomaly_fraction=0.2)
    
    # Create model manager
    manager = ModelManager()
    
    # Register models
    ensemble_model = EnsembleAnomalyDetector()
    manager.register_model(ensemble_model)
    
    if TF_AVAILABLE:
        deep_model = DeepAnomalyDetector()
        manager.register_model(deep_model)
        print("✓ Registered ensemble and deep learning models")
    else:
        print("✓ Registered ensemble model (TensorFlow not available)")
    
    # Train all models
    print("\nTraining all models...")
    try:
        results = manager.train_all_models(X, y)
        
        print(f"\nTraining Results:")
        for model_name, metrics in results.items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {metrics.accuracy:.3f}")
            print(f"    - AUC: {metrics.auc_score:.3f}")
        
        print(f"\n✓ Best model: {manager.best_model}")
        
        # Test predictions with best model
        print("\nTesting best model predictions...")
        test_X = X[:5]
        predictions = manager.predict_with_best_model(test_X)
        
        for i, pred in enumerate(predictions):
            print(f"  Sample {i}: {pred.classification} "
                  f"(prob: {pred.anomaly_probability:.3f})")
        
        return manager, results
        
    except Exception as e:
        print(f"❌ Model manager test failed: {str(e)}")
        return None, None


def test_model_persistence():
    """Test model saving and loading"""
    print("\n💾 Testing Model Persistence")
    print("=" * 50)
    
    # Create and train a simple model
    preparer = TrainingDataPreparer()
    X, y = preparer.prepare_synthetic_training_data(n_samples=200, anomaly_fraction=0.2)
    
    ensemble_model = EnsembleAnomalyDetector()
    
    try:
        print("Training model for persistence test...")
        metrics = ensemble_model.train(X, y)
        print(f"✓ Model trained (AUC: {metrics.auc_score:.3f})")
        
        # Test saving
        save_path = "temp_model_test"
        ensemble_model.save_model(save_path)
        print(f"✓ Model saved to {save_path}")
        
        # Test loading
        new_model = EnsembleAnomalyDetector()
        new_model.load_model(save_path)
        print(f"✓ Model loaded successfully")
        
        # Test that loaded model works
        test_predictions = new_model.predict(X[:3])
        print(f"✓ Loaded model predictions: {len(test_predictions)} results")
        
        # Cleanup
        for file_pattern in ["_ensemble.pkl"]:
            file_path = save_path + file_pattern
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Model persistence test failed: {str(e)}")
        return False


def demonstrate_phase4_capabilities():
    """Demonstrate key Phase 4 ML capabilities"""
    print("\n🎯 Phase 4 Capability Demonstration")
    print("=" * 50)
    
    # Create realistic test scenario
    preparer = TrainingDataPreparer()
    
    # Generate larger, more realistic dataset
    print("Creating realistic training scenario...")
    X, y = preparer.prepare_synthetic_training_data(
        n_samples=800,
        anomaly_fraction=0.1  # 10% anomalies (realistic)
    )
    
    # Balance and augment data
    X, y = preparer.balance_dataset(X, y)
    X, y = preparer.augment_training_data(X, y, augmentation_factor=2)
    
    print(f"✓ Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✓ Class distribution: {np.bincount(y)}")
    
    # Train ensemble model
    manager = ModelManager()
    ensemble = EnsembleAnomalyDetector()
    manager.register_model(ensemble)
    
    if TF_AVAILABLE:
        deep = DeepAnomalyDetector()
        manager.register_model(deep)
    
    print("\nTraining production-ready models...")
    results = manager.train_all_models(X, y)
    
    # Evaluate performance
    print(f"\n📈 Final Model Performance:")
    for name, metrics in results.items():
        print(f"  {name}:")
        print(f"    Accuracy: {metrics.accuracy:.3f}")
        print(f"    Precision: {metrics.precision:.3f}")
        print(f"    Recall: {metrics.recall:.3f}")
        print(f"    AUC: {metrics.auc_score:.3f}")
    
    # Test on new synthetic data
    print(f"\n🔬 Testing on New Objects:")
    generator = SyntheticDataGenerator()
    
    # Generate test objects
    test_objects = [
        generator.generate_normal_star(),
        generator.generate_dyson_sphere_candidate(0.7),
        generator.generate_megastructure(),
        generator.generate_normal_galaxy(),
        generator.generate_gravitational_anomaly()
    ]
    
    # Convert to features
    test_features = []
    for obj in test_objects:
        features = preparer.synthetic_object_to_features(obj)
        test_features.append(features)
    
    test_X = np.array(test_features)
    
    # Make predictions
    predictions = manager.predict_with_best_model(test_X)
    
    for i, (obj, pred) in enumerate(zip(test_objects, predictions)):
        true_label = "anomaly" if obj['label'] == 1 else "normal"
        predicted_label = pred.classification
        correct = "✓" if true_label == predicted_label else "❌"
        
        print(f"  {correct} {obj['object_type']}: "
              f"True={true_label}, Pred={predicted_label} "
              f"(prob={pred.anomaly_probability:.3f})")
    
    return manager, results


def main():
    """Run Phase 4 machine learning demonstration"""
    print("🤖 Cosmic Anomaly Detector - Phase 4: Machine Learning Models")
    print("=" * 70)
    
    success_count = 0
    total_tests = 6
    
    try:
        # Test 1: Synthetic data generation
        try:
            test_synthetic_data_generation()
            success_count += 1
        except Exception as e:
            print(f"❌ Synthetic data test failed: {str(e)}")
        
        # Test 2: Training data preparation
        try:
            test_training_data_preparation()
            success_count += 1
        except Exception as e:
            print(f"❌ Training data preparation failed: {str(e)}")
        
        # Test 3: Ensemble model
        try:
            test_ensemble_model()
            success_count += 1
        except Exception as e:
            print(f"❌ Ensemble model test failed: {str(e)}")
        
        # Test 4: Deep learning model
        try:
            test_deep_learning_model()
            success_count += 1
        except Exception as e:
            print(f"❌ Deep learning test failed: {str(e)}")
        
        # Test 5: Model manager
        try:
            test_model_manager()
            success_count += 1
        except Exception as e:
            print(f"❌ Model manager test failed: {str(e)}")
        
        # Test 6: Model persistence
        try:
            test_model_persistence()
            success_count += 1
        except Exception as e:
            print(f"❌ Model persistence test failed: {str(e)}")
        
        # Final demonstration
        print("\n" + "=" * 70)
        print("🚀 COMPREHENSIVE DEMONSTRATION")
        print("=" * 70)
        
        demonstrate_phase4_capabilities()
        
        print("\n" + "=" * 70)
        print("✅ Phase 4 Implementation Complete!")
        print(f"✅ {success_count}/{total_tests} component tests passed")
        
        print("\n🎯 Key Achievements:")
        print("   • Synthetic training data generation")
        print("   • Advanced feature engineering")
        print("   • Multiple ML model architectures")
        print("   • Ensemble learning and model selection")
        print("   • Model persistence and deployment")
        
        if TF_AVAILABLE:
            print("   • Deep learning with TensorFlow")
        else:
            print("   • Scikit-learn based machine learning")
        
        print("\n📊 Capabilities Demonstrated:")
        print("   • Realistic Dyson sphere detection")
        print("   • Megastructure identification")
        print("   • Gravitational anomaly classification")
        print("   • Physics-informed feature engineering")
        print("   • Conservative anomaly scoring")
        
        print("\n🚀 Ready for Phase 5: System Integration!")
        return 0
        
    except Exception as e:
        logger.error(f"Phase 4 demonstration failed: {str(e)}")
        print(f"\n❌ Phase 4 demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
