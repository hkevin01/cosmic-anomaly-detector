"""
Configuration Management for Cosmic Anomaly Detector

Handles configuration loading, validation, and management for the system.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing parameters"""
    max_image_size: int = 4096
    noise_reduction_sigma: float = 1.0
    edge_detection_threshold: float = 0.1
    object_detection_min_area: int = 100
    coordinate_system: str = "icrs"
    preprocessing_steps: list = field(default_factory=lambda: [
        "noise_reduction", "edge_enhancement", "normalization"
    ])


@dataclass
class GravitationalAnalysisConfig:
    """Configuration for gravitational analysis parameters"""
    gravitational_constant: float = 6.67430e-11  # m^3 kg^-1 s^-2
    kepler_tolerance: float = 0.05  # 5% tolerance for orbital mechanics
    lensing_detection_threshold: float = 0.01
    mass_estimation_method: str = "orbital_velocity"
    physics_validation_strict: bool = True


@dataclass
class ClassificationConfig:
    """Configuration for AI classification models"""
    model_type: str = "ensemble"
    confidence_threshold: float = 0.85
    false_positive_penalty: float = 10.0
    feature_extraction_method: str = "combined"
    cross_validation_folds: int = 5
    model_save_path: str = "models/"


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection parameters"""
    multi_modal_weights: Dict[str, float] = field(default_factory=lambda: {
        "visual": 0.4,
        "gravitational": 0.4,
        "spectral": 0.2
    })
    detection_threshold: float = 0.9
    consensus_required: int = 2  # Minimum number of methods that must agree
    adaptive_thresholds: bool = True


@dataclass
class SystemConfig:
    """Main system configuration"""
    # Component configurations
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    gravitational_analysis: GravitationalAnalysisConfig = field(default_factory=GravitationalAnalysisConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    
    # System settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    data_root_path: str = "data/"
    output_path: str = "output/"
    temp_path: str = "temp/"
    
    # Performance settings
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    gpu_enabled: bool = True
    batch_size: int = 16
    
    # Scientific settings
    scientific_notation: bool = True
    reproducible_random_seed: int = 42
    validation_strict: bool = True


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[SystemConfig] = None
        
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            "config.yaml",
            "config/config.yaml",
            os.path.expanduser("~/.cosmic_anomaly_detector/config.yaml"),
            "/etc/cosmic_anomaly_detector/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # Return default path if none found
        return "config.yaml"
    
    def load_config(self) -> SystemConfig:
        """Load configuration from file or create default"""
        if self._config is not None:
            return self._config
            
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                self._config = self._dict_to_config(config_dict)
                logging.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
                logging.info("Using default configuration")
                self._config = SystemConfig()
        else:
            logging.info("No config file found, using default configuration")
            self._config = SystemConfig()
            
        return self._config
    
    def save_config(self, config: SystemConfig, path: Optional[str] = None) -> None:
        """Save configuration to file"""
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(config)
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logging.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            raise
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object"""
        # This is a simplified conversion - in practice, you'd want more robust handling
        config = SystemConfig()
        
        if 'image_processing' in config_dict:
            config.image_processing = ImageProcessingConfig(**config_dict['image_processing'])
        if 'gravitational_analysis' in config_dict:
            config.gravitational_analysis = GravitationalAnalysisConfig(**config_dict['gravitational_analysis'])
        if 'classification' in config_dict:
            config.classification = ClassificationConfig(**config_dict['classification'])
        if 'anomaly_detection' in config_dict:
            config.anomaly_detection = AnomalyDetectionConfig(**config_dict['anomaly_detection'])
            
        # Update system settings
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in ['image_processing', 'gravitational_analysis', 'classification', 'anomaly_detection']:
                setattr(config, key, value)
                
        return config
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig object to dictionary"""
        return {
            'image_processing': config.image_processing.__dict__,
            'gravitational_analysis': config.gravitational_analysis.__dict__,
            'classification': config.classification.__dict__,
            'anomaly_detection': config.anomaly_detection.__dict__,
            'log_level': config.log_level,
            'log_format': config.log_format,
            'data_root_path': config.data_root_path,
            'output_path': config.output_path,
            'temp_path': config.temp_path,
            'max_workers': config.max_workers,
            'memory_limit_gb': config.memory_limit_gb,
            'gpu_enabled': config.gpu_enabled,
            'batch_size': config.batch_size,
            'scientific_notation': config.scientific_notation,
            'reproducible_random_seed': config.reproducible_random_seed,
            'validation_strict': config.validation_strict
        }
    
    def validate_config(self, config: SystemConfig) -> bool:
        """Validate configuration parameters"""
        try:
            # Validate paths exist or can be created
            for path_attr in ['data_root_path', 'output_path', 'temp_path']:
                path = getattr(config, path_attr)
                Path(path).mkdir(parents=True, exist_ok=True)
            
            # Validate numerical parameters
            assert 0 < config.gravitational_analysis.kepler_tolerance < 1
            assert 0 < config.classification.confidence_threshold <= 1
            assert 0 < config.anomaly_detection.detection_threshold <= 1
            assert config.max_workers > 0
            assert config.memory_limit_gb > 0
            
            # Validate multi-modal weights sum to 1
            weights = config.anomaly_detection.multi_modal_weights
            weight_sum = sum(weights.values())
            assert abs(weight_sum - 1.0) < 1e-6, f"Multi-modal weights sum to {weight_sum}, not 1.0"
            
            logging.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> SystemConfig:
    """Get the global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.load_config()


def set_config_path(path: str) -> None:
    """Set the configuration file path"""
    global _config_manager
    _config_manager = ConfigManager(path)


def reload_config() -> SystemConfig:
    """Reload configuration from file"""
    global _config_manager
    if _config_manager is not None:
        _config_manager._config = None
    return get_config()
