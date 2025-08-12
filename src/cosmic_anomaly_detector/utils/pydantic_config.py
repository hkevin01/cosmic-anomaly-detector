"""Pydantic-based configuration validation.

Provides a Pydantic model mirroring the dataclass SystemConfig for stricter
runtime validation and a helper to load/validate then return the legacy
dataclass instance for backwards compatibility.
"""
from __future__ import annotations  # mypy: ignore-errors

from pathlib import Path
from typing import Dict, Any

import yaml  # type: ignore
from pydantic import (  # type: ignore
    BaseModel,
    field_validator,
    model_validator,
)

from .config import (
    SystemConfig,
    ImageProcessingConfig,
    GravitationalAnalysisConfig,
    ClassificationConfig,
    AnomalyDetectionConfig,
)


class ImageProcessingModel(BaseModel):
    max_image_size: int = 4096
    noise_reduction_sigma: float = 1.0
    edge_detection_threshold: float = 0.1
    object_detection_min_area: int = 100
    coordinate_system: str = "icrs"
    preprocessing_steps: list[str] = [
        "noise_reduction",
        "edge_enhancement",
        "normalization",
    ]

    @field_validator("max_image_size")
    @classmethod
    def _size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_image_size must be positive")
        return v


class GravitationalAnalysisModel(BaseModel):
    gravitational_constant: float = 6.67430e-11
    kepler_tolerance: float = 0.05
    lensing_detection_threshold: float = 0.01
    mass_estimation_method: str = "orbital_velocity"
    physics_validation_strict: bool = True

    @field_validator("kepler_tolerance")
    @classmethod
    def _tol_range(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("kepler_tolerance must be (0,1)")
        return v


class ClassificationModel(BaseModel):
    model_type: str = "ensemble"
    confidence_threshold: float = 0.85
    false_positive_penalty: float = 10.0
    feature_extraction_method: str = "combined"
    cross_validation_folds: int = 5
    model_save_path: str = "models/"

    @field_validator("confidence_threshold")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("confidence_threshold must be (0,1]")
        return v


class AnomalyDetectionModel(BaseModel):
    multi_modal_weights: Dict[str, float] = {
        "visual": 0.4,
        "gravitational": 0.4,
        "spectral": 0.2,
    }
    detection_threshold: float = 0.9
    consensus_required: int = 2
    adaptive_thresholds: bool = True

    @model_validator(mode="after")
    def _weights_sum_one(
        self,
    ) -> "AnomalyDetectionModel":  # type: ignore[override]
        total = sum(self.multi_modal_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("multi_modal_weights must sum to 1.0")
        return self


class SystemModel(BaseModel):
    image_processing: ImageProcessingModel = ImageProcessingModel()
    gravitational_analysis: GravitationalAnalysisModel = (
        GravitationalAnalysisModel()
    )
    classification: ClassificationModel = ClassificationModel()
    anomaly_detection: AnomalyDetectionModel = AnomalyDetectionModel()
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    data_root_path: str = "data/"
    output_path: str = "output/"
    temp_path: str = "temp/"
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    gpu_enabled: bool = True
    batch_size: int = 16
    scientific_notation: bool = True
    reproducible_random_seed: int = 42
    validation_strict: bool = True

    @model_validator(mode="after")
    def _paths(self) -> "SystemModel":  # type: ignore[override]
        for p in [self.data_root_path, self.output_path, self.temp_path]:
            Path(p).mkdir(parents=True, exist_ok=True)
        return self


def load_pydantic_config(path: str | None = None) -> SystemConfig:
    """Load and validate config via Pydantic, return dataclass instance."""
    cfg_path = path or "config.yaml"
    data: Dict[str, Any] = {}
    if Path(cfg_path).exists():
        data = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    model = SystemModel(**data)
    # Convert back to dataclass structure for existing code
    return SystemConfig(
        image_processing=ImageProcessingConfig(
            **model.image_processing.model_dump()
        ),
        gravitational_analysis=GravitationalAnalysisConfig(
            **model.gravitational_analysis.model_dump()
        ),
        classification=ClassificationConfig(
            **model.classification.model_dump()
        ),
        anomaly_detection=AnomalyDetectionConfig(
            **model.anomaly_detection.model_dump()
        ),
        log_level=model.log_level,
        log_format=model.log_format,
        data_root_path=model.data_root_path,
        output_path=model.output_path,
        temp_path=model.temp_path,
        max_workers=model.max_workers,
        memory_limit_gb=model.memory_limit_gb,
        gpu_enabled=model.gpu_enabled,
        batch_size=model.batch_size,
        scientific_notation=model.scientific_notation,
        reproducible_random_seed=model.reproducible_random_seed,
        validation_strict=model.validation_strict,
    )
