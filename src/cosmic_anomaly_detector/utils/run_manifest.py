"""Run manifest writer.

Creates a JSON manifest capturing environment + config snapshot for
reproducibility of each analyze run.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import SystemConfig


def build_manifest(
    config: SystemConfig, extra: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": {
            "image_processing": config.image_processing.__dict__,
            "gravitational_analysis": config.gravitational_analysis.__dict__,
            "classification": config.classification.__dict__,
            "anomaly_detection": config.anomaly_detection.__dict__,
        },
    }
    if extra:
        info.update(extra)
    return info


def write_manifest(
    path: Path, config: SystemConfig, extra: Dict[str, Any] | None = None
) -> Path:
    manifest = build_manifest(config, extra)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
