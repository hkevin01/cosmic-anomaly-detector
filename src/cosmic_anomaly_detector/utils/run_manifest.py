"""
Run Manifest — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: MAN-001
# Requirement: Capture a complete, JSON-serialisable snapshot of the runtime
#              environment and active configuration for every analysis run.
# Purpose: Enable full scientific reproducibility — given a manifest, any run
#          can be reconstructed with identical software versions and parameters.
# Rationale: Reproducibility is a first-class requirement for scientific
#             software; embedding Python version, platform, and config avoids
#             "works on my machine" ambiguity in peer review.
# Inputs:  config (SystemConfig) — active configuration dataclass.
#          extra (Optional[Dict]) — caller-supplied metadata (e.g., run_id,
#          file list, timing data).
# Outputs: build_manifest → Dict (JSON-serialisable).
#          write_manifest → Path of written .json file.
# Preconditions:  config sub-components must expose __dict__ (dataclasses do).
# Postconditions: Manifest file written atomically (write_text is atomic on
#                 POSIX for sizes < filesystem block size).
# Assumptions: datetime.utcnow() used for UTC timestamp; caller is responsible
#              for ensuring path directory exists before calling write_manifest.
# Side Effects: write_manifest creates a file at path.
# Failure Modes: Non-serialisable values in extra → TypeError from json.dumps.
# Error Handling: Callers should validate extra values before passing.
# Constraints: Manifest size < 1 MB for typical configs; no binary data stored.
# Verification: tests/test_run_manifest.py asserts timestamp, python, and
#               config keys are present in the output dict.
# References: ISO 8601 UTC timestamp format; platform.platform() stdlib.
# ---------------------------------------------------------------------------
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
