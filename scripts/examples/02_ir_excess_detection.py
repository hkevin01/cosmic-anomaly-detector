#!/usr/bin/env python3
"""
Example 02 — Infrared Excess / Dyson Sphere SED Analysis

Demonstrates IRExcessDetector (Suazo et al. 2024 MNRAS 531, 695 methodology)
on a synthetic image containing:
  - A normal stellar point source (no IR excess)
  - A Dyson-sphere candidate (warm blackbody envelope at ~300 K, γ ≈ 0.3)

Shows the covering factor γ and best-fit DS temperature returned for each
detected object and how to interpret the anomaly score.

Usage
-----
    python scripts/examples/02_ir_excess_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import logging
import numpy as np
from astropy.io.fits import HDUList, PrimaryHDU

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')


def build_synthetic_scene(size: int = 256) -> tuple:
    """
    Return (image array, true_objects) for a 2-source scene.

    Source A: stellar point source, no IR excess (compact, symmetric).
    Source B: Dyson-sphere candidate (bright core + diffuse warm halo).
    """
    rng = np.random.default_rng(7)
    image = rng.standard_normal((size, size)).astype(np.float64) * 0.03
    image = np.clip(image, 0.0, None)

    y, x = np.ogrid[:size, :size]

    # Source A — normal star: narrow Gaussian peak
    cx_a, cy_a = 90, 90
    image += 0.8 * np.exp(-((x - cx_a) ** 2 + (y - cy_a) ** 2) / (2 * 2 ** 2))

    # Source B — DS candidate: bright core + broad warm halo
    cx_b, cy_b = 170, 160
    image += 0.9 * np.exp(-((x - cx_b) ** 2 + (y - cy_b) ** 2) / (2 * 2 ** 2))
    # Warm halo simulates re-radiated IR emission
    image += 0.35 * np.exp(-((x - cx_b) ** 2 + (y - cy_b) ** 2) / (2 * 18 ** 2))

    image = (image / image.max()).astype(np.float32)

    detected_objects = [
        {
            'id': 'obj_000', 'coordinates': [cy_a, cx_a],
            'bounding_box': [cy_a - 8, cx_a - 8, cy_a + 9, cx_a + 9],
            'brightness': 0.8, 'score': 0.5,
        },
        {
            'id': 'obj_001', 'coordinates': [cy_b, cx_b],
            'bounding_box': [cy_b - 25, cx_b - 25, cy_b + 26, cx_b + 26],
            'brightness': 0.9, 'score': 0.6,
        },
    ]
    return image, detected_objects


def main() -> None:
    from cosmic_anomaly_detector.processing.algorithms import IRExcessDetector

    image, detected_objects = build_synthetic_scene(256)
    print("=== IR Excess / Dyson Sphere SED Analysis Demo ===\n")
    print(f"Scene: {image.shape[0]}×{image.shape[1]} px, "
          f"{len(detected_objects)} injected sources\n")
    print("Source A (obj_000) — normal stellar point source (no warm halo)")
    print("Source B (obj_001) — Dyson-sphere candidate (core + warm diffuse halo)\n")

    detector = IRExcessDetector(
        ds_temp_range=(100.0, 700.0),
        ds_temp_steps=20,
        gamma_range=(0.01, 0.9),
        gamma_steps=25,
        rmse_threshold=0.20,
        min_gamma=0.05,
    )

    candidates = detector.detect(image, detected_objects)

    if not candidates:
        print("No IR-excess candidates found (all objects within normal SED bounds).")
        print("\nNote: 'normal' sources are expected to be filtered out — "
              "this is the desired behaviour.")
        return

    print(f"Found {len(candidates)} IR-excess candidate(s):\n")
    for cand in candidates:
        m = cand.metadata
        print(f"  ID             : {cand.id}")
        print(f"  Algorithm      : {cand.algorithm}")
        print(f"  Anomaly type   : {cand.anomaly_type}")
        print(f"  Position (r,c) : {cand.coordinates}")
        print(f"  Score          : {cand.score:.3f}  (0=normal, 1=strong anomaly)")
        print(f"  Covering factor γ : {m['covering_factor_gamma']:.3f}")
        print(f"  DS temperature    : {m['ds_temperature_k']:.0f} K")
        print(f"  SED RMSE (log10)  : {m['sed_rmse']:.4f}")
        print(f"  IR/opt flux ratio : {m['ir_to_optical_ratio']:.4f}")
        print(f"  Reference         : {m['reference']}")
        print()

    print("Interpretation:")
    print("  γ ≥ 0.05 and RMSE < 0.20 → object's IR excess is consistent with")
    print("  a Dyson sphere component at the listed DS temperature.")
    print("  Higher γ and lower RMSE → stronger evidence for anomalous IR emission.")


if __name__ == '__main__':
    main()
