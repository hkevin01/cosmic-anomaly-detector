#!/usr/bin/env python3
"""
Example 03 — Wavelet vs Baseline Source Detection

Compares the starlet wavelet multi-scale detector against the baseline
sigma-thresholding approach on a crowded synthetic stellar field with
injected faint sources.

Demonstrates that the wavelet detector recovers faint sources missed by
the simple threshold method (higher completeness at low SNR).

Usage
-----
    python scripts/examples/03_wavelet_detection_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')


def inject_sources(
    image: np.ndarray,
    positions: list,
    peak_flux: float,
    fwhm_px: float = 2.5,
) -> np.ndarray:
    """Inject Gaussian point sources at given pixel positions."""
    sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    for ry, cx in positions:
        image = image + peak_flux * np.exp(
            -((x - cx) ** 2 + (y - ry) ** 2) / (2 * sigma ** 2)
        )
    return image


def baseline_sigma_detect(image: np.ndarray, nsigma: float = 3.0) -> list:
    """Simple sigma-threshold detector for comparison."""
    from scipy.ndimage import gaussian_filter, label, center_of_mass
    smooth = gaussian_filter(image, sigma=1.0)
    bg = gaussian_filter(image, sigma=15.0)
    residual = smooth - bg
    threshold = nsigma * np.std(residual)
    mask = residual > threshold
    labelled, n = label(mask)
    positions = []
    for i in range(1, n + 1):
        cy, cx = center_of_mass(residual, labelled, i)
        positions.append((cy, cx))
    return positions


def main() -> None:
    from cosmic_anomaly_detector.processing.algorithms import (
        WaveletSourceDetector,
        MatchedFilterDetector,
    )

    rng = np.random.default_rng(2024)
    size = 300
    noise_level = 0.04

    # Background noise image
    image = (rng.standard_normal((size, size)) * noise_level).astype(np.float64)
    image = np.clip(image, 0.0, None)

    # Inject bright sources (easy to detect)
    bright_positions = [(60, 60), (60, 240), (240, 60), (240, 240), (150, 150)]
    image = inject_sources(image, bright_positions, peak_flux=0.80, fwhm_px=2.5)

    # Inject medium sources
    medium_positions = [(100, 200), (200, 100), (80, 160), (170, 80)]
    image = inject_sources(image, medium_positions, peak_flux=0.25, fwhm_px=2.5)

    # Inject faint sources (SNR ~ 3 — borderline for sigma-clipping)
    faint_positions = [(120, 50), (50, 170), (220, 200), (190, 240), (140, 220)]
    image = inject_sources(image, faint_positions, peak_flux=0.10, fwhm_px=2.5)

    true_positions = bright_positions + medium_positions + faint_positions
    total_injected = len(true_positions)
    image = np.clip(image / image.max(), 0.0, 1.0).astype(np.float32)

    print("=== Wavelet vs Baseline Source Detection Demo ===\n")
    print(f"Field: {size}×{size} px | Noise σ={noise_level:.2f} | "
          f"Total injected: {total_injected} sources")
    print(f"  Bright  ({len(bright_positions)}) : peak flux 0.80 → SNR ~20")
    print(f"  Medium  ({len(medium_positions)}) : peak flux 0.25 → SNR  ~6")
    print(f"  Faint   ({len(faint_positions)}) : peak flux 0.10 → SNR  ~3\n")

    # Helper: count matches between detected and true positions
    def count_matches(detected: list, true_pos: list, tol_px: float = 8.0) -> int:
        matched = 0
        for dy, dx in detected:
            for ty, tx in true_pos:
                if abs(dy - ty) < tol_px and abs(dx - tx) < tol_px:
                    matched += 1
                    break
        return matched

    # ── Baseline detector ─────────────────────────────────────────────────
    baseline_detections = baseline_sigma_detect(image, nsigma=3.0)
    baseline_coords = [(d[0], d[1]) for d in baseline_detections]
    baseline_match = count_matches(baseline_coords, true_positions)

    print(f"Baseline sigma-threshold (3σ):")
    print(f"  Detections : {len(baseline_detections)}")
    print(f"  Matched    : {baseline_match} / {total_injected} "
          f"({100*baseline_match/total_injected:.0f}% completeness)\n")

    # ── Wavelet detector ──────────────────────────────────────────────────
    wavelet_det = WaveletSourceDetector(n_scales=4, sigma_threshold=5.0)
    wavelet_candidates = wavelet_det.detect(image)
    wavelet_coords = [(c.coordinates[0], c.coordinates[1]) for c in wavelet_candidates]
    wavelet_match = count_matches(wavelet_coords, true_positions)

    print(f"Wavelet (starlet à trous, 4 scales, 5.0σ):")
    print(f"  Detections : {len(wavelet_candidates)}")
    print(f"  Matched    : {wavelet_match} / {total_injected} "
          f"({100*wavelet_match/total_injected:.0f}% completeness)\n")

    # ── Matched filter detector ───────────────────────────────────────────
    mf_det = MatchedFilterDetector(psf_fwhm_range=(1.5, 4.0), n_scales=4,
                                    detection_threshold=15.0)
    mf_candidates = mf_det.detect(image)
    mf_coords = [(c.coordinates[0], c.coordinates[1]) for c in mf_candidates]
    mf_match = count_matches(mf_coords, true_positions)

    print(f"Matched filter (Gaussian PSF, 4 FWHM scales, SNR>15):")
    print(f"  Detections : {len(mf_candidates)}")
    print(f"  Matched    : {mf_match} / {total_injected} "
          f"({100*mf_match/total_injected:.0f}% completeness)\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print("─" * 55)
    print(f"{'Method':<30} {'Detections':>12} {'Completeness':>12}")
    print("─" * 55)
    for name, n_det, n_match in [
        ('Baseline sigma-threshold', len(baseline_detections), baseline_match),
        ('Wavelet (starlet, 5.0σ)', len(wavelet_candidates), wavelet_match),
        ('Matched filter (SNR>15)', len(mf_candidates), mf_match),
    ]:
        comp = 100 * n_match / total_injected
        print(f"{name:<30} {n_det:>12} {comp:>11.0f}%")
    print("─" * 55)

    # Top wavelet candidates
    if wavelet_candidates:
        print(f"\nTop 5 wavelet candidates by score:")
        for c in wavelet_candidates[:5]:
            print(f"  {c.id:<30} score={c.score:.3f}"
                  f" snr={c.metadata.get('snr',0):.1f}"
                  f" scale={c.metadata.get('scale_pixels',0)}px")


if __name__ == '__main__':
    main()
