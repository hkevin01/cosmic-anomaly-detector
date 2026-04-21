#!/usr/bin/env python3
"""
Example 01 — JWST Data Access via MAST

Demonstrates how to query the Mikulski Archive for Space Telescopes (MAST)
for real James Webb Space Telescope observations and download calibrated
science files, then run the full anomaly detection pipeline on them.

Usage
-----
    python scripts/examples/01_jwst_data_access.py

Requirements
------------
    pip install astroquery astropy scipy numpy

Notes
-----
    Downloads are cached in data/jwst_downloads/. Subsequent runs reuse
    existing files. MAST data access requires no authentication for public
    data, but rate-limiting applies.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from project root without installing package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    from cosmic_anomaly_detector.utils.jwst_access import (
        JWSTDataFetcher,
        JWST_NOTABLE_TARGETS,
    )
    from cosmic_anomaly_detector.core.analyzer import CosmicAnomalyAnalyzer
    from cosmic_anomaly_detector.processing.image_processor import ImageProcessor

    # ── Print notable targets ─────────────────────────────────────────────
    print("\n=== Notable JWST Targets for Anomaly Analysis ===\n")
    for name, info in JWST_NOTABLE_TARGETS.items():
        prog_ids = ', '.join(map(str, info.get('program_ids', [])))
        print(f"  {name}")
        print(f"    RA={info['ra']:.4f}°  Dec={info['dec']:.4f}°"
              f"  Program(s): {prog_ids}")
        print(f"    {info['description']}\n")

    # ── Choose target ─────────────────────────────────────────────────────
    target = "Stephan's Quintet"
    print(f"Searching MAST for: {target!r} (NIRCam, public data)\n")

    fetcher = JWSTDataFetcher(
        output_dir='data/jwst_downloads',
        max_products=3,
    )

    paths = fetcher.search_and_download(
        target_name=target,
        instrument='NIRCam',
        radius_arcmin=5.0,
        max_products=3,
    )

    if not paths:
        print("No files downloaded (network unavailable or no public data).")
        print("Running pipeline on a synthetic FITS image instead.\n")
        _run_synthetic_demo()
        return

    print(f"\nDownloaded {len(paths)} FITS file(s):")
    for p in paths:
        print(f"  {p}")

    # ── Run pipeline on first downloaded file ─────────────────────────────
    fits_path = paths[0]
    print(f"\nRunning anomaly pipeline on: {fits_path.name}\n")

    processor = ImageProcessor()
    result = processor.process_image(str(fits_path))

    analyzer = CosmicAnomalyAnalyzer()
    analysis = analyzer.analyze(result)

    print(f"Objects detected   : {len(result.get('detected_objects', []))}")
    print(f"Anomaly score      : {analysis.get('overall_anomaly_score', 0):.3f}")
    print(f"Confidence         : {analysis.get('confidence', 0):.3f}")
    detections = analysis.get('anomalous_objects', [])
    print(f"Flagged anomalies  : {len(detections)}")
    for det in detections[:5]:
        print(f"  [{det.get('anomaly_type','?')}] score={det.get('score',0):.2f}"
              f" coords={det.get('coordinates','?')}")


def _run_synthetic_demo() -> None:
    """Fall-back demo with a synthetic FITS image."""
    import numpy as np
    from astropy.io.fits import HDUList, PrimaryHDU

    print("Creating synthetic 256×256 NIRCam-like test image …")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((256, 256)).astype(np.float32) * 0.05

    # Inject a compact point source
    cx, cy = 128, 128
    y, x = np.ogrid[:256, :256]
    data += 1.2 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3 ** 2))

    # Inject a diffuse warm source (simulated IR excess)
    data += 0.3 * np.exp(-((x - 80) ** 2 + (y - 170) ** 2) / (2 * 20 ** 2))

    hdr = {'TELESCOP': 'JWST', 'INSTRUME': 'NIRCAM', 'FILTER': 'F200W',
           'EXPTIME': 1200.0, 'CRVAL1': 339.0, 'CRVAL2': 33.9}
    hdu = PrimaryHDU(data)
    for k, v in hdr.items():
        hdu.header[k] = v

    out = Path('data/jwst_downloads')
    out.mkdir(parents=True, exist_ok=True)
    fits_path = out / 'synthetic_demo.fits'
    HDUList([hdu]).writeto(str(fits_path), overwrite=True)
    print(f"Synthetic FITS written to {fits_path}\n")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from cosmic_anomaly_detector.processing.image_processor import ImageProcessor
    from cosmic_anomaly_detector.core.analyzer import CosmicAnomalyAnalyzer

    processor = ImageProcessor()
    result = processor.process_image(str(fits_path))

    analyzer = CosmicAnomalyAnalyzer()
    analysis = analyzer.analyze(result)

    print(f"Objects detected   : {len(result.get('detected_objects', []))}")
    print(f"Anomaly score      : {analysis.get('overall_anomaly_score', 0):.3f}")
    print(f"Confidence         : {analysis.get('confidence', 0):.3f}")
    detections = analysis.get('anomalous_objects', [])
    print(f"Flagged anomalies  : {len(detections)}")


if __name__ == '__main__':
    main()
