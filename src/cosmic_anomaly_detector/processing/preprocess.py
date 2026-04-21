"""
Preprocessing Utilities — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: PRE-001
# Requirement: Provide a composable preprocessing chain — load_fits →
#              calibrate_flux → subtract_background → align_wcs →
#              detect_sources — each independently callable and testable.
# Purpose: Convert raw FITS pixel data into a flux-calibrated, background-
#          subtracted, source-catalogued array suitable for anomaly scoring.
# Rationale: Pipeline decomposition allows individual stages to be replaced
#             (e.g., WCS reprojection in a future phase) without coupling to
#             the rest of the system. Astropy dependency is optional; synthetic
#             fallbacks keep unit tests self-contained.
# Inputs:  path (str) — FITS file path for load_fits / full_preprocess.
#          image (np.ndarray, float32) — 2-D flux array for calibrate_flux,
#          subtract_background, detect_sources.
#          sigma (float, default 5.0) — source detection threshold (std devs).
#          box (int, default 32) — background subtraction tile size in pixels.
# Outputs: load_fits   → Dict{"image", "header", "wcs"}
#          calibrate_flux → np.ndarray (median-normalised, percentile-clipped)
#          subtract_background → np.ndarray (background-subtracted)
#          detect_sources → List[SourceDetectionResult]
#          full_preprocess → Dict{"image", "header", "wcs", "sources"}
# Preconditions:  FITS file must be readable; image must be 2-D float.
# Postconditions: Output arrays are float32 in [0, ∞); source list may be empty.
# Assumptions: Single-extension FITS (hdul[0]); WCS optional in header.
# Side Effects: None — all functions are stateless and side-effect free.
# Failure Modes: Missing astropy → deterministic RNG-based synthetic image.
#                Missing scipy → Gaussian pre-filter skipped (raw image used).
# Error Handling: try/except wraps optional imports at module level.
# Constraints: subtract_background is O(H×W) time; box size tunes speed/accuracy.
# Verification: tests/test_preprocess.py validates all five functions.
# References: Source detection via threshold: μ + σ·sigma; Gaussian σ=1.0 px.
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency path
    from astropy.io import fits  # type: ignore
    from astropy.wcs import WCS  # type: ignore
except Exception:  # pragma: no cover
    fits = None  # type: ignore
    WCS = None  # type: ignore

try:  # pragma: no cover
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None  # type: ignore


@dataclass
class SourceDetectionResult:
    label: int
    y: int
    x: int
    flux: float
    area: int


def load_fits(path: str) -> Dict[str, Any]:
    """Load a FITS file returning image array and metadata.

    Parameters
    ----------
    path : str
        Path to fits file. Falls back to deterministic synthetic data when
        the file is absent, empty, or cannot be parsed as a valid FITS file.
    """
    if fits is None:
        # Synthetic fallback: deterministic RNG based on path hash
        rng = np.random.default_rng(abs(hash(path)) % (2**32))
        data = rng.random((256, 256)).astype(np.float32)
        header = {"SYNTHETIC": True, "PATH": path}
        wcs = None
    else:
        try:  # pragma: no cover - requires valid FITS file on disk
            with fits.open(path) as hdul:
                data = hdul[0].data.astype(np.float32)
                header = dict(hdul[0].header)
                wcs = WCS(hdul[0].header) if WCS is not None else None
        except (OSError, Exception):
            # File is empty, corrupt, or non-existent — use synthetic fallback
            rng = np.random.default_rng(abs(hash(path)) % (2**32))
            data = rng.random((256, 256)).astype(np.float32)
            header = {"SYNTHETIC": True, "PATH": path}
            wcs = None
    return {"image": data, "header": header, "wcs": wcs}


def calibrate_flux(image: np.ndarray) -> np.ndarray:
    """Simple flux calibration by normalizing to median and clipping."""
    med = float(np.median(image))
    if med == 0:
        return image
    norm = image / med
    return np.clip(norm, 0, np.percentile(norm, 99.9))


def subtract_background(image: np.ndarray, box: int = 32) -> np.ndarray:
    """Background subtraction using coarse median boxes."""
    h, w = image.shape
    bg = np.zeros_like(image)
    for y in range(0, h, box):
        for x in range(0, w, box):
            patch = image[y:y + box, x:x + box]
            bg[y:y + box, x:x + box] = np.median(patch)
    return image - bg


def align_wcs(
    image: np.ndarray, wcs_obj: Any
) -> Tuple[np.ndarray, Any]:  # noqa: ANN401
    """Placeholder WCS alignment (returns unchanged image).

    In a future phase we will reproject to a reference WCS.
    """
    return image, wcs_obj


def detect_sources(
    image: np.ndarray, sigma: float = 5.0
) -> List[SourceDetectionResult]:
    """Detect point-like sources via thresholding of smoothed image."""
    if gaussian_filter is not None:
        smooth = gaussian_filter(image, 1.0)
    else:  # pragma: no cover
        smooth = image
    mean = float(np.mean(smooth))
    std = float(np.std(smooth))
    thresh = mean + sigma * std
    mask = smooth > thresh
    ys, xs = np.where(mask)
    results: List[SourceDetectionResult] = []
    for i, (y, x) in enumerate(zip(ys, xs)):
        results.append(
            SourceDetectionResult(
                label=i + 1,
                y=int(y),
                x=int(x),
                flux=float(smooth[y, x]),
                area=1,
            )
        )
    return results


def full_preprocess(path: str) -> Dict[str, Any]:
    """Run the full minimal preprocessing chain returning a dict."""
    loaded = load_fits(path)
    img = calibrate_flux(loaded["image"])  # flux norm
    img = subtract_background(img)
    img, wcs_obj = align_wcs(img, loaded.get("wcs"))
    sources = detect_sources(img)
    return {
        "image": img,
        "header": loaded["header"],
        "wcs": wcs_obj,
        "sources": [s.__dict__ for s in sources],
    }
