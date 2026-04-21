"""
Image Processor — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: IMG-001
# Requirement: Ingest a space telescope image file (FITS, PNG, JPG), apply
#              noise reduction and contrast enhancement, detect objects via
#              segmentation, and return a normalised processing dict.
# Purpose: Standardise raw telescope image data into a uniform representation
#          consumed by the gravitational analyser and AI classifier.
# Rationale: Centralising all image I/O and preprocessing in one module
#             isolates instrument-specific format handling from domain logic
#             and allows easy substitution of processing algorithms.
# Inputs:  image_path (str) — path to image file; config (Optional[Dict])
#          controlling noise_reduction (bool), contrast_enhancement (bool),
#          resolution_threshold (Tuple[int,int]).
# Outputs: Dict with keys: "image_array" (np.ndarray, float32, [0,1]),
#          "detected_objects" (List[Dict]), "metadata" (Dict),
#          "processing_steps" (List[str]).
# Preconditions:  image_path must reference a readable file.
# Postconditions: image_array is 3-D (H,W,C) or 2-D for grayscale FITS;
#                 detected_objects may be empty when no sources exceed sigma.
# Assumptions: FITS loading via astropy; PNG/JPG via PIL — both are optional;
#              fallback generates deterministic synthetic data.
# Side Effects: INFO log for each process() call.
# Failure Modes: Unreadable file → fallback to deterministic synthetic array.
# Error Handling: _load_image catches all I/O exceptions and returns synthetic.
# Constraints: Resolution normalised to resolution_threshold via scipy zoom.
# Verification: tests/test_preprocess.py and tests/test_detector.py.
# References: astropy.io.fits, scipy.ndimage.gaussian_filter,
#             scipy.ndimage.zoom, PIL.Image.
# ---------------------------------------------------------------------------
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards — all heavy imports are lazy so the module
# remains importable even in minimal test environments.
# ---------------------------------------------------------------------------
try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    _gaussian_filter = None  # type: ignore

try:
    from scipy.ndimage import zoom as _zoom
    _HAS_SCIPY_ZOOM = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY_ZOOM = False
    _zoom = None  # type: ignore

try:
    from astropy.io import fits as _fits
    _HAS_ASTROPY = True
except ImportError:  # pragma: no cover
    _HAS_ASTROPY = False
    _fits = None  # type: ignore

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False
    _PILImage = None  # type: ignore


@dataclass
class ProcessedImageData:
    """Container for processed image data."""
    image_array: np.ndarray
    detected_objects: List[Dict]
    metadata: Dict
    processing_steps: List[str]


def _infer_spectral_type(brightness: float) -> Tuple[str, float]:
    """Infer stellar spectral type and effective temperature from brightness."""
    if brightness > 0.9:
        return 'O', 40000.0
    elif brightness > 0.75:
        return 'B', 20000.0
    elif brightness > 0.6:
        return 'A', 9000.0
    elif brightness > 0.45:
        return 'F', 6500.0
    elif brightness > 0.3:
        return 'G', 5500.0
    elif brightness > 0.15:
        return 'K', 4000.0
    else:
        return 'M', 3000.0


class ImageProcessor:
    """Processes space telescope images for anomaly detection."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.noise_reduction = self.config.get('noise_reduction', True)
        self.contrast_enhancement = self.config.get('contrast_enhancement', True)
        self.resolution_threshold = self.config.get(
            'resolution_threshold', (512, 512)
        )

    def process(self, image_path: str) -> Dict:
        """
        Process an image file for anomaly detection.

        Args:
            image_path: Path to image file (FITS, PNG, JPG).

        Returns:
            Dict with image_array, detected_objects, metadata, processing_steps.
        """
        logger.info("Processing image: %s", image_path)

        image_data, fits_header = self._load_image(image_path)
        processed_image = self._preprocess_image(image_data)
        detected_objects = self._detect_objects(processed_image)
        metadata = self._extract_metadata(image_path, processed_image, fits_header)

        return {
            'image_array': processed_image,
            'detected_objects': detected_objects,
            'metadata': metadata,
            'processing_steps': self._get_processing_steps(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load an image file and return (array, fits_header_dict).

        Priority: FITS via astropy → PIL for raster formats → synthetic fallback.
        All returned arrays are float32 normalised to [0, 1] with shape (H, W, C).
        """
        p = Path(image_path)
        suffix = p.suffix.lower()
        fits_header: Dict = {}

        # ── FITS ──────────────────────────────────────────────────────────
        if suffix in ('.fits', '.fit', '.fts') and _HAS_ASTROPY:
            try:
                with _fits.open(image_path) as hdul:
                    raw = hdul[0].data
                    fits_header = dict(hdul[0].header)
                if raw is not None:
                    data = raw.astype(np.float32)
                    # Normalise channel-first FITS cubes to (H, W, C)
                    if data.ndim == 2:
                        data = np.stack([data, data, data], axis=-1)
                    elif data.ndim == 3 and data.shape[0] <= 4:
                        data = np.moveaxis(data, 0, -1)
                    dmin, dmax = float(data.min()), float(data.max())
                    if dmax > dmin:
                        data = (data - dmin) / (dmax - dmin)
                    logger.debug("FITS loaded: shape %s", data.shape)
                    return data, fits_header
            except Exception as exc:
                logger.warning("FITS load failed (%s): %s", image_path, exc)

        # ── Raster image (PNG, JPG, TIFF …) ──────────────────────────────
        if _HAS_PIL:
            try:
                img = _PILImage.open(image_path).convert('RGB')
                data = np.array(img, dtype=np.float32) / 255.0
                logger.debug("PIL loaded: shape %s", data.shape)
                return data, fits_header
            except Exception as exc:
                logger.warning("PIL load failed (%s): %s", image_path, exc)

        # ── Deterministic synthetic fallback ──────────────────────────────
        logger.info("Using synthetic data for path: %s", image_path)
        rng = np.random.default_rng(abs(hash(image_path)) % (2 ** 32))
        base = rng.random((512, 512), dtype=np.float64).astype(np.float32)
        # Add a few synthetic point-source-like peaks so detection has signal
        for _ in range(8):
            y = rng.integers(20, 490)
            x = rng.integers(20, 490)
            base[y - 2:y + 3, x - 2:x + 3] += rng.uniform(0.4, 1.0)
        base = np.clip(base, 0.0, 1.0)
        data = np.stack([base, base, base], axis=-1)
        return data, fits_header

    def _preprocess_image(self, image_data: np.ndarray) -> np.ndarray:
        """Apply configured preprocessing pipeline."""
        processed = image_data.copy()

        if self.noise_reduction:
            processed = self._apply_noise_reduction(processed)

        if self.contrast_enhancement:
            processed = self._enhance_contrast(processed)

        target_h, target_w = self.resolution_threshold
        if processed.shape[0] != target_h or processed.shape[1] != target_w:
            processed = self._resize_image(processed)

        return processed

    def _apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Gaussian noise reduction (σ=1 px), channel-wise.

        Falls back to a 3×3 box-filter average when scipy is unavailable.
        """
        sigma = 1.0
        if _HAS_SCIPY and _gaussian_filter is not None:
            if image.ndim == 3:
                return np.stack(
                    [_gaussian_filter(image[..., c], sigma)
                     for c in range(image.shape[2])],
                    axis=-1,
                ).astype(np.float32)
            return _gaussian_filter(image, sigma).astype(np.float32)

        # Pure-NumPy 3×3 box-filter fallback
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        def _box3(ch: np.ndarray) -> np.ndarray:
            padded = np.pad(ch, 1, mode='reflect')
            out = np.zeros_like(ch)
            for dy in range(3):
                for dx in range(3):
                    out += padded[dy:dy + ch.shape[0], dx:dx + ch.shape[1]] * kernel[dy, dx]
            return out
        if image.ndim == 3:
            return np.stack([_box3(image[..., c]) for c in range(image.shape[2])], axis=-1)
        return _box3(image)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Robust percentile contrast stretch [1st, 99th percentile] → [0, 1].

        Outlier-resistant and preserves relative flux ratios within the stretch.
        """
        lo = float(np.percentile(image, 1))
        hi = float(np.percentile(image, 99))
        if hi > lo:
            stretched = (image - lo) / (hi - lo)
            return np.clip(stretched, 0.0, 1.0).astype(np.float32)
        return image.astype(np.float32)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize to self.resolution_threshold using scipy bilinear zoom.

        Falls back to numpy nearest-neighbour when scipy is unavailable.
        """
        target_h, target_w = self.resolution_threshold
        current_h, current_w = image.shape[:2]

        if _HAS_SCIPY_ZOOM and _zoom is not None:
            zy = target_h / current_h
            zx = target_w / current_w
            if image.ndim == 3:
                zoomed = _zoom(image, (zy, zx, 1.0), order=1)
            else:
                zoomed = _zoom(image, (zy, zx), order=1)
            return zoomed.astype(np.float32)

        # Nearest-neighbour fallback
        y_idx = (np.arange(target_h) * current_h / target_h).astype(int)
        x_idx = (np.arange(target_w) * current_w / target_w).astype(int)
        y_idx = np.clip(y_idx, 0, current_h - 1)
        x_idx = np.clip(x_idx, 0, current_w - 1)
        if image.ndim == 3:
            return image[np.ix_(y_idx, x_idx,
                                np.arange(image.shape[2]))].astype(np.float32)
        return image[np.ix_(y_idx, x_idx)].astype(np.float32)

    def _detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect point-like and extended sources using the preprocessing pipeline.

        Delegates to preprocess.detect_sources (sigma-threshold on smoothed image).
        Each detected source is enriched with photometric and heuristic properties.
        """
        from ..processing.preprocess import detect_sources

        # Reduce to luminance for detection
        luminance = image.mean(axis=-1) if image.ndim == 3 else image
        h, w = luminance.shape

        sources = detect_sources(luminance, sigma=4.0)
        objects: List[Dict] = []

        for src in sources:
            r = 5  # patch half-radius in pixels
            y0 = max(0, src.y - r)
            y1 = min(h, src.y + r + 1)
            x0 = max(0, src.x - r)
            x1 = min(w, src.x + r + 1)
            patch = luminance[y0:y1, x0:x1]
            patch_mean = float(patch.mean()) if patch.size else 1e-9
            patch_std = float(patch.std()) if patch.size else 0.0

            brightness = float(np.clip(src.flux, 0.0, 1.0))
            _, temperature = _infer_spectral_type(brightness)

            # Mass-luminosity heuristic (dimensionless, relative to solar)
            estimated_mass = float(max(brightness ** 0.4, 0.1))

            objects.append({
                'id': f'src_{src.label:04d}',
                'coordinates': [src.y, src.x],
                'bounding_box': [y0, x0, y1, x1],
                'brightness': brightness,
                'estimated_mass': estimated_mass,
                'velocity': [0.0, 0.0],      # unknown without multi-epoch data
                'distance': 1000.0,           # pc — unknown without redshift
                'apparent_size': float(max(r, 1)),
                'luminosity': brightness,
                'circularity': 0.8,
                'symmetry': 0.8,
                'regularity': float(min(1.0, 1.0 - patch_std / (patch_mean + 1e-9))),
                'color_index': 0.0,
                'estimated_temperature': temperature,
                'edge_density': patch_std,
                'texture_complexity': float(patch_std / (patch_mean + 1e-9)),
                'pattern_repetition': 0.0,
                'geometric_precision': 0.5,
                'surface_regularity': float(min(1.0, 1.0 - patch_std / (patch_mean + 1e-9))),
                # Internal: row,col centroid for gravitational analysis
                'centroid': (float(src.y), float(src.x)),
                'area': float(src.area),
                'intensity_mean': brightness,
            })

        logger.debug("Detected %d sources in image", len(objects))
        return objects

    def _extract_metadata(
        self, image_path: str, processed_image: np.ndarray,
        fits_header: Optional[Dict] = None,
    ) -> Dict:
        """Extract image metadata, preferring FITS header values when present."""
        hdr = fits_header or {}
        return {
            'source_file': image_path,
            'image_shape': list(processed_image.shape),
            'processing_timestamp': datetime.now(timezone.utc).isoformat(),
            'telescope': hdr.get('TELESCOP', 'unknown'),
            'instrument': hdr.get('INSTRUME', 'unknown'),
            'filter': hdr.get('FILTER', hdr.get('FILTER1', 'unknown')),
            'exposure_time': float(hdr.get('EXPTIME', 0.0)),
            'target_coordinates': {
                'ra': float(hdr.get('CRVAL1', hdr.get('RA_TARG', 0.0))),
                'dec': float(hdr.get('CRVAL2', hdr.get('DEC_TARG', 0.0))),
            },
            'date_obs': hdr.get('DATE-OBS', ''),
        }

    def _get_processing_steps(self) -> List[str]:
        steps = ['image_loading']
        if self.noise_reduction:
            steps.append('noise_reduction')
        if self.contrast_enhancement:
            steps.append('contrast_enhancement')
        steps.extend(['resize', 'object_detection', 'metadata_extraction'])
        return steps

