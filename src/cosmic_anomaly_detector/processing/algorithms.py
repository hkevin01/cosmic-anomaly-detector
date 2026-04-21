"""
Advanced Detection Algorithms — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: ALG-001
# Requirement: Provide four independent, physics-grounded detection algorithms
#   (1) Starlet wavelet multi-scale source detection
#   (2) Gaussian matched-filter point-source detection
#   (3) Infrared excess / SED fitting (Hephaistos-style Dyson-sphere search)
#   (4) Microlensing magnification anomaly detection
#   that can each produce scored candidate lists from a 2-D image array.
# Purpose: Increase detection sensitivity and diversity beyond the baseline
#   sigma-thresholding in preprocess.py; each algorithm is sensitive to a
#   different physical signature of anomalous structures.
# Rationale:
#   - Wavelet (starlet) decomposition suppresses noise at the source scale
#     while preserving flux from genuine point sources — standard in X-ray /
#     optical astronomy (Starck & Murtagh, 2002; Bertin 1996).
#   - Matched filtering is the statistically optimal linear detector for a
#     known PSF in stationary Gaussian noise (Turin, 1960).
#   - IR-excess SED fitting follows Project Hephaistos (Suazo et al. 2024,
#     MNRAS 531, 695): model star + Dyson-sphere blackbody, grid-search γ and
#     T_DS, flag objects with RMSE < 0.2 mag.
#   - Microlensing magnification anomaly flags sources whose point-spread flux
#     exceeds the expected single-point-lens magnification curve, hinting at
#     dark extended mass concentrations (Hsiao et al. 2021; arxiv:2512.07924).
# Inputs:  2-D numpy array (float32, [0,1]) representing a normalised image.
# Outputs: List[Dict] of candidate objects with score, position, and algorithm
#          tag fields consistent with the detected_objects schema used by
#          ImageProcessor and AnomalyDetector.
# Preconditions:  Image must be 2-D (H,W); caller flattens colour channels.
# Postconditions: Each returned dict contains at minimum: id, coordinates,
#   bounding_box, brightness, score, algorithm, anomaly_type.
# Assumptions: scipy.ndimage and numpy always available; astropy optional for
#   physical unit conversions.
# Side Effects: INFO logging per algorithm call.
# Failure Modes: Empty image → returns []. Algorithm exception → logged, [].
# Error Handling: Each public method catches and logs exceptions.
# Constraints: Pure CPU; no GPU required. Wavelet levels capped at 5.
# Verification: scripts/examples/ integration tests demonstrate each algorithm.
# References:
#   Suazo et al. 2024 — arXiv:2405.02927 (Project Hephaistos II)
#   Starck & Murtagh 2002 — "Astronomical Image and Data Analysis"
#   Turin 1960 — IRE Trans. Info. Theory, 6(3):311-329
#   arxiv:2512.07924 — Microlensing of Dyson sphere-like structures
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmCandidate:
    """
    Unified candidate output produced by any detection algorithm.

    Fields mirror the detected_objects schema used throughout the pipeline so
    candidates can be merged without downstream schema changes.
    """
    id: str
    algorithm: str
    anomaly_type: str
    coordinates: List[float]          # [row, col] pixel centre
    bounding_box: List[int]           # [r0, c0, r1, c1]
    brightness: float                 # [0, 1] normalised peak flux
    score: float                      # algorithm-specific anomaly score [0, 1]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'algorithm': self.algorithm,
            'anomaly_type': self.anomaly_type,
            'coordinates': self.coordinates,
            'bounding_box': self.bounding_box,
            'brightness': self.brightness,
            'score': self.score,
            # populate fields expected by ImageProcessor / Classifier
            'estimated_mass': float(max(self.brightness ** 0.4, 0.1)),
            'velocity': [0.0, 0.0],
            'distance': 1000.0,
            'apparent_size': 5.0,
            'luminosity': self.brightness,
            'circularity': 0.8,
            'symmetry': 0.8,
            'regularity': self.score,
            'color_index': 0.0,
            'estimated_temperature': 5000.0,
            'edge_density': 0.3,
            'texture_complexity': 0.3,
            'pattern_repetition': 0.0,
            'geometric_precision': 0.5,
            'surface_regularity': 0.5,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# 1. Starlet (Isotropic Undecimated Wavelet) Source Detector
# ---------------------------------------------------------------------------

class WaveletSourceDetector:
    """
    Multi-scale source detection using the starlet (isotropic undecimated
    wavelet) transform.

    The à trous algorithm convolves the image with the B3-spline scaling
    function at each dyadic scale j: 1, 2, 4, 8, … pixels.  Sources appear as
    local maxima in individual wavelet planes above a sigma-clipped noise
    threshold.  This approach is robust against extended background and
    naturally handles the multi-scale nature of astronomical sources.

    Reference: Starck & Murtagh (2002), "Astronomical Image and Data Analysis"
    """

    def __init__(
        self,
        n_scales: int = 4,
        sigma_threshold: float = 3.5,
        min_peak_separation: int = 5,
    ) -> None:
        self.n_scales = min(n_scales, 5)  # cap at 5 to avoid edge artifacts
        self.sigma_threshold = sigma_threshold
        self.min_peak_separation = min_peak_separation

    def detect(self, image: np.ndarray) -> List[AlgorithmCandidate]:
        """
        Detect point-like and compact sources in a 2-D normalised image.

        Args:
            image: 2-D float32 array normalised to [0, 1].

        Returns:
            List of AlgorithmCandidate objects sorted by descending score.
        """
        try:
            return self._detect(image)
        except Exception as exc:
            logger.error("WaveletSourceDetector failed: %s", exc)
            return []

    def _detect(self, image: np.ndarray) -> List[AlgorithmCandidate]:
        # ── Build starlet planes ──────────────────────────────────────────
        planes = self._starlet_transform(image.astype(np.float64))

        candidates: List[AlgorithmCandidate] = []
        seen_positions: List[Tuple[int, int]] = []

        for j, plane in enumerate(planes):
            scale_px = 2 ** j          # characteristic scale in pixels
            noise = float(np.std(plane))
            if noise == 0:
                continue
            threshold = self.sigma_threshold * noise

            # Local maxima above threshold
            footprint_size = max(3, scale_px)
            local_max = maximum_filter(plane, size=footprint_size)
            mask = (plane == local_max) & (plane > threshold)
            rows, cols = np.where(mask)

            for r, c in zip(rows, cols):
                # Deduplicate by proximity
                too_close = any(
                    abs(r - pr) + abs(c - pc) < self.min_peak_separation
                    for pr, pc in seen_positions
                )
                if too_close:
                    continue
                seen_positions.append((r, c))

                h, w = image.shape
                hw = max(scale_px, 3)
                r0 = max(0, r - hw)
                r1 = min(h, r + hw + 1)
                c0 = max(0, c - hw)
                c1 = min(w, c + hw + 1)

                peak_flux = float(np.clip(image[r, c], 0.0, 1.0))
                snr = float(plane[r, c] / noise)
                score = float(min(1.0, (snr - self.sigma_threshold) /
                                  (self.sigma_threshold * 3)))

                cand = AlgorithmCandidate(
                    id=f'wav_s{j}_{r}_{c}',
                    algorithm='wavelet_starlet',
                    anomaly_type='point_source',
                    coordinates=[float(r), float(c)],
                    bounding_box=[r0, c0, r1, c1],
                    brightness=peak_flux,
                    score=max(0.0, score),
                    metadata={'scale_pixels': scale_px, 'snr': snr,
                              'wavelet_plane': j},
                )
                candidates.append(cand)

        candidates.sort(key=lambda x: x.score, reverse=True)
        logger.debug("Wavelet detector found %d candidates", len(candidates))
        return candidates

    @staticmethod
    def _b3_spline_kernel() -> np.ndarray:
        """1-D B3-spline coefficients [1, 4, 6, 4, 1] / 16."""
        return np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0

    def _starlet_transform(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Compute the starlet transform planes using the à trous algorithm.

        Returns n_scales wavelet planes; the last element is the coarse
        approximation (not a wavelet plane, excluded from detection).
        """
        from scipy.ndimage import convolve1d

        kernel_1d = self._b3_spline_kernel()
        current = image.copy()
        planes: List[np.ndarray] = []

        for j in range(self.n_scales):
            step = 2 ** j
            # Insert zeros between kernel coefficients (à trous)
            dilated = np.zeros(1 + step * (len(kernel_1d) - 1), dtype=np.float64)
            dilated[::step] = kernel_1d

            # Separable 2-D convolution via two 1-D passes
            smooth = convolve1d(current, dilated, axis=0, mode='mirror')
            smooth = convolve1d(smooth, dilated, axis=1, mode='mirror')

            planes.append(current - smooth)   # wavelet plane
            current = smooth

        return planes   # excludes final approximation


# ---------------------------------------------------------------------------
# 2. Matched-Filter Point-Source Detector
# ---------------------------------------------------------------------------

class MatchedFilterDetector:
    """
    Statistically optimal linear detector for point-like sources with a known
    (or assumed Gaussian) PSF in approximately stationary Gaussian noise.

    The matched filter maximises SNR for a known signal shape in additive
    white Gaussian noise.  The filter is applied at multiple PSF scales to
    handle sources with different apparent sizes.

    Reference: Turin (1960), IRE Trans. Info. Theory 6(3):311-329.
    """

    def __init__(
        self,
        psf_fwhm_range: Tuple[float, float] = (1.5, 5.0),
        n_scales: int = 4,
        detection_threshold: float = 5.0,   # SNR units
    ) -> None:
        self.psf_fwhm_range = psf_fwhm_range
        self.n_scales = n_scales
        self.detection_threshold = detection_threshold

    def detect(self, image: np.ndarray) -> List[AlgorithmCandidate]:
        try:
            return self._detect(image)
        except Exception as exc:
            logger.error("MatchedFilterDetector failed: %s", exc)
            return []

    def _detect(self, image: np.ndarray) -> List[AlgorithmCandidate]:
        fwhm_lo, fwhm_hi = self.psf_fwhm_range
        fwhm_values = np.linspace(fwhm_lo, fwhm_hi, self.n_scales)
        sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))

        # Estimate noise from MAD of high-pass residual
        background = gaussian_filter(image.astype(np.float64), sigma=10.0)
        residual = image.astype(np.float64) - background
        noise_sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        if noise_sigma < 1e-10:
            noise_sigma = 1e-10

        candidates: List[AlgorithmCandidate] = []
        seen: List[Tuple[int, int]] = []

        for fwhm in fwhm_values:
            sigma = fwhm / sigma_to_fwhm
            filtered = gaussian_filter(residual, sigma=sigma)
            snr_map = filtered / noise_sigma

            threshold_map = snr_map > self.detection_threshold
            labelled, n_obj = label(threshold_map)

            for obj_id in range(1, n_obj + 1):
                obj_mask = labelled == obj_id
                obj_rows, obj_cols = np.where(obj_mask)
                if len(obj_rows) == 0:
                    continue

                # Flux-weighted centroid
                fluxes = image[obj_rows, obj_cols]
                total_flux = float(fluxes.sum())
                if total_flux <= 0:
                    continue
                r_cen = float(np.average(obj_rows, weights=fluxes))
                c_cen = float(np.average(obj_cols, weights=fluxes))
                r_i = int(round(r_cen))
                c_i = int(round(c_cen))

                # Deduplicate
                too_close = any(
                    abs(r_i - pr) + abs(c_i - pc) < max(3, int(fwhm * 2))
                    for pr, pc in seen
                )
                if too_close:
                    continue
                seen.append((r_i, c_i))

                h, w = image.shape
                hw = max(int(fwhm * 1.5), 3)
                r0 = max(0, r_i - hw)
                r1 = min(h, r_i + hw + 1)
                c0 = max(0, c_i - hw)
                c1 = min(w, c_i + hw + 1)

                peak_snr = float(snr_map[r_i, c_i]) if 0 <= r_i < h and 0 <= c_i < w else float(np.max(snr_map[obj_mask]))
                peak_flux = float(np.clip(
                    image[r_i, c_i] if 0 <= r_i < h and 0 <= c_i < w else total_flux / len(obj_rows),
                    0.0, 1.0
                ))
                score = float(min(1.0, (peak_snr - self.detection_threshold) /
                                  (self.detection_threshold * 5)))

                candidates.append(AlgorithmCandidate(
                    id=f'mf_fwhm{fwhm:.1f}_{r_i}_{c_i}',
                    algorithm='matched_filter',
                    anomaly_type='point_source',
                    coordinates=[r_cen, c_cen],
                    bounding_box=[r0, c0, r1, c1],
                    brightness=peak_flux,
                    score=max(0.0, score),
                    metadata={'fwhm_pixels': float(fwhm), 'snr': peak_snr},
                ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        logger.debug("MatchedFilter found %d candidates", len(candidates))
        return candidates


# ---------------------------------------------------------------------------
# 3. Infrared Excess / SED Fitting (Hephaistos-style)
# ---------------------------------------------------------------------------

class IRExcessDetector:
    """
    Detect anomalous infrared excess consistent with Dyson-sphere waste heat.

    Implements the Suazo et al. (2024) MNRAS 531, 695 (Project Hephaistos II)
    methodology:

      1. Model each detected source as star + Dyson sphere (DS) blackbody.
      2. Combined magnitude: M = -2.5·log10(10^{-M_star/2.5}+10^{-M_DS/2.5})
      3. DS spectrum: M_DS = -2.5·log10(γ × BB(T_DS) / BB_ref)  (relative)
      4. Grid-search covering factor γ ∈ [0.01, 0.9] and
                         DS temperature T_DS ∈ [100, 700] K.
      5. Flag sources whose best-fit RMSE < rmse_threshold and γ > min_gamma.

    The image does not directly provide multi-band photometry, so IR excess is
    inferred from the spatial flux distribution: sources that are significantly
    brighter in a smoothed (low-frequency) channel relative to a high-pass
    channel exhibit the flat/extended SED shape characteristic of warm dust or
    waste-heat re-emission.

    Reference: arXiv:2405.02927 — Project Hephaistos II (Suazo et al. 2024)
    """

    # Planck function constant ratio (hc/k_B in µm·K)
    _HC_OVER_KB = 14387.77   # µm·K

    def __init__(
        self,
        ds_temp_range: Tuple[float, float] = (100.0, 700.0),
        ds_temp_steps: int = 13,
        gamma_range: Tuple[float, float] = (0.01, 0.9),
        gamma_steps: int = 18,
        rmse_threshold: float = 0.20,
        min_gamma: float = 0.05,
        reference_wavelength_um: float = 2.2,   # K-band (2MASS Ks)
        ir_wavelength_um: float = 12.0,          # WISE W3
    ) -> None:
        self.ds_temps = np.linspace(ds_temp_range[0], ds_temp_range[1],
                                    ds_temp_steps)
        self.gammas = np.linspace(gamma_range[0], gamma_range[1], gamma_steps)
        self.rmse_threshold = rmse_threshold
        self.min_gamma = min_gamma
        self.ref_wl = reference_wavelength_um
        self.ir_wl = ir_wavelength_um

    def detect(self, image: np.ndarray,
               detected_objects: Optional[List[Dict]] = None
               ) -> List[AlgorithmCandidate]:
        """
        Score each detected object for IR excess anomaly.

        Args:
            image: 2-D normalised float32 image.
            detected_objects: List of object dicts from ImageProcessor.

        Returns:
            Subset of objects with an IR-excess score ≥ min_gamma,
            as AlgorithmCandidate objects.
        """
        try:
            return self._detect(image, detected_objects or [])
        except Exception as exc:
            logger.error("IRExcessDetector failed: %s", exc)
            return []

    def _planck_ratio(self, wavelength_um: float, temperature_k: float) -> float:
        """
        Relative Planck blackbody intensity (arbitrary units, ν-space).
        B_ν ∝ ν³ / (exp(hν/kT)-1) ∝ λ^{-5} / (exp(hc/λkT)-1)
        """
        exponent = self._HC_OVER_KB / (wavelength_um * temperature_k)
        if exponent > 700:
            return 0.0
        return wavelength_um ** (-5) / (np.exp(exponent) - 1.0 + 1e-300)

    def _ds_model_flux_ratio(self, gamma: float, t_ds: float) -> float:
        """
        Fractional IR-to-optical flux ratio produced by a DS at temperature t_ds.

        Returns: (F_IR_DS) / (F_optical_star_before_obscuration)  [dimensionless]
        """
        bb_opt = self._planck_ratio(self.ref_wl, t_ds)
        bb_ir = self._planck_ratio(self.ir_wl, t_ds)
        if bb_opt <= 0:
            return 0.0
        return gamma * bb_ir / (bb_opt + 1e-300)

    def _detect(self, image: np.ndarray,
                detected_objects: List[Dict]) -> List[AlgorithmCandidate]:
        # Estimate optical (high-pass) and IR-proxy (low-pass) channels
        optical_channel = gaussian_filter(image.astype(np.float64), sigma=1.0)
        ir_channel = gaussian_filter(image.astype(np.float64), sigma=8.0)

        candidates: List[AlgorithmCandidate] = []

        for idx, obj in enumerate(detected_objects):
            coords = obj.get('coordinates', [0.0, 0.0])
            r = int(round(float(coords[0] if len(coords) > 0 else 0)))
            c = int(round(float(coords[1] if len(coords) > 1 else 0)))
            h, w = image.shape
            r = max(0, min(r, h - 1))
            c = max(0, min(c, w - 1))

            f_opt = float(np.clip(optical_channel[r, c], 1e-9, None))
            f_ir = float(np.clip(ir_channel[r, c], 1e-9, None))
            observed_ratio = f_ir / f_opt

            # ── Grid search over (gamma, T_DS) ───────────────────────────
            best_rmse = 1e9
            best_gamma = 0.0
            best_temp = 0.0

            for gamma in self.gammas:
                for t_ds in self.ds_temps:
                    predicted_ratio = self._ds_model_flux_ratio(gamma, t_ds)
                    if predicted_ratio <= 0:
                        continue
                    # RMSE in log-flux space (equivalent to RMSE in magnitudes)
                    rmse = abs(
                        np.log10(observed_ratio + 1e-9) -
                        np.log10(predicted_ratio + 1e-9)
                    )
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_gamma = gamma
                        best_temp = t_ds

            # ── Filter by quality criteria ────────────────────────────────
            if best_rmse > self.rmse_threshold or best_gamma < self.min_gamma:
                continue

            # Score: high γ and low RMSE → high score
            score = float(best_gamma * (1.0 - best_rmse / self.rmse_threshold))
            score = float(min(1.0, max(0.0, score)))

            bb = obj.get('bounding_box', [max(0, r-5), max(0, c-5),
                                           min(h, r+6), min(w, c+6)])

            candidates.append(AlgorithmCandidate(
                id=f'ir_excess_{idx}_{r}_{c}',
                algorithm='ir_excess_sed',
                anomaly_type='ir_excess_dyson_sphere',
                coordinates=[float(r), float(c)],
                bounding_box=bb,
                brightness=float(np.clip(image[r, c], 0.0, 1.0)),
                score=score,
                metadata={
                    'covering_factor_gamma': best_gamma,
                    'ds_temperature_k': best_temp,
                    'sed_rmse': best_rmse,
                    'ir_to_optical_ratio': observed_ratio,
                    'reference': 'Suazo+2024 MNRAS 531 695',
                },
            ))

        logger.debug("IRExcess detector flagged %d candidates", len(candidates))
        return candidates


# ---------------------------------------------------------------------------
# 4. Microlensing Magnification Anomaly Detector
# ---------------------------------------------------------------------------

class MicrolensingAnomalyDetector:
    """
    Flag sources whose flux profile is inconsistent with standard point-source
    microlensing, suggesting an extended or anomalous mass distribution
    consistent with a Dyson sphere or megastructure lens.

    Standard point-source Paczyński (1986) magnification curve:
        A(u) = (u² + 2) / (u · √(u² + 4))
    where u = (angular separation) / (Einstein radius θ_E).

    An anomaly is declared when a source's peak flux significantly exceeds
    the maximum magnification predicted for the estimated impact parameter,
    or when the flux profile is spatially extended in a way inconsistent with
    a pure point lens.

    Reference: arXiv:2512.07924 (microlensing of Dyson sphere-like structures)
    """

    def __init__(
        self,
        einstein_radius_px: float = 8.0,
        magnification_threshold: float = 1.3,
        min_source_flux: float = 0.05,
    ) -> None:
        self.einstein_radius_px = einstein_radius_px
        self.magnification_threshold = magnification_threshold
        self.min_source_flux = min_source_flux

    @staticmethod
    def _paczynski_magnification(u: float) -> float:
        """
        Point-source Paczyński magnification at impact parameter u.

        u: dimensionless separation in Einstein radius units.
        """
        if u <= 0:
            return 1e6   # caustic limit
        u2 = u * u
        return (u2 + 2.0) / (u * np.sqrt(u2 + 4.0))

    def detect(self, image: np.ndarray,
               detected_objects: Optional[List[Dict]] = None
               ) -> List[AlgorithmCandidate]:
        try:
            return self._detect(image, detected_objects or [])
        except Exception as exc:
            logger.error("MicrolensingAnomalyDetector failed: %s", exc)
            return []

    def _detect(self, image: np.ndarray,
                detected_objects: List[Dict]) -> List[AlgorithmCandidate]:
        h, w = image.shape
        background = float(np.median(image))
        candidates: List[AlgorithmCandidate] = []

        for idx, obj in enumerate(detected_objects):
            coords = obj.get('coordinates', [0.0, 0.0])
            r = int(round(float(coords[0] if len(coords) > 0 else 0)))
            c = int(round(float(coords[1] if len(coords) > 1 else 0)))
            r = max(0, min(r, h - 1))
            c = max(0, min(c, w - 1))

            peak_flux = float(image[r, c])
            if peak_flux < self.min_source_flux:
                continue

            # ── Radial flux profile ───────────────────────────────────────
            max_r = int(self.einstein_radius_px * 2)
            radii: List[float] = []
            fluxes: List[float] = []
            for dr in range(-max_r, max_r + 1):
                for dc in range(-max_r, max_r + 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        dist = np.sqrt(dr * dr + dc * dc)
                        radii.append(dist)
                        fluxes.append(float(image[rr, cc]))

            if len(radii) < 5:
                continue

            # Estimate impact parameter as the half-maximum radius
            sorted_pairs = sorted(zip(radii, fluxes))
            half_max = (peak_flux + background) / 2.0
            # Find radius where flux drops to half-max
            u_estimated = self.einstein_radius_px  # default
            for dist, flux in sorted_pairs:
                if flux < half_max and dist > 0:
                    u_estimated = dist / self.einstein_radius_px
                    break

            # Expected magnification for this impact parameter
            a_expected = self._paczynski_magnification(max(u_estimated, 0.01))
            # Observed "magnification" relative to background
            if background > 1e-9:
                a_observed = peak_flux / background
            else:
                a_observed = 1.0

            # Anomaly: observed magnification significantly exceeds Paczyński
            anomaly_ratio = a_observed / max(a_expected, 1.0)
            if anomaly_ratio < self.magnification_threshold:
                continue

            score = float(min(1.0, (anomaly_ratio - self.magnification_threshold) /
                              (self.magnification_threshold * 2)))

            bb = [max(0, r - max_r), max(0, c - max_r),
                  min(h, r + max_r + 1), min(w, c + max_r + 1)]

            candidates.append(AlgorithmCandidate(
                id=f'uL_{idx}_{r}_{c}',
                algorithm='microlensing_anomaly',
                anomaly_type='magnification_anomaly',
                coordinates=[float(r), float(c)],
                bounding_box=bb,
                brightness=float(np.clip(peak_flux, 0.0, 1.0)),
                score=score,
                metadata={
                    'a_observed': a_observed,
                    'a_expected_paczynski': a_expected,
                    'impact_parameter_u': u_estimated,
                    'anomaly_ratio': anomaly_ratio,
                    'reference': 'Paczynski1986; arXiv:2512.07924',
                },
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        logger.debug("Microlensing detector flagged %d candidates",
                     len(candidates))
        return candidates


# ---------------------------------------------------------------------------
# 5. Multi-Band IR Excess / SED Fitting (genuine photometric)
# ---------------------------------------------------------------------------

class MultiBandIRExcessDetector:
    """
    Genuine multi-band SED fitting for IR excess / Dyson sphere candidates.

    Unlike the single-band spatial-scale proxy (σ=1 vs σ=8 channel ratio),
    this class uses real inter-band photometric ratios measured across
    independent JWST filter images.  For each detected source an aperture
    flux is extracted in every supplied band, then the following model is
    simultaneously fitted to the full SED:

        F_model(λ) ∝ (1 − γ) · B_ν(T_star, λ)
                    +      γ  · B_ν(T_DS,  λ)

    The stellar blackbody temperature T_star, Dyson-sphere temperature T_DS,
    and covering factor γ are found by grid-search minimising RMSE in
    log₁₀-flux space (equivalent to RMSE in magnitudes — Suazo et al. 2024).

    A source is flagged when the best-fit RMSE < rmse_threshold AND γ > min_gamma.

    Usage
    -----
    >>> bands = {1.5: image_f150w, 2.0: image_f200w, 3.6: image_f356w}
    >>> det = MultiBandIRExcessDetector(bands)
    >>> candidates = det.detect(detected_objects)

    Reference: Suazo et al. 2024, MNRAS 531, 695 (Project Hephaistos II)
               arXiv:2405.02927
    """

    _HC_OVER_KB = 14387.77   # hc / k_B in µm·K

    def __init__(
        self,
        bands: Dict[float, np.ndarray],
        aperture_radius_px: int = 3,
        ds_temp_range: Tuple[float, float] = (100.0, 700.0),
        ds_temp_steps: int = 20,
        star_temp_range: Tuple[float, float] = (3000.0, 30000.0),
        star_temp_steps: int = 15,
        gamma_range: Tuple[float, float] = (0.01, 0.90),
        gamma_steps: int = 20,
        rmse_threshold: float = 0.15,
        min_gamma: float = 0.05,
        min_bands: int = 2,
    ) -> None:
        if len(bands) < min_bands:
            raise ValueError(
                f"MultiBandIRExcessDetector requires ≥{min_bands} bands, "
                f"got {len(bands)}."
            )
        self.bands = {float(wl): arr for wl, arr in bands.items()}
        self.wavelengths = sorted(self.bands)
        self.aperture_radius = aperture_radius_px
        self.ds_temps = np.linspace(ds_temp_range[0], ds_temp_range[1], ds_temp_steps)
        self.star_temps = np.linspace(star_temp_range[0], star_temp_range[1], star_temp_steps)
        self.gammas = np.linspace(gamma_range[0], gamma_range[1], gamma_steps)
        self.rmse_threshold = rmse_threshold
        self.min_gamma = min_gamma

    # ── Physics ────────────────────────────────────────────────────────────

    def _planck_bnu(self, wavelength_um: float, temperature_k: float) -> float:
        """B_ν ∝ ν³/(exp(hν/kT)−1) evaluated at wavelength_um."""
        if temperature_k <= 0:
            return 0.0
        x = self._HC_OVER_KB / (wavelength_um * temperature_k)
        if x > 700.0:
            return 0.0
        return wavelength_um ** (-5.0) / (np.expm1(x) + 1e-300)

    def _model_sed(
        self, gamma: float, t_star: float, t_ds: float
    ) -> np.ndarray:
        """
        Compute the normalised model SED at all band wavelengths.

        Returns log10 of the model flux vector (length = number of bands),
        normalised so that the brightest band = 0.
        """
        fluxes = np.array([
            (1.0 - gamma) * self._planck_bnu(wl, t_star)
            + gamma * self._planck_bnu(wl, t_ds)
            for wl in self.wavelengths
        ], dtype=np.float64)
        if fluxes.max() <= 0:
            return np.full(len(self.wavelengths), -30.0)
        fluxes = np.clip(fluxes, 1e-300, None)
        log_f = np.log10(fluxes)
        return log_f - log_f.max()   # normalise peak to 0

    # ── Photometry ─────────────────────────────────────────────────────────

    def _aperture_flux(self, image: np.ndarray, r: int, c: int) -> float:
        """Sum flux in a circular aperture of radius self.aperture_radius."""
        h, w = image.shape
        rad = self.aperture_radius
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(w, c + rad + 1)
        patch = image[r0:r1, c0:c1]
        rows_p, cols_p = np.meshgrid(
            np.arange(r0, r1) - r, np.arange(c0, c1) - c, indexing='ij'
        )
        circle = (rows_p ** 2 + cols_p ** 2) <= rad ** 2
        vals = patch[circle]
        return float(vals.sum()) if len(vals) else 0.0

    # ── Detection ──────────────────────────────────────────────────────────

    def detect(
        self,
        detected_objects: Optional[List[Dict]] = None,
    ) -> List[AlgorithmCandidate]:
        """
        Score each detected object against the multi-band DS SED model.

        Args:
            detected_objects: list of object dicts with 'coordinates' field.

        Returns:
            AlgorithmCandidate list for objects with a significant IR excess.
        """
        try:
            return self._detect(detected_objects or [])
        except Exception as exc:
            logger.error("MultiBandIRExcessDetector failed: %s", exc)
            return []

    def _detect(self, detected_objects: List[Dict]) -> List[AlgorithmCandidate]:
        # Reference image shape from first band
        ref_image = self.bands[self.wavelengths[0]]
        h, w = ref_image.shape

        candidates: List[AlgorithmCandidate] = []

        for idx, obj in enumerate(detected_objects):
            coords = obj.get('coordinates', [0.0, 0.0])
            r = int(round(float(coords[0])))
            c = int(round(float(coords[1])))
            r = max(0, min(r, h - 1))
            c = max(0, min(c, w - 1))

            # ── Extract multi-band aperture photometry ────────────────────
            obs_fluxes = np.array([
                self._aperture_flux(self.bands[wl], r, c)
                for wl in self.wavelengths
            ], dtype=np.float64)

            if obs_fluxes.max() <= 0:
                continue
            obs_fluxes = np.clip(obs_fluxes, 1e-300, None)
            log_obs = np.log10(obs_fluxes)
            log_obs -= log_obs.max()    # normalise peak to 0

            # ── Grid search (gamma, T_star, T_DS) ────────────────────────
            best_rmse = 1e9
            best_gamma = 0.0
            best_t_star = 5778.0
            best_t_ds = 300.0

            for gamma in self.gammas:
                for t_star in self.star_temps:
                    for t_ds in self.ds_temps:
                        log_model = self._model_sed(gamma, t_star, t_ds)
                        rmse = float(np.sqrt(np.mean((log_obs - log_model) ** 2)))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_gamma = gamma
                            best_t_star = t_star
                            best_t_ds = t_ds

            if best_rmse > self.rmse_threshold or best_gamma < self.min_gamma:
                continue

            score = float(min(1.0, best_gamma * (1.0 - best_rmse / self.rmse_threshold)))

            bb = obj.get('bounding_box', [
                max(0, r - 5), max(0, c - 5),
                min(h, r + 6), min(w, c + 6),
            ])
            peak_flux = float(np.clip(ref_image[r, c], 0.0, 1.0))

            candidates.append(AlgorithmCandidate(
                id=f'mb_ir_{idx}_{r}_{c}',
                algorithm='multiband_ir_excess',
                anomaly_type='ir_excess_dyson_sphere',
                coordinates=[float(r), float(c)],
                bounding_box=bb,
                brightness=peak_flux,
                score=score,
                metadata={
                    'covering_factor_gamma': best_gamma,
                    'ds_temperature_k': best_t_ds,
                    'star_temperature_k': best_t_star,
                    'sed_rmse_mag': best_rmse,
                    'n_bands': len(self.wavelengths),
                    'band_wavelengths_um': self.wavelengths,
                    'reference': 'Suazo+2024 MNRAS 531 695; arXiv:2405.02927',
                },
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        logger.debug("MultiBandIRExcess flagged %d candidates", len(candidates))
        return candidates


# ---------------------------------------------------------------------------
# 6. Multi-Epoch Microlensing Light-Curve Detector
# ---------------------------------------------------------------------------

class MultiEpochMicrolensingDetector:
    """
    Fit the standard Paczyński (1986) single-lens light curve to multi-epoch
    photometry and flag events that are anomalous (extended source, binary
    lens, or magnification excess — consistent with a Dyson sphere or
    megastructure lens).

    For each epoch a calibrated image is provided alongside its Julian date.
    The algorithm:

      1. Extracts aperture photometry at each source position in every epoch.
      2. Builds a light curve F(t) and tests for variability (χ²/dof vs flat).
      3. Grid-searches Paczyński parameters:
           u(t) = √(u₀² + ((t − t₀)/tE)²)
           A(t) = (u² + 2) / (u √(u² + 4))
           F_model(t) = f_s · A(t) + f_b
         over (t₀, tE, u₀) with f_s, f_b fitted analytically per grid point.
      4. Computes:
           chi2_flat:   χ²/dof of constant-flux model  (detects variability)
           chi2_paczynski: χ²/dof of best-fit Paczyński (detects lensing fit)
           delta_chi2:  chi2_flat − chi2_paczynski  (measures lensing signal)
      5. Anomaly declarations:
           A.  chi2_paczynski < chi2_threshold AND u₀ < u0_anomaly_threshold:
               source is *consistent* with lensing of a very compact object
               (potential megastructure lens).
           B.  chi2_flat > variability_threshold AND
               chi2_paczynski > chi2_threshold:
               source is variable but does NOT follow a single-lens curve —
               possible binary lens, extended source, or exotic magnification.

    Requires ≥ 3 epochs for meaningful fitting.

    References
    ----------
    Paczyński (1986), ApJ 304, 1.
    Suazo et al. (2024), MNRAS 531, 695.
    arXiv:2512.07924 — microlensing of Dyson sphere structures.
    """

    def __init__(
        self,
        epochs: List[Tuple[np.ndarray, float]],   # [(image, jd), ...]
        aperture_radius_px: int = 3,
        t0_steps: int = 20,
        te_range_days: Tuple[float, float] = (1.0, 100.0),
        te_steps: int = 15,
        u0_range: Tuple[float, float] = (0.01, 1.0),
        u0_steps: int = 15,
        variability_threshold: float = 5.0,   # chi2/dof to call source variable
        chi2_threshold: float = 3.0,           # chi2/dof to call Pac. fit good
        u0_anomaly_threshold: float = 0.3,     # compact-lens anomaly
    ) -> None:
        if len(epochs) < 3:
            raise ValueError(
                "MultiEpochMicrolensingDetector requires ≥ 3 epochs, "
                f"got {len(epochs)}."
            )
        # Sort epochs by JD
        self.epochs: List[Tuple[np.ndarray, float]] = sorted(
            epochs, key=lambda x: x[1]
        )
        self.times = np.array([jd for _, jd in self.epochs], dtype=np.float64)
        self.aperture_radius = aperture_radius_px

        t_lo, t_hi = self.times[0], self.times[-1]
        self.t0_values = np.linspace(t_lo, t_hi, t0_steps)
        self.te_values = np.linspace(te_range_days[0], te_range_days[1], te_steps)
        self.u0_values = np.linspace(u0_range[0], u0_range[1], u0_steps)

        self.variability_threshold = variability_threshold
        self.chi2_threshold = chi2_threshold
        self.u0_anomaly_threshold = u0_anomaly_threshold

    # ── Physics ────────────────────────────────────────────────────────────

    @staticmethod
    def _paczynski(t: np.ndarray, t0: float, tE: float, u0: float) -> np.ndarray:
        """Paczyński magnification curve A(t; t0, tE, u0)."""
        u = np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
        u = np.maximum(u, 1e-6)
        return (u ** 2 + 2.0) / (u * np.sqrt(u ** 2 + 4.0))

    # ── Photometry ─────────────────────────────────────────────────────────

    def _aperture_flux(self, image: np.ndarray, r: int, c: int) -> float:
        """Circular aperture sum centred at (r, c)."""
        h, w = image.shape
        rad = self.aperture_radius
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(w, c + rad + 1)
        patch = image[r0:r1, c0:c1]
        ri, ci = np.meshgrid(
            np.arange(r0, r1) - r, np.arange(c0, c1) - c, indexing='ij'
        )
        circle = (ri ** 2 + ci ** 2) <= rad ** 2
        vals = patch[circle]
        return float(vals.sum()) if len(vals) else 0.0

    # ── chi² helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _chi2_flat(fluxes: np.ndarray, errors: np.ndarray) -> float:
        """χ²/dof of the constant (weighted-mean) model."""
        w = 1.0 / np.maximum(errors ** 2, 1e-30)
        f_mean = np.sum(w * fluxes) / np.sum(w)
        chi2 = np.sum(w * (fluxes - f_mean) ** 2)
        dof = max(len(fluxes) - 1, 1)
        return float(chi2 / dof)

    @staticmethod
    def _fit_linear(A_curve: np.ndarray, fluxes: np.ndarray,
                    errors: np.ndarray) -> Tuple[float, float, float]:
        """
        Analytically fit f_s and f_b in F_model = f_s * A + f_b (least squares).
        Returns (f_s, f_b, chi2/dof).
        """
        w = 1.0 / np.maximum(errors ** 2, 1e-30)
        sw = np.sum(w)
        sA = np.sum(w * A_curve)
        sA2 = np.sum(w * A_curve ** 2)
        sF = np.sum(w * fluxes)
        sAF = np.sum(w * A_curve * fluxes)
        denom = sw * sA2 - sA ** 2
        if abs(denom) < 1e-30:
            return 0.0, float(sF / sw), 1e9
        f_s = (sw * sAF - sA * sF) / denom
        f_b = (sF - f_s * sA) / sw
        residuals = fluxes - (f_s * A_curve + f_b)
        chi2 = float(np.sum(w * residuals ** 2))
        dof = max(len(fluxes) - 2, 1)
        return float(f_s), float(f_b), chi2 / dof

    # ── Detection ──────────────────────────────────────────────────────────

    def detect(
        self,
        detected_objects: Optional[List[Dict]] = None,
    ) -> List[AlgorithmCandidate]:
        """
        Fit Paczyński light curves at each detected object's position.

        Args:
            detected_objects: object dicts with 'coordinates' field.

        Returns:
            AlgorithmCandidate list for objects with lensing-like or
            anomalous light curves.
        """
        try:
            return self._detect(detected_objects or [])
        except Exception as exc:
            logger.error("MultiEpochMicrolensingDetector failed: %s", exc)
            return []

    def _detect(self, detected_objects: List[Dict]) -> List[AlgorithmCandidate]:
        ref_image = self.epochs[0][0]
        h, w = ref_image.shape
        n_epoch = len(self.epochs)

        candidates: List[AlgorithmCandidate] = []

        for idx, obj in enumerate(detected_objects):
            coords = obj.get('coordinates', [0.0, 0.0])
            r = int(round(float(coords[0])))
            c = int(round(float(coords[1])))
            r = max(0, min(r, h - 1))
            c = max(0, min(c, w - 1))

            # ── Build light curve ─────────────────────────────────────────
            fluxes = np.array([
                self._aperture_flux(img, r, c)
                for img, _ in self.epochs
            ], dtype=np.float64)

            # Photon-noise error estimate (Poisson + readnoise floor)
            errors = np.sqrt(np.maximum(fluxes, 0.0)) + 1e-6

            if fluxes.mean() < 1e-10:
                continue

            # ── Variability test ──────────────────────────────────────────
            chi2_flat = self._chi2_flat(fluxes, errors)
            if chi2_flat < self.variability_threshold:
                continue   # source is not significantly variable

            # ── Paczyński grid search ─────────────────────────────────────
            best_chi2_pac = 1e9
            best_t0 = self.times.mean()
            best_te = 20.0
            best_u0 = 0.5
            best_fs = 0.0
            best_fb = 0.0

            for t0 in self.t0_values:
                for tE in self.te_values:
                    for u0 in self.u0_values:
                        A_curve = self._paczynski(self.times, t0, tE, u0)
                        f_s, f_b, chi2_pac = self._fit_linear(
                            A_curve, fluxes, errors
                        )
                        if chi2_pac < best_chi2_pac:
                            best_chi2_pac = chi2_pac
                            best_t0, best_te, best_u0 = t0, tE, u0
                            best_fs, best_fb = f_s, f_b

            # ── Anomaly classification ────────────────────────────────────
            delta_chi2 = chi2_flat - best_chi2_pac
            is_good_lens = best_chi2_pac < self.chi2_threshold
            is_anomalous_lens = (
                chi2_flat > self.variability_threshold
                and best_chi2_pac > self.chi2_threshold
            )
            is_compact_lens = (
                is_good_lens and best_u0 < self.u0_anomaly_threshold
            )

            if not (is_good_lens or is_anomalous_lens):
                continue

            # A_max at minimum impact parameter
            a_max = float(self._paczynski(
                np.array([best_t0]), best_t0, best_te, best_u0
            )[0])

            anomaly_type = (
                'microlensing_compact_lens' if is_compact_lens
                else 'microlensing_anomalous_variability' if is_anomalous_lens
                else 'microlensing_event'
            )

            # Score: higher for more variable + better lensing fit + compact u0
            score = float(min(1.0, (
                0.4 * min(delta_chi2 / 50.0, 1.0) +
                0.4 * max(0.0, 1.0 - best_chi2_pac / self.chi2_threshold) +
                0.2 * max(0.0, 1.0 - best_u0 / self.u0_anomaly_threshold)
            )))

            bb = [max(0, r - 5), max(0, c - 5),
                  min(h, r + 6), min(w, c + 6)]
            peak_flux = float(np.clip(fluxes.max() / max(fluxes.max(), 1e-9), 0.0, 1.0))

            candidates.append(AlgorithmCandidate(
                id=f'me_ul_{idx}_{r}_{c}',
                algorithm='multiepoch_microlensing',
                anomaly_type=anomaly_type,
                coordinates=[float(r), float(c)],
                bounding_box=bb,
                brightness=peak_flux,
                score=score,
                metadata={
                    't0_best_jd': best_t0,
                    'te_best_days': best_te,
                    'u0_best': best_u0,
                    'a_max': a_max,
                    'chi2_flat': chi2_flat,
                    'chi2_paczynski': best_chi2_pac,
                    'delta_chi2': delta_chi2,
                    'n_epochs': n_epoch,
                    'source_flux': best_fs,
                    'blend_flux': best_fb,
                    'is_compact_lens': is_compact_lens,
                    'is_anomalous': is_anomalous_lens,
                    'reference': 'Paczynski1986 ApJ304; arXiv:2512.07924',
                },
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        logger.debug("MultiEpochMicrolensing found %d candidates", len(candidates))
        return candidates


# ---------------------------------------------------------------------------
# Convenience: run all four algorithms and merge results
# ---------------------------------------------------------------------------

def run_all_algorithms(
    image: np.ndarray,
    detected_objects: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Run all four detection algorithms on a 2-D image and return merged results.

    Args:
        image: 2-D (H,W) float32 array normalised to [0,1].
        detected_objects: Pre-detected objects from ImageProcessor (optional).

    Returns:
        List of candidate dicts in the detected_objects schema, tagged by
        algorithm.
    """
    if image.ndim == 3:
        lum = image.mean(axis=-1)
    else:
        lum = image.astype(np.float32)

    objs = detected_objects or []
    results: List[Dict] = []

    for Detector, kwargs in [
        (WaveletSourceDetector, {'n_scales': 4, 'sigma_threshold': 3.5}),
        (MatchedFilterDetector, {'n_scales': 4, 'detection_threshold': 5.0}),
        (IRExcessDetector, {}),
        (MicrolensingAnomalyDetector, {}),
    ]:
        if Detector in (IRExcessDetector, MicrolensingAnomalyDetector):
            candidates = Detector(**kwargs).detect(lum, objs)
        else:
            candidates = Detector(**kwargs).detect(lum)
        results.extend([c.to_dict() for c in candidates])

    logger.info("All algorithms combined: %d candidates", len(results))
    return results
