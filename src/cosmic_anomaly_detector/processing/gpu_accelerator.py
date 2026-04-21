"""
GPU-Accelerated Processing — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: GPU-001
# Requirement: Provide drop-in GPU-accelerated replacements for the
#   computationally intensive operations in the detection pipeline:
#   Gaussian convolution, starlet wavelet transform, matched-filter SNR map,
#   and normalisation.  Fall back transparently to CPU (scipy/numpy) when no
#   CUDA device is available.
# Purpose: Reduce per-image processing time from O(seconds) to O(ms) on
#   hardware-equipped machines, enabling large-scale SETI/JWST batch runs.
# Rationale: PyTorch conv2d on CUDA is 20-200× faster than scipy.ndimage on
#   CPU for the kernel sizes used here (σ = 1-10 px, FWHM = 2-6 px).
#   Torch is already a declared dependency so no new requirement is added.
# Inputs:
#   image (np.ndarray): 2-D float32 normalised to [0, 1].
#   device (str): 'cuda', 'cpu', or 'auto' (default).
# Outputs: np.ndarray on CPU (same shape/dtype as input).
# Preconditions:  image must be 2-D.
# Postconditions: Output dtype = float32, range not guaranteed [0,1] for
#   intermediate maps (SNR maps may exceed 1).
# Assumptions: torch is importable; CUDA driver matching torch build present
#   for GPU path; CPU fallback always works.
# Side Effects: First call allocates CUDA context (≈ 300 ms one-time cost).
# Failure Modes: OOM on GPU → falls back to CPU; warns via logger.
# Error Handling: All public methods catch RuntimeError and retry on CPU.
# Constraints: Batch size limited by GPU VRAM; default TILE_SIZE = 2048 px.
# Verification: tests/test_gpu_accelerator.py.
# References: PyTorch conv2d; scipy.ndimage.gaussian_filter (CPU fallback).
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device(requested: str = 'auto') -> str:
    """
    Return the best available torch device string.

    Priority: cuda > cpu.  'auto' picks cuda when available.
    """
    try:
        import torch
        if requested == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        if requested == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            return 'cpu'
        return requested
    except ImportError:
        return 'cpu'


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Build a normalised 2-D Gaussian kernel."""
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1
    x = np.arange(size, dtype=np.float64) - radius
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    return np.outer(g, g).astype(np.float32)


def _b3_spline_kernel_1d() -> np.ndarray:
    """1-D B3-spline coefficients used in the starlet à trous transform."""
    return np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0


# ---------------------------------------------------------------------------
# Core GPU-accelerated ops
# ---------------------------------------------------------------------------

class GPUAccelerator:
    """
    GPU-accelerated image processing operations for the anomaly detector.

    All public methods accept and return NumPy float32 arrays so callers
    need not be aware of the GPU/CPU backend.

    Usage
    -----
    >>> acc = GPUAccelerator(device='auto')
    >>> smoothed = acc.gaussian_filter(image, sigma=2.0)
    >>> wavelet_planes = acc.starlet_transform(image, n_scales=4)
    >>> snr_map = acc.matched_filter_snr(image, fwhm=3.0)
    """

    def __init__(self, device: str = 'auto') -> None:
        self.device = _select_device(device)
        self._torch_available = self._check_torch()
        if self._torch_available and self.device == 'cuda':
            logger.info("GPUAccelerator: using CUDA GPU acceleration")
        elif self._torch_available:
            logger.info("GPUAccelerator: using PyTorch CPU (no CUDA device found)")
        else:
            logger.info("GPUAccelerator: torch not importable — pure numpy/scipy fallback")

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _check_torch() -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def _to_tensor(self, image: np.ndarray):
        """Convert 2-D numpy array to (1,1,H,W) torch float32 tensor."""
        import torch
        return torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

    def _from_tensor(self, t) -> np.ndarray:
        """Convert (1,1,H,W) tensor back to 2-D numpy float32."""
        return t.squeeze().cpu().numpy().astype(np.float32)

    def _conv2d_kernel(self, kernel: np.ndarray) -> 'torch.Tensor':
        """Wrap a 2-D numpy kernel as a (1,1,kH,kW) torch tensor."""
        import torch
        return torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).to(self.device)

    # ── public API ────────────────────────────────────────────────────────

    def gaussian_filter(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply a Gaussian blur with the given sigma.

        Falls back to scipy.ndimage.gaussian_filter on CPU when torch
        is not available.
        """
        if not self._torch_available:
            from scipy.ndimage import gaussian_filter as _gf
            return _gf(image.astype(np.float64), sigma=sigma).astype(np.float32)

        import torch
        import torch.nn.functional as F

        try:
            kernel_np = _gaussian_kernel_2d(sigma)
            pad = kernel_np.shape[0] // 2
            img_t = self._to_tensor(image)
            ker_t = self._conv2d_kernel(kernel_np)
            out = F.conv2d(img_t, ker_t, padding=pad)
            return self._from_tensor(out)
        except RuntimeError as exc:
            logger.warning("GPU gaussian_filter failed (%s); retrying on CPU", exc)
            self.device = 'cpu'
            return self.gaussian_filter(image, sigma)

    def starlet_transform(
        self, image: np.ndarray, n_scales: int = 4
    ) -> List[np.ndarray]:
        """
        Compute the starlet (isotropic undecimated wavelet) à trous transform.

        Returns n_scales wavelet planes as numpy arrays; each plane is the
        difference between successive B3-spline approximations.

        GPU path: each separable 1-D convolution is performed as a 2-D
        depthwise conv with a dilated kernel.
        """
        if not self._torch_available:
            return self._starlet_cpu(image, n_scales)

        import torch
        import torch.nn.functional as F

        try:
            return self._starlet_gpu(image, n_scales)
        except RuntimeError as exc:
            logger.warning("GPU starlet failed (%s); retrying on CPU", exc)
            self.device = 'cpu'
            return self._starlet_cpu(image, n_scales)

    def _starlet_gpu(self, image: np.ndarray, n_scales: int) -> List[np.ndarray]:
        import torch
        import torch.nn.functional as F

        k1d = _b3_spline_kernel_1d()   # (5,) float32
        current = self._to_tensor(image)   # (1,1,H,W)
        planes: List[np.ndarray] = []

        for j in range(n_scales):
            step = 2 ** j
            # Build dilated 1-D → 2-D kernels
            dilated_len = 1 + step * (len(k1d) - 1)
            k_row = torch.zeros(1, 1, 1, dilated_len, device=self.device, dtype=torch.float32)
            k_col = torch.zeros(1, 1, dilated_len, 1, device=self.device, dtype=torch.float32)
            for idx, val in enumerate(k1d):
                k_row[0, 0, 0, idx * step] = float(val)
                k_col[0, 0, idx * step, 0] = float(val)

            pad_r = dilated_len // 2
            smooth = F.conv2d(current, k_row, padding=(0, pad_r))
            smooth = F.conv2d(smooth, k_col, padding=(pad_r, 0))
            planes.append(self._from_tensor(current - smooth))
            current = smooth

        return planes

    def _starlet_cpu(self, image: np.ndarray, n_scales: int) -> List[np.ndarray]:
        from scipy.ndimage import convolve1d
        k1d = _b3_spline_kernel_1d().astype(np.float64)
        current = image.astype(np.float64)
        planes: List[np.ndarray] = []
        for j in range(n_scales):
            step = 2 ** j
            dilated = np.zeros(1 + step * (len(k1d) - 1))
            dilated[::step] = k1d
            smooth = convolve1d(current, dilated, axis=0, mode='mirror')
            smooth = convolve1d(smooth, dilated, axis=1, mode='mirror')
            planes.append((current - smooth).astype(np.float32))
            current = smooth
        return planes

    def matched_filter_snr(
        self, image: np.ndarray, fwhm: float
    ) -> Tuple[np.ndarray, float]:
        """
        Apply a Gaussian matched filter and return (snr_map, noise_sigma).

        snr_map[i,j] = filtered[i,j] / noise_sigma (MAD-estimated).
        """
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        bg = self.gaussian_filter(image, sigma=10.0)
        residual = image.astype(np.float32) - bg
        noise_sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        noise_sigma = max(noise_sigma, 1e-10)
        filtered = self.gaussian_filter(residual, sigma=sigma)
        snr_map = filtered / noise_sigma
        return snr_map, noise_sigma

    def normalise(self, image: np.ndarray,
                  lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
        """Percentile-based normalisation to [0, 1]."""
        lo, hi = np.percentile(image[np.isfinite(image)], [lo_pct, hi_pct])
        if hi <= lo:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def batch_normalise(
        self, images: List[np.ndarray], **kwargs
    ) -> List[np.ndarray]:
        """Normalise a list of images; uses GPU batching when available."""
        if self._torch_available and self.device == 'cuda' and len(images) > 1:
            return self._batch_normalise_gpu(images, **kwargs)
        return [self.normalise(img, **kwargs) for img in images]

    def _batch_normalise_gpu(
        self, images: List[np.ndarray],
        lo_pct: float = 1.0, hi_pct: float = 99.0
    ) -> List[np.ndarray]:
        import torch
        results = []
        for img in images:
            try:
                t = self._to_tensor(img)
                flat = t.flatten().cpu().numpy()
                lo, hi = np.percentile(flat[np.isfinite(flat)], [lo_pct, hi_pct])
                if hi <= lo:
                    results.append(np.zeros_like(img, dtype=np.float32))
                    continue
                normed = ((t - lo) / (hi - lo)).clamp(0.0, 1.0)
                results.append(self._from_tensor(normed))
            except RuntimeError:
                results.append(self.normalise(img, lo_pct, hi_pct))
        return results

    # ── diagnostics ───────────────────────────────────────────────────────

    def device_info(self) -> Dict[str, object]:
        """Return a dict describing the current compute device."""
        info: Dict[str, object] = {'device': self.device, 'torch': self._torch_available}
        if self._torch_available:
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_memory_gb'] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2
                )
        return info


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_default_accelerator: Optional[GPUAccelerator] = None


def get_accelerator(device: str = 'auto') -> GPUAccelerator:
    """Return (and cache) the default GPUAccelerator instance."""
    global _default_accelerator
    if _default_accelerator is None or _default_accelerator.device != _select_device(device):
        _default_accelerator = GPUAccelerator(device=device)
    return _default_accelerator
