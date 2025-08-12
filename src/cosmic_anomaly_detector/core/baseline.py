"""Baseline anomaly scoring (vertical slice)

Implements a simple brightness / residual z-score based anomaly scorer that
operates on processed images. This provides an interpretable starting point
before advanced ML models are integrated into the real-time path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:  # pragma: no cover
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None  # type: ignore


@dataclass
class BaselineCandidate:
    y: int
    x: int
    score: float
    local_flux: float


class BaselineAnomalyScorer:
    """Compute simple anomaly scores using local z-score heuristics."""

    def __init__(self, sigma: float = 3.0, max_candidates: int = 20):
        self.sigma = sigma
        self.max_candidates = max_candidates

    def score(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        if image.ndim == 3:  # reduce RGB-like to luminance
            img = image.mean(axis=2)
        else:
            img = image
        if gaussian_filter is not None:
            smooth = gaussian_filter(img, 1.0)
        else:  # pragma: no cover
            smooth = img
        mean = float(np.mean(smooth))
        std = float(np.std(smooth)) or 1.0
        z = (smooth - mean) / std
        mask = z > self.sigma
        ys, xs = np.where(mask)
        candidates: List[BaselineCandidate] = []
        for y, x in zip(ys, xs):
            candidates.append(
                BaselineCandidate(
                    int(y), int(x), float(z[y, x]), float(smooth[y, x])
                )
            )
        # sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        top = candidates[: self.max_candidates]
        return {
            "candidates": [c.__dict__ for c in top],
            "total_candidates": len(candidates),
            "mean": mean,
            "std": std,
        }
