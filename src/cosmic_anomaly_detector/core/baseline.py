"""
Baseline Anomaly Scorer — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: BAS-001
# Requirement: Compute local-flux z-score anomaly candidates from a 2-D image
#              array and return the top-N candidates ranked by score.
# Purpose: Provide an interpretable, dependency-light anomaly signal that
#          operates before the ML classifier to catch statistically extreme
#          brightness events independently of trained model weights.
# Rationale: Z-score thresholding is deterministic, fast, and auditable —
#             essential for scientific reproducibility and baseline comparison.
# Inputs:  image (np.ndarray) — 2-D or 3-D float array in arbitrary flux units.
#          sigma (float, default 3.0) — detection threshold in standard deviations.
#          max_candidates (int, default 20) — cap on returned results.
# Outputs: Dict with keys: "candidates" (List[Dict] with y, x, score, local_flux),
#          "total_candidates" (int), "mean" (float), "std" (float).
# Preconditions:  image must be non-empty; std > 0 (homogeneous images return 0 candidates).
# Postconditions: Candidates are sorted descending by z-score; length ≤ max_candidates.
# Assumptions: scipy.ndimage.gaussian_filter is available for pre-smoothing;
#              falls back to raw image when scipy is absent.
# Side Effects: None — pure function with no I/O or state mutation.
# Failure Modes: All-zero image → std=0 → no candidates returned (safe default).
# Error Handling: 3-D image reduced to luminance via mean(axis=2) before scoring.
# Constraints: O(H×W) time and space; suitable for images up to 8192×8192.
# Verification: tests/test_baseline.py injects a bright pixel and asserts it
#               appears as the top candidate.
# References: Z-score: z = (x − μ) / σ; Gaussian pre-filter σ=1.0 px.
# ---------------------------------------------------------------------------
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
