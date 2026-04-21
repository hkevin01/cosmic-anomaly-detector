"""
Detection Evaluation — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: EVL-001
# Requirement: Given a list of detected anomalies and a list of injected ground-
#              truth positions, compute true positives (TP), false positives (FP),
#              false negatives (FN), precision, and recall within a spatial
#              tolerance window.
# Purpose: Quantify detection performance against synthetic injection datasets
#          to support model comparison and threshold tuning.
# Rationale: Tolerance-based matching (rather than exact pixel) reflects the
#             inherent centroid localisation uncertainty in astronomical images
#             and avoids penalising good detections for sub-pixel offsets.
# Inputs:  anomalies (Iterable[Dict]) — detector output; each dict must contain
#          either a "location" key ([y, x] list) or "y"/"x" int keys.
#          injected (Iterable[Tuple[int,int]]) — ground-truth (y, x) positions.
#          tol (int, default 5) — matching tolerance in pixels (Chebyshev distance).
# Outputs: Dict with keys: "tp" (int), "fp" (int), "fn" (int),
#          "precision" (float ∈ [0,1]), "recall" (float ∈ [0,1]).
# Preconditions:  Both inputs may be empty; tol must be > 0.
# Postconditions: Each injected position matched at most once (greedy first-match).
#                 Each detected position contributes to at most one TP.
# Assumptions: Coordinate system is row-major (y, x); tol is in pixel units.
# Side Effects: None — pure function with no I/O or state mutation.
# Failure Modes: Missing location keys → anomaly coordinate treated as (-9999, -9999)
#                and will not match any injected position (counted as FP).
# Error Handling: .get() with sentinel defaults for missing dict keys.
# Constraints: O(N×M) matching; acceptable for N,M < 10 000 in evaluation context.
# Verification: tests/test_gravity_adapter.py exercises match_detections with
#               known TP/FP/FN counts.
# References: Precision = TP/(TP+FP); Recall = TP/(TP+FN).
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def match_detections(
    anomalies: Iterable[Dict],
    injected: Iterable[Tuple[int, int]],
    tol: int = 5,
) -> Dict[str, float]:
    injected_list = list(injected)
    detected_coords: List[Tuple[int, int]] = []
    for a in anomalies:
        loc = a.get("location")
        if isinstance(loc, list) and len(loc) >= 2:
            detected_coords.append((int(loc[0]), int(loc[1])))
        else:
            y = int(a.get("y", -9999))
            x = int(a.get("x", -9999))
            if y != -9999:
                detected_coords.append((y, x))
    tp = 0
    used = set()
    for iy, ix in injected_list:
        for j, (dy, dx) in enumerate(detected_coords):
            if j in used:
                continue
            if abs(iy - dy) <= tol and abs(ix - dx) <= tol:
                tp += 1
                used.add(j)
                break
    fp = max(0, len(detected_coords) - tp)
    fn = max(0, len(injected_list) - tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }
