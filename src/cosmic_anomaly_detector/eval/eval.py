"""Evaluation helpers (scaffold)."""
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
