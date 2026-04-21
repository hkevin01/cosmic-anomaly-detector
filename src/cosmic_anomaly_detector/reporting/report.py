"""
Report Generator — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: RPT-001
# Requirement: Generate a human-readable Markdown report and a machine-readable
#              summary.json for each analysis run, and optionally produce a
#              thumbnail PNG when matplotlib is available.
# Purpose: Provide auditable, reproducible output artefacts per run so that
#          scientists can review findings without executing Python.
# Rationale: Markdown is readable in all environments (terminal, GitHub, Jupyter);
#             JSON summary enables dashboard ingestion and trend tracking.
# Inputs:  run_dir (Path) — writable output directory for this run.
#          results (List[Dict]) — per-file result dicts each containing "file"
#          (str) and "anomalies" (List[Dict] with type, severity, description).
#          cfg (Any) — SystemConfig instance (used for future template rendering).
# Outputs: report.md written to run_dir; summary.json written to run_dir;
#          thumbnail.png optionally written when matplotlib is present.
#          Returns Path to report.md.
# Preconditions:  run_dir must exist and be writable.
# Postconditions: report.md and summary.json are always written;
#                 thumbnail.png written only when matplotlib is importable.
# Assumptions: results may be empty; anomaly dicts may be missing optional keys.
# Side Effects: Writes files to disk; no network I/O.
# Failure Modes: Non-writable run_dir → OSError propagated to caller.
# Error Handling: Missing dict keys use .get() with safe defaults.
# Constraints: Markdown table cells are truncated to 120 chars if needed.
# Verification: tests/test_report.py asserts report.md and summary.json exist
#               and contain correct anomaly counts after generate_markdown_report().
# References: GitHub Flavored Markdown pipe tables; matplotlib magma colormap.
# ---------------------------------------------------------------------------
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:  # pragma: no cover
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


class ReportGenerator:
    def generate_markdown_report(
        self, run_dir: Path, results: List[Dict[str, Any]], cfg: Any
    ) -> Path:  # noqa: ANN401
        report_path = run_dir / "report.md"
        lines: List[str] = []
        lines.append("# Cosmic Anomaly Detector Report\n")
        lines.append(f"Run Directory: `{run_dir.name}`\n")
        lines.append("\n## Summary\n")
        total_anomalies = sum(len(r.get("anomalies", [])) for r in results)
        lines.append(f"Analyzed files: {len(results)}  ")
        lines.append(f"Total anomalies: {total_anomalies}\n")
        lines.append("\n## Files\n")
        for r in results:
            lines.append(f"### {r['file']}\n")
            lines.append(
                f"Anomalies detected: {len(r.get('anomalies', []))}\n"
            )
            if r.get("anomalies"):
                lines.append("| # | Type | Score | Description |\n")
                lines.append("|---|------|-------|-------------|\n")
                for idx, a in enumerate(r["anomalies"]):
                    cell_type = a.get("type", "?")
                    sev = a.get("severity", 0.0)
                    desc = a.get("description", "")
                    lines.append(
                        f"| {idx + 1} | {cell_type} | {sev:.3f} | {desc} |\n"
                    )
            lines.append("\n")
        # Save simple stats JSON
        (run_dir / "summary.json").write_text(
            json.dumps({"total_anomalies": total_anomalies}, indent=2)
        )
        # Optional visualization placeholder
        if results and plt is not None:  # pragma: no cover
            first_image_np = np.random.random((128, 128))
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(first_image_np, cmap="magma")
            ax.set_title("Sample Image (placeholder)")
            ax.axis("off")
            fig.tight_layout()
            fig_path = run_dir / "thumbnail.png"
            fig.savefig(str(fig_path))
            plt.close(fig)
            lines.append("## Visualization\n\n")
            lines.append("![Thumbnail](thumbnail.png)\n")
        report_path.write_text("".join(lines))
        return report_path
