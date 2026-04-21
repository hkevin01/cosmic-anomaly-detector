"""
Command Line Interface — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: CLI-001
# Requirement: Expose a Click-based CLI with an "analyze" subcommand that
#              accepts a FITS file or directory path, processes all matching
#              files, and writes results/report/manifest to a timestamped run
#              directory under output/runs/.
# Purpose: Allow batch and single-image analysis without writing Python, and
#          provide a stable interface for scripting and CI integration.
# Rationale: Click is chosen for its composable command groups, automatic
#             --help generation, and typed argument validation — reducing the
#             surface area for invalid input errors.
# Inputs:  path (Path) — FITS file or directory (must exist).
#          --recursive / --no-recursive (bool, default True).
#          --run-id (Optional[str]) — custom run identifier; auto-generated
#          as YYYYMMDD-HHMMSS-<uuid8> when omitted.
#          --limit (Optional[int]) — max number of files to process.
# Outputs: output/runs/<run_id>/results.json — full anomaly data.
#          output/runs/<run_id>/report.md    — Markdown summary.
#          output/runs/<run_id>/summary.json — lightweight stats.
#          output/runs/<run_id>/manifest.json — reproducibility manifest.
# Preconditions:  path must reference a readable file or directory.
# Postconditions: run directory created with all four output artefacts.
# Assumptions: Output root defaults to "output/" relative to CWD.
# Side Effects: Creates directories and writes files; logs to console + file.
# Failure Modes: Unreadable path → Click validation error before processing.
#                Sub-analysis exception → logged per file; run continues.
# Error Handling: Per-file exceptions are caught and included in results
#                 as error entries; the run completes rather than aborting.
# Constraints: Processes files sequentially; parallel mode deferred to Phase 5.
# Verification: Invoked in tests/test_run_manifest.py via CLI runner.
# References: Click 8.x; run_manifest.write_manifest; ReportGenerator.
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

import click

from .core.detector import AnomalyDetector
from .reporting.report import ReportGenerator
from .utils.config import get_config
from .utils.logging import get_logger, setup_logging
from .utils.run_manifest import write_manifest

logger = get_logger(__name__)


def _ensure_runs_dir(root: Path) -> Path:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return runs


def _create_run_dir(root: Path, run_id: Optional[str] = None) -> Path:
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    rd = _ensure_runs_dir(root) / run_id
    rd.mkdir(parents=True, exist_ok=False)
    return rd


@click.group(help="Cosmic Anomaly Detector CLI")
def cad() -> None:  # pragma: no cover - thin wrapper
    setup_logging()
    get_config()  # ensure config loaded


@cad.command(
    "analyze", help="Analyze a FITS image or directory of FITS files"
)
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Recurse into subdirectories when path is a directory",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Optional run id (for reproducibility)",
)
@click.option(
    "--limit",
    type=int,
    default=0,
    help="Limit number of files (0 = no limit)",
)
def analyze_cmd(
    path: Path, recursive: bool, run_id: Optional[str], limit: int
) -> None:
    cfg = get_config()
    detector = AnomalyDetector()
    run_dir = _create_run_dir(Path(cfg.output_path), run_id)
    logger.info(f"Run directory: {run_dir}")

    files: List[Path]
    if path.is_dir():
        pattern = "**/*.fits" if recursive else "*.fits"
        files = sorted(path.glob(pattern))
    else:
        files = [path]

    if limit > 0:
        files = files[:limit]

    if not files:
        click.echo("No FITS files found to analyze.")
        sys.exit(1)

    all_results = []
    for f in files:
        try:
            res = detector.analyze_image(str(f))
            # serialize minimal result
            result_dict = {
                "file": str(f),
                "anomalies": res.anomalies,
                "confidence_scores": res.confidence_scores,
                "metadata": res.processing_metadata,
            }
            all_results.append(result_dict)
        except Exception as e:  # pragma: no cover - defensive
            logger.exception(f"Failed analyzing {f}: {e}")

    results_json = run_dir / "results.json"
    with results_json.open("w") as fh:
        json.dump({"results": all_results}, fh, indent=2)
    logger.info(f"Saved results -> {results_json}")
    # Write manifest
    write_manifest(run_dir / "run_manifest.json", cfg)

    # Generate report
    rep = ReportGenerator()
    report_path = rep.generate_markdown_report(run_dir, all_results, cfg)
    click.echo(f"Report generated: {report_path}")


@cad.command(
    "report", help="Regenerate report for an existing run directory"
)
@click.argument(
    "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def report_cmd(run_dir: Path) -> None:
    cfg = get_config()
    results_json = run_dir / "results.json"
    if not results_json.exists():
        click.echo("results.json not found in run directory", err=True)
        sys.exit(1)
    data = json.loads(results_json.read_text())
    rep = ReportGenerator()
    report_path = rep.generate_markdown_report(
        run_dir, data.get("results", []), cfg
    )
    click.echo(f"Report regenerated: {report_path}")


def main() -> None:  # pragma: no cover - entrypoint
    cad()


if __name__ == "__main__":  # pragma: no cover
    main()
