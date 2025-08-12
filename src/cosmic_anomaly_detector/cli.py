"""Command Line Interface for Cosmic Anomaly Detector

Provides entrypoints:
  cad analyze <path>  - Analyze a FITS image or directory of FITS images
  cad report <run_id> - Regenerate / display a report for an existing run

Implements a minimal vertical slice: load FITS -> preprocessing -> baseline
anomaly scoring -> report/manifest creation.
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
