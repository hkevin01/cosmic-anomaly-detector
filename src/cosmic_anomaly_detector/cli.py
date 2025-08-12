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


@cad.command("analyze", help="Analyze a FITS image or directory of FITS files")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--recursive/--no-recursive", default=True, help="Recurse into subdirectories when path is a directory")
@click.option("--run-id", type=str, default=None, help="Optional run id (for reproducibility)")
@click.option("--limit", type=int, default=0, help="Limit number of files (0 = no limit)")
def analyze_cmd(path: Path, recursive: bool, run_id: Optional[str], limit: int) -> None:
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

    # Generate report
    rep = ReportGenerator()
    report_path = rep.generate_markdown_report(run_dir, all_results, cfg)
    click.echo(f"Report generated: {report_path}")


@cad.command("report", help="Regenerate report for an existing run directory")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report_cmd(run_dir: Path) -> None:
    cfg = get_config()
    results_json = run_dir / "results.json"
    if not results_json.exists():
        click.echo("results.json not found in run directory", err=True)
        sys.exit(1)
    data = json.loads(results_json.read_text())
    rep = ReportGenerator()
    report_path = rep.generate_markdown_report(run_dir, data.get("results", []), cfg)
    click.echo(f"Report regenerated: {report_path}")


def main() -> None:  # pragma: no cover - entrypoint
    cad()


if __name__ == "__main__":  # pragma: no cover
    main()
"""Command Line Interface for Cosmic Anomaly Detector

Provides entrypoints:
  cad analyze <path>  - Analyze a FITS image or directory of FITS images
  cad report <run_id> - Regenerate / display a report for an existing run

Implements a minimal vertical slice: load FITS -> preprocessing -> baseline
anomaly scoring -> report/manifest creation.
"""

from __future__ import annotations

from pathlib import Path

import click

from .utils.logging import get_logger

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


@cad.command("analyze", help="Analyze a FITS image or directory of FITS files")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--recursive/--no-recursive", default=True, help="Recurse into subdirectories when path is a directory")
@click.option("--run-id", type=str, default=None, help="Optional run id (for reproducibility)")
@click.option("--limit", type=int, default=0, help="Limit number of files (0 = no limit)")
def analyze_cmd(path: Path, recursive: bool, run_id: Optional[str], limit: int) -> None:
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

    # Generate report
    rep = ReportGenerator()
    report_path = rep.generate_markdown_report(run_dir, all_results, cfg)
    click.echo(f"Report generated: {report_path}")


@cad.command("report", help="Regenerate report for an existing run directory")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report_cmd(run_dir: Path) -> None:
    cfg = get_config()
    results_json = run_dir / "results.json"
    if not results_json.exists():
        click.echo("results.json not found in run directory", err=True)
        sys.exit(1)
    data = json.loads(results_json.read_text())
    rep = ReportGenerator()
    report_path = rep.generate_markdown_report(run_dir, data.get("results", []), cfg)
    click.echo(f"Report regenerated: {report_path}")


def main() -> None:  # pragma: no cover - entrypoint
    cad()


if __name__ == "__main__":  # pragma: no cover
    main()
#!/usr/bin/env python3
"""
Command Line Interface for Cosmic Anomaly Detector

Provides command-line access to image analysis functionality.
"""

import argparse
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cosmic_anomaly_detector.core.detector import AnomalyDetector
from cosmic_anomaly_detector.utils.config import get_config, set_config_path
from cosmic_anomaly_detector.utils.logging import get_logger, setup_logging


def analyze_single_image(image_path: str, output_dir: Optional[str] = None) -> None:
    """Analyze a single FITS image"""
    logger = get_logger(__name__)

    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        return

    logger.info(f"Analyzing image: {image_path}")

    # Initialize detector
    config = get_config()
    detector = AnomalyDetector(config)

    try:
        # Run analysis
        results = detector.analyze_image(image_path)

        if results:
            # Print summary
            num_objects = len(results.get('objects', []))
            artificial_candidates = sum(
                1 for obj in results.get('objects', [])
                if obj.get('artificial_probability', 0) > 0.8
            )

            print(f"\nAnalysis Results for {Path(image_path).name}:")
            print(f"  Total objects detected: {num_objects}")
            print(f"  Artificial candidates: {artificial_candidates}")

            # Save results if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                import json
                result_file = output_path / f"{Path(image_path).stem}_results.json"

                # Convert results to serializable format
                serializable_results = {
                    'source_file': str(image_path),
                    'num_objects': num_objects,
                    'artificial_candidates': artificial_candidates,
                    'objects': [
                        {
                            'centroid': obj.get('centroid', []),
                            'area': obj.get('area', 0),
                            'artificial_probability': obj.get('artificial_probability', 0)
                        }
                        for obj in results.get('objects', [])
                    ]
                }

                with open(result_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)

                logger.info(f"Results saved to: {result_file}")
        else:
            logger.warning("No results returned from analysis")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")


def analyze_batch(image_paths: List[str], output_dir: Optional[str] = None) -> None:
    """Analyze multiple FITS images"""
    logger = get_logger(__name__)

    logger.info(f"Starting batch analysis of {len(image_paths)} images")

    for i, image_path in enumerate(image_paths, 1):
        print(f"\nProgress: {i}/{len(image_paths)} - {Path(image_path).name}")
        analyze_single_image(image_path, output_dir)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Cosmic Anomaly Detector - Analyze JWST images for artificial structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.fits                    # Analyze single image
  %(prog)s *.fits -o results/            # Analyze multiple images
  %(prog)s --config custom.yaml image.fits  # Use custom configuration
  %(prog)s --gui                         # Launch GUI interface
        """
    )

    parser.add_argument(
        'images',
        nargs='*',
        help='FITS image files to analyze'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output directory for results'
    )

    parser.add_argument(
        '-c', '--config',
        help='Configuration file path'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI interface'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Cosmic Anomaly Detector 0.1.0'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(
        log_level=args.log_level,
        console_output=True,
        file_output=True
    )

    # Load custom configuration if specified
    if args.config:
        set_config_path(args.config)

    # Launch GUI if requested
    if args.gui:
        try:
            from cosmic_anomaly_detector.gui.main_window import \
                main as gui_main
            gui_main()
        except ImportError as e:
            print(f"GUI not available: {e}")
            print("Install GUI dependencies: pip install PyQt5 pyqtgraph")
            sys.exit(1)
        return

    # Check if images provided
    if not args.images:
        parser.print_help()
        print("\nError: No image files specified")
        sys.exit(1)

    # Expand glob patterns
    image_files = []
    for pattern in args.images:
        if '*' in pattern or '?' in pattern:
            import glob
            matches = glob.glob(pattern)
            image_files.extend(matches)
        else:
            image_files.append(pattern)

    # Validate files exist
    valid_files = []
    for img_file in image_files:
        if Path(img_file).exists():
            valid_files.append(img_file)
        else:
            print(f"Warning: File not found - {img_file}")

    if not valid_files:
        print("Error: No valid image files found")
        sys.exit(1)

    # Run analysis
    if len(valid_files) == 1:
        analyze_single_image(valid_files[0], args.output)
    else:
        analyze_batch(valid_files, args.output)


if __name__ == "__main__":
    main()
