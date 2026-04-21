#!/usr/bin/env python3
"""
Batch Analysis Script — Cosmic Anomaly Detector

Run the full 6-algorithm SETI / Dyson-sphere detection pipeline over
large collections of JWST FITS images using GPU-accelerated parallel
processing.

Usage examples
--------------
# Analyse all FITS in a local directory (8 workers, GPU auto-select)
  python scripts/batch_analyze.py --input-dir data/jwst/ --workers 8

# Use SETI priority target catalog to source files from a download folder
  python scripts/batch_analyze.py --seti --data-dir data/seti_downloads/ --workers 4

# Custom threshold, CSV output, force CPU
  python scripts/batch_analyze.py --input-dir data/ --threshold 0.4 \\
      --format csv --output results/ --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(name)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('batch_analyze')


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {'.fits', '.fit', '.fts', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}


def find_images(directory: str, recursive: bool = True) -> List[str]:
    """Discover all image / FITS files in *directory*."""
    base = Path(directory)
    if not base.exists():
        logger.error("Input directory not found: %s", directory)
        return []
    paths = base.rglob('*') if recursive else base.glob('*')
    return sorted(str(p) for p in paths if p.suffix.lower() in _IMAGE_EXTS and p.is_file())


def load_fits_list(list_file: str) -> List[str]:
    """Load file paths from a plain-text list (one path per line)."""
    with open(list_file) as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith('#')]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def export_csv(result, output_path: Path) -> None:
    """Export top detections to CSV."""
    try:
        import csv
        rows = []
        for fr in result.file_results:
            if fr.status != 'ok' or not fr.candidates:
                continue
            for c in fr.candidates:
                rows.append({
                    'file': Path(fr.file_path).name,
                    'algorithm': c.get('algorithm', ''),
                    'anomaly_type': c.get('anomaly_type', ''),
                    'score': c.get('score', ''),
                    'brightness': c.get('brightness', ''),
                    'coord_x': c.get('coordinates', [None, None])[0],
                    'coord_y': c.get('coordinates', [None, None])[1],
                })
        csv_path = output_path / 'detections.csv'
        if rows:
            with open(csv_path, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("CSV → %s  (%d rows)", csv_path, len(rows))
        else:
            logger.info("No detections to write to CSV")
    except Exception as exc:
        logger.warning("CSV export failed: %s", exc)


def export_html(result, output_path: Path) -> None:
    """Export a self-contained HTML report."""
    try:
        rows_html = ''
        for fr in result.file_results:
            if fr.status != 'ok' or not fr.candidates:
                continue
            for c in fr.candidates:
                score = c.get('score', 0)
                colour = '#c0392b' if score >= 0.7 else ('#e67e22' if score >= 0.4 else '#27ae60')
                rows_html += (
                    f'<tr>'
                    f'<td>{Path(fr.file_path).name}</td>'
                    f'<td>{c.get("algorithm","")}</td>'
                    f'<td>{c.get("anomaly_type","")}</td>'
                    f'<td><b style="color:{colour}">{score:.3f}</b></td>'
                    f'<td>{c.get("brightness","")}</td>'
                    f'</tr>\n'
                )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cosmic Anomaly Detector — Batch Report</title>
<style>
  body {{ font-family: monospace; background:#111; color:#eee; padding:20px; }}
  h1   {{ color:#4fc3f7; }}
  table {{ border-collapse:collapse; width:100%; }}
  th,td {{ border:1px solid #444; padding:6px 10px; text-align:left; }}
  th   {{ background:#1e88e5; color:#fff; }}
  tr:nth-child(even) {{ background:#1a1a2e; }}
</style>
</head>
<body>
<h1>Cosmic Anomaly Detector — Batch Report</h1>
<p>Run ID: <code>{result.run_id}</code></p>
<p>Files: {result.total_files} submitted &nbsp;|&nbsp;
        {result.n_ok} OK &nbsp;|&nbsp;
        {result.n_errors} errors &nbsp;|&nbsp;
        {result.high_score_files} with candidates</p>
<p>Elapsed: {result.elapsed_s:.1f}s</p>
<hr>
<table>
<thead><tr>
  <th>File</th><th>Algorithm</th><th>Type</th><th>Score</th><th>Brightness</th>
</tr></thead>
<tbody>
{rows_html if rows_html else '<tr><td colspan="5">No detections above threshold</td></tr>'}
</tbody>
</table>
</body>
</html>"""
        html_path = output_path / 'report.html'
        html_path.write_text(html)
        logger.info("HTML → %s", html_path)
    except Exception as exc:
        logger.warning("HTML export failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Batch JWST/SETI anomaly detection with GPU parallel processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument('--input-dir',  metavar='DIR',
                     help='Directory of FITS/image files to analyse')
    src.add_argument('--fits-list',  metavar='FILE',
                     help='Text file listing one FITS path per line')
    src.add_argument('--seti',       action='store_true',
                     help='Source targets from SETI priority catalog + download from MAST')

    parser.add_argument('--data-dir',  default='data/seti_downloads',
                        metavar='DIR',
                        help='Local data directory when using --seti (default: data/seti_downloads)')
    parser.add_argument('--workers',   type=int,   default=0,
                        help='Worker processes (0 = auto: cpu_count//2)')
    parser.add_argument('--device',    default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Compute device (default: auto)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Minimum anomaly score to report (default: 0.3)')
    parser.add_argument('--output',    default='output/batch',
                        metavar='DIR',
                        help='Output directory for results (default: output/batch)')
    parser.add_argument('--format',    default='json',
                        choices=['json', 'csv', 'html', 'all'],
                        help='Output format (default: json)')
    parser.add_argument('--sigma',     type=float, default=4.0,
                        help='Source extraction sigma threshold (default: 4.0)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not recurse into subdirectories')
    parser.add_argument('--max-files', type=int, default=0,
                        help='Limit total files (0 = all)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable DEBUG logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Collect file list ────────────────────────────────────────────────
    file_list: List[str] = []

    if args.seti:
        logger.info("Sourcing targets from SETI priority catalog…")
        from cosmic_anomaly_detector.utils.seti_catalog import SETICatalog
        catalog = SETICatalog()
        cat_result = catalog.get_targets(priority_min='medium')
        catalog.summary(cat_result)

        # Try to find locally downloaded FITS in --data-dir
        data_dir = Path(args.data_dir)
        if data_dir.exists():
            file_list = find_images(str(data_dir), recursive=not args.no_recursive)
            logger.info("Found %d local FITS files in %s", len(file_list), data_dir)
        else:
            logger.info("data_dir %s not found — attempting MAST download…", data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            top_targets = cat_result.targets[:10]
            downloaded = catalog.download_jwst_data(
                top_targets,
                output_dir=str(data_dir),
                max_products_per_target=3,
            )
            for paths in downloaded.values():
                file_list.extend(paths)
            logger.info("Downloaded %d files", len(file_list))

    elif args.fits_list:
        file_list = load_fits_list(args.fits_list)
        logger.info("Loaded %d paths from %s", len(file_list), args.fits_list)

    elif args.input_dir:
        file_list = find_images(args.input_dir, recursive=not args.no_recursive)
        logger.info("Discovered %d files in %s", len(file_list), args.input_dir)

    else:
        parser.print_help()
        sys.exit(0)

    if args.max_files and len(file_list) > args.max_files:
        logger.info("Limiting to first %d files (--max-files)", args.max_files)
        file_list = file_list[:args.max_files]

    if not file_list:
        logger.error("No files to process.  Check --input-dir or --seti --data-dir.")
        sys.exit(1)

    # ── Device info ──────────────────────────────────────────────────────
    try:
        from cosmic_anomaly_detector.processing.gpu_accelerator import get_accelerator
        acc = get_accelerator(args.device)
        info = acc.device_info()
        logger.info(
            "Device: %s | CUDA available: %s | torch: %s",
            info.get('device'), info.get('cuda_available'), info.get('torch_version', 'n/a'),
        )
    except Exception as exc:
        logger.warning("Could not get device info: %s", exc)

    # ── Build BatchConfig ────────────────────────────────────────────────
    from cosmic_anomaly_detector.processing.parallel_pipeline import BatchConfig, BatchPipeline

    n_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 4) // 2)
    cfg = BatchConfig(
        output_dir=args.output,
        n_workers=n_workers,
        score_threshold=args.threshold,
        device=args.device,
        sigma_threshold=args.sigma,
    )

    logger.info(
        "Batch config: workers=%d  threshold=%.2f  device=%s",
        cfg.n_workers, cfg.score_threshold, cfg.device,
    )

    # ── Run ──────────────────────────────────────────────────────────────
    pipeline = BatchPipeline(cfg)
    result = pipeline.run(file_list)

    # ── Export ──────────────────────────────────────────────────────────
    out_dir = Path(args.output) / result.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt in ('csv', 'all'):
        export_csv(result, out_dir)
    if fmt in ('html', 'all'):
        export_html(result, out_dir)
    # JSON is always written by BatchPipeline (results.jsonl + summary.json)

    print(f"\nResults → {out_dir}")
    print(f"  results.jsonl : line-delimited per-file JSON")
    print(f"  summary.json  : aggregate statistics")
    if fmt in ('csv', 'all'):
        print(f"  detections.csv: tabular detections")
    if fmt in ('html', 'all'):
        print(f"  report.html   : human-readable report")

    # Exit code reflects whether anomalies were found
    sys.exit(0 if result.n_ok > 0 else 1)


if __name__ == '__main__':
    main()
