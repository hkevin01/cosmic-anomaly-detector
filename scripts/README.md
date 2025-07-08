# Scripts

This directory contains utility scripts for the Cosmic Anomaly Detector project.

## Available Scripts

- `download_jwst_data.py` - Download JWST images from NASA archives
- `batch_analyze.py` - Batch analysis of multiple images
- `generate_report.py` - Generate analysis reports
- `train_models.py` - Train and update detection models
- `visualize_results.py` - Create visualizations of detection results

## Usage

Each script includes detailed help documentation. Use `python script_name.py --help` for usage information.

## Examples

```bash
# Download recent JWST images
python download_jwst_data.py --target "Andromeda Galaxy" --days 30

# Analyze a batch of images
python batch_analyze.py --input-dir /path/to/images --output results.json

# Generate a report
python generate_report.py --results results.json --format html
```
