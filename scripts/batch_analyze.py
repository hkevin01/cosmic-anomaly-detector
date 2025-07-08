#!/usr/bin/env python3
"""
Batch Analysis Script

Analyze multiple JWST images for anomalies in batch mode.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cosmic_anomaly_detector import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main batch analysis function"""
    parser = argparse.ArgumentParser(
        description="Batch analyze JWST images for anomalies"
    )
    parser.add_argument(
        "--input-dir", 
        required=True,
        help="Directory containing images to analyze"
    )
    parser.add_argument(
        "--output", 
        default="batch_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--config", 
        help="Configuration file path"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.7,
        help="Confidence threshold for reporting anomalies"
    )
    parser.add_argument(
        "--format", 
        choices=["json", "csv", "html"], 
        default="json",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    config = load_config(args.config) if args.config else None
    detector = AnomalyDetector(config)
    
    # Find images
    image_paths = find_images(args.input_dir)
    logger.info(f"Found {len(image_paths)} images to analyze")
    
    # Process images
    results = detector.batch_analyze(image_paths)
    
    # Filter by threshold
    filtered_results = filter_results(results, args.threshold)
    
    # Generate output
    save_results(filtered_results, args.output, args.format)
    
    # Print summary
    stats = detector.get_detection_statistics(results)
    print_summary(stats)


def load_config(config_path: str) -> dict:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def find_images(input_dir: str) -> List[str]:
    """Find image files in directory"""
    image_extensions = {'.fits', '.png', '.jpg', '.jpeg', '.tiff'}
    image_paths = []
    
    for path in Path(input_dir).rglob('*'):
        if path.suffix.lower() in image_extensions:
            image_paths.append(str(path))
    
    return sorted(image_paths)


def filter_results(results: List, threshold: float) -> List:
    """Filter results by confidence threshold"""
    filtered = []
    for result in results:
        high_conf_anomalies = result.get_high_confidence_anomalies(threshold)
        if high_conf_anomalies:
            filtered.append(result)
    return filtered


def save_results(results: List, output_path: str, format_type: str):
    """Save results in specified format"""
    if format_type == "json":
        save_json_results(results, output_path)
    elif format_type == "csv":
        save_csv_results(results, output_path)
    elif format_type == "html":
        save_html_results(results, output_path)


def save_json_results(results: List, output_path: str):
    """Save results as JSON"""
    serializable_results = []
    for result in results:
        serializable_results.append({
            'anomalies': result.anomalies,
            'confidence_scores': result.confidence_scores,
            'metadata': result.processing_metadata
        })
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def save_csv_results(results: List, output_path: str):
    """Save results as CSV"""
    # Placeholder for CSV export
    logger.info(f"CSV export to {output_path} (not implemented)")


def save_html_results(results: List, output_path: str):
    """Save results as HTML report"""
    # Placeholder for HTML export
    logger.info(f"HTML export to {output_path} (not implemented)")


def print_summary(stats: dict):
    """Print analysis summary"""
    print("\n" + "="*50)
    print("BATCH ANALYSIS SUMMARY")
    print("="*50)
    print(f"Images analyzed: {stats['total_images_analyzed']}")
    print(f"Anomalies detected: {stats['total_anomalies_detected']}")
    print(f"High confidence: {stats['high_confidence_anomalies']}")
    print(f"Detection rate: {stats['anomaly_detection_rate']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
