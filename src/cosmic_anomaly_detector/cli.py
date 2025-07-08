#!/usr/bin/env python3
"""
Command Line Interface for Cosmic Anomaly Detector

Provides command-line access to image analysis functionality.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

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
            from cosmic_anomaly_detector.gui.main_window import main as gui_main
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
