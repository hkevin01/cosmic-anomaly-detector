<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Cosmic Anomaly Detector - Copilot Instructions

This project analyzes James Webb Space Telescope images to detect artificial structures and gravitational anomalies that might indicate intelligent extraterrestrial life.

## Project Context

- **Purpose**: AI-powered detection of Dyson spheres, megastructures, and objects violating gravitational physics
- **Data Source**: James Webb Space Telescope FITS images and other astronomical data
- **Technologies**: Python, Computer Vision, Deep Learning, Astrophysics

## Key Components

1. **Image Processing**: JWST FITS file handling, noise reduction, object detection
2. **Gravitational Analysis**: Physics validation using Kepler's laws and gravitational lensing
3. **AI Classification**: ML models for natural vs artificial structure classification
4. **Anomaly Detection**: Combined analysis to identify potential signs of intelligence

## Coding Guidelines

- Use astropy for astronomical data handling (FITS files, coordinates, units)
- Apply proper scientific computing practices with numpy/scipy
- Implement robust error handling for space data edge cases
- Include comprehensive logging for scientific reproducibility
- Follow physics-based validation principles
- Use type hints for all functions

## Domain-Specific Considerations

- JWST data comes in FITS format with specific header information
- Gravitational analysis requires careful handling of mass estimates and orbital mechanics
- Classification models should be conservative to minimize false positives
- All detections should include confidence scores and reasoning
- Consider observational biases and instrument limitations

## Architecture Patterns

- Modular design with separate processing, analysis, and classification components
- Data pipeline approach for batch processing multiple images
- Results should be serializable for further scientific analysis
- Support for both real-time analysis and batch processing modes
