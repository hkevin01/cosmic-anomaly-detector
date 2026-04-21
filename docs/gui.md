# Cosmic Anomaly Detector - Quick Start Guide

## 🚀 Quick Start with run.sh

The easiest way to get started is using the automated `run.sh` script:

### 1. First Time Setup
```bash
./run.sh setup
```
This will:
- Check Python version (requires 3.9+)
- Create virtual environment
- Install all dependencies
- Create sample JWST data for testing

### 2. Launch GUI Interface
```bash
./run.sh gui
# or simply
./run.sh
```

### 3. Use Command Line Interface
```bash
./run.sh cli data/samples/jwst_with_anomaly.fits -o results/
```

### 4. Run Tests
```bash
./run.sh test
```

## 🖥️ GUI Interface Features

### Main Features
- **Image Display**: High-quality FITS image visualization with adjustable contrast/brightness
- **Real-time Analysis**: Background processing to keep GUI responsive
- **Object Detection**: Automatic detection and highlighting of astronomical objects
- **Geometric Analysis**: Advanced analysis of object regularity and artificial probability
- **Interactive Results**: Clickable object tables with detailed statistics
- **Export Capabilities**: Save results in JSON format for further analysis

### GUI Components

#### 1. Image Analysis Tab
- **Image Viewer**: Interactive image display with zoom and pan
- **Overlay Controls**: Toggle object detection overlays
- **Image Controls**: Adjust contrast and brightness
- **Processing Log**: Real-time analysis progress and messages

#### 2. Object Analysis Tab
- **Detection Table**: Detailed object properties and coordinates
- **Statistics Panel**: Summary of detection results
- **Confidence Scoring**: Artificial structure probability analysis

#### 3. Configuration Tab
- **Processing Parameters**: Adjust noise reduction and detection thresholds
- **Analysis Settings**: Configure anomaly detection sensitivity
- **Real-time Updates**: Apply settings without restarting

### Supported File Formats
- **.fits, .fit, .fts**: JWST and other space telescope FITS images
- **JSON Export**: Analysis results with object coordinates and properties

## 📊 Computer Vision Pipeline

### Advanced Noise Reduction
- **Adaptive Bilateral Filtering**: Preserves edges while reducing noise
- **Non-local Means Denoising**: Advanced algorithm for astronomical images
- **Wavelet Denoising**: Multi-scale noise reduction (requires PyWavelets)

### Object Detection & Segmentation
- **Multi-scale Detection**: Combines local and global thresholding
- **Watershed Segmentation**: Separates touching astronomical objects
- **Morphological Operations**: Cleans and refines object boundaries

### Geometric Analysis
- **Regularity Scoring**: Quantifies geometric perfection unusual in nature
- **Symmetry Analysis**: Detects bilateral and radial symmetry
- **Edge Sharpness**: Identifies artificially sharp boundaries
- **Shape Classification**: Distinguishes between natural and artificial geometries

### Scale-Invariant Features
- **ORB Feature Detection**: Robust keypoint detection
- **Pattern Analysis**: Identifies regular arrangements and alignments
- **Clustering Analysis**: Detects non-random feature distributions

## 🛠️ Development & Testing

### Run All Tests
```bash
./run.sh test
```

### Create Sample Data
```bash
./run.sh sample-data
```

### Check System Requirements
```bash
./run.sh check
```

### Clean Project
```bash
./run.sh clean
```

## 📁 Project Structure After Setup

```
cosmic-anomaly-detector/
├── run.sh                     # Main run script
├── venv/                      # Python virtual environment
├── data/
│   └── samples/               # Sample JWST FITS files
├── results/                   # Analysis output directory
├── logs/                      # Application logs
├── src/cosmic_anomaly_detector/
│   ├── gui/                   # PyQt5 GUI application
│   ├── processing/            # Computer vision pipeline
│   ├── core/                  # Detection and analysis engines
│   └── utils/                 # Configuration and logging
└── tests/                     # Comprehensive test suite
```

## 🔬 Scientific Methodology

### Detection Criteria
The system identifies potential artificial structures based on:

1. **Geometric Regularity**: Perfect circles, squares, or regular polygons
2. **Bilateral Symmetry**: Symmetric structures unusual in natural formations
3. **Edge Sharpness**: Artificially sharp boundaries and transitions
4. **Pattern Repetition**: Regular arrangements and spacing
5. **Scale Consistency**: Features that maintain properties across scales

### Confidence Scoring
- **Conservative Approach**: High thresholds to minimize false positives
- **Multi-modal Analysis**: Combines visual, geometric, and pattern analysis
- **Physics Validation**: Future integration with gravitational anomaly detection
- **Ensemble Methods**: Multiple detection algorithms for robust results

### Scientific Rigor
- **Reproducible Analysis**: All parameters logged for scientific validation
- **Comprehensive Testing**: Unit tests ensure algorithm correctness
- **Configuration Management**: Systematic parameter tracking
- **Export Capabilities**: Results in standard formats for peer review

## 🚨 Known Limitations

1. **GPU Acceleration**: Not yet implemented (CPU-only processing)
2. **Large Images**: Memory usage may be high for very large FITS files
3. **Real-time Processing**: Analysis time depends on image size and complexity
4. **Training Data**: Initial models use synthetic data (real training data needed)

## 🔧 Troubleshooting

### Common Issues

**PyQt5 Installation Fails**
```bash
# Ubuntu/Debian
sudo apt-get install python3-pyqt5 python3-pyqt5-dev

# Install from source if needed
pip install PyQt5 --no-cache-dir
```

**FITS File Loading Errors**
- Ensure astropy is properly installed
- Check FITS file integrity
- Verify file permissions

**Memory Issues with Large Images**
- Reduce image size in configuration
- Increase system memory limit
- Use image tiling for very large files

**GUI Not Starting**
```bash
# Check display environment
echo $DISPLAY

# For remote connections
ssh -X username@hostname
```

### Performance Optimization

**Speed Up Analysis**
- Reduce image resolution for initial testing
- Adjust object detection thresholds
- Enable multi-threading in configuration

**Memory Optimization**
- Close unused applications
- Adjust batch processing size
- Use image compression for storage

## 📞 Support & Contributing

### Getting Help
- Check the logs in `logs/` directory
- Run `./run.sh check` to verify system requirements
- Examine sample data with `./run.sh sample-data`

### Contributing
- Follow the development setup in the main README
- Run tests before submitting changes
- Use the configuration system for new parameters
- Add comprehensive logging for debugging

## 🎯 Next Steps

After Phase 2 completion, the system will expand to include:
- **Physics Validation**: Gravitational anomaly detection
- **Machine Learning Models**: Trained classification systems
- **Multi-wavelength Analysis**: Integration with multiple telescope data
- **Advanced Visualization**: 3D structure analysis and temporal tracking

---

*For detailed technical documentation, see the full project README and API documentation.*
