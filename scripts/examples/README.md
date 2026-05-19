# Cosmic Anomaly Detector — Example Scripts

Example scripts demonstrating new detection algorithms and JWST data access.

## Scripts

### 01 — JWST Data Access
```bash
python scripts/examples/01_jwst_data_access.py
```
Queries MAST for real JWST observations of **Stephan's Quintet** via `astroquery.mast`,
downloads up to 3 calibrated science FITS files, and runs the full anomaly detection
pipeline. Falls back to a synthetic FITS image if no network is available.

**Targets pre-configured:**
| <sub>Target</sub> | <sub>Instrument</sub> | <sub>Program</sub> | <sub>Description</sub> |
|--------|-----------|---------|-------------|
| <sub>Stephan's Quintet</sub> | <sub>NIRCam</sub> | <sub>2732</sub> | <sub>Compact galaxy group — tidal forces</sub> |
| <sub>Carina Nebula</sub> | <sub>NIRCam</sub> | <sub>2731</sub> | <sub>Star-forming region</sub> |
| <sub>SMACS 0723</sub> | <sub>NIRCam</sub> | <sub>2736</sub> | <sub>First deep field — gravitational lensing</sub> |
| <sub>Pillars of Creation</sub> | <sub>NIRCam</sub> | <sub>2739</sub> | <sub>Eagle Nebula star-forming pillars</sub> |
| <sub>Cartwheel Galaxy</sub> | <sub>NIRCam</sub> | <sub>2727</sub> | <sub>Ringed collision remnant</sub> |
| <sub>WR 140</sub> | <sub>MIRI</sub> | <sub>2024</sub> | <sub>Wolf-Rayet binary — concentric dust shells</sub> |

### 02 — IR Excess / Dyson Sphere SED Analysis
```bash
python scripts/examples/02_ir_excess_detection.py
```
Demonstrates `IRExcessDetector` on a synthetic 2-source scene:
- **Source A**: Normal stellar point source (compact, symmetric)
- **Source B**: Dyson-sphere candidate (bright core + diffuse warm halo)

The detector fits a star + blackbody SED grid, reporting the best-fit covering
factor **γ** and Dyson-sphere temperature. Based on:

> Suazo et al. 2024, MNRAS 531, 695 — *Project Hephaistos II: Dyson sphere
> candidates from Gaia, 2MASS, and WISE* (arXiv:2405.02927)

### 03 — Wavelet vs Baseline Source Detection
```bash
python scripts/examples/03_wavelet_detection_demo.py
```
Compares three detection methods on a crowded synthetic field with **14 injected
point sources** at three flux levels (bright/medium/faint):

| <sub>Method</sub> | <sub>Algorithm</sub> | <sub>Reference</sub> |
|--------|-----------|-----------|
| <sub>Baseline sigma-threshold</sub> | <sub>Simple 3σ clipping</sub> | <sub>Standard practice</sub> |
| <sub>Wavelet (starlet à trous)</sub> | <sub>B3-spline multi-scale decomposition</sub> | <sub>Starck & Murtagh 2002</sub> |
| <sub>Matched filter</sub> | <sub>Optimal linear filter (Gaussian PSF)</sub> | <sub>Turin 1960</sub> |

## Requirements
```
pip install astropy astroquery scipy numpy
```

## Algorithm References

| <sub>Algorithm</sub> | <sub>Implementation</sub> | <sub>Paper</sub> |
|-----------|----------------|-------|
| <sub>IR Excess / SED fitting</sub> | <sub>`IRExcessDetector`</sub> | <sub>Suazo et al. 2024, MNRAS 531, 695</sub> |
| <sub>Starlet wavelet detection</sub> | <sub>`WaveletSourceDetector`</sub> | <sub>Starck & Murtagh 2002</sub> |
| <sub>Matched filter</sub> | <sub>`MatchedFilterDetector`</sub> | <sub>Turin 1960, IRE Trans.</sub> |
| <sub>Microlensing anomaly</sub> | <sub>`MicrolensingAnomalyDetector`</sub> | <sub>Paczyński 1986; arXiv:2512.07924</sub> |

All algorithms are in `src/cosmic_anomaly_detector/processing/algorithms.py`.
JWST data access is in `src/cosmic_anomaly_detector/utils/jwst_access.py`.