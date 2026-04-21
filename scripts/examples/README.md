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
| Target | Instrument | Program | Description |
|--------|-----------|---------|-------------|
| Stephan's Quintet | NIRCam | 2732 | Compact galaxy group — tidal forces |
| Carina Nebula | NIRCam | 2731 | Star-forming region |
| SMACS 0723 | NIRCam | 2736 | First deep field — gravitational lensing |
| Pillars of Creation | NIRCam | 2739 | Eagle Nebula star-forming pillars |
| Cartwheel Galaxy | NIRCam | 2727 | Ringed collision remnant |
| WR 140 | MIRI | 2024 | Wolf-Rayet binary — concentric dust shells |

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

| Method | Algorithm | Reference |
|--------|-----------|-----------|
| Baseline sigma-threshold | Simple 3σ clipping | Standard practice |
| Wavelet (starlet à trous) | B3-spline multi-scale decomposition | Starck & Murtagh 2002 |
| Matched filter | Optimal linear filter (Gaussian PSF) | Turin 1960 |

## Requirements
```
pip install astropy astroquery scipy numpy
```

## Algorithm References

| Algorithm | Implementation | Paper |
|-----------|----------------|-------|
| IR Excess / SED fitting | `IRExcessDetector` | Suazo et al. 2024, MNRAS 531, 695 |
| Starlet wavelet detection | `WaveletSourceDetector` | Starck & Murtagh 2002 |
| Matched filter | `MatchedFilterDetector` | Turin 1960, IRE Trans. |
| Microlensing anomaly | `MicrolensingAnomalyDetector` | Paczyński 1986; arXiv:2512.07924 |

All algorithms are in `src/cosmic_anomaly_detector/processing/algorithms.py`.
JWST data access is in `src/cosmic_anomaly_detector/utils/jwst_access.py`.
