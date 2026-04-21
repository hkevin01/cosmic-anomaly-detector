<div align="center">
  <h1>🛸 Cosmic Anomaly Detector</h1>
  <p><em>AI-powered analysis of James Webb Space Telescope imagery to detect Dyson spheres, megastructures, and gravitational anomalies — potential signatures of intelligent extraterrestrial life.</em></p>
</div>

<div align="center">

[![License](https://img.shields.io/github/license/hkevin01/cosmic-anomaly-detector?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/hkevin01/cosmic-anomaly-detector?style=flat-square)](https://github.com/hkevin01/cosmic-anomaly-detector/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/hkevin01/cosmic-anomaly-detector?style=flat-square)](https://github.com/hkevin01/cosmic-anomaly-detector/network)
[![Last Commit](https://img.shields.io/github/last-commit/hkevin01/cosmic-anomaly-detector?style=flat-square)](https://github.com/hkevin01/cosmic-anomaly-detector/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/hkevin01/cosmic-anomaly-detector?style=flat-square)](https://github.com/hkevin01/cosmic-anomaly-detector)
[![Issues](https://img.shields.io/github/issues/hkevin01/cosmic-anomaly-detector?style=flat-square)](https://github.com/hkevin01/cosmic-anomaly-detector/issues)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Astropy](https://img.shields.io/badge/astropy-7.x-orange?style=flat-square&logo=python)](https://www.astropy.org)
[![astroquery](https://img.shields.io/badge/astroquery-0.4%2B-blueviolet?style=flat-square)](https://astroquery.readthedocs.io)
[![NumPy](https://img.shields.io/badge/numpy-scientific%20computing-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![Tests](https://img.shields.io/badge/tests-12%20passing-brightgreen?style=flat-square)](tests/)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Detection Algorithms](#detection-algorithms)
- [Real JWST Results](#real-jwst-results)
- [Architecture](#architecture)
- [Usage Flow](#usage-flow)
- [Detection Breakdown](#detection-breakdown)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Example Scripts](#example-scripts)
- [Core Capabilities](#core-capabilities)
- [Project Roadmap](#project-roadmap)
- [Development Status](#development-status)
- [Contributing](#contributing)
- [License & Acknowledgements](#license--acknowledgements)

---

## Overview

**Cosmic Anomaly Detector** is a scientific Python platform that applies computer vision, orbital mechanics validation, and ensemble machine learning to raw FITS imagery from the James Webb Space Telescope. It automates the search for structures inconsistent with natural astrophysical processes — including Dyson spheres, megastructures, and objects that violate Kepler's laws — flagging candidates with confidence scores for follow-up scientific investigation.

The system is designed for astrophysicists, SETI researchers, and data scientists who need a reproducible, batch-capable pipeline for hypothesis-driven anomaly detection at scale. Every detection is accompanied by gravitational compliance scores, orbital parameter analysis, and structured JSON output for downstream analysis.

> [!IMPORTANT]
> All four detection algorithm implementations are active and physics-validated: starlet wavelet decomposition, Gaussian matched filter, IR excess SED fitting (Suazo et al. 2024), and microlensing magnification anomaly detection (Paczyński 1986). JWST data is accessible live via MAST using `astroquery`. Classification ensemble uses heuristic models; real pretrained weights are planned for Phase 5.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Key Features

| Icon | Feature | Description | Impact | Status |
|------|---------|-------------|--------|--------|
| 🔭 | FITS Image Pipeline | Automated ingestion and preprocessing of JWST FITS files using astropy | High | ✅ Stable |
| ⚖️ | Gravitational Validation | Kepler's law compliance scoring, orbital mechanics analysis, mass anomaly detection | High | ✅ Stable |
| 🌊 | Wavelet Source Detection | Starlet (isotropic undecimated wavelet) à trous decomposition — recovers faint sources missed by sigma-clipping | High | ✅ Stable |
| 🎯 | Matched Filter Detection | Statistically optimal Gaussian PSF matched filter at multiple FWHM scales; MAD noise estimate | High | ✅ Stable |
| ♨️ | Multi-Band IR Excess / Dyson Sphere SED | Genuine aperture photometry across ≥2 JWST bands; joint (T★, T_DS, γ) SED grid-search in log-flux space | High | ✅ Stable |
| 🌀 | Multi-Epoch Microlensing | Paczyński (1986) χ² light-curve fitting across time-ordered epoch images; flags compact-lens & anomalous variability | High | ✅ Stable |
| 🛰️ | JWST Live Data Access | `JWSTDataFetcher` queries MAST via astroquery, downloads calibrated Level 2/3 FITS products | High | ✅ Stable |
| 🤖 | Ensemble Classifier | Multi-model AI stack: Dyson sphere detector, megastructure classifier, geometric anomaly detector | High | 🟡 Heuristic |
| 🔍 | Gravitational Lensing Analysis | Einstein radius calculation, magnification factor, distortion pattern analysis | Medium | ✅ Stable |
| 📊 | Structured Reporting | JSON results, Markdown reports, summary statistics per run | High | ✅ Stable |
| 🖥️ | CLI & GUI | `cosmic-analyze` CLI for batch runs; PyQt GUI for interactive exploration | Medium | ✅ Stable |
| 🔁 | Batch Processing | Multi-file recursive directory analysis with run manifest tracking | Medium | ✅ Stable |
| 📋 | Run Manifest | Reproducible run tracking with metadata, timestamps, and output artifacts | Medium | ✅ Stable |

**Highlight capabilities:**
- Validates orbital physics against Kepler's third law with configurable tolerance thresholds
- Detects gravitational lensing signatures including Einstein radius and background source distortion
- Conservative classification design minimizes false positives — only high-confidence detections are elevated
- All outputs are serializable JSON/Markdown for integration with downstream scientific toolchains
- Supports both single-image and recursive directory batch analysis modes

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Detection Algorithms

Four independent, peer-reviewed detection algorithms are implemented in `src/cosmic_anomaly_detector/processing/algorithms.py`. Each targets a distinct physical signature of anomalous structures.

| Algorithm | Class | Physical Basis | Reference |
|-----------|-------|----------------|-----------|
| **Starlet Wavelet Detection** | `WaveletSourceDetector` | B3-spline à trous decomposition across 4 dyadic scales; local maxima above MAD-σ threshold | Starck & Murtagh 2002 |
| **Matched Filter Detection** | `MatchedFilterDetector` | Gaussian PSF convolution at multiple FWHM scales; MAD noise estimation; flux-weighted centroid | Turin 1960, IRE Trans. |
| **Multi-Band IR Excess** | `MultiBandIRExcessDetector` | Aperture photometry in ≥2 JWST bands; grid-search (γ, T★, T_DS) to fit $F = (1-γ)B_ν(T_\star) + γB_ν(T_{DS})$; RMSE in magnitudes | Suazo et al. 2024, MNRAS 531, 695 |
| **Multi-Epoch Microlensing** | `MultiEpochMicrolensingDetector` | Paczyński χ² grid-search over (t₀, tE, u₀); flags compact-lens events and anomalous variability | Paczyński 1986 ApJ 304; arXiv:2512.07924 |

```python
from cosmic_anomaly_detector.processing.algorithms import run_all_algorithms
import numpy as np

image = np.random.rand(512, 512).astype('float32')  # or load a FITS array
candidates = run_all_algorithms(image)

for c in candidates:
    print(f"[{c['algorithm']}] score={c['score']:.3f}  pos={c['coordinates']}")
```

All algorithms share the `AlgorithmCandidate` result schema which maps directly to the `detected_objects` format used throughout the pipeline — candidates can be merged with standard detections without downstream changes.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Real JWST Results

The pipeline was run against two real JWST Early Release Observation datasets downloaded live from the [MAST archive](https://mast.stsci.edu/). The detection map below shows the three-panel output for SMACS J0723.3-7327 observed with MIRI at 7.7 µm.

![Cosmic Anomaly Detector — SMACS J0723 MIRI F770W detection map](docs/detection_map_smacs0723_miri_f770w.png)

*Left: raw normalised FITS image. Centre: sigma-clipping source catalogue (500 sources, 3σ). Right: multi-algorithm candidates — cyan circles = wavelet, yellow squares = matched filter.*

### SMACS J0723.3-7327 — MIRI/F770W (7.7 µm)

| Property | Value |
|----------|-------|
| Program | JWST ERO 02736 (Early Release Observation 10) |
| Target | SMACS J0723.3-7327 — gravitational lens cluster, z ≈ 0.39 |
| Instrument / Filter | MIRI / F770W — 7.7 µm mid-infrared |
| Exposure | 557.8 s |
| Image size | 1032 × 1024 px · 0.11″/px · 1.9′ × 1.9′ FoV |
| DQ-masked pixels | 483,790 / 1,056,768 (45.8%) — MIRI detector gaps |
| **Wavelet detections** | **1,020** sources across 4 spatial scales |
| **Matched-filter sources** | **216** point sources, peak SNR = 14.7 |
| Top wavelet source | RA = 110.820°, Dec = −73.465° · score = 0.841 (near cluster core) |
| Top matched-filter source | RA = 110.853°, Dec = −73.457° · SNR = 14.7 |

### NGC 7320 / Stephan's Quintet — NIRCam/F200W (2.0 µm)

| Property | Value |
|----------|-------|
| Program | JWST ERO 02732 (Early Release Observation 6) |
| Target | NGC 7320 / Stephan's Quintet — compact group of galaxies |
| Instrument / Filter | NIRCam / F200W — 2.0 µm near-infrared |
| Exposure | 236.2 s |
| Image size | 2048 × 2048 px · 0.031″/px · 1.1′ × 1.1′ FoV |
| DQ-masked pixels | 140,710 / 4,194,304 (3.4%) — very clean detector |
| **Wavelet detections** | **379** sources, 30 high-confidence (score > 0.3) |
| Top wavelet source | RA = 338.931°, Dec = 33.967° · score = 0.478 |

### Detection Methodology

**Preprocessing.** Raw Level-2b calibrated FITS (`_cal.fits`) files are loaded from the `SCI` extension (flux in MJy/sr). The `DQ` (data quality) bitmask flags bad pixels (charge migration, saturated, cosmic rays) which are replaced with the science-region median. A 150 px hard-edge exclusion plus 30 px binary erosion produces a `source_mask` that keeps source positions ≥ 180 px from any masked boundary — preventing the IR Gaussian filter from sampling fill values across the detector edge. Flux is normalised to [0, 1] using the 1st–99th percentile of science pixels only.

**Starlet wavelet detection** (`WaveletSourceDetector`). The à trous algorithm convolves the image with a B₃-spline kernel dilated by factors of 1, 2, 4, 8 pixels. Wavelet planes are the difference between consecutive approximations; local maxima above a MAD-σ threshold in each plane correspond to sources at that spatial scale. The approach is scale-adaptive — compact point sources appear at scale j=0, extended galaxies at j=2–3 — and naturally rejects large-scale background gradients.

**Matched-filter detection** (`MatchedFilterDetector`). A Gaussian PSF model is convolved with the background-subtracted image at FWHM = 2–6 px. The peak SNR map is thresholded at 6σ (MAD-estimated noise). Flux-weighted centroids are computed for each connected region. Deduplication prevents the same source appearing at multiple FWHM scales. This is the statistically optimal linear estimator for a known PSF in stationary Gaussian noise (Turin 1960).

**Multi-band IR excess** (`MultiBandIRExcessDetector`). Requires ≥ 2 independent JWST band images covering optical and mid-IR wavelengths. Circular aperture photometry (radius 3 px) is extracted at each source position in every band. A joint SED model $F_\lambda = (1-\gamma)\,B_\nu(T_\star,\lambda) + \gamma\,B_\nu(T_{DS},\lambda)$ is fitted via grid-search over stellar temperature T★ ∈ [3000, 30000 K], Dyson-sphere temperature T_DS ∈ [100, 700 K], and covering factor γ ∈ [0.01, 0.9]. The fit quality is measured as RMSE in log₁₀-flux space (equivalent to RMSE in magnitudes). Sources with RMSE < 0.15 mag and γ > 0.05 are flagged — following the Project Hephaistos methodology (Suazo et al. 2024, MNRAS 531, 695).

**Multi-epoch microlensing** (`MultiEpochMicrolensingDetector`). Accepts a time-ordered list of (image, JD) tuples (minimum 3 epochs). Aperture photometry at each source position builds a light curve F(t). A variability test (χ²/dof vs flat model) gates further analysis. The standard Paczyński (1986) single-lens curve $A(t) = (u^2+2)\,(u\sqrt{u^2+4})^{-1}$ with $u(t)=\sqrt{u_0^2+((t-t_0)/t_E)^2}$ is fitted by grid-searching (t₀, tE, u₀) with f_source and f_blend solved analytically per grid point (weighted least squares). Two anomaly classes are declared: (A) *compact lens* — the Paczyński fit is good and u₀ < 0.3 θ_E (requires a very compact, dense lens); (B) *anomalous variability* — the source is variable but the Paczyński model does not fit (binary lens, extended source, or exotic magnification).

> [!NOTE]
> Results from a single FITS frame (single-band, single-epoch) are used to demonstrate wavelet and matched-filter performance. The multi-band IR excess and multi-epoch microlensing detectors require genuine multi-band or time-series JWST data respectively — run `scripts/analyze_real_jwst.py --fits <path>` against your own FITS files.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Architecture

The system is organized into four primary layers: ingestion, processing, analysis, and reporting. Each layer is independently testable and loosely coupled via data-class interfaces.

```mermaid
flowchart TD
    A[JWST FITS Image / MAST Download] --> B[ImageProcessor]
    B --> C[Preprocessing & Noise Reduction]
    C --> D[Object Detection]
    C --> W[algorithms.py]
    W --> W1[WaveletSourceDetector]
    W --> W2[MatchedFilterDetector]
    W --> W3[IRExcessDetector]
    W --> W4[MicrolensingAnomalyDetector]
    W1 & W2 & W3 & W4 --> D
    D --> E{Analysis Pipeline}
    E --> F[OrbitalMechanicsCalculator]
    E --> G[MassEstimator]
    E --> H[LensingAnalyzer]
    F & G & H --> I[GravitationalAnalyzer]
    I --> J[BaselineAnomalyScorer]
    J --> K[ArtificialStructureClassifier]
    K --> L{Confidence Threshold}
    L -->|≥ 0.7| M[DetectionResult - Flagged]
    L -->|< 0.7| N[DetectionResult - Natural]
    M & N --> O[ReportGenerator]
    O --> P[results.json]
    O --> Q[report.md]
    O --> R[summary.json]
```

**Component responsibilities:**

| Component | Module | Role |
|-----------|--------|------|
| `ImageProcessor` | `processing/image_processor.py` | FITS ingestion, noise reduction, object segmentation |
| `WaveletSourceDetector` | `processing/algorithms.py` | Starlet à trous multi-scale source detection |
| `MatchedFilterDetector` | `processing/algorithms.py` | Optimal Gaussian PSF matched filter |
| `IRExcessDetector` | `processing/algorithms.py` | Dyson sphere SED fitting (Suazo et al. 2024) |
| `MicrolensingAnomalyDetector` | `processing/algorithms.py` | Paczyński magnification anomaly detection |
| `JWSTDataFetcher` | `utils/jwst_access.py` | MAST archive search and FITS download via astroquery |
| `OrbitalMechanicsCalculator` | `core/analyzer.py` | Orbital parameter derivation, Kepler validation |
| `MassEstimator` | `core/analyzer.py` | Mass estimation via photometry and dynamics |
| `GravitationalAnalyzer` | `core/analyzer.py` | Aggregates gravitational anomaly scores |
| `BaselineAnomalyScorer` | `core/baseline.py` | Statistical baseline comparison |
| `ArtificialStructureClassifier` | `core/classifier.py` | Ensemble ML classification (natural vs. artificial) |
| `AnomalyDetector` | `core/detector.py` | Orchestrates full detection pipeline |
| `ReportGenerator` | `reporting/report.py` | Structured output generation |

Data flows from raw FITS bytes through physics-validated orbital mechanics, then into the ML classifier. Each stage emits structured dataclasses (`DetectionResult`, `GravitationalAnomalyResult`, `LensingSignature`) so any stage can be tested or replaced independently.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Usage Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as cosmic-analyze CLI
    participant Detector as AnomalyDetector
    participant Physics as GravitationalAnalyzer
    participant AI as ArtificialStructureClassifier
    participant FS as Output Files

    User->>CLI: cosmic-analyze analyze data/jwst/ --recursive
    CLI->>Detector: analyze_image(fits_path)
    Detector->>Physics: validate_kepler_laws(orbital_params)
    Physics-->>Detector: kepler_compliance_score, anomaly_scores
    Detector->>AI: classify(processed_data)
    AI-->>Detector: classification_results {confidence, type}
    Detector-->>CLI: DetectionResult
    CLI->>FS: results.json, report.md, summary.json
    FS-->>User: Structured output under output/runs/<run_id>/
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Detection Breakdown

The classifier ensemble targets four primary anomaly categories. The current distribution of detection targets reflects the project's research priorities:

```mermaid
pie title Anomaly Classification Targets
    "Dyson Sphere Candidates" : 35
    "Megastructure Signatures" : 25
    "Gravitational Physics Violations" : 25
    "Geometric Anomalies" : 15
```

| Category | Detector Model | Key Signal | False Positive Risk |
|----------|---------------|------------|---------------------|
| Dyson Sphere | `IRExcessDetector` + `dyson_sphere_detector` | IR excess (γ ≥ 0.05) + orbital compliance failure | Low |
| Megastructure | `megastructure_classifier` + `WaveletSourceDetector` | Geometric regularity + multi-scale source profile | Medium |
| Gravity Violation | `OrbitalMechanicsCalculator` + `MicrolensingAnomalyDetector` | Kepler deviation > threshold; magnification ratio anomaly | Low |
| Geometric Anomaly | `geometric_anomaly_detector` + `MatchedFilterDetector` | Non-natural shape signatures + SNR-resolved point sources | Medium |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Technology Stack

| Technology | Purpose | Why Chosen | Alternatives |
|------------|---------|------------|--------------|
| Python 3.9+ | Core runtime | Scientific ecosystem maturity | Julia (less tooling) |
| astropy 7.x | FITS I/O, coordinates, units, WCS | Astronomy standard library | fitsio (no WCS/units) |
| astroquery 0.4+ | MAST archive access, SIMBAD resolution | Official STScI-supported JWST interface | manual MAST API calls |
| NumPy / SciPy | Orbital mechanics, wavelet transform, matched filter | Performance + scientific correctness | PyTorch tensors (overkill) |
| OpenCV / advanced_cv | Object detection, image segmentation | Real-time CV pipelines | scikit-image (slower) |
| scikit-learn | Baseline ML, ensemble classifiers | Rapid iteration, interpretability | PyTorch (future phase) |
| PyQt5 / GUI | Interactive exploration interface | Native desktop performance | Electron (heavy) |
| Pydantic | Config validation and schema enforcement | Type-safe config at boundaries | dataclasses (no validation) |
| pytest + coverage | Test suite, 80% coverage gate | CI reproducibility | unittest (verbose) |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Setup & Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- pip 23+
- Git

> [!NOTE]
> A CUDA-capable GPU is optional but recommended for large batch analysis runs. The CPU-only path is fully supported.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/hkevin01/cosmic-anomaly-detector.git
cd cosmic-anomaly-detector

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install runtime dependencies
pip install -r requirements.txt

# 4. Install development dependencies (optional)
pip install -r requirements-dev.txt

# 5. Install the package in editable mode
pip install -e .

# 6. Verify installation
cosmic-analyze --help
```

> [!TIP]
> MAST archive access works without any API key for public data. The `JWSTDataFetcher` in `utils/jwst_access.py` handles authentication automatically for proprietary data if MAST credentials are configured via `~/.astropy/config/astroquery.cfg`.

### Verify with sample data

```bash
cosmic-analyze analyze scripts/ --limit 1 --run-id VERIFY_001
# Expected: output/runs/VERIFY_001/ created with results.json and report.md
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Usage

### CLI

```bash
# Analyze a single FITS file
cosmic-analyze analyze data/sample.fits

# Analyze a directory recursively
cosmic-analyze analyze data/jwst/ --recursive

# Limit files and assign a run ID
cosmic-analyze analyze data/jwst/ --limit 10 --run-id SURVEY_2026_Q2
```

Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to gracefully stop a running batch job. The run manifest saves progress so the job can be resumed.

**Output artifacts per run** (written to `output/runs/<run_id>/`):

| File | Description |
|------|-------------|
| `results.json` | Full structured anomaly data with confidence scores |
| `report.md` | Markdown summary report for human review |
| `summary.json` | Lightweight statistics for dashboard ingestion |
| `thumbnail.png` | Annotated image visualization |

### Python API

```python
from cosmic_anomaly_detector.core.detector import AnomalyDetector

detector = AnomalyDetector()
result = detector.analyze_image("data/jwst_image.fits")

print(f"Total anomalies: {len(result.anomalies)}")
print(f"High-confidence: {len(result.get_high_confidence_anomalies(threshold=0.8))}")
print(f"Gravitational analysis: {result.gravitational_analysis}")
```

### Batch Script

```bash
python scripts/batch_analyze.py --input data/jwst_batch/ --output output/batch_001/
```

### JWST Live Data

```python
from cosmic_anomaly_detector.utils.jwst_access import JWSTDataFetcher

fetcher = JWSTDataFetcher(output_dir='data/jwst_downloads', max_products=5)

# Download by well-known target name
paths = fetcher.search_and_download(
    target_name="Stephan's Quintet",
    instrument='NIRCam',
)

# Download by coordinates
paths = fetcher.search_and_download(
    ra_deg=339.0142, dec_deg=33.9656,
    instrument='MIRI',
    radius_arcmin=3.0,
)

# Download by program ID
paths = fetcher.download_by_program_id(program_id=2731, instrument='NIRCam')
```

### New Detection Algorithms

```python
from cosmic_anomaly_detector.processing.algorithms import (
    WaveletSourceDetector,
    IRExcessDetector,
    MicrolensingAnomalyDetector,
    run_all_algorithms,
)
import numpy as np

image = np.load('my_jwst_image.npy')  # 2-D float32, normalised [0,1]

# Run a single algorithm
wavelet = WaveletSourceDetector(n_scales=4, sigma_threshold=3.5)
candidates = wavelet.detect(image)

# Or run all four and get a merged list
all_candidates = run_all_algorithms(image)
for c in all_candidates:
    print(f"{c['algorithm']:25s}  score={c['score']:.3f}  "
          f"pos={c['coordinates']}")
```

<details>
<summary>📋 Full CLI Option Reference</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `--recursive` | False | Recurse into subdirectories |
| `--limit N` | None | Max files to process |
| `--run-id ID` | auto | Custom run identifier |
| `--config PATH` | `config.yaml` | Path to config override file |
| `--log-level` | INFO | Logging verbosity (DEBUG/INFO/WARNING) |
| `--output-dir` | `output/runs/` | Base directory for run artifacts |

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Example Scripts

Ready-to-run examples are in `scripts/examples/`:

| Script | What it demonstrates |
|--------|----------------------|
| [01_jwst_data_access.py](scripts/examples/01_jwst_data_access.py) | Query MAST for Stephan's Quintet NIRCam data; run full pipeline; synthetic fallback if offline |
| [02_ir_excess_detection.py](scripts/examples/02_ir_excess_detection.py) | SED grid-search on a 2-source scene — flags DS candidate (γ=0.31, 637 K) vs normal star |
| [03_wavelet_detection_demo.py](scripts/examples/03_wavelet_detection_demo.py) | 14-source crowded field comparison: baseline sigma-clipping vs wavelet vs matched filter |

```bash
# Run any example directly from the project root
python scripts/examples/02_ir_excess_detection.py
python scripts/examples/03_wavelet_detection_demo.py
python scripts/examples/01_jwst_data_access.py   # requires network for MAST download
```

See [scripts/examples/README.md](scripts/examples/README.md) for full details and algorithm references.

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Core Capabilities

### ⚖️ Gravitational Physics Validation

The `OrbitalMechanicsCalculator` derives orbital parameters from position/velocity observations and validates compliance with Kepler's third law ($T^2 \propto a^3$). Deviations beyond the configured tolerance are scored as anomalies.

```python
from cosmic_anomaly_detector.core.analyzer import OrbitalMechanicsCalculator
import numpy as np

calc = OrbitalMechanicsCalculator()
params = calc.calculate_orbital_parameters(
    positions=np.array([1.0, 0.0, 0.0]),   # AU
    velocities=np.array([0.0, 6.28, 0.0]),  # AU/year
    central_mass=1.0                         # solar masses
)
compliance = calc.validate_kepler_laws(params, observed_period=1.0)
print(f"Kepler compliance score: {compliance:.3f}")  # 1.0 = perfect
```

> [!WARNING]
> Gravitational analysis requires accurate mass estimates. Poorly constrained mass inputs will inflate anomaly scores. Always validate input data quality before interpreting results.

### 🔭 Gravitational Lensing Analysis

The `LensingAnalyzer` computes Einstein radius, magnification factor, and background source distortion patterns. Anomalous lensing signatures — where observed deflection does not match predicted mass — are flagged as potential indicators of hidden mass concentrations.

### 🌊 Wavelet & Matched-Filter Source Detection

The `WaveletSourceDetector` implements the starlet (isotropic undecimated wavelet) transform via the à trous algorithm. The image is convolved with a B3-spline scaling function at dyadic scales (1, 2, 4, 8 px). Sources appear as local maxima in individual wavelet planes above a sigma-clipped threshold — naturally handling multi-scale source morphology.

The `MatchedFilterDetector` applies the statistically optimal linear filter for a Gaussian PSF in additive white noise at four FWHM scales, using MAD-based noise estimation for robustness against non-Gaussian tails.

### ♨️ Multi-Band IR Excess / Dyson Sphere SED Fitting

The `MultiBandIRExcessDetector` implements genuine multi-band photometric SED fitting following Project Hephaistos (Suazo et al. 2024, MNRAS 531, 695):

- Extracts circular aperture photometry at each source position in every supplied band
- Simultaneously fits stellar blackbody + Dyson sphere component:
  $$F_\lambda = (1-\gamma)\,B_\nu(T_\star,\lambda) + \gamma\,B_\nu(T_{DS},\lambda)$$
- Grid-searches: $T_\star \in [3000, 30000]\,\text{K}$, $T_{DS} \in [100, 700]\,\text{K}$, $\gamma \in [0.01, 0.9]$
- Flags detections where best-fit RMSE $< 0.15$ mag and $\gamma \geq 0.05$

Unlike a single-band spatial-scale proxy, this uses real inter-band flux ratios — the correct approach for detecting warm dust re-emission characteristic of a Dyson sphere.

```python
from cosmic_anomaly_detector.processing.algorithms import MultiBandIRExcessDetector
import numpy as np

# Supply two or more registered JWST band images (same pixel grid)
bands = {
    1.5:  image_f150w,   # NIRCam F150W — stellar continuum
    3.6:  image_f356w,   # NIRCam F356W — warm dust begins
    12.0: image_miri,    # MIRI F1130W  — DS waste heat peak
}
det = MultiBandIRExcessDetector(bands, rmse_threshold=0.15, min_gamma=0.05)
candidates = det.detect(detected_objects)

for c in candidates:
    m = c.metadata
    print(f"pos={c.coordinates}  γ={m['covering_factor_gamma']:.2f}  "
          f"T_DS={m['ds_temperature_k']:.0f}K  "
          f"T★={m['star_temperature_k']:.0f}K  "
          f"RMSE={m['sed_rmse_mag']:.3f} mag")
```

### 🌀 Multi-Epoch Microlensing Anomaly Detection

The `MultiEpochMicrolensingDetector` fits the Paczyński (1986) single-lens light curve to aperture photometry extracted from a time series of JWST images:

$$A(t) = \frac{u(t)^2 + 2}{u(t)\,\sqrt{u(t)^2 + 4}}, \quad u(t) = \sqrt{u_0^2 + \left(\frac{t - t_0}{t_E}\right)^2}$$

Parameters (t₀, tE, u₀) are found by grid-search; f_source and f_blend are solved analytically per grid point. Two anomaly categories are declared:

| Category | Condition | Interpretation |
|----------|-----------|----------------|
| **Compact lens** | Good Paczyński fit + u₀ < 0.3θ_E | Very dense compact lens — consistent with a megastructure or black hole |
| **Anomalous variability** | Variable source + poor Paczyński fit | Binary lens, extended emission, or exotic magnification |

```python
from cosmic_anomaly_detector.processing.algorithms import MultiEpochMicrolensingDetector

# epochs: list of (calibrated_image_array, julian_date) tuples, sorted by time
epochs = [
    (image_epoch1, 2460000.0),
    (image_epoch2, 2460010.0),
    (image_epoch3, 2460020.0),
    # ... more epochs
]
det = MultiEpochMicrolensingDetector(
    epochs,
    variability_threshold=5.0,   # chi2/dof to call source variable
    chi2_threshold=3.0,           # chi2/dof for good Paczynski fit
    u0_anomaly_threshold=0.3,    # compact-lens flag threshold
)
candidates = det.detect(detected_objects)

for c in candidates:
    m = c.metadata
    print(f"pos={c.coordinates}  type={c.anomaly_type}  score={c.score:.3f}")
    print(f"  t0={m['t0_best_jd']:.1f} JD  tE={m['te_best_days']:.1f}d  "
          f"u0={m['u0_best']:.3f}  A_max={m['a_max']:.2f}")
    print(f"  chi2_flat={m['chi2_flat']:.1f}  chi2_Pac={m['chi2_paczynski']:.2f}")
```

### 🤖 Ensemble AI Classification

Four specialized models target distinct anomaly classes:

<details>
<summary>🤖 Classifier Model Details</summary>

| Model | Target | Key Features Used |
|-------|--------|-------------------|
| `dyson_sphere_detector` | Infrared excess, stellar occlusion patterns | Photometric residuals, IR flux ratio, γ score |
| `megastructure_classifier` | Large-scale geometric regularity | Shape descriptors, spatial frequency |
| `geometric_anomaly_detector` | Non-natural structural geometry | Symmetry metrics, fractal dimension |
| `material_composition_analyzer` | Spectral composition vs. natural stellar matter | Spectral index, absorption features |

All models feed into a confidence-weighted ensemble. Only results exceeding the `confidence_threshold` (default: 0.7) are elevated to `DetectionResult.anomalies`.

</details>

### 📊 Structured Reporting & Run Manifest

Every analysis run generates a timestamped manifest capturing input files, configuration, processing metadata, and output artifact paths. This enables full scientific reproducibility.

<details>
<summary>📄 Sample results.json Structure</summary>

```json
{
  "run_id": "SURVEY_2026_Q2",
  "timestamp": "2026-04-20T14:32:00Z",
  "files_processed": 42,
  "anomalies": [
    {
      "object_id": "JWST-OBJ-00147",
      "anomaly_type": "dyson_sphere_candidate",
      "kepler_compliance_score": 0.41,
      "overall_anomaly_score": 0.87,
      "confidence": 0.83,
      "follow_up_priority": "HIGH",
      "physical_explanation": "Significant infrared excess with orbital period deviation of 59% from Kepler prediction"
    }
  ],
  "summary": {
    "total_objects_analyzed": 1840,
    "high_confidence_detections": 1,
    "mean_kepler_compliance": 0.97
  }
}
```

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Project Roadmap

```mermaid
gantt
    title Cosmic Anomaly Detector Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1 — Foundation
        FITS Pipeline & Image Processing    :done,    p1a, 2025-07-01, 2025-08-01
        Gravitational Analysis Engine       :done,    p1b, 2025-08-01, 2025-09-01
    section Phase 2 — Core Detection
        Baseline Anomaly Scoring            :done,    p2a, 2025-09-01, 2025-10-01
        CLI & Run Manifest                  :done,    p2b, 2025-10-01, 2025-11-01
    section Phase 3 — Classification
        Ensemble Classifier Framework       :done,    p3a, 2025-11-01, 2026-01-01
        Reporting & GUI                     :done,    p3b, 2026-01-01, 2026-03-01
    section Phase 4 — Advanced Algorithms
        Wavelet / Matched Filter Detection  :done,    p4a, 2026-01-01, 2026-03-01
        IR Excess SED Fitting (Hephaistos)  :done,    p4b, 2026-01-01, 2026-03-01
        Microlensing Anomaly Detection      :done,    p4c, 2026-02-01, 2026-04-01
        MAST Archive Live Integration       :done,    p4d, 2026-03-01, 2026-04-01
    section Phase 5 — Intelligence
        Pretrained Model Weights            :active,  p5a, 2026-04-01, 2026-09-01
        Distributed Batch Processing        :         p5b, 2026-09-01, 2026-12-01
        Community Dataset & Open Release    :         p5c, 2026-12-01, 2027-03-01
```

| Phase | Goals | Target | Status |
|-------|-------|--------|--------|
| 1 — Foundation | FITS pipeline, gravitational engine | Q3 2025 | ✅ Complete |
| 2 — Core Detection | Baseline scoring, CLI, run manifest | Q4 2025 | ✅ Complete |
| 3 — Classification | Ensemble framework, reporting, GUI | Q1 2026 | ✅ Complete |
| 4 — Advanced Algorithms | Wavelet, matched filter, IR excess SED, microlensing, MAST live data | Q2 2026 | ✅ Complete |
| 5 — Intelligence | Pretrained model weights, distributed batch, open release | Q3 2026+ | 🟡 In Progress |

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Development Status

| Version | Stability | Tests | Known Limitations |
|---------|-----------|-------|-------------------|
| 0.4.x (current) | Beta | 12/12 passing | ML model weights are heuristic; pretrained weights planned for v0.5 |
| 0.5.x (planned) | Beta | Target 90% coverage | First pretrained weights; GPU acceleration |
| 1.0.0 (planned) | Stable | Target 95% coverage | Full production release |

```mermaid
mindmap
  root((Cosmic Anomaly Detector))
    Image Processing
      FITS Ingestion
      Noise Reduction
      Object Segmentation
    Detection Algorithms
      Wavelet Starlet à trous
      Gaussian Matched Filter
      IR Excess SED Fitting
      Microlensing Anomaly
    JWST Data Access
      MAST Archive Query
      Level 2/3 FITS Download
      7 Notable Targets
    Gravitational Analysis
      Kepler Validation
      Orbital Mechanics
      Lensing Signatures
      Mass Estimation
    AI Classification
      Dyson Sphere Detector
      Megastructure Classifier
      Geometric Anomaly Detector
      Material Composition Analyzer
    Output
      results.json
      report.md
      summary.json
      Run Manifest
    Interfaces
      CLI cosmic-analyze
      Python API
      PyQt GUI
      Example Scripts
```

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## Contributing

We welcome contributions from astrophysicists, ML engineers, and scientific software developers.

**Workflow:**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit with descriptive messages: `git commit -m "feat: add lensing distortion metric"`
4. Push and open a Pull Request against `main`

<details>
<summary>📋 Contribution Guidelines</summary>

**Code Style**
- Format with `black` (line length 79) and sort imports with `isort`
- Add type hints to all new functions
- Use the `get_logger(__name__)` utility from `utils/logging.py` — never `print()`

**Testing Requirements**
- All new features require unit tests in `tests/`
- Gravitational and physics calculations must include `@pytest.mark.scientific` tests
- Maintain ≥ 80% coverage (enforced by CI gate)

**Running Tests**
```bash
pytest                          # Full suite
pytest -m "not slow"            # Skip slow integration tests
pytest tests/test_detector.py   # Single module
```

**Scientific Reproducibility**
- All detection runs must produce identical output given the same inputs and `config.yaml`
- Log all scientific events using `log_scientific_event()` for auditability

</details>

<p align="right">(<a href="#top">back to top ↑</a>)</p>

---

## License & Acknowledgements

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this software with attribution. See [LICENSE](LICENSE) for the full text.

**Acknowledgements:**
- [NASA James Webb Space Telescope](https://www.jwst.nasa.gov/) for the extraordinary imaging data this project is designed to analyze
- [Astropy Project](https://www.astropy.org/) — the foundational astronomy Python library powering FITS I/O and coordinate transforms
- SETI Institute and the broader astrobiology research community for inspiration and scientific framing
- [shields.io](https://shields.io/) for badge infrastructure

<p align="right">(<a href="#top">back to top ↑</a>)</p>

## 🙏 Acknowledgments

- NASA James Webb Space Telescope Team
- Astropy Community
- OpenCV and TensorFlow communities
- SETI Institute for inspiration

---

*"Two possibilities exist: either we are alone in the Universe or we are not. Both are equally terrifying."* - Arthur C. Clarke
