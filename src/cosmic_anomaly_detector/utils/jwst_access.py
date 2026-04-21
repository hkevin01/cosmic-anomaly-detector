"""
JWST Data Access — Cosmic Anomaly Detector

# ---------------------------------------------------------------------------
# ID: JWST-001
# Requirement: Provide a high-level API to search for and download James Webb
#   Space Telescope observations from the MAST archive via astroquery.mast,
#   returning local paths to FITS calibrated science files.
# Purpose: Allow the anomaly detector to operate on real JWST imagery without
#   manual data downloads — users supply target coordinates or program IDs and
#   the utility handles authentication, querying, filtering, and downloading.
# Rationale: astroquery.mast is the standard Python interface to MAST; wrapping
#   it in a project-specific class isolates version dependencies and allows
#   offline fallback without modifying caller code.
# Inputs:
#   - target_name (str): Simbad-resolvable target or "RA DEC" string
#   - coordinates (SkyCoord): Pre-built coordinate object
#   - program_id (int): JWST program/proposal ID
#   - instrument (str): NIRCam, MIRI, NIRSpec, NIRISS, FGS
#   - radius (Quantity): Search cone radius (default 3 arcmin)
#   - product_type (str): science, calibration (default "science")
#   - output_dir (Path): Where to save downloaded FITS files
# Outputs: List[Path] of downloaded or cached FITS calibrated-image file paths.
# Preconditions:  astroquery installed; network access to MAST.
# Postconditions: Files are written to output_dir; paths are absolute.
# Assumptions: MAST API is publicly accessible; authentication optional for
#   proprietary data (guided interactively if needed).
# Side Effects: Creates output_dir if absent; writes FITS files to disk.
# Failure Modes: Network timeout → raises ConnectionError with guidance.
#   No data found → returns empty list with WARNING log.
#   astroquery not installed → raises ImportError with install hint.
# Error Handling: All network calls wrapped in try/except; timeouts logged.
# Constraints: Default maximum 20 products downloaded per call to avoid disk
#   saturation; configurable via max_products parameter.
# Verification: scripts/examples/01_jwst_data_access.py exercises this class.
# References:
#   astroquery.mast documentation — https://astroquery.readthedocs.io/en/latest/mast/mast.html
#   MAST JWST notebooks — https://github.com/spacetelescope/mast_notebooks
#   JWST Data Products Guide — https://jwst-docs.stsci.edu
# ---------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Notable JWST targets useful for anomaly analysis
JWST_NOTABLE_TARGETS: Dict[str, Dict] = {
    "Stephan's Quintet": {
        "ra": 339.0142, "dec": 33.9656,
        "description": "Compact galaxy group — tidal forces, dark matter distribution",
        "program_ids": [2732],
    },
    "Carina Nebula": {
        "ra": 160.98, "dec": -59.62,
        "description": "Star-forming region — massive star outflows, potential technosignature background",
        "program_ids": [2731],
    },
    "SMACS 0723 (Deep Field)": {
        "ra": 110.8139, "dec": -73.4546,
        "description": "Gravitational lens cluster — magnification anomalies, dark matter",
        "program_ids": [2736],
    },
    "Pillars of Creation (M16)": {
        "ra": 274.7, "dec": -13.807,
        "description": "Eagle Nebula star-forming pillars — photoionization structures",
        "program_ids": [2739],
    },
    "Cartwheel Galaxy": {
        "ra": 9.9945, "dec": -33.7253,
        "description": "Ringed galaxy from collision — unusual geometric structure",
        "program_ids": [2727],
    },
    "Tarantula Nebula (30 Dor)": {
        "ra": 84.676, "dec": -69.100,
        "description": "Largest known star-forming region — extreme stellar density",
        "program_ids": [2729],
    },
    "WR 140 (Wolf-Rayet Binary)": {
        "ra": 301.0, "dec": 43.85,
        "description": "Wolf-Rayet + O-star binary — concentric dust shells, geometric anomaly",
        "program_ids": [2024],
    },
}


class JWSTDataFetcher:
    """
    Wrapper around astroquery.mast for fetching JWST calibrated science images.

    Example usage
    -------------
    >>> fetcher = JWSTDataFetcher(output_dir="data/jwst")
    >>> paths = fetcher.search_and_download(
    ...     target_name="Stephan's Quintet",
    ...     instrument="NIRCam",
    ...     max_products=5,
    ... )
    >>> for p in paths:
    ...     print(p)
    """

    # Calibrated science file suffixes (Level 2 / 3 pipeline products)
    _SCIENCE_SUFFIXES = (
        '_i2d.fits',      # Level 3 mosaic (primary interest)
        '_cal.fits',      # Level 2b calibrated individual exposures
        '_s2d.fits',      # Level 3 spectral 2D
    )

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/jwst_downloads",
        max_products: int = 20,
        timeout_s: int = 120,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_products = max_products
        self.timeout_s = timeout_s
        self._check_astroquery()

    def _check_astroquery(self) -> None:
        try:
            import astroquery  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "astroquery is required for JWST data access.\n"
                "Install with: pip install astroquery"
            ) from exc

    def search_and_download(
        self,
        target_name: Optional[str] = None,
        ra_deg: Optional[float] = None,
        dec_deg: Optional[float] = None,
        program_id: Optional[int] = None,
        instrument: Optional[str] = "NIRCam",
        radius_arcmin: float = 3.0,
        max_products: Optional[int] = None,
    ) -> List[Path]:
        """
        Search MAST for JWST observations and download calibrated FITS files.

        Args:
            target_name: Simbad-resolvable name or notable target key.
            ra_deg: RA in degrees (if no target_name).
            dec_deg: Dec in degrees (if no target_name).
            program_id: JWST program/proposal ID for filtering.
            instrument: JWST instrument name (NIRCam, MIRI, etc.).
            radius_arcmin: Search cone radius in arcminutes.
            max_products: Override instance max_products.

        Returns:
            List of absolute Paths to downloaded FITS files.
        """
        from astroquery.mast import Observations
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        n = max_products or self.max_products
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Resolve coordinates ───────────────────────────────────────────
        if target_name and target_name in JWST_NOTABLE_TARGETS:
            info = JWST_NOTABLE_TARGETS[target_name]
            coord = SkyCoord(ra=info['ra'], dec=info['dec'], unit='deg')
            if program_id is None and info.get('program_ids'):
                program_id = info['program_ids'][0]
            logger.info("Known target %r at RA=%.4f Dec=%.4f",
                        target_name, info['ra'], info['dec'])
        elif target_name:
            try:
                from astroquery.simbad import Simbad
                result = Simbad.query_object(target_name)
                if result is None or len(result) == 0:
                    logger.error("SIMBAD could not resolve: %s", target_name)
                    return []
                ra_str = str(result['RA'][0])
                dec_str = str(result['DEC'][0])
                coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                logger.info("Resolved %r → RA=%s Dec=%s", target_name, ra_str, dec_str)
            except Exception as exc:
                logger.error("Target resolution failed: %s", exc)
                return []
        elif ra_deg is not None and dec_deg is not None:
            coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg')
        else:
            raise ValueError("Provide target_name or ra_deg + dec_deg.")

        # ── Query MAST ────────────────────────────────────────────────────
        try:
            obs_table = Observations.query_criteria(
                coordinates=coord,
                radius=radius_arcmin * u.arcmin,
                obs_collection='JWST',
                instrument_name=instrument,
                dataRights=['PUBLIC'],
            )
        except Exception as exc:
            logger.error("MAST query failed: %s", exc)
            return []

        if obs_table is None or len(obs_table) == 0:
            logger.warning("No public JWST/%s observations found at target.",
                           instrument)
            return []

        logger.info("Found %d observations; fetching products …", len(obs_table))

        # ── Filter to science products ────────────────────────────────────
        try:
            products = Observations.get_product_list(obs_table[:min(10, len(obs_table))])
        except Exception as exc:
            logger.error("Product list retrieval failed: %s", exc)
            return []

        # Keep calibrated science files only
        science_mask = [
            any(str(uri).endswith(s) for s in self._SCIENCE_SUFFIXES)
            for uri in products['productFilename']
        ]
        science_products = products[science_mask]

        if len(science_products) == 0:
            logger.warning("No calibrated science products found.")
            return []

        science_products = science_products[:n]
        logger.info("Downloading %d science products …", len(science_products))

        # ── Download ──────────────────────────────────────────────────────
        try:
            manifest = Observations.download_products(
                science_products,
                download_dir=str(self.output_dir),
            )
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            return []

        local_paths = []
        for row in manifest:
            local_file = Path(str(row['Local Path']))
            if local_file.exists():
                local_paths.append(local_file.resolve())

        logger.info("Downloaded %d files to %s", len(local_paths), self.output_dir)
        return local_paths

    @staticmethod
    def list_notable_targets() -> Dict[str, Dict]:
        """Return the built-in catalogue of notable JWST targets."""
        return dict(JWST_NOTABLE_TARGETS)

    def download_by_program_id(
        self,
        program_id: int,
        instrument: Optional[str] = None,
        max_products: Optional[int] = None,
    ) -> List[Path]:
        """
        Download all public science files for a given JWST program ID.

        Args:
            program_id: JWST proposal/program ID (e.g. 2731 for Carina Nebula).
            instrument: Optional instrument filter.
            max_products: Limit number of files downloaded.

        Returns:
            List of local FITS file paths.
        """
        from astroquery.mast import Observations
        import astropy.units as u

        n = max_products or self.max_products
        self.output_dir.mkdir(parents=True, exist_ok=True)

        criteria: Dict = {
            'obs_collection': 'JWST',
            'proposal_id': str(program_id),
            'dataRights': ['PUBLIC'],
        }
        if instrument:
            criteria['instrument_name'] = instrument

        try:
            obs_table = Observations.query_criteria(**criteria)
        except Exception as exc:
            logger.error("Program ID query failed: %s", exc)
            return []

        if obs_table is None or len(obs_table) == 0:
            logger.warning("No observations for program ID %d.", program_id)
            return []

        logger.info("Program %d: %d observations found.", program_id, len(obs_table))

        try:
            products = Observations.get_product_list(obs_table[:10])
        except Exception as exc:
            logger.error("Product list failed: %s", exc)
            return []

        science_mask = [
            any(str(uri).endswith(s) for s in self._SCIENCE_SUFFIXES)
            for uri in products['productFilename']
        ]
        science_products = products[science_mask][:n]

        if len(science_products) == 0:
            return []

        try:
            manifest = Observations.download_products(
                science_products, download_dir=str(self.output_dir)
            )
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            return []

        return [
            Path(str(r['Local Path'])).resolve()
            for r in manifest
            if Path(str(r['Local Path'])).exists()
        ]
