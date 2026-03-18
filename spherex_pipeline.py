"""
SPHEREx End-to-End Investigation Pipeline
==========================================

Takes a sky position and produces a complete multi-panel investigation
report using tools from spherex_tools.py.

Usage:
    python spherex_pipeline.py                          # Orion Nebula default
    python spherex_pipeline.py --ra 83.822 --dec -5.391 --name "Orion Nebula"

Author: Generated for Fred
Date: 2026-03-18
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.gridspec import GridSpec
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import our tool suite
from spherex_tools import (
    SPHEREX_BANDS, FLAG_DEFINITIONS,
    download_cutout, download_cutouts_parallel,
    get_pixel_wavelength, extract_spectrum_at_position,
    interpret_flags, create_flag_mask, flag_quality_summary,
    get_psf_for_position, check_psf_version, psf_fwhm,
    subtract_background, detect_sources, aperture_photometry,
    measure_ice_band_depth, classify_ice_spectrum, ICE_FEATURES,
    get_observation_metadata, get_band_info, check_data_quality,
    get_variance_and_snr,
)
from spherex_analysis import (
    query_spherex_region, resolve_datalink, download_fits,
)


# Default output directories
OUTPUT_DIR = Path("./spherex_investigation")
DATA_DIR = Path("./spherex_data")


def investigate_position(ra, dec, name="Target", size_deg=0.2,
                         output_dir=None, data_dir=None, max_images=6):
    """
    Full investigation pipeline for any sky position.

    Steps:
        1. Query IRSA for SPHEREx observations
        2. Download FITS files (or use cached)
        3. Analyze each image: flags, WCS, band info
        4. Extract spectrum at target position
        5. Detect sources and measure ice features
        6. Generate multi-panel investigation report
        7. Save all outputs

    Parameters
    ----------
    ra, dec : float
        Target position in degrees (J2000).
    name : str
        Target name for labels and filenames.
    size_deg : float
        Search/cutout radius in degrees.
    output_dir : str or Path, optional
        Directory for output plots and catalogs.
    data_dir : str or Path, optional
        Directory for downloaded FITS files.
    max_images : int
        Maximum number of images to download/analyze.

    Returns
    -------
    report : dict
        Complete investigation results.
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / name.replace(" ", "_")
    dat_dir = Path(data_dir) if data_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dat_dir.mkdir(parents=True, exist_ok=True)

    safe_name = name.replace(" ", "_").lower()
    report = {"name": name, "ra": ra, "dec": dec, "images": [], "warnings": []}

    print(f"\n{'='*70}")
    print(f"  SPHEREx Investigation: {name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"{'='*70}\n")

    # ----------------------------------------------------------------
    # Step 1: Query IRSA
    # ----------------------------------------------------------------
    print("[Step 1] Querying IRSA for SPHEREx observations...")
    try:
        results = query_spherex_region(ra, dec, radius_deg=size_deg)
        n_found = len(results)
        print(f"  Found {n_found} observations.\n")
    except Exception as e:
        print(f"  Query failed: {e}")
        report["warnings"].append(f"TAP query failed: {e}")
        results = None
        n_found = 0

    # ----------------------------------------------------------------
    # Step 2: Download FITS files
    # ----------------------------------------------------------------
    fits_paths = []
    if results is not None and n_found > 0:
        print(f"[Step 2] Downloading up to {max_images} FITS files...")
        for i, row in enumerate(results):
            if i >= max_images:
                break
            try:
                access_url = str(row['access_url'])
                fits_url = resolve_datalink(access_url)
                if fits_url:
                    fpath = download_fits(fits_url, output_dir=str(dat_dir))
                    fits_paths.append(fpath)
            except Exception as e:
                print(f"  Skipping image {i}: {e}")
        print(f"  Downloaded {len(fits_paths)} files.\n")
    else:
        # Check for existing files in data directory
        existing = list(dat_dir.glob("*.fits"))
        if existing:
            print(f"[Step 2] Using {len(existing)} existing FITS files from {dat_dir}")
            fits_paths = existing[:max_images]

    if not fits_paths:
        print("No FITS files available. Cannot proceed with analysis.")
        report["warnings"].append("No FITS files available")
        return report

    # ----------------------------------------------------------------
    # Step 3: Analyze each image
    # ----------------------------------------------------------------
    print(f"[Step 3] Analyzing {len(fits_paths)} images...")
    image_analyses = []

    for fpath in fits_paths:
        analysis = {"path": str(fpath)}
        try:
            with fits.open(fpath) as hdul:
                # Metadata
                meta = get_observation_metadata(hdul)
                analysis["metadata"] = meta

                # Band info
                band = get_band_info(hdul)
                analysis["band"] = band

                # Data quality
                quality = check_data_quality(hdul)
                analysis["quality"] = quality

                # PSF info
                version, needs_fix = check_psf_version(hdul)
                analysis["psf_version"] = version
                analysis["psf_needs_fix"] = needs_fix
                analysis["psf_fwhm"] = psf_fwhm(hdul)

                # Flag summary
                for hdu in hdul:
                    if hdu.name == 'FLAGS' and hdu.data is not None:
                        analysis["flag_summary"] = flag_quality_summary(hdu.data)
                        break

                image_analyses.append(analysis)
        except Exception as e:
            print(f"  Error analyzing {fpath}: {e}")
            report["warnings"].append(f"Analysis error: {e}")

    report["images"] = image_analyses
    print(f"  Analyzed {len(image_analyses)} images.\n")

    # ----------------------------------------------------------------
    # Step 4: Extract spectrum at target position
    # ----------------------------------------------------------------
    print("[Step 4] Extracting spectrum at target position...")
    wavelengths, fluxes, bandwidths = extract_spectrum_at_position(
        ra, dec, fits_paths
    )
    report["spectrum"] = {
        "wavelengths": wavelengths,
        "fluxes": fluxes,
        "bandwidths": bandwidths,
        "n_channels": len(wavelengths),
    }
    print(f"  Extracted {len(wavelengths)} spectral channels.\n")

    # ----------------------------------------------------------------
    # Step 5: Source detection on first image
    # ----------------------------------------------------------------
    print("[Step 5] Detecting sources...")
    sources = None
    source_image_path = fits_paths[0]
    try:
        with fits.open(source_image_path) as hdul:
            image_data = hdul['IMAGE'].data

            flags_data = None
            for hdu in hdul:
                if hdu.name == 'FLAGS':
                    flags_data = hdu.data
                    break

            bkg_sub, bkg = subtract_background(image_data, flags_data)
            sources, n_sources = detect_sources(bkg_sub, flags_data)
            report["sources"] = {
                "n_detected": n_sources,
                "image": str(source_image_path),
            }
    except Exception as e:
        print(f"  Source detection failed: {e}")
        report["warnings"].append(f"Source detection failed: {e}")
    print()

    # ----------------------------------------------------------------
    # Step 6: Ice feature analysis
    # ----------------------------------------------------------------
    print("[Step 6] Analyzing ice features...")
    if len(wavelengths) > 5:
        classification, dominant = classify_ice_spectrum(wavelengths, fluxes)
        report["ice"] = {
            "classification": classification,
            "dominant_feature": dominant,
        }
        for feat_name, info in classification.items():
            status = "DETECTED" if info["detected"] else "not detected"
            depth = info["depth"]
            if not np.isnan(depth):
                print(f"  {info['label']}: depth={depth:.3f} ({status})")
            else:
                print(f"  {info['label']}: insufficient data")
    else:
        print("  Insufficient spectral coverage for ice analysis.")
        report["ice"] = None
    print()

    # ----------------------------------------------------------------
    # Step 7: Generate multi-panel report
    # ----------------------------------------------------------------
    print("[Step 7] Generating investigation report...")
    report_path = _generate_report(
        report, fits_paths, bkg_sub if sources is not None else None,
        sources, wavelengths, fluxes, out_dir, safe_name
    )
    report["report_path"] = str(report_path)

    # ----------------------------------------------------------------
    # Step 8: Save catalog
    # ----------------------------------------------------------------
    if sources is not None and len(sources) > 0:
        catalog_path = out_dir / f"{safe_name}_sources.csv"
        _save_source_catalog(sources, catalog_path)
        report["catalog_path"] = str(catalog_path)

    print(f"\n{'='*70}")
    print(f"  Investigation complete: {name}")
    print(f"  Report: {report.get('report_path', 'N/A')}")
    print(f"  Catalog: {report.get('catalog_path', 'N/A')}")
    print(f"  Warnings: {len(report['warnings'])}")
    print(f"{'='*70}\n")

    return report


def _generate_report(report, fits_paths, bkg_sub, sources,
                     wavelengths, fluxes, out_dir, safe_name):
    """Generate a 6-panel investigation report figure."""

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"SPHEREx Investigation: {report['name']}\n"
                 f"RA={report['ra']:.4f}, Dec={report['dec']:.4f}",
                 fontsize=16, fontweight='bold')

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Panel 1: Multi-band image ----
    ax1 = fig.add_subplot(gs[0, 0])
    try:
        with fits.open(fits_paths[0]) as hdul:
            img = hdul['IMAGE'].data
            valid = img[np.isfinite(img)]
            vmin = np.percentile(valid, 1)
            vmax = np.percentile(valid, 99)
            ax1.imshow(img, origin='lower', cmap='inferno',
                       vmin=vmin, vmax=vmax)

            # Mark target position
            wcs = WCS(hdul['IMAGE'].header)
            coord = SkyCoord(ra=report['ra']*u.deg, dec=report['dec']*u.deg)
            px, py = wcs.world_to_pixel(coord)
            ax1.plot(float(px), float(py), 'c+', ms=15, mew=2)

            band = report["images"][0].get("band", {}) if report["images"] else {}
            band_num = band.get("band_number", "?")
            ax1.set_title(f"Band {band_num} Image", fontsize=11)
    except Exception:
        ax1.text(0.5, 0.5, "Image not available", ha='center', va='center',
                 transform=ax1.transAxes)
        ax1.set_title("Field Image", fontsize=11)

    # ---- Panel 2: Extracted spectrum ----
    ax2 = fig.add_subplot(gs[0, 1])
    if len(wavelengths) > 0:
        ax2.plot(wavelengths, fluxes, 'b-', lw=1.2, alpha=0.8)
        ax2.scatter(wavelengths, fluxes, c='navy', s=15, zorder=5)

        # Mark ice features
        colors = {'water_ice': 'cyan', 'co2_ice': 'orange',
                  'co_ice': 'red', 'ch3oh_ice': 'green'}
        for feat_name, feat_info in ICE_FEATURES.items():
            ax2.axvline(feat_info["center_um"], color=colors.get(feat_name, 'gray'),
                        ls='--', alpha=0.5, label=feat_info["label"])

        ax2.set_xlabel("Wavelength (um)")
        ax2.set_ylabel("Flux (MJy/sr)")
        ax2.set_title("Extracted Spectrum", fontsize=11)
        ax2.legend(fontsize=7, loc='upper right')
    else:
        ax2.text(0.5, 0.5, "No spectrum extracted", ha='center', va='center',
                 transform=ax2.transAxes)
        ax2.set_title("Extracted Spectrum", fontsize=11)

    # ---- Panel 3: Source detection overlay ----
    ax3 = fig.add_subplot(gs[0, 2])
    if bkg_sub is not None and sources is not None:
        valid = bkg_sub[np.isfinite(bkg_sub)]
        vmin = np.percentile(valid, 5)
        vmax = np.percentile(valid, 95)
        ax3.imshow(bkg_sub, origin='lower', cmap='gray_r',
                   vmin=vmin, vmax=vmax)

        if len(sources) > 0:
            ax3.scatter(sources['x'], sources['y'], s=20,
                        facecolors='none', edgecolors='lime', linewidths=0.8)
        ax3.set_title(f"Source Detection ({len(sources)} found)", fontsize=11)
    else:
        ax3.text(0.5, 0.5, "Source detection\nnot available", ha='center',
                 va='center', transform=ax3.transAxes)
        ax3.set_title("Source Detection", fontsize=11)

    # ---- Panel 4: Zodi comparison ----
    ax4 = fig.add_subplot(gs[1, 0])
    try:
        with fits.open(fits_paths[0]) as hdul:
            zodi_data = None
            for hdu in hdul:
                if hdu.name == 'ZODI' and hdu.data is not None:
                    zodi_data = hdu.data
                    break

            if zodi_data is not None:
                img = hdul['IMAGE'].data
                subtracted = img - zodi_data
                valid_sub = subtracted[np.isfinite(subtracted)]
                vmin = np.percentile(valid_sub, 5)
                vmax = np.percentile(valid_sub, 95)
                ax4.imshow(subtracted, origin='lower', cmap='viridis',
                           vmin=vmin, vmax=vmax)
                ax4.set_title("Zodiacal-Subtracted", fontsize=11)
            else:
                ax4.text(0.5, 0.5, "ZODI extension\nnot found", ha='center',
                         va='center', transform=ax4.transAxes)
                ax4.set_title("Zodiacal Subtraction", fontsize=11)
    except Exception:
        ax4.text(0.5, 0.5, "Not available", ha='center', va='center',
                 transform=ax4.transAxes)
        ax4.set_title("Zodiacal Subtraction", fontsize=11)

    # ---- Panel 5: Variance / SNR map ----
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        with fits.open(fits_paths[0]) as hdul:
            variance, snr = get_variance_and_snr(hdul)
            valid_snr = snr[np.isfinite(snr) & (snr > 0)]
            if len(valid_snr) > 0:
                vmax_snr = np.percentile(valid_snr, 98)
                ax5.imshow(snr, origin='lower', cmap='plasma',
                           vmin=0, vmax=vmax_snr)
                ax5.set_title(f"SNR Map (median={np.median(valid_snr):.1f})",
                              fontsize=11)
            else:
                ax5.text(0.5, 0.5, "SNR computation\nfailed", ha='center',
                         va='center', transform=ax5.transAxes)
                ax5.set_title("SNR Map", fontsize=11)
    except Exception:
        ax5.text(0.5, 0.5, "VARIANCE extension\nnot found", ha='center',
                 va='center', transform=ax5.transAxes)
        ax5.set_title("SNR Map", fontsize=11)

    # ---- Panel 6: Data quality summary ----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary_lines = [f"Target: {report['name']}"]
    summary_lines.append(f"RA: {report['ra']:.4f}  Dec: {report['dec']:.4f}")
    summary_lines.append(f"Images analyzed: {len(report['images'])}")
    summary_lines.append(f"Spectral channels: {report.get('spectrum', {}).get('n_channels', 0)}")

    if report.get("sources"):
        summary_lines.append(f"Sources detected: {report['sources']['n_detected']}")

    if report.get("ice") and report["ice"].get("classification"):
        summary_lines.append("")
        summary_lines.append("Ice Features:")
        for feat_name, info in report["ice"]["classification"].items():
            if info["detected"]:
                summary_lines.append(f"  {info['label']}: depth={info['depth']:.3f}")

    if report["images"]:
        img0 = report["images"][0]
        summary_lines.append("")
        summary_lines.append(f"Pipeline: v{img0.get('psf_version', '?')}")
        fwhm = img0.get('psf_fwhm')
        if fwhm:
            summary_lines.append(f"PSF FWHM: {fwhm:.2f}\"")
        q = img0.get('quality', {})
        dqa = q.get('l2_dqa', 'unknown')
        summary_lines.append(f"L2 DQA: {dqa}")

    if report["warnings"]:
        summary_lines.append("")
        summary_lines.append(f"Warnings: {len(report['warnings'])}")
        for w in report["warnings"][:3]:
            summary_lines.append(f"  - {w}")

    summary_text = "\n".join(summary_lines)
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       alpha=0.8))
    ax6.set_title("Investigation Summary", fontsize=11)

    # Save
    report_path = out_dir / f"{safe_name}_investigation.png"
    fig.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Report saved: {report_path}")

    return report_path


def _save_source_catalog(sources, catalog_path):
    """Save detected sources to CSV."""
    import csv

    fields = ['x', 'y', 'flux', 'a', 'b', 'theta', 'flag']
    available = [f for f in fields if f in sources.dtype.names]

    with open(catalog_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(available)
        for src in sources:
            writer.writerow([src[field] for field in available])

    print(f"  Source catalog saved: {catalog_path} ({len(sources)} sources)")


def batch_investigate(target_list, output_dir=None, data_dir=None):
    """
    Run investigate_position() for multiple targets.

    Parameters
    ----------
    target_list : list of dict
        Each dict has keys: "name", "ra", "dec" and optional "size_deg".
    output_dir, data_dir : str or Path, optional

    Returns
    -------
    reports : list of dict
        Investigation reports for each target.
    """
    reports = []
    for i, target in enumerate(target_list):
        print(f"\n{'#'*70}")
        print(f"  Target {i+1}/{len(target_list)}: {target['name']}")
        print(f"{'#'*70}")

        report = investigate_position(
            ra=target["ra"],
            dec=target["dec"],
            name=target["name"],
            size_deg=target.get("size_deg", 0.2),
            output_dir=output_dir,
            data_dir=data_dir,
        )
        reports.append(report)

    print(f"\nBatch investigation complete: {len(reports)} targets processed.")
    return reports


# ============================================================================
# PREDEFINED TARGETS
# ============================================================================

INVESTIGATION_TARGETS = [
    {"name": "Orion Nebula",     "ra": 83.822,  "dec": -5.391,  "size_deg": 0.3},
    {"name": "Rho Ophiuchi",     "ra": 246.787, "dec": -24.538, "size_deg": 0.3},
    {"name": "Galactic Center",  "ra": 266.417, "dec": -29.008, "size_deg": 0.2},
    {"name": "North Ecliptic Pole", "ra": 270.0, "dec": 66.561, "size_deg": 0.2},
    {"name": "Crab Nebula",      "ra": 83.633,  "dec": 22.015,  "size_deg": 0.1},
    {"name": "Eagle Nebula",     "ra": 274.700, "dec": -13.807, "size_deg": 0.2},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPHEREx Investigation Pipeline")
    parser.add_argument("--ra", type=float, default=83.822,
                        help="Right ascension in degrees (default: Orion)")
    parser.add_argument("--dec", type=float, default=-5.391,
                        help="Declination in degrees (default: Orion)")
    parser.add_argument("--name", type=str, default="Orion Nebula",
                        help="Target name")
    parser.add_argument("--size", type=float, default=0.2,
                        help="Search radius in degrees")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch investigation on predefined targets")

    args = parser.parse_args()

    if args.batch:
        batch_investigate(INVESTIGATION_TARGETS)
    else:
        investigate_position(args.ra, args.dec, args.name, args.size)
