"""
SPHEREx Science Investigation Toolkit
======================================

Modular tools for investigating SPHEREx data from IRSA, informed by:
- IRSA tutorials (Data Overview, Cutouts, PSF Models, Source Discovery)
- SPHEREx Explanatory Supplement v1.4 (2026-01-23)
- AWS Open Data access patterns

Sections:
    A. Cutout Service — download and assemble spatial cutouts
    B. Spectral WCS — wavelength extraction using CWAVE/CBAND
    C. Flag Interpretation — 20-flag bitmap with preset masks
    D. PSF Handling — zone lookup across 11x11 grid
    E. Background Subtraction & Source Detection — via sep
    F. SPLICES Catalog Cross-Matching
    G. Ice Feature Detection — water, CO2, CO absorption
    H. AWS S3 Direct Access
    I. Data Quality & Header Utilities

Requirements:
    pip install numpy astropy pyvo matplotlib scipy sep s3fs

Author: Generated for Fred
Date: 2026-03-18
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import pyvo
    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False

try:
    import sep
    HAS_SEP = True
except ImportError:
    HAS_SEP = False

try:
    import s3fs
    HAS_S3FS = True
except ImportError:
    HAS_S3FS = False


# IRSA TAP service endpoint
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# SPHEREx band definitions (from Explanatory Supplement Table)
SPHEREX_BANDS = {
    1: {"wl_min": 0.744, "wl_max": 1.116, "R": 39, "detector": "SWIR"},
    2: {"wl_min": 1.099, "wl_max": 1.651, "R": 41, "detector": "SWIR"},
    3: {"wl_min": 1.636, "wl_max": 2.421, "R": 41, "detector": "SWIR"},
    4: {"wl_min": 2.423, "wl_max": 3.822, "R": 35, "detector": "MWIR"},
    5: {"wl_min": 3.809, "wl_max": 4.420, "R": 112, "detector": "MWIR"},
    6: {"wl_min": 4.412, "wl_max": 5.002, "R": 128, "detector": "MWIR"},
}


# ============================================================================
# SECTION A: CUTOUT SERVICE
# ============================================================================

def download_cutout(fits_url, ra, dec, size_deg=0.1, output_dir="./spherex_data"):
    """
    Download a spatial cutout by appending center/size params to a FITS URL.

    The IRSA cutout service accepts URL parameters:
        ?center=RA,DEC&size=SIZE_DEG

    Parameters
    ----------
    fits_url : str
        Base FITS file URL from IRSA.
    ra, dec : float
        Center position in degrees (J2000).
    size_deg : float
        Cutout size in degrees.
    output_dir : str
        Directory to save the cutout.

    Returns
    -------
    hdulist : astropy.io.fits.HDUList or None
        The downloaded cutout, or None on failure.
    """
    import urllib.request

    separator = "&" if "?" in fits_url else "?"
    cutout_url = f"{fits_url}{separator}center={ra},{dec}&size={size_deg}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"cutout_ra{ra:.3f}_dec{dec:.3f}_s{size_deg}.fits"
    filepath = output_dir / filename

    try:
        urllib.request.urlretrieve(cutout_url, filepath)
        hdulist = fits.open(filepath)
        print(f"Downloaded cutout: {filepath.name} ({len(hdulist)} HDUs)")
        return hdulist
    except Exception as e:
        print(f"Cutout download failed: {e}")
        return None


def download_cutouts_parallel(url_list, ra, dec, size_deg=0.1,
                               max_workers=10, output_dir="./spherex_data"):
    """
    Download multiple cutouts in parallel using ThreadPoolExecutor.

    Achieves ~14x speedup over serial downloads per IRSA tutorials.

    Parameters
    ----------
    url_list : list of str
        FITS file URLs to download cutouts from.
    ra, dec : float
        Center position in degrees (J2000).
    size_deg : float
        Cutout size in degrees.
    max_workers : int
        Number of parallel download threads.
    output_dir : str
        Directory to save cutouts.

    Returns
    -------
    results : list of astropy.io.fits.HDUList
        Successfully downloaded cutouts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _download_one(url_idx):
        url, idx = url_idx
        separator = "&" if "?" in url else "?"
        cutout_url = f"{url}{separator}center={ra},{dec}&size={size_deg}"
        filepath = output_dir / f"cutout_{idx:03d}.fits"
        try:
            urllib.request.urlretrieve(cutout_url, filepath)
            return fits.open(filepath)
        except Exception:
            return None

    results = []
    print(f"Downloading {len(url_list)} cutouts with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, (url, i)): i
                   for i, url in enumerate(url_list)}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    print(f"Successfully downloaded {len(results)}/{len(url_list)} cutouts.")
    return results


def assemble_cutout_mef(cutout_hdus_list, output_path):
    """
    Assemble downloaded cutouts into a single multi-extension FITS file.

    Parameters
    ----------
    cutout_hdus_list : list of HDUList
        List of cutout FITS HDU lists.
    output_path : str
        Output file path.

    Returns
    -------
    output_path : str
        Path to the assembled MEF.
    """
    primary = fits.PrimaryHDU()
    hdu_list = [primary]

    for i, hdul in enumerate(cutout_hdus_list):
        for hdu in hdul:
            if hdu.data is not None and hdu.name != '':
                hdu.name = f"{hdu.name}_{i:03d}"
                hdu_list.append(hdu)

    assembled = fits.HDUList(hdu_list)
    assembled.writeto(output_path, overwrite=True)
    print(f"Assembled {len(cutout_hdus_list)} cutouts into {output_path}")
    return output_path


# ============================================================================
# SECTION B: SPECTRAL WCS & WAVELENGTH EXTRACTION
# ============================================================================

def get_pixel_wavelength(hdulist, x, y, method="cwave"):
    """
    Extract wavelength and bandwidth at a pixel position.

    Per the Explanatory Supplement (Sect. 4.3.1):
    - CWAVE extension: per-pixel central wavelengths (um) — for science
    - CBAND extension: per-pixel bandwidths (um) — for science
    - WAVE-TAB WCS (key='W'): approximate, for visualization only

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.
    x, y : int
        Pixel coordinates.
    method : str
        "cwave" (preferred, uses CWAVE/CBAND extensions) or
        "wcs" (fallback, uses WAVE-TAB WCS approximation).

    Returns
    -------
    wavelength : float
        Central wavelength in microns.
    bandwidth : float
        Bandwidth in microns.
    """
    if method == "cwave":
        # Preferred: use dedicated spectral WCS extensions
        cwave_data = None
        cband_data = None
        for hdu in hdulist:
            if hdu.name == 'CWAVE':
                cwave_data = hdu.data
            elif hdu.name == 'CBAND':
                cband_data = hdu.data

        if cwave_data is not None:
            wavelength = float(cwave_data[y, x])
            bandwidth = float(cband_data[y, x]) if cband_data is not None else 0.0
            return wavelength, bandwidth

        # If CWAVE not in file, fall through to WCS method
        print("CWAVE extension not found, falling back to WAVE-TAB WCS.")

    # Fallback: try WCS-WAVE binary table extension directly
    for hdu in hdulist:
        if hdu.name == 'WCS-WAVE' and hasattr(hdu, 'data') and hdu.data is not None:
            try:
                # WCS-WAVE is a binary table with VALUES, X, Y columns
                values = hdu.data['VALUES']
                x_coords = hdu.data['X']
                y_coords = hdu.data['Y']
                # Interpolate to get wavelength at (x, y)
                from scipy.interpolate import RegularGridInterpolator
                if values.ndim >= 2:
                    # values shape is typically (nx, ny, 2) for wavelength and bandpass
                    interp_wl = RegularGridInterpolator(
                        (x_coords, y_coords), values[:, :, 0],
                        method='linear', bounds_error=False, fill_value=None
                    )
                    interp_bw = RegularGridInterpolator(
                        (x_coords, y_coords), values[:, :, 1],
                        method='linear', bounds_error=False, fill_value=None
                    )
                    wavelength = float(interp_wl([x, y])[0])
                    bandwidth = float(interp_bw([x, y])[0])
                    return wavelength, bandwidth
            except Exception:
                pass

    # Last resort: WAVE-TAB WCS (approximate, for visualization)
    try:
        header = hdulist['IMAGE'].header
        wcs_wave = WCS(header, fobj=hdulist, key='W')
        # pixel_to_world may return Quantity objects or tuples
        result = wcs_wave.pixel_to_world(x, y)
        if isinstance(result, (list, tuple)) and len(result) == 2:
            wavelength = float(result[0].value) if hasattr(result[0], 'value') else float(result[0])
            bandwidth = float(result[1].value) if hasattr(result[1], 'value') else float(result[1])
        elif hasattr(result, 'value'):
            wavelength = float(result.value)
            bandwidth = 0.0
        else:
            wavelength = float(result)
            bandwidth = 0.0
        return wavelength, bandwidth
    except Exception as e:
        print(f"WCS wavelength extraction failed: {e}")
        return None, None


def load_spectral_wcs_file(detector, output_dir="./spherex_data"):
    """
    Download and load the per-detector spectral WCS calibration file from IRSA.

    These files contain full-resolution CWAVE and CBAND extensions for
    each detector band, suitable for science analysis.

    Parameters
    ----------
    detector : int
        Detector number (1-6).
    output_dir : str
        Directory to save/cache the file.

    Returns
    -------
    hdulist : HDUList
        The spectral WCS calibration file.
    """
    if not HAS_PYVO:
        raise ImportError("pyvo is required.")

    tap_service = pyvo.dal.TAPService(IRSA_TAP_URL)
    query = f"""
    SELECT TOP 1 a.uri
    FROM spherex.artifact AS a
    JOIN spherex.plane AS p ON a.plane_id = p.plane_id
    WHERE a.uri LIKE '%spectral_wcs%D{detector}%'
      AND a.content_type = 'application/fits'
    """
    try:
        results = tap_service.search(query)
        if len(results) > 0:
            url = str(results[0]['uri'])
            from spherex_analysis import download_fits
            filepath = download_fits(url, output_dir)
            return fits.open(filepath)
    except Exception as e:
        print(f"Could not load spectral WCS for detector {detector}: {e}")
    return None


def extract_spectrum_at_position(ra, dec, fits_paths):
    """
    Build a full 0.75-5.0 um spectrum at a sky position from multiple images.

    Each SPHEREx image covers a single wavelength band. By combining
    flux measurements across bands, we build the full infrared spectrum.

    Parameters
    ----------
    ra, dec : float
        Sky position in degrees (J2000).
    fits_paths : list of str or Path
        Paths to SPHEREx FITS files covering different bands.

    Returns
    -------
    wavelengths : array
        Central wavelengths in microns.
    fluxes : array
        Flux values in MJy/sr.
    bandwidths : array
        Bandwidth of each channel in microns.
    """
    wavelengths = []
    fluxes = []
    bandwidths = []

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    for fpath in fits_paths:
        try:
            with fits.open(fpath) as hdul:
                image_hdu = hdul['IMAGE']
                wcs = WCS(image_hdu.header)

                # Convert RA/Dec to pixel coordinates
                px, py = wcs.world_to_pixel(coord)
                ix, iy = int(round(float(px))), int(round(float(py)))

                # Check bounds
                ny, nx = image_hdu.data.shape
                if 0 <= ix < nx and 0 <= iy < ny:
                    flux = float(image_hdu.data[iy, ix])
                    wl, bw = get_pixel_wavelength(hdul, ix, iy)
                    if wl is not None and np.isfinite(flux):
                        wavelengths.append(wl)
                        fluxes.append(flux)
                        bandwidths.append(bw)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

    # Sort by wavelength
    order = np.argsort(wavelengths)
    return (np.array(wavelengths)[order],
            np.array(fluxes)[order],
            np.array(bandwidths)[order])


# ============================================================================
# SECTION C: FLAG INTERPRETATION
# ============================================================================

# Complete flag table from Explanatory Supplement Table 16
# Keys are bit positions in the FLAGS bitmap
FLAG_DEFINITIONS = {
    0:  "TRANSIENT",       # Cosmic ray / transient event
    1:  "OVERFLOW",        # Pixel saturated / overflowed
    2:  "SUR_ERROR",       # Sample-up-the-ramp fitting error
    6:  "NONFUNC",         # Permanently non-functioning pixel
    7:  "DICHROIC",        # Dichroic filter attenuation >= 50% (Bands 3,4 only)
    9:  "MISSING_DATA",    # Missing telemetry data
    10: "HOT",             # Hot pixel
    11: "COLD",            # Cold pixel
    12: "FULLSAMPLE",      # Full-sample readout mode
    14: "PHANMISS",        # Phantom pixel missing
    15: "NONLINEAR",       # Strong non-linearity detected
    17: "PERSIST",         # Persistence contamination above threshold
    19: "OUTLIER",         # Statistical outlier in temporal sequence
    21: "SOURCE",          # Known astronomical source (from masking catalog)
    22: "GHOST",           # Optical ghost artifact
    24: "GHOST_EXT",       # Extended ghost region
    26: "BLOOM",           # Charge bloom from bright source
    27: "SNOWBALL",        # Snowball event (localized charge deposit)
    28: "HALO",            # Scattered light halo
    29: "SATELLITE_HALO",  # Satellite streak halo
}

# Reverse lookup: name -> bit position
FLAG_BITS = {name: bit for bit, name in FLAG_DEFINITIONS.items()}

# Preset flag masks from Explanatory Supplement Sect. 3.3
BACKGROUND_MASK_FLAGS = [
    "OVERFLOW", "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD",
    "NONLINEAR", "PERSIST", "OUTLIER", "SOURCE", "TRANSIENT",
    "BLOOM", "SNOWBALL", "HALO", "SATELLITE_HALO"
]

PHOTOMETRY_MASK_FLAGS = [
    "SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD",
    "NONLINEAR", "PERSIST", "BLOOM", "SNOWBALL", "HALO", "SATELLITE_HALO"
]


def interpret_flags(flag_value):
    """
    Decode a flag bitmap into a list of active flag names.

    Parameters
    ----------
    flag_value : int
        Integer bitmap from the FLAGS extension.

    Returns
    -------
    active_flags : list of str
        Names of all active flags.
    """
    active = []
    for bit, name in sorted(FLAG_DEFINITIONS.items()):
        if flag_value & (1 << bit):
            active.append(name)
    return active


def create_flag_mask(flags_data, mode="photometry", custom_flags=None):
    """
    Create a boolean mask excluding pixels with specified bad flags.

    Parameters
    ----------
    flags_data : ndarray
        2D array from the FLAGS extension.
    mode : str
        "photometry" — mask per Expsupp photometry recommendations (12 flags)
        "background" — mask per Expsupp background estimation (15 flags)
        "custom" — use custom_flags list
    custom_flags : list of str, optional
        Flag names to mask when mode="custom".

    Returns
    -------
    good_mask : ndarray of bool
        True for pixels that are NOT flagged (safe to use).
    """
    if mode == "background":
        flag_list = BACKGROUND_MASK_FLAGS
    elif mode == "custom" and custom_flags is not None:
        flag_list = custom_flags
    else:
        flag_list = PHOTOMETRY_MASK_FLAGS

    # Build combined bitmask
    combined_mask = 0
    for name in flag_list:
        if name in FLAG_BITS:
            combined_mask |= (1 << FLAG_BITS[name])

    # Good pixels have none of the bad flags set
    good_mask = (flags_data & combined_mask) == 0
    return good_mask


def flag_quality_summary(flags_data):
    """
    Report pixel counts per flag type, mirroring L2_N_* header keywords.

    Parameters
    ----------
    flags_data : ndarray
        2D array from the FLAGS extension.

    Returns
    -------
    summary : dict
        {flag_name: pixel_count} for each flag with non-zero count.
    """
    total_pixels = flags_data.size
    summary = {}
    for bit, name in sorted(FLAG_DEFINITIONS.items()):
        count = int(np.count_nonzero(flags_data & (1 << bit)))
        if count > 0:
            summary[name] = count

    flagged_any = int(np.count_nonzero(flags_data))
    clean = total_pixels - flagged_any
    print(f"Flag Quality Summary ({total_pixels:,} total pixels):")
    print(f"  Clean pixels: {clean:,} ({100*clean/total_pixels:.1f}%)")
    for name, count in summary.items():
        pct = 100 * count / total_pixels
        print(f"  {name:20s}: {count:>8,} ({pct:.2f}%)")

    return summary


# ============================================================================
# SECTION D: PSF HANDLING
# ============================================================================

def get_psf_zone_index(hdulist, x_pixel, y_pixel):
    """
    Find which of the 121 PSF zones a pixel falls in.

    The PSF data cube has 121 zones on an 11x11 grid. Zone centers and
    widths are encoded in the PSF header as XCTR_*, YCTR_*, XWID_*, YWID_*.

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.
    x_pixel, y_pixel : float
        Pixel coordinates on the detector.

    Returns
    -------
    zone_index : int
        Index (0-120) into the PSF data cube's first axis.
    zone_ix, zone_iy : int
        Grid coordinates (0-10) of the zone.
    """
    psf_hdu = None
    for hdu in hdulist:
        if hdu.name in ('PSF', 'PSF-DATA-CUBE') or \
           (hdu.data is not None and hdu.data.ndim == 3 and hdu.data.shape[0] == 121):
            psf_hdu = hdu
            break

    if psf_hdu is None:
        raise ValueError("No PSF extension found in FITS file.")

    header = psf_hdu.header

    # Read zone centers from header
    # Zone naming follows: XCTR_a through XCTR_v (11 x-centers per y-row)
    # and YCTR_b through YCTR_w (11 y-centers per x-column)
    # Letters cycle through: a,c,e,g,i,k,m,p,r,t,v for x-centers
    x_center_keys = ['a', 'c', 'e', 'g', 'i', 'k', 'm', 'p', 'r', 't', 'v']

    x_centers = []
    for key in x_center_keys:
        kw = f'XCTR_{key}'
        if kw in header:
            x_centers.append(float(header[kw]))

    y_centers = []
    # y-centers use different letter keys
    y_center_keys = ['b', 'd', 'f', 'h', 'j', 'l', 'o', 'q', 's', 'u', 'w']
    for key in y_center_keys:
        kw = f'YCTR_{key}'
        if kw in header:
            y_centers.append(float(header[kw]))

    if len(x_centers) < 11 or len(y_centers) < 11:
        # Fallback: compute uniform grid assuming 2040x2040 detector
        x_centers = [93.22 + i * 186.45 for i in range(11)]
        y_centers = [93.22 + i * 186.45 for i in range(11)]

    # Find nearest zone
    x_dists = [abs(x_pixel - xc) for xc in x_centers]
    y_dists = [abs(y_pixel - yc) for yc in y_centers]
    zone_ix = int(np.argmin(x_dists))
    zone_iy = int(np.argmin(y_dists))

    # Zone index in the 121-element cube: varies by data layout
    # Standard ordering: zone_index = zone_iy * 11 + zone_ix
    zone_index = zone_iy * 11 + zone_ix

    return zone_index, zone_ix, zone_iy


def get_psf_for_position(hdulist, x_pixel, y_pixel):
    """
    Extract the 101x101 oversampled PSF for a given pixel position.

    The PSF is at 10x oversampling (0.615"/px), normalized to unit integral,
    deconvolved from pixel response via accelerated Richardson-Lucy.

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.
    x_pixel, y_pixel : float
        Pixel coordinates.

    Returns
    -------
    psf : ndarray
        101x101 oversampled PSF image.
    zone_info : dict
        Zone index and grid coordinates.
    """
    zone_index, zone_ix, zone_iy = get_psf_zone_index(hdulist, x_pixel, y_pixel)

    psf_data = None
    for hdu in hdulist:
        if hdu.data is not None and hdu.data.ndim == 3 and hdu.data.shape[0] == 121:
            psf_data = hdu.data
            break

    if psf_data is None:
        raise ValueError("No PSF data cube found.")

    psf = psf_data[zone_index]

    return psf, {
        "zone_index": zone_index,
        "zone_ix": zone_ix,
        "zone_iy": zone_iy,
        "shape": psf.shape,
        "oversampling": 10,
        "pixel_scale_arcsec": 0.615,
    }


def check_psf_version(hdulist):
    """
    Check data pipeline version and warn if PSF header fix is needed.

    Pipeline versions <= 6.5.5 have a known header bug affecting PSF
    zone center coordinates.

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.

    Returns
    -------
    version : str
        Pipeline version string.
    needs_fix : bool
        True if version has the known PSF header bug.
    """
    version = "unknown"
    for hdu in hdulist:
        if 'VERSION' in hdu.header:
            version = str(hdu.header['VERSION']).strip()
            break

    needs_fix = False
    try:
        parts = version.split('.')
        if len(parts) >= 2:
            major, minor = int(parts[0]), int(parts[1])
            patch = int(parts[2]) if len(parts) >= 3 else 0
            if (major < 6) or (major == 6 and minor < 5) or \
               (major == 6 and minor == 5 and patch <= 5):
                needs_fix = True
                print(f"WARNING: Pipeline version {version} has known PSF header bug.")
                print("  PSF zone centers may be incorrect. Use uniform grid fallback.")
    except (ValueError, IndexError):
        pass

    return version, needs_fix


def psf_fwhm(hdulist):
    """
    Return PSF FWHM from header or measure from PSF data.

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.

    Returns
    -------
    fwhm_arcsec : float
        PSF FWHM in arcseconds.
    """
    # Try header first
    for hdu in hdulist:
        if 'PSF_FWHM' in hdu.header:
            return float(hdu.header['PSF_FWHM'])

    # Measure from PSF data: use central zone
    try:
        psf, info = get_psf_for_position(hdulist, 1020, 1020)
        # Find half-maximum
        peak = psf.max()
        half_max = peak / 2
        above_half = np.sum(psf >= half_max)
        # Approximate FWHM as diameter of circle with same area
        fwhm_px = 2 * np.sqrt(above_half / np.pi)
        fwhm_arcsec = fwhm_px * 0.615  # oversampled pixel scale
        return fwhm_arcsec
    except Exception:
        return None


# ============================================================================
# SECTION E: BACKGROUND SUBTRACTION & SOURCE DETECTION
# ============================================================================

def subtract_background(image_data, flags_data=None, bw=64, bh=64, fw=11, fh=11):
    """
    Subtract background using sep (SExtractor in Python).

    Parameters
    ----------
    image_data : ndarray
        2D image data (must be C-contiguous float).
    flags_data : ndarray, optional
        Flag bitmap array for masking.
    bw, bh : int
        Background mesh box width and height.
    fw, fh : int
        Background filter width and height.

    Returns
    -------
    subtracted : ndarray
        Background-subtracted image.
    bkg : sep.Background
        The background model.
    """
    if not HAS_SEP:
        raise ImportError("sep is required. Install with: pip install sep")

    # sep requires C-contiguous float32
    data = np.ascontiguousarray(image_data, dtype=np.float32)

    # Create mask from flags if provided
    mask = None
    if flags_data is not None:
        mask = create_flag_mask(flags_data, mode="background")
        mask = ~mask  # sep expects True = masked

    bkg = sep.Background(data, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh)
    subtracted = data - bkg.back()

    print(f"Background: global={bkg.globalback:.4g}, rms={bkg.globalrms:.4g}")
    return subtracted, bkg


def detect_sources(image_data, flags_data=None, threshold=3.0,
                   min_area=5, deblend_cont=0.005):
    """
    Run source detection on a SPHEREx image using sep.

    Parameters
    ----------
    image_data : ndarray
        2D image data (background-subtracted recommended).
    flags_data : ndarray, optional
        Flag bitmap array for masking.
    threshold : float
        Detection threshold in units of background RMS.
    min_area : int
        Minimum number of pixels for a source.
    deblend_cont : float
        Deblending contrast ratio.

    Returns
    -------
    sources : numpy structured array
        Detected sources with positions, fluxes, shapes.
        Key fields: x, y, flux, a, b, theta, flag
    n_sources : int
        Number of detected sources.
    """
    if not HAS_SEP:
        raise ImportError("sep is required. Install with: pip install sep")

    data = np.ascontiguousarray(image_data, dtype=np.float32)

    # Create mask from flags
    mask = None
    if flags_data is not None:
        good = create_flag_mask(flags_data, mode="photometry")
        mask = ~good

    # Estimate background if not already subtracted
    bkg = sep.Background(data, mask=mask)
    data_sub = data - bkg

    sources = sep.extract(data_sub, threshold, err=bkg.globalrms,
                          mask=mask, minarea=min_area,
                          deblend_cont=deblend_cont)

    print(f"Detected {len(sources)} sources (threshold={threshold}sigma)")
    return sources, len(sources)


def aperture_photometry(image_data, sources, radii=None, error=None):
    """
    Perform aperture photometry on detected sources at multiple radii.

    Parameters
    ----------
    image_data : ndarray
        2D image data.
    sources : structured array
        Detected sources from detect_sources().
    radii : list of float
        Aperture radii in pixels.
    error : ndarray or float, optional
        Per-pixel error or global error estimate.

    Returns
    -------
    photometry : dict
        {radius: (flux, flux_err, flag)} for each aperture radius.
    """
    if not HAS_SEP:
        raise ImportError("sep is required.")

    if radii is None:
        radii = [3.0, 4.0, 5.0]

    data = np.ascontiguousarray(image_data, dtype=np.float32)
    photometry = {}

    for r in radii:
        flux, flux_err, flag = sep.sum_circle(
            data, sources['x'], sources['y'], r, err=error
        )
        photometry[r] = {
            'flux': flux,
            'flux_err': flux_err,
            'flag': flag,
        }
        print(f"  Aperture r={r:.1f}px: median flux = {np.median(flux):.4g}")

    return photometry


def compute_fit_quality(image_data, model_data, variance_data, x, y, radius=2.5):
    """
    Compute the fit quality metric from Explanatory Supplement Eq. 18.

    fit_quality = mean(|pixel - model| / sigma) within a circular aperture.
    Should be close to 1.0 for good fits.

    Parameters
    ----------
    image_data, model_data : ndarray
        Observed and model images.
    variance_data : ndarray
        Variance map.
    x, y : float
        Source position in pixels.
    radius : float
        Aperture radius in pixels.

    Returns
    -------
    fit_quality : float
        Mean absolute deviation in units of sigma.
    """
    ny, nx = image_data.shape
    yy, xx = np.mgrid[:ny, :nx]
    dist = np.sqrt((xx - x)**2 + (yy - y)**2)
    aperture = dist <= radius

    sigma = np.sqrt(variance_data[aperture])
    residual = np.abs(image_data[aperture] - model_data[aperture])

    valid = sigma > 0
    if not np.any(valid):
        return np.nan

    fit_quality = np.mean(residual[valid] / sigma[valid])
    return fit_quality


# ============================================================================
# SECTION F: SPLICES CATALOG CROSS-MATCHING
# ============================================================================

def query_splices_region(ra, dec, radius_arcsec=5.0):
    """
    Query the SPLICES catalog of ~9M ice sources via IRSA TAP.

    SPLICES (SPHEREx Library of Ice and Cosmic Evolution Spectra)
    contains ice absorption feature measurements.

    Parameters
    ----------
    ra, dec : float
        Position in degrees.
    radius_arcsec : float
        Search radius in arcseconds.

    Returns
    -------
    results : astropy.table.Table or pyvo result set
        Matching SPLICES sources.
    """
    if not HAS_PYVO:
        raise ImportError("pyvo is required.")

    tap_service = pyvo.dal.TAPService(IRSA_TAP_URL)
    radius_deg = radius_arcsec / 3600.0

    query = f"""
    SELECT *
    FROM splices.sources
    WHERE CONTAINS(POINT('ICRS', ra, dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
    """

    try:
        results = tap_service.search(query)
        print(f"SPLICES: {len(results)} ice sources within {radius_arcsec}\" of ({ra}, {dec})")
        return results
    except Exception as e:
        print(f"SPLICES query failed: {e}")
        return None


def crossmatch_sources(source_ra, source_dec, catalog_ra, catalog_dec,
                       max_sep_arcsec=3.0):
    """
    Positional cross-match between two source lists.

    Parameters
    ----------
    source_ra, source_dec : array-like
        Positions of detected sources (degrees).
    catalog_ra, catalog_dec : array-like
        Positions of catalog sources (degrees).
    max_sep_arcsec : float
        Maximum separation for a match.

    Returns
    -------
    matches : list of tuples
        (source_idx, catalog_idx, separation_arcsec) for each match.
    """
    sources = SkyCoord(ra=source_ra*u.deg, dec=source_dec*u.deg)
    catalog = SkyCoord(ra=catalog_ra*u.deg, dec=catalog_dec*u.deg)

    idx, sep2d, _ = sources.match_to_catalog_sky(catalog)
    matches = []
    for i, (cat_idx, sep) in enumerate(zip(idx, sep2d)):
        sep_arcsec = sep.arcsec
        if sep_arcsec <= max_sep_arcsec:
            matches.append((i, int(cat_idx), float(sep_arcsec)))

    print(f"Cross-match: {len(matches)} matches within {max_sep_arcsec}\"")
    return matches


# ============================================================================
# SECTION G: ICE FEATURE DETECTION
# ============================================================================

# Key ice absorption features in SPHEREx wavelength range
ICE_FEATURES = {
    "water_ice":   {"center_um": 3.0,  "width_um": 0.3, "label": "H2O ice (3.0 um)"},
    "co2_ice":     {"center_um": 4.26, "width_um": 0.1, "label": "CO2 ice (4.26 um)"},
    "co_ice":      {"center_um": 4.67, "width_um": 0.05, "label": "CO ice (4.67 um)"},
    "ch3oh_ice":   {"center_um": 3.53, "width_um": 0.1, "label": "CH3OH ice (3.53 um)"},
}


def measure_ice_band_depth(wavelengths, fluxes, feature="water_ice"):
    """
    Measure the optical depth of an ice absorption feature.

    Computes: tau = 1 - F(feature) / F_continuum

    Parameters
    ----------
    wavelengths : array
        Wavelengths in microns.
    fluxes : array
        Flux values (any unit, continuum-relative).
    feature : str
        Key from ICE_FEATURES dict.

    Returns
    -------
    band_depth : float
        Absorption depth (0 = no absorption, 1 = complete absorption).
    continuum : float
        Estimated continuum flux at the feature.
    feature_flux : float
        Measured flux at the feature center.
    """
    feat = ICE_FEATURES.get(feature)
    if feat is None:
        raise ValueError(f"Unknown feature '{feature}'. Options: {list(ICE_FEATURES.keys())}")

    center = feat["center_um"]
    width = feat["width_um"]

    wl = np.array(wavelengths)
    fl = np.array(fluxes)

    # Feature region
    in_feature = (wl >= center - width/2) & (wl <= center + width/2)

    # Continuum: sample from flanking regions
    blue_cont = (wl >= center - 3*width) & (wl < center - width)
    red_cont = (wl > center + width) & (wl <= center + 3*width)

    if not np.any(in_feature):
        return np.nan, np.nan, np.nan

    feature_flux = np.nanmean(fl[in_feature])

    # Estimate continuum via linear interpolation of flanking regions
    cont_wl = np.concatenate([wl[blue_cont], wl[red_cont]]) if np.any(blue_cont) and np.any(red_cont) else wl[blue_cont | red_cont]
    cont_fl = np.concatenate([fl[blue_cont], fl[red_cont]]) if np.any(blue_cont) and np.any(red_cont) else fl[blue_cont | red_cont]

    if len(cont_wl) < 2:
        return np.nan, np.nan, feature_flux

    # Linear fit to continuum
    coeffs = np.polyfit(cont_wl, cont_fl, 1)
    continuum = np.polyval(coeffs, center)

    if continuum <= 0:
        return np.nan, continuum, feature_flux

    band_depth = 1.0 - feature_flux / continuum
    return band_depth, continuum, feature_flux


def classify_ice_spectrum(wavelengths, fluxes):
    """
    Classify a spectrum by its ice content.

    Parameters
    ----------
    wavelengths : array
        Wavelengths in microns.
    fluxes : array
        Flux values.

    Returns
    -------
    classification : dict
        {feature_name: {"depth": float, "detected": bool, "label": str}}
    dominant : str or None
        Name of the deepest detected feature, or None.
    """
    classification = {}
    max_depth = 0
    dominant = None

    for name in ICE_FEATURES:
        depth, continuum, feat_flux = measure_ice_band_depth(wavelengths, fluxes, name)
        detected = (not np.isnan(depth)) and (depth > 0.05)  # 5% threshold
        classification[name] = {
            "depth": depth,
            "detected": detected,
            "label": ICE_FEATURES[name]["label"],
            "continuum": continuum,
            "feature_flux": feat_flux,
        }
        if detected and depth > max_depth:
            max_depth = depth
            dominant = name

    return classification, dominant


# ============================================================================
# SECTION H: AWS S3 DIRECT ACCESS
# ============================================================================

# SPHEREx AWS bucket info
S3_BUCKET = "nasa-irsa-spherex"
S3_REGION = "us-east-1"


def open_fits_from_s3(s3_path):
    """
    Open a SPHEREx FITS file directly from AWS S3 (no download needed).

    Uses fsspec with anonymous access (no AWS credentials required).
    Bucket: nasa-irsa-spherex, region: us-east-1

    Parameters
    ----------
    s3_path : str
        Path within the bucket (e.g., "qr2/level2/...")

    Returns
    -------
    hdulist : HDUList
        The FITS file opened from S3.
    """
    if not HAS_S3FS:
        raise ImportError("s3fs is required. Install with: pip install s3fs")

    full_path = f"s3://{S3_BUCKET}/{s3_path}"
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": S3_REGION})

    with fs.open(full_path, 'rb') as f:
        hdulist = fits.open(f, memmap=False)
        # Read into memory since we'll close the S3 connection
        for hdu in hdulist:
            if hdu.data is not None:
                _ = hdu.data.copy()

    print(f"Opened from S3: {s3_path} ({len(hdulist)} HDUs)")
    return hdulist


def list_s3_observations(prefix="qr2/level2/", max_files=100):
    """
    List available observations in the SPHEREx S3 bucket.

    Parameters
    ----------
    prefix : str
        S3 key prefix to search.
    max_files : int
        Maximum number of files to return.

    Returns
    -------
    files : list of str
        S3 paths to FITS files.
    """
    if not HAS_S3FS:
        raise ImportError("s3fs is required. Install with: pip install s3fs")

    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": S3_REGION})
    full_prefix = f"{S3_BUCKET}/{prefix}"

    all_files = fs.ls(full_prefix, detail=False)
    fits_files = [f for f in all_files if f.endswith('.fits')][:max_files]

    # Strip bucket prefix for cleaner output
    fits_files = [f.replace(f"{S3_BUCKET}/", "") for f in fits_files]
    print(f"Found {len(fits_files)} FITS files under s3://{S3_BUCKET}/{prefix}")
    return fits_files


# ============================================================================
# SECTION I: DATA QUALITY & HEADER UTILITIES
# ============================================================================

def get_observation_metadata(hdulist):
    """
    Extract key observation metadata from FITS headers.

    Fields from Explanatory Supplement Appendix A.

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.

    Returns
    -------
    metadata : dict
        Key observation parameters.
    """
    meta = {}
    header = hdulist[0].header if len(hdulist) > 0 else {}

    # Try IMAGE extension header (more complete)
    for hdu in hdulist:
        if hdu.name == 'IMAGE':
            header = hdu.header
            break

    # Core identification
    meta["obs_id"] = header.get("OBSID", "unknown")
    meta["detector"] = header.get("DETECTOR", None)
    meta["date_obs"] = header.get("DATE-OBS", None)
    meta["date_avg"] = header.get("DATE-AVG", None)
    meta["exposure_time"] = header.get("XPOSURE", None)
    meta["elapsed_time"] = header.get("TELAPSE", None)

    # Pipeline info
    meta["version"] = header.get("VERSION", "unknown")
    meta["l1_dqa"] = header.get("L1DQAFLG", None)
    meta["l2_dqa"] = header.get("L2DQAFLG", None)
    meta["finast"] = header.get("FINAST", None)

    # PSF
    meta["psf_fwhm"] = header.get("PSF_FWHM", None)

    # Astrometry
    meta["crval1"] = header.get("CRVAL1", None)  # RA at reference pixel
    meta["crval2"] = header.get("CRVAL2", None)  # Dec at reference pixel

    # Pixel statistics
    meta["omega_median"] = header.get("OMEGA_MEDIAN", None)  # Solid angle median (arcsec^2)
    meta["pxsrcmsk"] = header.get("PXSRCMSK", None)  # % pixels with SOURCE flag

    # Flag counts
    flag_count_keys = [
        "L2_N_TRANSIENT", "L2_N_OVERFLOW", "L2_N_SUR_ERROR",
        "L2_N_NONFUNC", "L2_N_DICHROIC", "L2_N_MISSING",
        "L2_N_HOT", "L2_N_COLD", "L2_N_FULLSAMPLE", "L2_N_PHANMISS",
        "L2_N_NONLINEAR", "L2_N_PERSIST", "L2_N_OUTLIER", "L2_N_SOURCE",
    ]
    meta["flag_counts"] = {}
    for key in flag_count_keys:
        val = header.get(key, None)
        if val is not None:
            flag_name = key.replace("L2_N_", "")
            meta["flag_counts"][flag_name] = int(val)

    # Spacecraft position
    meta["spacecraft"] = {
        "x_km": header.get("X_SC", None),
        "y_km": header.get("Y_SC", None),
        "z_km": header.get("Z_SC", None),
        "frame": header.get("XYZ_SC_SYSTEM", "GEOCENTER"),
    }

    return meta


def get_band_info(hdulist):
    """
    Determine which SPHEREx band this image covers.

    Uses DETECTOR number (1-6) and/or wavelength information.

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.

    Returns
    -------
    band_info : dict
        Band number, wavelength range, spectral resolution, detector type.
    """
    header = hdulist[0].header
    for hdu in hdulist:
        if hdu.name == 'IMAGE':
            header = hdu.header
            break

    detector = header.get("DETECTOR", None)

    if detector and detector in SPHEREX_BANDS:
        band = SPHEREX_BANDS[detector].copy()
        band["band_number"] = detector
        return band

    # Fallback: try to determine from wavelength WCS
    try:
        wl, _ = get_pixel_wavelength(hdulist, 1020, 1020)
        if wl:
            for bnum, binfo in SPHEREX_BANDS.items():
                if binfo["wl_min"] <= wl <= binfo["wl_max"]:
                    band = binfo.copy()
                    band["band_number"] = bnum
                    band["measured_wl"] = wl
                    return band
    except Exception:
        pass

    return {"band_number": None, "wl_min": None, "wl_max": None, "R": None}


def check_data_quality(hdulist):
    """
    Check data quality using DQA flags and flag pixel counts.

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.

    Returns
    -------
    quality : dict
        DQA results and quality assessment.
    """
    meta = get_observation_metadata(hdulist)

    quality = {
        "l1_dqa": meta["l1_dqa"],
        "l2_dqa": meta["l2_dqa"],
        "finast": meta["finast"],
        "passed": True,
        "warnings": [],
    }

    # Check DQA flags
    if meta["l1_dqa"] and "Fail" in str(meta["l1_dqa"]):
        quality["passed"] = False
        quality["warnings"].append("L1 DQA failed")

    if meta["l2_dqa"] and "Fail" in str(meta["l2_dqa"]):
        quality["passed"] = False
        quality["warnings"].append("L2 DQA failed")

    if meta["finast"] is not None and meta["finast"] != 0:
        quality["warnings"].append(f"Fine astrometry flag = {meta['finast']}")

    # Check flag counts
    total_pixels = 2040 * 2040
    for flag_name, count in meta.get("flag_counts", {}).items():
        pct = 100 * count / total_pixels
        if pct > 10:
            quality["warnings"].append(f"{flag_name}: {pct:.1f}% of pixels flagged")

    # Report
    status = "PASS" if quality["passed"] else "FAIL"
    print(f"Data Quality: {status}")
    for w in quality["warnings"]:
        print(f"  Warning: {w}")

    return quality


def get_variance_and_snr(hdulist):
    """
    Extract variance map and compute signal-to-noise ratio.

    SNR = IMAGE / sqrt(VARIANCE)

    Parameters
    ----------
    hdulist : HDUList
        Open SPHEREx FITS file.

    Returns
    -------
    variance : ndarray
        Variance map in (MJy/sr)^2.
    snr : ndarray
        Signal-to-noise ratio map.
    """
    image_data = None
    variance_data = None

    for hdu in hdulist:
        if hdu.name == 'IMAGE' and hdu.data is not None:
            image_data = hdu.data.astype(np.float64)
        elif hdu.name == 'VARIANCE' and hdu.data is not None:
            variance_data = hdu.data.astype(np.float64)

    if image_data is None or variance_data is None:
        raise ValueError("IMAGE and/or VARIANCE extensions not found.")

    # Compute SNR, avoiding division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = np.sqrt(np.maximum(variance_data, 0))
        snr = np.where(sigma > 0, image_data / sigma, 0.0)

    return variance_data, snr
