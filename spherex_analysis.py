"""
SPHEREx All-Sky Infrared Survey — Data Analysis Toolkit
========================================================

A comprehensive analysis script for NASA's SPHEREx mission data,
covering visualization, statistical analysis, and machine learning.

SPHEREx maps the entire sky in 102 infrared bands (0.75–5.0 µm).
Data is accessed via IRSA's TAP service and SIA (Simple Image Access).

Requirements:
    pip install numpy astropy pyvo matplotlib scipy scikit-learn

Author: Generated for Fred
Date: 2026-03-17
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Optional imports — install as needed
try:
    import pyvo
    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False
    print("⚠ pyvo not installed. Run: pip install pyvo")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠ scikit-learn not installed. Run: pip install scikit-learn")

from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: DATA ACCESS
# ============================================================================

# IRSA TAP service endpoint
IRSA_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

# SPHEREx Quick Release 2 collections available via SIA:
#   'spherex_qr2'      — Wide Survey Spectral Image MEFs
#   'spherex_qr2_cal'  — Calibration files
#   'spherex_qr2_deep' — Deep Survey Spectral Image MEFs


def query_spherex_region(ra, dec, radius_deg=0.5, collection="spherex_qr2"):
    """
    Query SPHEREx spectral images covering a sky region via IRSA TAP.

    Parameters
    ----------
    ra, dec : float
        Right ascension and declination in degrees (J2000).
    radius_deg : float
        Search radius in degrees.
    collection : str
        SPHEREx collection name at IRSA.

    Returns
    -------
    results : astropy.table.Table or pyvo result set
        Matching spectral image metadata.
    """
    if not HAS_PYVO:
        raise ImportError("pyvo is required. Install with: pip install pyvo")

    tap_service = pyvo.dal.TAPService(IRSA_TAP_URL)

    # ADQL query using the correct SPHEREx table at IRSA
    query = f"""
    SELECT TOP 100 obs_id, s_ra, s_dec, access_url, access_format,
           target_name, em_min, em_max, s_xel1, s_xel2, t_min, calib_level
    FROM spherex.obscore
    WHERE CONTAINS(POINT('ICRS', s_ra, s_dec),
                   CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
      AND obs_collection = '{collection}'
    ORDER BY t_min DESC
    """

    print(f"Querying IRSA TAP for SPHEREx images near (RA={ra}, Dec={dec})...")
    results = tap_service.search(query)
    print(f"Found {len(results)} spectral images.")
    return results


def query_spherex_sia(ra, dec, radius_deg=0.1, collection="spherex_qr2"):
    """
    Alternative: Query SPHEREx via Simple Image Access (SIA) protocol.
    """
    if not HAS_PYVO:
        raise ImportError("pyvo is required.")

    sia_url = f"https://irsa.ipac.caltech.edu/SIA?COLLECTION={collection}"
    sia_service = pyvo.dal.SIA2Service(sia_url)

    pos = (ra, dec, radius_deg)
    results = sia_service.search(pos=pos)
    print(f"SIA query returned {len(results)} results.")
    return results


def download_fits(url, output_dir="./spherex_data"):
    """
    Download a FITS file from a data access URL.

    Parameters
    ----------
    url : str
        The access_url from a TAP/SIA query result.
    output_dir : str
        Local directory to save the file.

    Returns
    -------
    filepath : Path
        Path to the downloaded FITS file.
    """
    import urllib.request

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1].split("?")[0]
    if not filename.endswith(".fits"):
        filename += ".fits"

    filepath = output_dir / filename

    if filepath.exists():
        print(f"File already exists: {filepath}")
        return filepath

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"Saved to {filepath}")
    return filepath


def resolve_datalink(access_url):
    """
    Resolve a datalink URL to the direct FITS download URL.

    SPHEREx access_url from obscore returns datalink XML, not direct FITS.
    This parses the datalink response to find the semantics="#this" entry.

    Parameters
    ----------
    access_url : str
        The access_url from a TAP/SIA query result.

    Returns
    -------
    fits_url : str or None
        Direct URL to the FITS file, or None if not found.
    """
    if not HAS_PYVO:
        raise ImportError("pyvo is required. Install with: pip install pyvo")

    try:
        dl_result = pyvo.dal.adhoc.DatalinkResults.from_result_url(access_url)
        for row in dl_result:
            if hasattr(row, 'semantics') and row.semantics == "#this":
                return row.access_url
        # Fallback: return first result with a FITS URL
        for row in dl_result:
            url = str(row.access_url)
            if '.fits' in url.lower():
                return url
    except Exception as e:
        print(f"Datalink resolution failed: {e}")
    return None


def query_spherex_cutouts(ra, dec, size_deg=0.1, collection="spherex_qr2"):
    """
    Query for cutout-ready URLs using the artifact/plane tables.

    This uses the spherex.artifact + spherex.plane JOIN pattern from
    the IRSA cutouts tutorial, which returns URLs that support spatial
    cutouts via appended query parameters.

    Parameters
    ----------
    ra, dec : float
        Right ascension and declination in degrees (J2000).
    size_deg : float
        Cutout size in degrees.
    collection : str
        SPHEREx collection name at IRSA.

    Returns
    -------
    results : astropy.table.Table or pyvo result set
        Matching artifact metadata with cutout-ready access URLs.
    """
    if not HAS_PYVO:
        raise ImportError("pyvo is required. Install with: pip install pyvo")

    tap_service = pyvo.dal.TAPService(IRSA_TAP_URL)

    query = f"""
    SELECT TOP 50 a.uri, a.content_type, a.content_length,
           p.em_min, p.em_max, p.t_min, p.t_max,
           p.s_ra, p.s_dec
    FROM spherex.artifact AS a
    JOIN spherex.plane AS p ON a.plane_id = p.plane_id
    WHERE CONTAINS(POINT('ICRS', p.s_ra, p.s_dec),
                   CIRCLE('ICRS', {ra}, {dec}, {size_deg})) = 1
      AND a.content_type = 'application/fits'
    ORDER BY p.t_min DESC
    """

    print(f"Querying IRSA for cutout-ready SPHEREx images near (RA={ra}, Dec={dec})...")
    results = tap_service.search(query)
    print(f"Found {len(results)} cutout-ready artifacts.")
    return results


# ============================================================================
# SECTION 2: FITS FILE EXPLORATION
# ============================================================================

def explore_fits(filepath):
    """
    Print the structure of a SPHEREx Multi-Extension FITS (MEF) file.

    SPHEREx Level 2 MEFs typically contain:
        HDU 0: Primary (minimal metadata)
        HDU 1: IMAGE     — Calibrated fluxes in MJy/sr
        HDU 2: FLAGS     — Per-pixel status/processing flags
        HDU 3: VARIANCE  — Per-pixel variance estimates
        HDU 4: ZODI      — Zodiacal dust background model
        HDU 5: PSF       — 3D Point Spread Function cube
        HDU 6: WCS-WAVE  — Spectral WCS lookup (pixel → wavelength)
    """
    with fits.open(filepath) as hdul:
        print(f"\n{'='*60}")
        print(f"SPHEREx FITS Structure: {Path(filepath).name}")
        print(f"{'='*60}")
        hdul.info()

        for i, hdu in enumerate(hdul):
            if hdu.data is not None:
                print(f"\n  HDU {i} ({hdu.name}):")
                print(f"    Shape: {hdu.data.shape}")
                print(f"    Dtype: {hdu.data.dtype}")
                if np.issubdtype(hdu.data.dtype, np.floating):
                    valid = hdu.data[np.isfinite(hdu.data)]
                    if len(valid) > 0:
                        print(f"    Range: [{valid.min():.4g}, {valid.max():.4g}]")
                        print(f"    Median: {np.median(valid):.4g}")

    return hdul


def extract_spectrum_at_pixel(filepath, x, y):
    """
    Extract the infrared spectrum at a given pixel from a SPHEREx spectral image.

    Parameters
    ----------
    filepath : str or Path
        Path to a SPHEREx MEF FITS file.
    x, y : int
        Pixel coordinates.

    Returns
    -------
    wavelengths : ndarray
        Wavelengths in microns.
    fluxes : ndarray
        Flux values in MJy/sr.
    variances : ndarray
        Variance estimates.
    """
    with fits.open(filepath) as hdul:
        image_data = hdul['IMAGE'].data      # (n_bands, ny, nx) or (ny, nx)
        variance_data = hdul['VARIANCE'].data

        # Get wavelength information from WCS-WAVE extension
        wcs_wave = hdul['WCS-WAVE'].data  # wavelength lookup table

        if image_data.ndim == 3:
            # Spectral cube: extract along wavelength axis
            fluxes = image_data[:, y, x]
            variances = variance_data[:, y, x]
            wavelengths = wcs_wave[:, y, x] if wcs_wave.ndim == 3 else wcs_wave
        elif image_data.ndim == 2:
            # Single-band image
            fluxes = np.array([image_data[y, x]])
            variances = np.array([variance_data[y, x]])
            wavelengths = np.array([wcs_wave.flatten()[0]])

    return wavelengths, fluxes, variances


# ============================================================================
# SECTION 3: VISUALIZATION
# ============================================================================

def plot_spectral_image(filepath, band_index=0, cmap='inferno', vmin=None, vmax=None):
    """
    Plot a single band from a SPHEREx spectral image with WCS projection.
    """
    with fits.open(filepath) as hdul:
        image_data = hdul['IMAGE'].data

        if image_data.ndim == 3:
            data = image_data[band_index]
            title_suffix = f" (Band {band_index})"
        else:
            data = image_data
            title_suffix = ""

        # Try to get WCS
        try:
            wcs = WCS(hdul['IMAGE'].header, naxis=2)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                                   subplot_kw={'projection': wcs})
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
        except Exception:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.set_xlabel('X pixel')
            ax.set_ylabel('Y pixel')

        # Mask invalid values
        plot_data = np.where(np.isfinite(data), data, np.nan)

        im = ax.imshow(plot_data, origin='lower', cmap=cmap,
                       norm=LogNorm(vmin=vmin, vmax=vmax) if vmin and vmax
                       else None)
        plt.colorbar(im, ax=ax, label='Flux [MJy/sr]')
        ax.set_title(f'SPHEREx Spectral Image{title_suffix}')
        plt.tight_layout()
        return fig, ax


def plot_spectrum(wavelengths, fluxes, variances=None, label=None):
    """
    Plot an infrared spectrum with optional error bars.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    if variances is not None:
        errors = np.sqrt(np.abs(variances))
        ax.errorbar(wavelengths, fluxes, yerr=errors, fmt='o-',
                     markersize=3, capsize=2, alpha=0.8, label=label)
    else:
        ax.plot(wavelengths, fluxes, 'o-', markersize=3, alpha=0.8,
                label=label)

    ax.set_xlabel('Wavelength [µm]')
    ax.set_ylabel('Flux [MJy/sr]')
    ax.set_title('SPHEREx Infrared Spectrum')
    ax.set_xlim(0.75, 5.0)  # SPHEREx wavelength range
    ax.grid(True, alpha=0.3)

    # Mark key spectral features
    features = {
        1.5: 'H₂O ice',
        3.0: 'H₂O ice (strong)',
        3.3: 'PAH',
        4.26: 'CO₂ ice',
        4.67: 'CO ice',
    }
    for wl, name in features.items():
        if wavelengths.min() <= wl <= wavelengths.max():
            ax.axvline(wl, color='gray', linestyle='--', alpha=0.5)
            ax.text(wl, ax.get_ylim()[1] * 0.95, name, fontsize=7,
                    ha='center', va='top', rotation=90, color='gray')

    if label:
        ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_rgb_composite(filepath, r_band=80, g_band=50, b_band=10):
    """
    Create an RGB false-color composite from three SPHEREx bands.

    Parameters
    ----------
    filepath : str or Path
        Path to SPHEREx MEF.
    r_band, g_band, b_band : int
        Band indices for R, G, B channels.
    """
    with fits.open(filepath) as hdul:
        data = hdul['IMAGE'].data

        if data.ndim != 3:
            print("Need 3D spectral cube for RGB composite.")
            return

        def normalize(arr):
            arr = np.where(np.isfinite(arr), arr, 0)
            p2, p98 = np.percentile(arr[arr > 0], [2, 98]) if np.any(arr > 0) else (0, 1)
            return np.clip((arr - p2) / (p98 - p2), 0, 1)

        r = normalize(data[r_band])
        g = normalize(data[g_band])
        b = normalize(data[b_band])

        rgb = np.stack([r, g, b], axis=-1)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb, origin='lower')
        ax.set_title(f'SPHEREx False Color (R={r_band}, G={g_band}, B={b_band})')
        ax.set_xlabel('X pixel')
        ax.set_ylabel('Y pixel')
        plt.tight_layout()
        return fig, ax


def plot_sky_coverage(results_table, title="SPHEREx Sky Coverage"):
    """
    Plot the sky positions of queried SPHEREx observations on an Aitoff projection.
    """
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='aitoff')

    # Extract RA/Dec from query results
    try:
        ra = np.array(results_table['s_ra'])
        dec = np.array(results_table['s_dec'])
    except (KeyError, TypeError):
        ra = np.array(results_table['ra'])
        dec = np.array(results_table['dec'])

    # Convert to radians for Aitoff, wrap RA to [-π, π]
    ra_rad = np.deg2rad(ra)
    ra_rad = np.where(ra_rad > np.pi, ra_rad - 2 * np.pi, ra_rad)
    dec_rad = np.deg2rad(dec)

    ax.scatter(ra_rad, dec_rad, s=2, alpha=0.5, c='cyan', edgecolors='none')
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ============================================================================
# SECTION 4: STATISTICAL ANALYSIS
# ============================================================================

def compute_band_statistics(filepath):
    """
    Compute per-band statistics across a SPHEREx spectral cube.

    Returns a dict with per-band: mean, median, std, skewness, kurtosis.
    """
    with fits.open(filepath) as hdul:
        data = hdul['IMAGE'].data

        if data.ndim != 3:
            print("Need 3D spectral cube for band statistics.")
            return None

        n_bands = data.shape[0]
        stats_dict = {
            'band': [], 'mean': [], 'median': [], 'std': [],
            'skewness': [], 'kurtosis': [], 'valid_frac': []
        }

        for i in range(n_bands):
            band = data[i]
            valid = band[np.isfinite(band)]

            stats_dict['band'].append(i)
            stats_dict['valid_frac'].append(len(valid) / band.size)

            if len(valid) > 10:
                stats_dict['mean'].append(np.mean(valid))
                stats_dict['median'].append(np.median(valid))
                stats_dict['std'].append(np.std(valid))
                stats_dict['skewness'].append(float(stats.skew(valid)))
                stats_dict['kurtosis'].append(float(stats.kurtosis(valid)))
            else:
                for key in ['mean', 'median', 'std', 'skewness', 'kurtosis']:
                    stats_dict[key].append(np.nan)

        return stats_dict


def plot_band_statistics(stats_dict, wavelengths=None):
    """
    Visualize per-band statistics across the SPHEREx wavelength range.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    x = wavelengths if wavelengths is not None else stats_dict['band']
    xlabel = 'Wavelength [µm]' if wavelengths is not None else 'Band Index'

    # Mean and median flux
    axes[0].plot(x, stats_dict['mean'], 'b-', label='Mean', alpha=0.8)
    axes[0].plot(x, stats_dict['median'], 'r--', label='Median', alpha=0.8)
    axes[0].fill_between(x,
                          np.array(stats_dict['mean']) - np.array(stats_dict['std']),
                          np.array(stats_dict['mean']) + np.array(stats_dict['std']),
                          alpha=0.2, color='blue')
    axes[0].set_ylabel('Flux [MJy/sr]')
    axes[0].set_title('Per-Band Flux Statistics')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Skewness
    axes[1].plot(x, stats_dict['skewness'], 'g-', alpha=0.8)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Skewness')
    axes[1].grid(True, alpha=0.3)

    # Valid pixel fraction
    axes[2].plot(x, stats_dict['valid_frac'], 'purple', alpha=0.8)
    axes[2].set_ylabel('Valid Pixel Fraction')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def spectral_correlation_matrix(filepath, sample_size=5000):
    """
    Compute and plot the correlation matrix between SPHEREx bands.

    This reveals which wavelength bands vary together — useful for
    identifying correlated emission mechanisms.
    """
    with fits.open(filepath) as hdul:
        data = hdul['IMAGE'].data

        if data.ndim != 3:
            return None

        n_bands, ny, nx = data.shape

        # Flatten spatial dims and sample
        flat = data.reshape(n_bands, -1).T  # (n_pixels, n_bands)
        valid_mask = np.all(np.isfinite(flat), axis=1)
        flat_valid = flat[valid_mask]

        if len(flat_valid) > sample_size:
            idx = np.random.choice(len(flat_valid), sample_size, replace=False)
            flat_valid = flat_valid[idx]

        corr = np.corrcoef(flat_valid.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Pearson Correlation')
        ax.set_xlabel('Band Index')
        ax.set_ylabel('Band Index')
        ax.set_title('Inter-Band Correlation Matrix')
        plt.tight_layout()
        return fig, ax, corr


# ============================================================================
# SECTION 5: MACHINE LEARNING
# ============================================================================

def spectral_clustering(filepath, n_clusters=5, sample_size=10000):
    """
    Cluster pixels by their spectral signatures using K-Means.

    This identifies distinct spectral populations in the field —
    e.g., stars, galaxies, nebulae, background regions.

    Parameters
    ----------
    filepath : str or Path
        SPHEREx MEF FITS file.
    n_clusters : int
        Number of clusters for K-Means.
    sample_size : int
        Max pixels to cluster (for performance).

    Returns
    -------
    labels_map : ndarray
        2D array of cluster labels (same shape as image).
    cluster_centers : ndarray
        Mean spectrum for each cluster.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required. pip install scikit-learn")

    with fits.open(filepath) as hdul:
        data = hdul['IMAGE'].data

        if data.ndim != 3:
            raise ValueError("Need 3D spectral cube.")

        n_bands, ny, nx = data.shape
        flat = data.reshape(n_bands, -1).T  # (n_pixels, n_bands)

        # Mask invalid pixels
        valid_mask = np.all(np.isfinite(flat), axis=1)
        valid_indices = np.where(valid_mask)[0]
        flat_valid = flat[valid_mask]

        print(f"Valid pixels: {len(flat_valid)} / {flat.shape[0]}")

        # Subsample if needed
        if len(flat_valid) > sample_size:
            sample_idx = np.random.choice(len(flat_valid), sample_size, replace=False)
            flat_sample = flat_valid[sample_idx]
        else:
            flat_sample = flat_valid
            sample_idx = np.arange(len(flat_valid))

        # Standardize
        scaler = StandardScaler()
        flat_scaled = scaler.fit_transform(flat_sample)

        # K-Means clustering
        print(f"Running K-Means with k={n_clusters}...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_sample = kmeans.fit_predict(flat_scaled)

        # Predict on all valid pixels
        all_scaled = scaler.transform(flat_valid)
        all_labels = kmeans.predict(all_scaled)

        # Reconstruct 2D label map
        labels_map = np.full(ny * nx, -1)
        labels_map[valid_indices] = all_labels
        labels_map = labels_map.reshape(ny, nx)

        # Cluster centers in original units
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

        return labels_map, cluster_centers


def plot_cluster_map_and_spectra(labels_map, cluster_centers, wavelengths=None):
    """
    Visualize the K-Means clustering results: spatial map and mean spectra.
    """
    n_clusters = cluster_centers.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cluster map
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    im = axes[0].imshow(labels_map, origin='lower', cmap=cmap, vmin=-0.5,
                         vmax=n_clusters - 0.5, interpolation='nearest')
    plt.colorbar(im, ax=axes[0], ticks=range(n_clusters), label='Cluster')
    axes[0].set_title('Spectral Cluster Map')
    axes[0].set_xlabel('X pixel')
    axes[0].set_ylabel('Y pixel')

    # Cluster spectra
    x = wavelengths if wavelengths is not None else np.arange(cluster_centers.shape[1])
    xlabel = 'Wavelength [µm]' if wavelengths is not None else 'Band Index'

    for i in range(n_clusters):
        axes[1].plot(x, cluster_centers[i], color=colors[i], linewidth=2,
                     label=f'Cluster {i}')

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Flux [MJy/sr]')
    axes[1].set_title('Mean Spectra per Cluster')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def pca_spectral_decomposition(filepath, n_components=5, sample_size=10000):
    """
    PCA decomposition of SPHEREx spectral cubes.

    Identifies the principal spectral components — the dominant modes
    of variation in the infrared across the field of view.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required.")

    with fits.open(filepath) as hdul:
        data = hdul['IMAGE'].data
        if data.ndim != 3:
            raise ValueError("Need 3D cube.")

        n_bands, ny, nx = data.shape
        flat = data.reshape(n_bands, -1).T
        valid_mask = np.all(np.isfinite(flat), axis=1)
        flat_valid = flat[valid_mask]

        if len(flat_valid) > sample_size:
            idx = np.random.choice(len(flat_valid), sample_size, replace=False)
            flat_valid = flat_valid[idx]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(flat_valid)

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(scaled)

        print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
        print(f"Total explained: {pca.explained_variance_ratio_.sum():.1%}")

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Variance explained
        axes[0].bar(range(n_components), pca.explained_variance_ratio_,
                    color='steelblue')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Spectral Decomposition')

        # Eigenvectors (principal spectral components)
        for i in range(min(3, n_components)):
            axes[1].plot(pca.components_[i], label=f'PC{i+1}', linewidth=1.5)

        axes[1].set_xlabel('Band Index')
        axes[1].set_ylabel('Component Weight')
        axes[1].set_title('Principal Spectral Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes, pca, components


def anomaly_detection(filepath, contamination=0.05, sample_size=10000):
    """
    Detect spectrally anomalous pixels using Isolation Forest.

    Anomalies might be: rare source types, artifacts, transient events,
    or objects with unusual spectral energy distributions.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required.")

    from sklearn.ensemble import IsolationForest

    with fits.open(filepath) as hdul:
        data = hdul['IMAGE'].data
        if data.ndim != 3:
            raise ValueError("Need 3D cube.")

        n_bands, ny, nx = data.shape
        flat = data.reshape(n_bands, -1).T
        valid_mask = np.all(np.isfinite(flat), axis=1)
        valid_indices = np.where(valid_mask)[0]
        flat_valid = flat[valid_mask]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(flat_valid)

        print(f"Running Isolation Forest (contamination={contamination})...")
        iso = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso.fit_predict(scaled)

        # Map back to 2D
        anomaly_map = np.zeros(ny * nx)
        anomaly_map[valid_indices] = predictions
        anomaly_map = anomaly_map.reshape(ny, nx)

        n_anomalies = np.sum(predictions == -1)
        print(f"Detected {n_anomalies} anomalous pixels ({n_anomalies/len(predictions):.1%})")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(anomaly_map, origin='lower', cmap='RdYlBu', vmin=-1, vmax=1)
        ax.set_title(f'Anomaly Map ({n_anomalies} anomalous pixels)')
        ax.set_xlabel('X pixel')
        ax.set_ylabel('Y pixel')
        plt.tight_layout()

        return fig, ax, anomaly_map


# ============================================================================
# SECTION 6: INTERESTING TARGETS TO EXPLORE
# See also: spherex_3d_visualization.py for an interactive 3D view of these
# targets (and more) placed in a 360-degree sphere around Earth.
# ============================================================================

INTERESTING_TARGETS = {
    "Orion Nebula (M42)":     {"ra": 83.822,  "dec": -5.391,  "why": "Rich star-forming region with PAH + ice features"},
    "Galactic Center":        {"ra": 266.417, "dec": -29.008, "why": "Dense stellar field, dust, CO/CO₂ absorption"},
    "Crab Nebula (M1)":       {"ra": 83.633,  "dec": 22.015,  "why": "Supernova remnant — synchrotron + line emission"},
    "Large Magellanic Cloud":  {"ra": 80.894,  "dec": -69.756, "why": "Nearby galaxy — varied stellar populations"},
    "North Ecliptic Pole":    {"ra": 270.0,   "dec": 66.561,  "why": "SPHEREx deep field — deepest coverage"},
    "South Ecliptic Pole":    {"ra": 90.0,    "dec": -66.561, "why": "SPHEREx deep field — deepest coverage"},
    "Rho Ophiuchi cloud":     {"ra": 246.79,  "dec": -24.54,  "why": "Nearby molecular cloud — ice features at 3 µm"},
}


def print_targets():
    """Print the table of interesting SPHEREx analysis targets."""
    print(f"\n{'Target':<30s} {'RA':>8s} {'Dec':>8s}  {'Science Case'}")
    print("-" * 90)
    for name, info in INTERESTING_TARGETS.items():
        print(f"{name:<30s} {info['ra']:8.3f} {info['dec']:8.3f}  {info['why']}")


# ============================================================================
# SECTION 7: QUICK-START WORKFLOW
# ============================================================================

def quickstart(target_name="Orion Nebula (M42)", n_clusters=5):
    """
    Complete end-to-end analysis workflow for a named target.

    1. Query IRSA for SPHEREx data
    2. Download the first matching file
    3. Explore FITS structure
    4. Visualize the field
    5. Compute statistics
    6. Run K-Means spectral clustering
    7. PCA decomposition

    Usage:
        quickstart("Orion Nebula (M42)")
    """
    if target_name not in INTERESTING_TARGETS:
        print(f"Unknown target. Choose from:")
        print_targets()
        return

    target = INTERESTING_TARGETS[target_name]
    print(f"\n{'='*60}")
    print(f"SPHEREx Analysis: {target_name}")
    print(f"RA={target['ra']}, Dec={target['dec']}")
    print(f"Science: {target['why']}")
    print(f"{'='*60}\n")

    # Step 1: Query
    results = query_spherex_region(target['ra'], target['dec'])

    if len(results) == 0:
        print("No data found for this region yet. SPHEREx coverage is still growing.")
        print("Try the ecliptic poles (deep fields) or check back later.")
        return

    # Step 2: Download first result
    url = results[0]['access_url']
    filepath = download_fits(url)

    # Step 3: Explore
    explore_fits(filepath)

    # Step 4: Visualize
    print("\nGenerating visualizations...")
    plot_spectral_image(filepath)
    plt.savefig('spherex_field.png', dpi=150, bbox_inches='tight')

    # Step 5: Statistics
    print("\nComputing band statistics...")
    band_stats = compute_band_statistics(filepath)
    if band_stats:
        plot_band_statistics(band_stats)
        plt.savefig('spherex_stats.png', dpi=150, bbox_inches='tight')

    # Step 6: Clustering
    print("\nRunning spectral clustering...")
    labels_map, centers = spectral_clustering(filepath, n_clusters=n_clusters)
    plot_cluster_map_and_spectra(labels_map, centers)
    plt.savefig('spherex_clusters.png', dpi=150, bbox_inches='tight')

    # Step 7: PCA
    print("\nRunning PCA decomposition...")
    pca_spectral_decomposition(filepath)
    plt.savefig('spherex_pca.png', dpi=150, bbox_inches='tight')

    print("\nDone! Plots saved to current directory.")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("SPHEREx Analysis Toolkit")
    print("=" * 40)
    print()
    print_targets()
    print()
    print("Quick start:")
    print('  >>> from spherex_analysis import quickstart')
    print('  >>> quickstart("Orion Nebula (M42)")')
    print()
    print("Or step by step:")
    print('  >>> results = query_spherex_region(83.822, -5.391)')
    print('  >>> filepath = download_fits(results[0]["access_url"])')
    print('  >>> explore_fits(filepath)')
    print('  >>> labels, centers = spectral_clustering(filepath, n_clusters=5)')
    print()
    print("Required packages: numpy, astropy, pyvo, matplotlib, scipy, scikit-learn")
