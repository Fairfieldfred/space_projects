"""
Gaia DR3 3D Star Map Visualizer
================================

Interactive 3D visualization of the solar neighborhood using real star data
from the ESA Gaia DR3 catalog. Stars are positioned using measured parallax
distances and colored by spectral type (bp_rp color index).

Earth/Sun sits at the origin. Stars are rendered with log-scaled radial
distances and grouped by spectral type for interactive legend toggling.

Uses Plotly for browser-based interactivity (orbit, zoom, pan, hover tooltips).

Usage:
    python gaia_3d_visualization.py
    python gaia_3d_visualization.py --distance 100 --proper-motion

    Or from Python:
        from gaia_3d_visualization import render_gaia_3d
        render_gaia_3d()

Requirements:
    pip install numpy astropy plotly pyvo pandas

Author: Generated for Fred
Date: 2026-03-23
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import time
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("plotly not installed. Run: pip install plotly")

try:
    import pyvo
    HAS_PYVO = True
except ImportError:
    HAS_PYVO = False
    print("pyvo not installed. Run: pip install pyvo")


# ============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# ============================================================================

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

CACHE_DIR = Path("./gaia_cache")
DEFAULT_MAX_DISTANCE_PC = 50
MAX_RENDERED_POINTS = 10_000
CACHE_EXPIRY_DAYS = 7

# Spectral type classification by bp_rp color index
# (low_bp_rp, high_bp_rp, label, hex_color)
SPECTRAL_BOUNDARIES = [
    (-99,  -0.1, "O/B", "#6699FF"),
    (-0.1,  0.3, "A",   "#AABBFF"),
    (0.3,   0.6, "F",   "#F0F0FF"),
    (0.6,   0.9, "G",   "#FFFF88"),
    (0.9,   1.5, "K",   "#FFAA44"),
    (1.5,   99,  "M",   "#FF4422"),
]

# Anchor points for smooth bp_rp -> hex color interpolation
# (bp_rp_value, R, G, B)
COLOR_ANCHORS = [
    (-0.5,  51,  102, 255),  # deep blue (O stars)
    (-0.1, 102,  153, 255),  # blue
    ( 0.0, 170,  187, 255),  # blue-white (A stars)
    ( 0.3, 240,  240, 255),  # white (F stars)
    ( 0.6, 255,  255, 136),  # yellow-white (G stars)
    ( 1.0, 255,  215,   0),  # yellow (late G / early K)
    ( 1.5, 255,  140,   0),  # orange (K stars)
    ( 2.0, 255,   69,   0),  # red-orange (early M)
    ( 3.0, 204,   34,   0),  # deep red (late M)
]

# Notable nearby stars for labeling and highlighting.
# RA/Dec are approximate (J2000); distance_pc is used to verify matches
# since high proper-motion stars shift significantly by Gaia's epoch (J2016).
NOTABLE_STARS = {
    "Proxima Centauri":  {"ra": 217.429, "dec": -62.680, "distance_pc": 1.30},
    "Alpha Centauri B":  {"ra": 219.914, "dec": -60.839, "distance_pc": 1.34},
    "Barnard's Star":    {"ra": 269.452, "dec":   4.694, "distance_pc": 1.83},
    "Sirius B":          {"ra": 101.287, "dec": -16.716, "distance_pc": 2.64},
    "Tau Ceti":          {"ra":  26.017, "dec": -15.937, "distance_pc": 3.65},
    "Epsilon Eridani":   {"ra":  53.233, "dec":  -9.458, "distance_pc": 3.22},
    "Lacaille 9352":     {"ra": 346.467, "dec": -35.853, "distance_pc": 3.29},
    "Ross 128":          {"ra": 176.937, "dec":   0.800, "distance_pc": 3.37},
    "Wolf 359":          {"ra": 164.120, "dec":   7.015, "distance_pc": 2.39},
    "Lalande 21185":     {"ra": 165.834, "dec":  35.970, "distance_pc": 2.55},
    "61 Cygni A":        {"ra": 316.720, "dec":  38.750, "distance_pc": 3.50},
    "Pollux":            {"ra": 116.329, "dec":  28.026, "distance_pc": 10.36},
    "Capella":           {"ra":  79.172, "dec":  45.998, "distance_pc": 13.04},
    "Aldebaran":         {"ra":  68.980, "dec":  16.509, "distance_pc": 20.43},
}
# Note: Sirius A (G~-1.4), Vega (G~0.0), Alpha Centauri A (G~-0.01),
# Procyon (G~0.4), Altair (G~0.8), Arcturus (G~-0.3), Fomalhaut (G~1.2)
# are too bright for reliable Gaia photometry and are excluded from the catalog.


# ============================================================================
# SECTION 2: GAIA DATA ACQUISITION
# ============================================================================

def _build_gaia_adql(max_distance_pc=50, mag_limit=15.0):
    """
    Build an ADQL query for Gaia DR3 stars within a distance limit.

    Parameters
    ----------
    max_distance_pc : float
        Maximum distance in parsecs.
    mag_limit : float
        Faintest G-band magnitude to include.

    Returns
    -------
    query : str
        ADQL query string.
    """
    min_parallax = 1000.0 / max_distance_pc  # mas
    return f"""
    SELECT
        source_id, designation, ra, dec,
        parallax, parallax_error, parallax_over_error,
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp,
        teff_gspphot,
        pmra, pmdec, radial_velocity
    FROM gaiadr3.gaia_source
    WHERE parallax > {min_parallax}
      AND parallax_over_error > 3
      AND phot_g_mean_mag < {mag_limit}
    ORDER BY phot_g_mean_mag ASC
    """


def _cache_path(max_distance_pc, mag_limit, cache_dir=None):
    """Return the cache file path for given query parameters."""
    d = cache_dir or CACHE_DIR
    return Path(d) / f"gaia_dr3_{max_distance_pc}pc_mag{mag_limit}.csv"


def _load_cache(cache_path):
    """Load cached results if the file exists and is fresh."""
    p = Path(cache_path)
    if not p.exists():
        return None
    age_days = (time.time() - p.stat().st_mtime) / 86400
    if age_days > CACHE_EXPIRY_DAYS:
        print(f"Cache expired ({age_days:.1f} days old). Will re-query.")
        return None
    print(f"Loading cached Gaia data from {p} ({age_days:.1f} days old)...")
    return pd.read_csv(p)


def _save_cache(df, cache_path):
    """Save query results to CSV cache."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Cached {len(df)} stars to {p}")


def query_gaia_nearby(max_distance_pc=50, mag_limit=15.0,
                      use_cache=True, cache_dir=None):
    """
    Query Gaia DR3 for stars within max_distance_pc.

    Uses pyvo TAP to query ESA's Gaia archive. Results are cached
    locally as CSV to avoid re-querying on subsequent runs.

    Parameters
    ----------
    max_distance_pc : float
        Maximum distance in parsecs. Default 50.
    mag_limit : float
        Faintest G-band magnitude to include. Default 15.0.
    use_cache : bool
        If True, load from cache when available.
    cache_dir : str or Path, optional
        Directory for cache files. Defaults to ./gaia_cache/

    Returns
    -------
    df : pandas.DataFrame
        Gaia DR3 star data.
    """
    cp = _cache_path(max_distance_pc, mag_limit, cache_dir)

    if use_cache:
        cached = _load_cache(cp)
        if cached is not None:
            print(f"  {len(cached)} stars loaded from cache.")
            return cached

    if not HAS_PYVO:
        raise ImportError("pyvo is required. Install with: pip install pyvo")

    adql = _build_gaia_adql(max_distance_pc, mag_limit)
    print(f"Querying Gaia DR3 for stars within {max_distance_pc} pc "
          f"(mag < {mag_limit})...")
    print(f"  TAP endpoint: {GAIA_TAP_URL}")

    tap = pyvo.dal.TAPService(GAIA_TAP_URL)
    result = tap.search(adql, maxrec=MAX_RENDERED_POINTS + 5000)

    df = result.to_table().to_pandas()
    print(f"  Query returned {len(df)} stars.")

    # Filter distance edge cases (parallax inversion can exceed max)
    df["distance_pc"] = 1000.0 / df["parallax"]
    df = df[df["distance_pc"] <= max_distance_pc].copy()
    df = df[df["distance_pc"] > 0].copy()

    if len(df) > MAX_RENDERED_POINTS:
        print(f"  Warning: {len(df)} stars exceeds rendering limit of "
              f"{MAX_RENDERED_POINTS}. Keeping brightest.")
        df = df.nsmallest(MAX_RENDERED_POINTS, "phot_g_mean_mag")

    print(f"  Final count: {len(df)} stars within {max_distance_pc} pc.")
    _save_cache(df, cp)
    return df


# ============================================================================
# SECTION 3: DATA PROCESSING
# ============================================================================

def compute_distances(df):
    """
    Compute distance and absolute magnitude from parallax.

    Adds columns: distance_pc, distance_ly, abs_mag.
    """
    df = df.copy()
    df["distance_pc"] = 1000.0 / df["parallax"]
    df["distance_ly"] = df["distance_pc"] * 3.26156
    df["abs_mag"] = (df["phot_g_mean_mag"]
                     - 5.0 * np.log10(df["distance_pc"])
                     + 5.0)
    return df


def compute_cartesian(df, log_scale=True):
    """
    Convert RA/Dec/distance to Cartesian x, y, z.

    Uses log10(distance_pc) as radial coordinate when log_scale=True.
    Adds columns: x, y, z, log_distance.
    """
    df = df.copy()
    ra_rad = np.deg2rad(df["ra"].values)
    dec_rad = np.deg2rad(df["dec"].values)
    dist = df["distance_pc"].values

    df["log_distance"] = np.log10(np.maximum(dist, 0.1))
    r = df["log_distance"].values if log_scale else dist

    df["x"] = r * np.cos(dec_rad) * np.cos(ra_rad)
    df["y"] = r * np.cos(dec_rad) * np.sin(ra_rad)
    df["z"] = r * np.sin(dec_rad)
    return df


def assign_spectral_colors(df):
    """
    Map bp_rp color index to hex colors and spectral type labels.

    Stars without bp_rp data get gray (#888888) and type 'Unknown'.
    Falls back to teff_gspphot if bp_rp is unavailable.
    Adds columns: color_hex, spectral_type.
    """
    df = df.copy()
    colors = []
    types = []

    for _, row in df.iterrows():
        bp_rp = row.get("bp_rp")
        teff = row.get("teff_gspphot")

        if pd.notna(bp_rp):
            colors.append(bp_rp_to_hex(bp_rp))
            types.append(bp_rp_to_spectral_type(bp_rp))
        elif pd.notna(teff):
            colors.append(teff_to_hex(teff))
            types.append(teff_to_spectral_type(teff))
        else:
            colors.append("#888888")
            types.append("Unknown")

    df["color_hex"] = colors
    df["spectral_type"] = types
    return df


def assign_marker_sizes(df, min_size=2, max_size=10):
    """
    Map apparent G magnitude to marker size (brighter = larger).

    Adds column: marker_size.
    """
    df = df.copy()
    mag = df["phot_g_mean_mag"].values
    mag_min = np.nanmin(mag)
    mag_max = np.nanmax(mag)
    mag_range = mag_max - mag_min
    if mag_range < 0.1:
        mag_range = 1.0

    # Invert: bright (low mag) → large size
    normalized = 1.0 - (mag - mag_min) / mag_range
    df["marker_size"] = min_size + normalized * (max_size - min_size)
    return df


def identify_notable_stars(df):
    """
    Cross-match DataFrame against NOTABLE_STARS by coordinate proximity.

    Uses astropy SkyCoord for angular separation matching (5 arcsec).
    Adds column: notable_name (NaN for non-notable stars).
    """
    df = df.copy()
    df["notable_name"] = np.nan

    if len(df) == 0:
        return df

    cat_coords = SkyCoord(ra=df["ra"].values * u.deg,
                          dec=df["dec"].values * u.deg)

    # Use 300 arcsec tolerance to account for proper motion shifts
    # between reference epoch (J2000) and Gaia DR3 epoch (J2016.0).
    # Verify matches by checking distance is within 50% of expected.
    match_radius_arcsec = 300.0

    for name, info in NOTABLE_STARS.items():
        star_coord = SkyCoord(ra=info["ra"] * u.deg,
                              dec=info["dec"] * u.deg)
        sep = star_coord.separation(cat_coords)
        min_idx = np.argmin(sep)
        if sep[min_idx].arcsec < match_radius_arcsec:
            # Verify distance is roughly consistent
            expected_dist = info.get("distance_pc")
            actual_dist = df.iloc[min_idx]["distance_pc"]
            if expected_dist and abs(actual_dist - expected_dist) / expected_dist > 0.5:
                continue  # distance mismatch — likely wrong star
            df.iloc[min_idx, df.columns.get_loc("notable_name")] = name

    n_found = df["notable_name"].notna().sum()
    print(f"  Matched {n_found}/{len(NOTABLE_STARS)} notable stars.")
    return df


# ============================================================================
# SECTION 4: COLOR AND SPECTRAL MAPPING
# ============================================================================

def bp_rp_to_hex(bp_rp):
    """
    Convert bp_rp color index to a hex color string using smooth interpolation.

    Returns '#888888' for NaN values.
    """
    if pd.isna(bp_rp):
        return "#888888"

    anchors_x = [a[0] for a in COLOR_ANCHORS]
    anchors_r = [a[1] for a in COLOR_ANCHORS]
    anchors_g = [a[2] for a in COLOR_ANCHORS]
    anchors_b = [a[3] for a in COLOR_ANCHORS]

    r = int(np.clip(np.interp(bp_rp, anchors_x, anchors_r), 0, 255))
    g = int(np.clip(np.interp(bp_rp, anchors_x, anchors_g), 0, 255))
    b = int(np.clip(np.interp(bp_rp, anchors_x, anchors_b), 0, 255))

    return f"#{r:02x}{g:02x}{b:02x}"


def bp_rp_to_spectral_type(bp_rp):
    """Map bp_rp color index to a spectral type string."""
    if pd.isna(bp_rp):
        return "Unknown"
    for low, high, label, _ in SPECTRAL_BOUNDARIES:
        if low <= bp_rp < high:
            return label
    return "M"  # very red stars


def teff_to_hex(teff):
    """
    Convert effective temperature to hex color (blackbody approximation).
    """
    if pd.isna(teff):
        return "#888888"
    # Approximate bp_rp from teff and delegate
    # Rough relation: bp_rp ~ 7000/teff - 0.5 (empirical fit)
    approx_bp_rp = 7000.0 / max(teff, 2000) - 0.5
    return bp_rp_to_hex(approx_bp_rp)


def teff_to_spectral_type(teff):
    """Map effective temperature to spectral type."""
    if pd.isna(teff):
        return "Unknown"
    if teff > 30000:
        return "O/B"
    elif teff > 10000:
        return "O/B"
    elif teff > 7500:
        return "A"
    elif teff > 6000:
        return "F"
    elif teff > 5200:
        return "G"
    elif teff > 3700:
        return "K"
    else:
        return "M"


def _build_colorscale():
    """Build a Plotly-compatible colorscale from the spectral color anchors."""
    bp_rp_min = COLOR_ANCHORS[0][0]
    bp_rp_max = COLOR_ANCHORS[-1][0]
    bp_rp_range = bp_rp_max - bp_rp_min

    scale = []
    for bp_rp, r, g, b in COLOR_ANCHORS:
        frac = (bp_rp - bp_rp_min) / bp_rp_range
        scale.append([frac, f"rgb({r},{g},{b})"])
    return scale


# ============================================================================
# SECTION 5: STRUCTURAL OVERLAYS
# ============================================================================

def _format_distance(distance_pc):
    """Format a distance in parsecs to a human-readable string."""
    if distance_pc < 1000:
        return f"{distance_pc:.1f} pc"
    elif distance_pc < 1e6:
        return f"{distance_pc / 1000:.1f} kpc"
    else:
        return f"{distance_pc / 1e6:.1f} Mpc"


def _create_distance_shells(log_scale=True):
    """
    Create wireframe spheres at key distances as spatial references.
    Adapted for the solar neighborhood scale (5-50 pc).
    """
    shells = [
        ("5 pc", 5),
        ("10 pc", 10),
        ("25 pc", 25),
        ("50 pc", 50),
    ]
    shell_colors = [
        "rgba(255, 255, 255, 0.08)",
        "rgba(200, 200, 255, 0.08)",
        "rgba(150, 150, 255, 0.08)",
        "rgba(100, 100, 255, 0.08)",
    ]

    traces = []
    n_lines = 8
    n_points = 60

    for (label, dist_pc), color in zip(shells, shell_colors):
        radius = np.log10(dist_pc) if log_scale else dist_pc

        # Meridians
        for j in range(n_lines):
            phi = 2 * np.pi * j / n_lines
            theta = np.linspace(0, np.pi, n_points)
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
                hoverinfo="text",
                hovertext=label,
            ))

        # Equatorial circle
        phi = np.linspace(0, 2 * np.pi, n_points)
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        z = np.zeros(n_points)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=color, width=1),
            showlegend=False,
            hoverinfo="text",
            hovertext=label,
        ))

        # Label at top
        traces.append(go.Scatter3d(
            x=[0], y=[0], z=[radius],
            mode="text",
            text=[label],
            textfont=dict(size=9, color="rgba(180, 180, 220, 0.6)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


def _create_milky_way_arms(log_scale=True):
    """
    Generate Milky Way spiral arm traces in Galactocentric coordinates,
    then transform to heliocentric for plotting.
    """
    sun_x_kpc = -8.122
    sun_y_kpc = 0.0

    n_arms = 4
    traces = []
    arm_names = ["Perseus Arm", "Sagittarius Arm",
                 "Scutum-Centaurus Arm", "Norma Arm"]
    arm_colors = [
        "rgba(100, 150, 255, 0.3)",
        "rgba(100, 150, 255, 0.25)",
        "rgba(100, 150, 255, 0.2)",
        "rgba(100, 150, 255, 0.15)",
    ]

    theta = np.linspace(0, 4 * np.pi, 400)

    for i in range(n_arms):
        offset = i * (2 * np.pi / n_arms)
        r_kpc = 2.0 * np.exp(0.21 * theta)
        mask = r_kpc <= 18
        r_kpc = r_kpc[mask]
        t = theta[mask] + offset

        gx = r_kpc * np.cos(t)
        gy = r_kpc * np.sin(t)

        hx = gx - sun_x_kpc
        hy = gy - sun_y_kpc

        dist_pc = np.sqrt(hx**2 + hy**2) * 1000
        dist_pc = np.maximum(dist_pc, 1)

        if log_scale:
            scale = np.log10(dist_pc)
        else:
            scale = dist_pc

        angle = np.arctan2(hy, hx)
        px = scale * np.cos(angle)
        py = scale * np.sin(angle)
        pz = np.zeros_like(px)

        traces.append(go.Scatter3d(
            x=px, y=py, z=pz,
            mode="lines",
            line=dict(color=arm_colors[i], width=3),
            name=arm_names[i],
            legendgroup="milky_way",
            legendgrouptitle_text="Milky Way (schematic)",
            showlegend=(i == 0),
            hoverinfo="text",
            hovertext=arm_names[i],
        ))

    return traces


def _create_galactic_plane(log_scale=True, max_radius_pc=200):
    """
    Create a subtle grid on the Galactic plane for orientation.
    """
    radius = np.log10(max_radius_pc) if log_scale else max_radius_pc

    n = 40
    t = np.linspace(0, 2 * np.pi, n)
    traces = []

    for frac in [0.3, 0.6, 0.9]:
        r = radius * frac
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros(n)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color="rgba(60, 60, 100, 0.15)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


# ============================================================================
# SECTION 6: 3D SCENE CONSTRUCTION
# ============================================================================

def build_star_field_traces(df, log_scale=True):
    """
    Create Scatter3d traces for the star field, grouped by spectral type.

    Each spectral type gets its own trace for legend toggling.
    """
    traces = []
    type_order = ["O/B", "A", "F", "G", "K", "M", "Unknown"]
    type_colors = {b[2]: b[3] for b in SPECTRAL_BOUNDARIES}
    type_colors["Unknown"] = "#888888"

    for spec_type in type_order:
        group = df[df["spectral_type"] == spec_type]
        if len(group) == 0:
            continue

        hover_texts = []
        for _, row in group.iterrows():
            name = row["notable_name"] if pd.notna(row.get("notable_name")) else row.get("designation", "")
            teff_str = f"{row['teff_gspphot']:.0f} K" if pd.notna(row.get("teff_gspphot")) else "N/A"
            bp_rp_str = f"{row['bp_rp']:.2f}" if pd.notna(row.get("bp_rp")) else "N/A"
            hover_texts.append(
                f"<b>{name}</b><br>"
                f"Distance: {_format_distance(row['distance_pc'])} "
                f"({row['distance_ly']:.1f} ly)<br>"
                f"G mag: {row['phot_g_mean_mag']:.2f} "
                f"(M_G: {row['abs_mag']:.2f})<br>"
                f"Spectral type: {spec_type} (BP-RP: {bp_rp_str})<br>"
                f"T_eff: {teff_str}"
            )

        traces.append(go.Scatter3d(
            x=group["x"].values,
            y=group["y"].values,
            z=group["z"].values,
            mode="markers",
            marker=dict(
                size=group["marker_size"].values,
                color=group["color_hex"].values,
                opacity=0.85,
                line=dict(width=0),
            ),
            name=f"{spec_type} ({len(group)})",
            legendgroup=f"spectral_{spec_type}",
            legendgrouptitle_text="Spectral Types",
            hoverinfo="text",
            hovertext=hover_texts,
        ))

    return traces


def build_notable_star_traces(df, log_scale=True):
    """
    Highlight notable stars with glowing ring halos and text labels.
    """
    notable = df[df["notable_name"].notna()].copy()
    if len(notable) == 0:
        return []

    traces = []
    n_ring = 30

    # Ring halos
    ring_xs, ring_ys, ring_zs = [], [], []
    ring_r = 0.06  # radius in log-distance units

    for _, row in notable.iterrows():
        ox, oy, oz = row["x"], row["y"], row["z"]
        dist = np.sqrt(ox**2 + oy**2 + oz**2)
        if dist < 1e-6:
            continue

        er = np.array([ox, oy, oz]) / dist
        up = np.array([0, 0, 1]) if abs(er[2]) < 0.9 else np.array([1, 0, 0])
        e1 = np.cross(er, up)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(er, e1)

        theta = np.linspace(0, 2 * np.pi, n_ring)
        for t in theta:
            pt = np.array([ox, oy, oz]) + ring_r * (np.cos(t) * e1 + np.sin(t) * e2)
            ring_xs.append(pt[0])
            ring_ys.append(pt[1])
            ring_zs.append(pt[2])
        ring_xs.append(None)
        ring_ys.append(None)
        ring_zs.append(None)

    # Ring trace
    traces.append(go.Scatter3d(
        x=ring_xs, y=ring_ys, z=ring_zs,
        mode="lines",
        line=dict(color="rgba(0, 255, 200, 0.5)", width=3),
        name="Notable Stars",
        legendgroup="notable",
        legendgrouptitle_text="Highlights",
        hoverinfo="skip",
    ))

    # Text labels
    traces.append(go.Scatter3d(
        x=notable["x"].values,
        y=notable["y"].values,
        z=notable["z"].values,
        mode="text",
        text=["  " + n for n in notable["notable_name"].values],
        textposition="top center",
        textfont=dict(size=9, color="rgba(0, 255, 200, 0.8)"),
        name="Star Labels",
        legendgroup="notable",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Hover points
    hover_texts = []
    for _, row in notable.iterrows():
        teff_str = f"{row['teff_gspphot']:.0f} K" if pd.notna(row.get("teff_gspphot")) else "N/A"
        hover_texts.append(
            f"<b>{row['notable_name']}</b><br>"
            f"Distance: {_format_distance(row['distance_pc'])} "
            f"({row['distance_ly']:.1f} ly)<br>"
            f"G mag: {row['phot_g_mean_mag']:.2f}<br>"
            f"Spectral type: {row['spectral_type']}<br>"
            f"T_eff: {teff_str}"
        )

    traces.append(go.Scatter3d(
        x=notable["x"].values,
        y=notable["y"].values,
        z=notable["z"].values,
        mode="markers",
        marker=dict(size=10, color="rgba(0, 255, 200, 0.15)",
                    symbol="circle", line=dict(width=0)),
        name="Notable Info",
        legendgroup="notable",
        showlegend=False,
        hoverinfo="text",
        hovertext=hover_texts,
    ))

    return traces


def build_proper_motion_traces(df, log_scale=True, time_scale_yr=50000):
    """
    Show proper motion as arrows from each star's current position.

    Only rendered for stars with proper motion > 100 mas/yr.
    Off by default — enable with show_proper_motion=True.

    Parameters
    ----------
    df : DataFrame
        Must include pmra, pmdec, ra, dec, distance_pc, x, y, z.
    log_scale : bool
        Use log-scaled coordinates.
    time_scale_yr : float
        Time projection in years for arrow length.
    """
    # Filter to stars with significant proper motion
    has_pm = df["pmra"].notna() & df["pmdec"].notna()
    pm_total = np.sqrt(df["pmra"]**2 + df["pmdec"]**2)
    mask = has_pm & (pm_total > 100)
    stars = df[mask].copy()

    if len(stars) == 0:
        return []

    line_xs, line_ys, line_zs = [], [], []

    for _, row in stars.iterrows():
        # Current position
        x0, y0, z0 = row["x"], row["y"], row["z"]

        # Proper motion: convert mas/yr to degrees over time_scale
        dra_deg = (row["pmra"] / 3600000.0) * time_scale_yr / np.cos(np.deg2rad(row["dec"]))
        ddec_deg = (row["pmdec"] / 3600000.0) * time_scale_yr

        future_ra = row["ra"] + dra_deg
        future_dec = row["dec"] + ddec_deg
        dist = row["distance_pc"]

        r = np.log10(max(dist, 0.1)) if log_scale else dist
        ra_rad = np.deg2rad(future_ra)
        dec_rad = np.deg2rad(future_dec)

        x1 = r * np.cos(dec_rad) * np.cos(ra_rad)
        y1 = r * np.cos(dec_rad) * np.sin(ra_rad)
        z1 = r * np.sin(dec_rad)

        line_xs.extend([x0, x1, None])
        line_ys.extend([y0, y1, None])
        line_zs.extend([z0, z1, None])

    return [go.Scatter3d(
        x=line_xs, y=line_ys, z=line_zs,
        mode="lines",
        line=dict(color="rgba(255, 200, 50, 0.3)", width=1.5),
        name=f"Proper Motion ({len(stars)} stars)",
        legendgroup="motion",
        hoverinfo="skip",
    )]


def build_3d_scene(df, log_scale=True, show_mw_arms=True,
                   show_distance_shells=True, show_galactic_plane=True,
                   show_proper_motion=False, show_notable_labels=True):
    """
    Construct the full interactive 3D Plotly figure.

    Parameters
    ----------
    df : DataFrame
        Processed star data with x, y, z, color_hex, etc.
    log_scale : bool
        Log-scaled radial coordinates.
    show_mw_arms : bool
        Show Milky Way spiral arm schematic.
    show_distance_shells : bool
        Show wireframe distance reference spheres.
    show_galactic_plane : bool
        Show galactic plane grid.
    show_proper_motion : bool
        Show proper motion vectors (off by default).
    show_notable_labels : bool
        Show labels and highlights for notable stars.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Run: pip install plotly")

    fig = go.Figure()

    # Earth / Sun at origin
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(
            size=7,
            color="#FFE44D",
            symbol="diamond",
            line=dict(width=2, color="white"),
        ),
        text=["  Sun"],
        textposition="top center",
        textfont=dict(size=11, color="#FFE44D"),
        name="Sun",
        hoverinfo="text",
        hovertext="<b>The Sun</b><br>You are here<br>Origin of all measurements",
    ))

    # Star field
    for trace in build_star_field_traces(df, log_scale):
        fig.add_trace(trace)

    # Notable star highlights
    if show_notable_labels:
        for trace in build_notable_star_traces(df, log_scale):
            fig.add_trace(trace)

    # Proper motion vectors
    if show_proper_motion:
        for trace in build_proper_motion_traces(df, log_scale):
            fig.add_trace(trace)

    # Structural overlays
    if show_distance_shells:
        for trace in _create_distance_shells(log_scale):
            fig.add_trace(trace)

    if show_mw_arms:
        for trace in _create_milky_way_arms(log_scale):
            fig.add_trace(trace)

    if show_galactic_plane:
        for trace in _create_galactic_plane(log_scale):
            fig.add_trace(trace)

    # Layout
    max_dist = df["distance_pc"].max() if len(df) > 0 else 50
    axis_range = np.log10(max_dist) + 0.3 if log_scale else max_dist * 1.1

    axis_template = dict(
        backgroundcolor="rgb(5, 5, 20)",
        gridcolor="rgba(50, 50, 80, 0.3)",
        showticklabels=False,
        title="",
        range=[-axis_range, axis_range],
    )

    n_stars = len(df)
    max_d = df["distance_pc"].max() if len(df) > 0 else 50

    fig.update_layout(
        scene=dict(
            xaxis=axis_template,
            yaxis=axis_template,
            zaxis=axis_template,
            bgcolor="rgb(5, 5, 20)",
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="rgb(5, 5, 20)",
        plot_bgcolor="rgb(5, 5, 20)",
        font=dict(color="white", family="Arial"),
        title=dict(
            text=(
                "The Solar Neighborhood: Gaia DR3 Star Map<br>"
                f"<sup style='color:gray'>{n_stars:,} stars within "
                f"{max_d:.0f} pc | Colored by spectral type</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="Legend", font=dict(size=12)),
            bgcolor="rgba(10, 10, 30, 0.8)",
            bordercolor="rgba(100, 100, 150, 0.3)",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    # Camera preset buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(30, 30, 60, 0.8)",
                font=dict(color="white", size=10),
                buttons=[
                    dict(
                        label="Oblique View",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 0.8}}],
                    ),
                    dict(
                        label="Top Down (N Galactic Pole)",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 3.0}}],
                    ),
                    dict(
                        label="Edge On (Galactic Plane)",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 3.0, "y": 0, "z": 0.01}}],
                    ),
                    dict(
                        label="From Galactic Center",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": -2.5, "y": 0, "z": 0.3}}],
                    ),
                ],
            ),
        ],
    )

    return fig


# ============================================================================
# SECTION 7: HERTZSPRUNG-RUSSELL DIAGRAM
# ============================================================================

def render_hr_diagram(df, output_html="gaia_hr_diagram.html", auto_open=False):
    """
    Create an interactive Hertzsprung-Russell diagram.

    X-axis: BP-RP color index (blue-left to red-right)
    Y-axis: Absolute G magnitude (bright at top, inverted)
    Color: spectral type colors

    Parameters
    ----------
    df : DataFrame
        Processed star data with bp_rp, abs_mag, color_hex columns.
    output_html : str
        Output HTML file path.
    auto_open : bool
        Open in browser.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required.")

    # Filter to stars with bp_rp
    hr = df[df["bp_rp"].notna()].copy()

    hover_texts = []
    for _, row in hr.iterrows():
        name = row["notable_name"] if pd.notna(row.get("notable_name")) else row.get("designation", "")
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"BP-RP: {row['bp_rp']:.2f}<br>"
            f"M_G: {row['abs_mag']:.2f}<br>"
            f"Distance: {_format_distance(row['distance_pc'])}<br>"
            f"Type: {row['spectral_type']}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=hr["bp_rp"].values,
        y=hr["abs_mag"].values,
        mode="markers",
        marker=dict(
            size=4,
            color=hr["color_hex"].values,
            opacity=0.7,
            line=dict(width=0),
        ),
        hoverinfo="text",
        hovertext=hover_texts,
        name="Stars",
    ))

    # Highlight notable stars
    notable_hr = hr[hr["notable_name"].notna()]
    if len(notable_hr) > 0:
        fig.add_trace(go.Scattergl(
            x=notable_hr["bp_rp"].values,
            y=notable_hr["abs_mag"].values,
            mode="markers+text",
            marker=dict(
                size=10,
                color="rgba(0, 255, 200, 0.8)",
                symbol="star",
                line=dict(width=1, color="white"),
            ),
            text=notable_hr["notable_name"].values,
            textposition="top right",
            textfont=dict(size=8, color="rgba(0, 255, 200, 0.7)"),
            name="Notable Stars",
            hoverinfo="text",
            hovertext=[
                f"<b>{row['notable_name']}</b><br>"
                f"BP-RP: {row['bp_rp']:.2f}, M_G: {row['abs_mag']:.2f}"
                for _, row in notable_hr.iterrows()
            ],
        ))

    n_stars = len(hr)
    fig.update_layout(
        paper_bgcolor="rgb(5, 5, 20)",
        plot_bgcolor="rgb(10, 10, 30)",
        font=dict(color="white", family="Arial"),
        title=dict(
            text=(
                "Hertzsprung-Russell Diagram: Gaia DR3<br>"
                f"<sup style='color:gray'>{n_stars:,} stars | "
                "Color = spectral type</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(
            title="BP - RP Color Index",
            gridcolor="rgba(50, 50, 80, 0.3)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Absolute G Magnitude (M_G)",
            autorange="reversed",  # bright at top
            gridcolor="rgba(50, 50, 80, 0.3)",
            zeroline=False,
        ),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(10, 10, 30, 0.8)",
            bordercolor="rgba(100, 100, 150, 0.3)",
            font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=60, b=50),
    )

    fig.write_html(output_html, include_plotlyjs=True)
    print(f"HR diagram saved to {output_html}")

    if auto_open:
        fig.show()

    return fig


# ============================================================================
# SECTION 8: MAIN ENTRY POINT
# ============================================================================

def render_gaia_3d(output_html="gaia_3d_starmap.html",
                   max_distance_pc=50, mag_limit=15.0,
                   log_scale=True, show_mw_arms=True,
                   show_proper_motion=False,
                   show_hr_diagram=True,
                   use_cache=True, auto_open=True):
    """
    Build and save the interactive Gaia DR3 3D star map.

    Parameters
    ----------
    output_html : str
        Path for the self-contained HTML file.
    max_distance_pc : float
        Maximum distance in parsecs. Default 50.
    mag_limit : float
        Faintest G magnitude to include.
    log_scale : bool
        Use logarithmic distance scaling.
    show_mw_arms : bool
        Show schematic Milky Way spiral arms.
    show_proper_motion : bool
        Show proper motion vectors (off by default).
    show_hr_diagram : bool
        Also generate a companion HR diagram HTML.
    use_cache : bool
        Use cached query results.
    auto_open : bool
        Open the HTML in the default browser.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Run: pip install plotly")

    # 1. Query Gaia DR3
    print("=" * 50)
    print("Gaia DR3 3D Star Map Visualizer")
    print("=" * 50)
    print()

    df = query_gaia_nearby(max_distance_pc, mag_limit, use_cache)

    # 2. Process data
    print("\nProcessing star data...")
    df = compute_distances(df)
    df = compute_cartesian(df, log_scale)
    df = assign_spectral_colors(df)
    df = assign_marker_sizes(df)
    df = identify_notable_stars(df)

    # Print summary
    print(f"\n  Distance range: {df['distance_pc'].min():.1f} - "
          f"{df['distance_pc'].max():.1f} pc")
    print(f"  Magnitude range: {df['phot_g_mean_mag'].min():.1f} - "
          f"{df['phot_g_mean_mag'].max():.1f}")
    print(f"  Spectral types:")
    for st in ["O/B", "A", "F", "G", "K", "M", "Unknown"]:
        count = (df["spectral_type"] == st).sum()
        if count > 0:
            print(f"    {st:>6s}: {count:>5,}")

    # 3. Build 3D scene
    print("\nBuilding 3D scene...")
    fig = build_3d_scene(
        df,
        log_scale=log_scale,
        show_mw_arms=show_mw_arms,
        show_proper_motion=show_proper_motion,
    )

    # 4. Save
    print(f"Saving to {output_html}...")
    fig.write_html(output_html, include_plotlyjs=True)
    print(f"3D star map saved to {output_html}")

    # 5. HR diagram
    if show_hr_diagram:
        hr_path = output_html.replace(".html", "_hr.html")
        if hr_path == output_html:
            hr_path = "gaia_hr_diagram.html"
        print(f"\nGenerating HR diagram...")
        render_hr_diagram(df, hr_path, auto_open=False)

    # 6. Open
    if auto_open:
        fig.show()

    return fig


# ============================================================================
# SECTION 9: CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gaia DR3 3D Star Map Visualizer"
    )
    parser.add_argument(
        "--distance", type=float, default=DEFAULT_MAX_DISTANCE_PC,
        help=f"Max distance in pc (default: {DEFAULT_MAX_DISTANCE_PC})"
    )
    parser.add_argument(
        "--mag-limit", type=float, default=15.0,
        help="Faintest G magnitude (default: 15.0)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force fresh Gaia query (ignore cache)"
    )
    parser.add_argument(
        "--proper-motion", action="store_true",
        help="Show proper motion vectors"
    )
    parser.add_argument(
        "--no-hr", action="store_true",
        help="Skip HR diagram generation"
    )
    parser.add_argument(
        "--output", default="gaia_3d_starmap.html",
        help="Output HTML filename (default: gaia_3d_starmap.html)"
    )
    args = parser.parse_args()

    render_gaia_3d(
        output_html=args.output,
        max_distance_pc=args.distance,
        mag_limit=args.mag_limit,
        use_cache=not args.no_cache,
        show_proper_motion=args.proper_motion,
        show_hr_diagram=not args.no_hr,
    )
