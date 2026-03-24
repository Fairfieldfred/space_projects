"""
3D Milky Way Model — 4-Layer Visualization
============================================

Interactive 3D model of the Milky Way galaxy built from 4 layers of real
astronomical data:

  Layer 1: Gaia DR3 stars with parallax distances (< 5 kpc)
  Layer 2: Spiral arm tracers — masers, HII regions, open clusters
  Layer 3: Distant Gaia giants with photometric distances (5-15 kpc)
  Layer 4: Analytical 3D dust model showing interstellar dust lanes

Centered on the Galactic Center in Galactocentric coordinates (x, y, z kpc).
The Sun is marked at (-8.1, 0, 0) kpc.

Uses Plotly for browser-based interactivity (orbit, zoom, pan, hover tooltips).

Usage:
    python milkyway_3d_visualization.py
    python milkyway_3d_visualization.py --no-dust --no-cache

    Or from Python:
        from milkyway_3d_visualization import render_milkyway_3d
        render_milkyway_3d()

Requirements:
    pip install numpy astropy plotly pyvo pandas

Author: Generated for Fred
Date: 2026-03-23
"""

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u
import time
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
CACHE_DIR = Path("./milkyway_cache")
CACHE_EXPIRY_DAYS = 7

# Sun's position in Galactocentric frame (kpc)
SUN_X_KPC = -8.122
SUN_Y_KPC = 0.0
SUN_Z_KPC = 0.021

# Point budgets per layer (total must stay under ~10,000 for Plotly)
LAYER1_BUDGET = 3000  # Gaia parallax stars
LAYER3_BUDGET = 2000  # Gaia distant giants

# Spiral arm colors
ARM_COLORS = {
    "Perseus":           "#4488FF",
    "Sagittarius-Carina": "#44DD66",
    "Scutum-Centaurus":  "#FF8844",
    "Norma-Outer":       "#FF4444",
    "Local (Orion)":     "#00DDDD",
}

# Spiral arm model parameters (logarithmic spirals)
# Reference: Reid et al. (2019), Vallée (2017)
# (arm_name, R_ref_kpc, theta_ref_deg, pitch_deg)
ARM_PARAMS = [
    ("Perseus",            9.9,  14.0, 9.4),
    ("Sagittarius-Carina", 6.0, -10.0, 12.0),
    ("Scutum-Centaurus",   5.0, -55.0, 13.0),
    ("Norma-Outer",        4.5, -95.0, 11.0),
    ("Local (Orion)",      8.0,   8.0, 12.0),
]

# Spectral type colors (reused from gaia_3d_visualization.py)
SPECTRAL_BOUNDARIES = [
    (-99,  -0.1, "O/B", "#6699FF"),
    (-0.1,  0.3, "A",   "#AABBFF"),
    (0.3,   0.6, "F",   "#F0F0FF"),
    (0.6,   0.9, "G",   "#FFFF88"),
    (0.9,   1.5, "K",   "#FFAA44"),
    (1.5,   99,  "M",   "#FF4422"),
]

COLOR_ANCHORS = [
    (-0.5,  51, 102, 255),
    (-0.1, 102, 153, 255),
    ( 0.0, 170, 187, 255),
    ( 0.3, 240, 240, 255),
    ( 0.6, 255, 255, 136),
    ( 1.0, 255, 215,   0),
    ( 1.5, 255, 140,   0),
    ( 2.0, 255,  69,   0),
    ( 3.0, 204,  34,   0),
]


# ============================================================================
# SECTION 2: CURATED SPIRAL ARM TRACER CATALOGS
# ============================================================================

# Key maser parallaxes from Reid et al. (2019) BeSSeL survey.
# Gold-standard VLBI astrometry tracing star-forming regions in spiral arms.
# Format: (name, l_deg, b_deg, distance_kpc, arm)
MASER_TRACERS = [
    # Perseus Arm
    ("W3(OH)",          133.95,  1.06,  2.0, "Perseus"),
    ("NGC 7538",        111.54,  0.78,  2.65, "Perseus"),
    ("W51 Main",         49.49, -0.37,  5.41, "Perseus"),
    ("G108.20+0.59",    108.20,  0.59,  4.4, "Perseus"),
    ("G133.95+1.07",    133.95,  1.07,  2.0, "Perseus"),
    ("G111.54+0.78",    111.54,  0.78,  2.65, "Perseus"),
    ("G98.04+1.45",      98.04,  1.45,  4.4, "Perseus"),
    ("G183.72-3.66",    183.72, -3.66,  1.6, "Perseus"),
    ("G188.95+0.89",    188.95,  0.89,  2.1, "Perseus"),
    ("G192.60-0.05",    192.60, -0.05,  1.5, "Perseus"),
    ("G196.45-1.68",    196.45, -1.68,  5.3, "Perseus"),
    ("G209.01-19.4",    209.01,-19.38,  0.41, "Perseus"),
    # Sagittarius-Carina Arm
    ("W33",              12.81, -0.19,  2.4, "Sagittarius-Carina"),
    ("G35.20-1.74",      35.20, -1.74,  2.2, "Sagittarius-Carina"),
    ("G23.01-0.41",      23.01, -0.41,  4.6, "Sagittarius-Carina"),
    ("G29.96-0.02",      29.96, -0.02,  5.3, "Sagittarius-Carina"),
    ("G10.62-0.38",      10.62, -0.38,  4.95, "Sagittarius-Carina"),
    ("G14.33-0.64",      14.33, -0.64,  1.1, "Sagittarius-Carina"),
    ("G16.59-0.05",      16.59, -0.05,  3.6, "Sagittarius-Carina"),
    ("G351.78-0.54",    351.78, -0.54,  1.0, "Sagittarius-Carina"),
    ("G345.01+1.79",    345.01,  1.79,  2.4, "Sagittarius-Carina"),
    ("G339.88-1.26",    339.88, -1.26,  2.1, "Sagittarius-Carina"),
    ("G326.48+0.70",    326.48,  0.70,  1.8, "Sagittarius-Carina"),
    ("G305.20+0.21",    305.20,  0.21,  4.0, "Sagittarius-Carina"),
    ("G284.35-0.42",    284.35, -0.42,  5.9, "Sagittarius-Carina"),
    # Scutum-Centaurus Arm
    ("G5.89-0.39",        5.89, -0.39,  1.28, "Scutum-Centaurus"),
    ("G6.80-0.26",        6.80, -0.26,  2.2, "Scutum-Centaurus"),
    ("G9.62+0.19",        9.62,  0.19,  5.2, "Scutum-Centaurus"),
    ("G12.89+0.49",      12.89,  0.49,  2.5, "Scutum-Centaurus"),
    ("G15.03-0.68",      15.03, -0.68,  1.9, "Scutum-Centaurus"),
    ("G18.34+1.78",      18.34,  1.78,  2.0, "Scutum-Centaurus"),
    ("G23.44-0.18",      23.44, -0.18,  5.9, "Scutum-Centaurus"),
    ("G31.28+0.06",      31.28,  0.06,  7.0, "Scutum-Centaurus"),
    ("G35.03+0.35",      35.03,  0.35,  3.2, "Scutum-Centaurus"),
    ("G43.17+0.01",      43.17,  0.01, 11.1, "Scutum-Centaurus"),
    ("G318.05+0.09",    318.05,  0.09,  2.4, "Scutum-Centaurus"),
    ("G333.13-0.43",    333.13, -0.43,  3.6, "Scutum-Centaurus"),
    # Norma-Outer Arm
    ("G337.92-0.48",    337.92, -0.48,  3.5, "Norma-Outer"),
    ("G336.99-0.03",    336.99, -0.03, 10.3, "Norma-Outer"),
    ("G330.88-0.37",    330.88, -0.37,  5.3, "Norma-Outer"),
    # Local / Orion Spur
    ("Orion KL",        209.01, -19.38, 0.41, "Local (Orion)"),
    ("S252 (Gem OB1)",  188.95,  0.89,  2.1, "Local (Orion)"),
    ("MonR2",           213.71, -12.60, 0.83, "Local (Orion)"),
    ("NGC 281",         123.07, -6.31,  2.8, "Local (Orion)"),
    ("Cep A",           109.87,  2.11,  0.70, "Local (Orion)"),
    ("IRAS 20126",       78.12,  3.63,  1.64, "Local (Orion)"),
]

# Key HII regions from WISE catalog with measured distances
HII_REGIONS = [
    # (name, l_deg, b_deg, distance_kpc, arm)
    ("M42 (Orion)",       209.0, -19.4,  0.41, "Local (Orion)"),
    ("M8 (Lagoon)",         6.0,  -1.2,  1.25, "Sagittarius-Carina"),
    ("M16 (Eagle)",        17.0,   0.8,  1.74, "Sagittarius-Carina"),
    ("M17 (Omega)",        15.1,  -0.7,  1.98, "Sagittarius-Carina"),
    ("M20 (Trifid)",        7.0,  -0.3,  1.68, "Sagittarius-Carina"),
    ("NGC 3603",          291.6,  -0.5,  7.6, "Scutum-Centaurus"),
    ("Carina Nebula",     287.6,  -0.7,  2.3, "Sagittarius-Carina"),
    ("NGC 6334",          351.2,   0.7,  1.35, "Sagittarius-Carina"),
    ("NGC 6357",          353.2,   0.9,  1.7, "Sagittarius-Carina"),
    ("RCW 49",            284.3,  -0.3,  4.2, "Scutum-Centaurus"),
    ("W49A",               43.2,   0.0, 11.1, "Scutum-Centaurus"),
    ("W43",                30.8,  -0.0,  5.5, "Scutum-Centaurus"),
    ("NGC 7538",          111.5,   0.8,  2.65, "Perseus"),
    ("IC 1795/W3",        133.7,   1.2,  2.0, "Perseus"),
    ("Rosette Nebula",    206.3,  -2.1,  1.6, "Local (Orion)"),
    ("IC 1396",           099.3,   3.7,  0.87, "Local (Orion)"),
    ("Sh2-235",           173.5,   2.3,  1.8, "Perseus"),
    ("NGC 2024 (Flame)",  206.5, -16.4,  0.41, "Local (Orion)"),
    ("Sh2-140",           106.8,   5.3,  0.76, "Local (Orion)"),
    ("NGC 6611 (Eagle)",   17.0,   0.8,  1.74, "Sagittarius-Carina"),
]

# Well-known open clusters with distances
OPEN_CLUSTERS = [
    # (name, l_deg, b_deg, distance_kpc, arm)
    ("Pleiades (M45)",    166.6, -23.5,  0.136, "Local (Orion)"),
    ("Hyades",            180.1, -22.3,  0.047, "Local (Orion)"),
    ("Praesepe (M44)",    205.9,  32.5,  0.187, "Local (Orion)"),
    ("Alpha Persei",      147.6, -5.9,   0.172, "Local (Orion)"),
    ("NGC 2516",          273.8, -15.9,  0.35, "Local (Orion)"),
    ("NGC 6231",          343.5,  1.2,   1.6, "Sagittarius-Carina"),
    ("Trumpler 14",       287.4, -0.6,   2.5, "Sagittarius-Carina"),
    ("Trumpler 16",       287.6, -0.6,   2.3, "Sagittarius-Carina"),
    ("NGC 3293",          285.9,  0.1,   2.3, "Sagittarius-Carina"),
    ("NGC 6611",           17.0,  0.8,   1.74, "Sagittarius-Carina"),
    ("Berkeley 87",        75.8,  0.4,   0.95, "Local (Orion)"),
    ("NGC 6823",           59.4, -0.1,   2.0, "Local (Orion)"),
    ("NGC 869 (h Per)",   134.6, -3.7,   2.3, "Perseus"),
    ("NGC 884 (chi Per)", 135.0, -3.6,   2.3, "Perseus"),
    ("Stock 8",           173.4, -0.2,   2.0, "Perseus"),
    ("IC 1805",           134.7,  0.9,   2.0, "Perseus"),
    ("NGC 663",           129.5, -0.9,   2.1, "Perseus"),
    ("NGC 457",           126.6, -4.4,   2.5, "Perseus"),
    ("Westerlund 2",      284.3, -0.3,   4.2, "Scutum-Centaurus"),
    ("RSGC1",              25.3, -0.2,   6.6, "Scutum-Centaurus"),
]

# Milky Way globular clusters (Harris 1996 catalog, 2010 edition)
GLOBULAR_CLUSTERS = [
    # (name, l_deg, b_deg, distance_kpc)
    ("47 Tuc (NGC 104)",    305.9, -44.9,  4.5),
    ("Omega Cen (NGC 5139)", 309.1,  14.97, 5.2),
    ("M4 (NGC 6121)",       351.0,  16.0,  2.2),
    ("M13 (NGC 6205)",       59.0,  40.9,  7.1),
    ("M22 (NGC 6656)",        9.9,  -7.6,  3.2),
    ("M3 (NGC 5272)",       42.2,  78.7, 10.2),
    ("M5 (NGC 5904)",        3.9,  46.8,  7.5),
    ("M15 (NGC 7078)",       65.0, -27.3, 10.4),
    ("M92 (NGC 6341)",       68.3,  34.9,  8.3),
    ("M2 (NGC 7089)",        53.4, -35.8, 11.5),
    ("Palomar 5",             0.9,  45.9, 23.2),
    ("NGC 6397",            338.2, -12.0,  2.3),
    ("NGC 6752",            336.5, -25.6,  4.0),
    ("Terzan 5",              3.8,   1.7,  5.9),
    ("NGC 6388",            345.6,  -6.7,  9.9),
    ("NGC 362",             301.5, -46.2,  8.6),
    ("NGC 288",             152.3, -89.4,  8.9),
    ("NGC 1851",            244.5, -35.0, 12.1),
    ("NGC 2808",            282.2, -11.3,  9.6),
    ("NGC 6441",            353.5,  -5.0, 11.6),
    ("M80 (NGC 6093)",      352.7,  19.5, 10.0),
    ("M9 (NGC 6333)",        5.5,  10.7,  7.8),
    ("Terzan 7",              3.4, -20.1, 22.8),
    ("M71 (NGC 6838)",       56.7,  -4.6,  4.0),
    ("M107 (NGC 6171)",       3.4,  23.0,  6.4),
]

# Notable Milky Way landmarks
NOTABLE_LANDMARKS = [
    # (name, x_kpc, y_kpc, z_kpc, description)
    ("Galactic Center", 0.0, 0.0, 0.0, "Sagittarius A* — supermassive black hole"),
    ("Sun", SUN_X_KPC, SUN_Y_KPC, SUN_Z_KPC, "Our location in the Milky Way"),
]


# ============================================================================
# SECTION 3: DATA ACQUISITION (Gaia TAP queries)
# ============================================================================

def _load_cache(cache_path):
    """Load cached results if file exists and is fresh."""
    p = Path(cache_path)
    if not p.exists():
        return None
    age_days = (time.time() - p.stat().st_mtime) / 86400
    if age_days > CACHE_EXPIRY_DAYS:
        print(f"  Cache expired ({age_days:.1f} days old). Will re-query.")
        return None
    print(f"  Loading from cache: {p} ({age_days:.1f} days old)")
    return pd.read_csv(p)


def _save_cache(df, cache_path):
    """Save query results to CSV cache."""
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"  Cached {len(df)} rows to {p}")


def query_layer1_stars(use_cache=True):
    """
    Query Gaia DR3 for stars with reliable parallax distances out to 5 kpc.

    Uses stratified sampling across 4 distance bins for even coverage.
    Returns ~3,000 stars total.
    """
    cache_path = CACHE_DIR / "layer1_gaia_stars.csv"

    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"  {len(cached)} Layer 1 stars loaded.")
            return cached

    if not HAS_PYVO:
        raise ImportError("pyvo is required. Install: pip install pyvo")

    tap = pyvo.dal.TAPService(GAIA_TAP_URL)
    per_bin = LAYER1_BUDGET // 4

    # Distance bins in parallax (mas): 5kpc=0.2, 3kpc=0.33, 1.5kpc=0.67, 0.5kpc=2.0
    bins = [
        ("0-500 pc",    2.0, 999.0, per_bin),
        ("500-1500 pc", 0.667, 2.0, per_bin),
        ("1500-3000 pc", 0.333, 0.667, per_bin),
        ("3000-5000 pc", 0.2, 0.333, per_bin),
    ]

    all_dfs = []
    for label, plx_min, plx_max, n in bins:
        adql = f"""
        SELECT TOP {n * 3}
            source_id, designation, ra, dec, l, b,
            parallax, parallax_error, parallax_over_error,
            phot_g_mean_mag, bp_rp, teff_gspphot
        FROM gaiadr3.gaia_source
        WHERE parallax > {plx_min}
          AND parallax < {plx_max}
          AND parallax_over_error > 10
          AND phot_g_mean_mag < 16
          AND bp_rp IS NOT NULL
          AND random_index < 200000000
        ORDER BY random_index ASC
        """
        print(f"  Querying {label}...")
        result = tap.search(adql, maxrec=n * 3)
        df = result.to_table().to_pandas()
        df["distance_pc"] = 1000.0 / df["parallax"]
        # Take requested count
        if len(df) > n:
            df = df.sample(n=n, random_state=42)
        all_dfs.append(df)
        print(f"    Got {len(df)} stars")

    combined = pd.concat(all_dfs, ignore_index=True)
    _save_cache(combined, cache_path)
    return combined


def query_layer3_giants(use_cache=True):
    """
    Query Gaia DR3 for luminous giants at 5-15 kpc using photometric distances.

    Selects red giants (low logg) with GSP-Phot distances beyond the parallax limit.
    Returns ~2,000 stars.
    """
    cache_path = CACHE_DIR / "layer3_gaia_giants.csv"

    if use_cache:
        cached = _load_cache(cache_path)
        if cached is not None:
            print(f"  {len(cached)} Layer 3 giants loaded.")
            return cached

    if not HAS_PYVO:
        raise ImportError("pyvo is required. Install: pip install pyvo")

    tap = pyvo.dal.TAPService(GAIA_TAP_URL)

    adql = f"""
    SELECT TOP {LAYER3_BUDGET * 3}
        source_id, designation, ra, dec, l, b,
        parallax, phot_g_mean_mag, bp_rp,
        teff_gspphot, logg_gspphot, distance_gspphot
    FROM gaiadr3.gaia_source
    WHERE distance_gspphot > 5000
      AND distance_gspphot < 15000
      AND logg_gspphot < 2.5
      AND teff_gspphot IS NOT NULL
      AND bp_rp IS NOT NULL
      AND phot_g_mean_mag < 16
      AND random_index < 100000000
    ORDER BY random_index ASC
    """

    print(f"  Querying distant giants (5-15 kpc)...")
    result = tap.search(adql, maxrec=LAYER3_BUDGET * 3)
    df = result.to_table().to_pandas()
    df["distance_pc"] = df["distance_gspphot"]

    if len(df) > LAYER3_BUDGET:
        df = df.sample(n=LAYER3_BUDGET, random_state=42)

    print(f"    Got {len(df)} giants")
    _save_cache(df, cache_path)
    return df


# ============================================================================
# SECTION 4: COORDINATE TRANSFORMS
# ============================================================================

def icrs_to_galactocentric(ra, dec, distance_pc):
    """
    Convert ICRS RA/Dec/distance to Galactocentric (x, y, z) in kpc.

    Parameters
    ----------
    ra, dec : array-like
        Degrees (ICRS).
    distance_pc : array-like
        Distance in parsecs.

    Returns
    -------
    x, y, z : arrays
        Galactocentric coordinates in kpc.
    """
    coords = SkyCoord(
        ra=np.asarray(ra) * u.deg,
        dec=np.asarray(dec) * u.deg,
        distance=np.asarray(distance_pc) * u.pc,
    )
    gc = coords.transform_to(Galactocentric())
    return gc.x.to(u.kpc).value, gc.y.to(u.kpc).value, gc.z.to(u.kpc).value


def galactic_to_galactocentric(l_deg, b_deg, distance_kpc):
    """
    Convert Galactic (l, b, distance) to Galactocentric (x, y, z) in kpc.

    Parameters
    ----------
    l_deg, b_deg : float or array
        Galactic longitude and latitude in degrees.
    distance_kpc : float or array
        Distance in kpc.

    Returns
    -------
    x, y, z : float or arrays
        Galactocentric coordinates in kpc.
    """
    coords = SkyCoord(
        l=np.asarray(l_deg) * u.deg,
        b=np.asarray(b_deg) * u.deg,
        distance=np.asarray(distance_kpc) * u.kpc,
        frame='galactic',
    )
    gc = coords.transform_to(Galactocentric())
    return gc.x.to(u.kpc).value, gc.y.to(u.kpc).value, gc.z.to(u.kpc).value


def process_gaia_layer(df, distance_col="distance_pc"):
    """Add Galactocentric x, y, z columns to a Gaia DataFrame."""
    df = df.copy()
    dist_pc = df[distance_col].values
    x, y, z = icrs_to_galactocentric(df["ra"].values, df["dec"].values, dist_pc)
    df["x_kpc"] = x
    df["y_kpc"] = y
    df["z_kpc"] = z
    return df


# ============================================================================
# SECTION 5: ANALYTICAL DUST MODEL
# ============================================================================

def _spiral_arm_distance(x_kpc, y_kpc):
    """
    Compute minimum distance to the nearest spiral arm centerline.

    Returns distance in kpc.
    """
    min_dist = 999.0
    R_point = np.sqrt(x_kpc**2 + y_kpc**2)
    theta_point = np.arctan2(y_kpc, x_kpc)

    for _, R_ref, theta_ref_deg, pitch_deg in ARM_PARAMS:
        pitch_rad = np.deg2rad(pitch_deg)
        theta_ref = np.deg2rad(theta_ref_deg)
        tan_pitch = np.tan(pitch_rad)

        # Sample points along this arm
        thetas = np.linspace(-2 * np.pi, 4 * np.pi, 600)
        R_arm = R_ref * np.exp(tan_pitch * (thetas - theta_ref))
        mask = (R_arm > 1) & (R_arm < 18)
        R_arm = R_arm[mask]
        thetas = thetas[mask]

        ax = R_arm * np.cos(thetas)
        ay = R_arm * np.sin(thetas)

        dists = np.sqrt((ax - x_kpc)**2 + (ay - y_kpc)**2)
        arm_min = np.min(dists)
        if arm_min < min_dist:
            min_dist = arm_min

    return min_dist


def _compute_dust_density(x_kpc, y_kpc, z_kpc):
    """
    Compute analytical dust density at a Galactocentric position.

    Model:
    - Exponential disk: exp(-|z|/h_z) * exp(-R/h_R)
    - Enhanced near spiral arms (Gaussian with width 0.5 kpc)
    - Suppressed in the central bar region
    """
    R = np.sqrt(x_kpc**2 + y_kpc**2)
    h_z = 0.12  # scale height in kpc
    h_R = 3.5   # radial scale length in kpc

    # Base exponential disk
    disk = np.exp(-abs(z_kpc) / h_z) * np.exp(-R / h_R)

    # Spiral arm enhancement
    arm_dist = _spiral_arm_distance(x_kpc, y_kpc)
    arm_width = 0.5  # kpc
    arm_boost = 2.0 * np.exp(-arm_dist**2 / (2 * arm_width**2))

    # Suppress in the inner bulge (R < 1 kpc)
    if R < 1.0:
        disk *= R  # taper toward center

    # Suppress very far out
    if R > 14:
        disk *= max(0, (18 - R) / 4)

    return max(0, disk * (1.0 + arm_boost))


def build_dust_grid(n_radial=25, n_azimuth=40, n_z=3):
    """
    Generate a dust density grid in the Galactic plane.

    Returns DataFrame with x_kpc, y_kpc, z_kpc, dust_density columns.
    ~2,000 points total.
    """
    radii = np.linspace(1.5, 14, n_radial)
    azimuths = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    z_values = np.linspace(-0.15, 0.15, n_z)

    rows = []
    for r in radii:
        for az in azimuths:
            for z in z_values:
                x = r * np.cos(az)
                y = r * np.sin(az)
                density = _compute_dust_density(x, y, z)
                if density > 0.02:  # skip very low density regions
                    rows.append({"x_kpc": x, "y_kpc": y, "z_kpc": z,
                                 "dust_density": density})

    df = pd.DataFrame(rows)
    print(f"  Dust grid: {len(df)} points with density > 0.02")
    return df


# ============================================================================
# SECTION 6: SPIRAL ARM MODEL
# ============================================================================

def _spiral_arm_curve(arm_name, R_ref, theta_ref_deg, pitch_deg,
                      r_min=2.0, r_max=15.0, n_points=200):
    """
    Generate a logarithmic spiral arm curve.

    Returns x_kpc, y_kpc arrays in Galactocentric coordinates.
    """
    pitch_rad = np.deg2rad(pitch_deg)
    theta_ref = np.deg2rad(theta_ref_deg)
    tan_pitch = np.tan(pitch_rad)

    thetas = np.linspace(-2 * np.pi, 4 * np.pi, n_points)
    R = R_ref * np.exp(tan_pitch * (thetas - theta_ref))

    mask = (R >= r_min) & (R <= r_max)
    R = R[mask]
    thetas = thetas[mask]

    x = R * np.cos(thetas)
    y = R * np.sin(thetas)
    return x, y


def build_spiral_arm_traces():
    """
    Build Plotly traces for the 4 major spiral arms + Local/Orion spur.
    """
    traces = []
    for arm_name, R_ref, theta_ref, pitch in ARM_PARAMS:
        x, y = _spiral_arm_curve(arm_name, R_ref, theta_ref, pitch)
        z = np.zeros_like(x)
        color = ARM_COLORS.get(arm_name, "#FFFFFF")

        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=color, width=4),
            opacity=0.4,
            name=arm_name,
            legendgroup="arms",
            legendgrouptitle_text="Spiral Arms",
            showlegend=True,
            hoverinfo="text",
            hovertext=arm_name,
        ))

    return traces


# ============================================================================
# SECTION 7: GALAXY STRUCTURE OVERLAYS
# ============================================================================

def _format_distance_kpc(distance_kpc):
    """Format distance in kpc for display."""
    if distance_kpc < 1:
        return f"{distance_kpc * 1000:.0f} pc"
    return f"{distance_kpc:.1f} kpc"


def _create_disk_outline():
    """Create a subtle circular outline for the Milky Way disk."""
    traces = []
    theta = np.linspace(0, 2 * np.pi, 100)

    # Disk edge at ~15 kpc
    r = 15.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(100)
    traces.append(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color="rgba(100, 100, 150, 0.2)", width=2),
        name="Disk Edge (15 kpc)",
        legendgroup="structure",
        legendgrouptitle_text="Galaxy Structure",
        showlegend=True,
        hoverinfo="skip",
    ))

    return traces


def _create_galactic_bar():
    """
    Create the central Galactic bar.

    The bar is ~5 kpc half-length, tilted ~25° from the Sun-GC line.
    """
    bar_half_length = 4.5  # kpc
    bar_half_width = 0.8   # kpc
    bar_angle = np.deg2rad(25)  # tilt from x-axis

    # Bar outline as a rotated rectangle
    corners_x = [-bar_half_length, bar_half_length,
                  bar_half_length, -bar_half_length, -bar_half_length]
    corners_y = [-bar_half_width, -bar_half_width,
                  bar_half_width, bar_half_width, -bar_half_width]

    # Rotate
    rx = [cx * np.cos(bar_angle) - cy * np.sin(bar_angle) for cx, cy in zip(corners_x, corners_y)]
    ry = [cx * np.sin(bar_angle) + cy * np.cos(bar_angle) for cx, cy in zip(corners_x, corners_y)]
    rz = [0.0] * 5

    traces = [go.Scatter3d(
        x=rx, y=ry, z=rz,
        mode="lines",
        line=dict(color="rgba(255, 200, 50, 0.3)", width=3),
        name="Galactic Bar",
        legendgroup="structure",
        showlegend=True,
        hoverinfo="text",
        hovertext="Galactic Bar (~9 kpc, tilted 25°)",
    )]

    # Fill with subtle scatter for visual weight
    n_fill = 60
    fill_x = np.random.RandomState(42).uniform(-bar_half_length * 0.9,
                                                 bar_half_length * 0.9, n_fill)
    fill_y = np.random.RandomState(43).uniform(-bar_half_width * 0.7,
                                                bar_half_width * 0.7, n_fill)
    # Rotate fill points
    fx = fill_x * np.cos(bar_angle) - fill_y * np.sin(bar_angle)
    fy = fill_x * np.sin(bar_angle) + fill_y * np.cos(bar_angle)
    fz = np.zeros(n_fill)

    traces.append(go.Scatter3d(
        x=fx, y=fy, z=fz,
        mode="markers",
        marker=dict(size=3, color="rgba(255, 200, 50, 0.15)"),
        name="Bar fill",
        legendgroup="structure",
        showlegend=False,
        hoverinfo="skip",
    ))

    return traces


def _create_distance_rings():
    """Concentric rings at 2, 5, 10, 15 kpc from the Galactic Center."""
    traces = []
    theta = np.linspace(0, 2 * np.pi, 80)

    for radius in [2, 5, 10, 15]:
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros(80)
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color="rgba(80, 80, 120, 0.15)", width=1),
            showlegend=False,
            hoverinfo="text",
            hovertext=f"{radius} kpc",
        ))
        # Label at top
        traces.append(go.Scatter3d(
            x=[0], y=[radius], z=[0],
            mode="text",
            text=[f"{radius} kpc"],
            textfont=dict(size=8, color="rgba(150, 150, 200, 0.5)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


def _create_landmark_labels():
    """Text labels for key landmarks."""
    traces = []

    # Galactic Center
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=8, color="#FFD700", symbol="diamond",
                    line=dict(width=2, color="white")),
        text=["  Galactic Center"],
        textposition="top center",
        textfont=dict(size=10, color="#FFD700"),
        name="Galactic Center",
        hoverinfo="text",
        hovertext="<b>Sagittarius A*</b><br>Supermassive black hole<br>~4 million solar masses",
    ))

    # Sun
    traces.append(go.Scatter3d(
        x=[SUN_X_KPC], y=[SUN_Y_KPC], z=[SUN_Z_KPC],
        mode="markers+text",
        marker=dict(size=8, color="#FFE44D", symbol="circle",
                    line=dict(width=2, color="white")),
        text=["  Sun"],
        textposition="top center",
        textfont=dict(size=10, color="#FFE44D"),
        name="Sun",
        hoverinfo="text",
        hovertext=(
            "<b>The Sun</b><br>"
            f"Galactocentric distance: {abs(SUN_X_KPC):.1f} kpc<br>"
            "Located in the Local (Orion) Spur"
        ),
    ))

    # Arm labels at midpoints
    for arm_name, R_ref, theta_ref_deg, pitch_deg in ARM_PARAMS:
        x, y = _spiral_arm_curve(arm_name, R_ref, theta_ref_deg, pitch_deg,
                                  n_points=50)
        if len(x) > 0:
            mid = len(x) // 2
            color = ARM_COLORS.get(arm_name, "#FFFFFF")
            traces.append(go.Scatter3d(
                x=[x[mid]], y=[y[mid]], z=[0.3],
                mode="text",
                text=[arm_name],
                textfont=dict(size=8, color=color),
                showlegend=False,
                hoverinfo="skip",
            ))

    return traces


# ============================================================================
# SECTION 8: 3D SCENE CONSTRUCTION
# ============================================================================

def bp_rp_to_hex(bp_rp):
    """Convert bp_rp color index to hex color."""
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
    """Map bp_rp to spectral type."""
    if pd.isna(bp_rp):
        return "Unknown"
    for low, high, label, _ in SPECTRAL_BOUNDARIES:
        if low <= bp_rp < high:
            return label
    return "M"


def _assign_colors(df):
    """Assign spectral colors and types to a Gaia DataFrame."""
    df = df.copy()
    df["color_hex"] = df["bp_rp"].apply(bp_rp_to_hex)
    df["spectral_type"] = df["bp_rp"].apply(bp_rp_to_spectral_type)
    return df


def build_layer1_traces(df):
    """
    Build traces for Layer 1: Gaia parallax stars colored by spectral type.
    """
    df = _assign_colors(df)
    traces = []
    type_order = ["O/B", "A", "F", "G", "K", "M", "Unknown"]

    for spec_type in type_order:
        group = df[df["spectral_type"] == spec_type]
        if len(group) == 0:
            continue

        hover = []
        for _, row in group.iterrows():
            d_kpc = row["distance_pc"] / 1000
            hover.append(
                f"<b>{row.get('designation', '')}</b><br>"
                f"Distance: {_format_distance_kpc(d_kpc)}<br>"
                f"G mag: {row['phot_g_mean_mag']:.1f}<br>"
                f"Type: {spec_type}"
            )

        traces.append(go.Scatter3d(
            x=group["x_kpc"].values,
            y=group["y_kpc"].values,
            z=group["z_kpc"].values,
            mode="markers",
            marker=dict(
                size=2.5,
                color=group["color_hex"].values,
                opacity=0.6,
            ),
            name=f"{spec_type} ({len(group)})",
            legendgroup="layer1",
            legendgrouptitle_text="Layer 1: Gaia Stars (< 5 kpc)",
            hoverinfo="text",
            hovertext=hover,
        ))

    return traces


def build_layer2_traces():
    """
    Build traces for Layer 2: Spiral arm tracers.

    Masers, HII regions, open clusters, and globular clusters.
    """
    traces = []

    # -- Masers --
    for arm_name in ARM_COLORS:
        arm_masers = [(n, l, b, d, a) for n, l, b, d, a in MASER_TRACERS if a == arm_name]
        if not arm_masers:
            continue
        names, ls, bs, dists, _ = zip(*arm_masers)
        x, y, z = galactic_to_galactocentric(
            np.array(ls), np.array(bs), np.array(dists))
        color = ARM_COLORS[arm_name]
        hover = [f"<b>{n}</b> (maser)<br>{arm_name}<br>"
                 f"Distance: {_format_distance_kpc(d)}"
                 for n, d in zip(names, dists)]
        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=5, color=color, symbol="diamond",
                        line=dict(width=1, color="white"), opacity=0.9),
            name=f"Masers — {arm_name}",
            legendgroup="layer2_masers",
            legendgrouptitle_text="Layer 2: Arm Tracers",
            showlegend=(arm_name == list(ARM_COLORS.keys())[0]),
            hoverinfo="text",
            hovertext=hover,
        ))

    # -- HII Regions --
    names, ls, bs, dists, arms = zip(*HII_REGIONS)
    x, y, z = galactic_to_galactocentric(
        np.array(ls), np.array(bs), np.array(dists))
    colors = [ARM_COLORS.get(a, "#FFFFFF") for a in arms]
    hover = [f"<b>{n}</b> (HII region)<br>{a}<br>"
             f"Distance: {_format_distance_kpc(d)}"
             for n, d, a in zip(names, dists, arms)]
    traces.append(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=6, color=colors, symbol="circle",
                    line=dict(width=1, color="white"), opacity=0.9),
        name=f"HII Regions ({len(names)})",
        legendgroup="layer2_hii",
        hoverinfo="text",
        hovertext=hover,
    ))

    # -- Open Clusters --
    names, ls, bs, dists, arms = zip(*OPEN_CLUSTERS)
    x, y, z = galactic_to_galactocentric(
        np.array(ls), np.array(bs), np.array(dists))
    colors = [ARM_COLORS.get(a, "#FFFFFF") for a in arms]
    hover = [f"<b>{n}</b> (open cluster)<br>{a}<br>"
             f"Distance: {_format_distance_kpc(d)}"
             for n, d, a in zip(names, dists, arms)]
    traces.append(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=5, color=colors, symbol="square",
                    line=dict(width=1, color="white"), opacity=0.85),
        name=f"Open Clusters ({len(names)})",
        legendgroup="layer2_oc",
        hoverinfo="text",
        hovertext=hover,
    ))

    # -- Globular Clusters --
    names, ls, bs, dists = zip(*GLOBULAR_CLUSTERS)
    x, y, z = galactic_to_galactocentric(
        np.array(ls), np.array(bs), np.array(dists))
    hover = [f"<b>{n}</b> (globular cluster)<br>"
             f"Distance: {_format_distance_kpc(d)}"
             for n, d in zip(names, dists)]
    traces.append(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=6, color="#CC88FF", symbol="cross",
                    line=dict(width=1, color="white"), opacity=0.85),
        name=f"Globular Clusters ({len(names)})",
        legendgroup="layer2_gc",
        hoverinfo="text",
        hovertext=hover,
    ))

    return traces


def build_layer3_traces(df):
    """
    Build traces for Layer 3: Distant Gaia giants (5-15 kpc).
    """
    df = _assign_colors(df)
    traces = []
    type_order = ["O/B", "A", "F", "G", "K", "M", "Unknown"]

    for spec_type in type_order:
        group = df[df["spectral_type"] == spec_type]
        if len(group) == 0:
            continue

        hover = []
        for _, row in group.iterrows():
            d_kpc = row["distance_pc"] / 1000
            teff = row.get("teff_gspphot", float('nan'))
            teff_str = f"{teff:.0f} K" if pd.notna(teff) else "N/A"
            hover.append(
                f"<b>{row.get('designation', '')}</b> (giant)<br>"
                f"Distance: {_format_distance_kpc(d_kpc)}<br>"
                f"G mag: {row['phot_g_mean_mag']:.1f}<br>"
                f"Type: {spec_type}, T_eff: {teff_str}"
            )

        traces.append(go.Scatter3d(
            x=group["x_kpc"].values,
            y=group["y_kpc"].values,
            z=group["z_kpc"].values,
            mode="markers",
            marker=dict(
                size=2,
                color=group["color_hex"].values,
                opacity=0.4,
            ),
            name=f"{spec_type} giants ({len(group)})",
            legendgroup="layer3",
            legendgrouptitle_text="Layer 3: Distant Giants (5-15 kpc)",
            hoverinfo="text",
            hovertext=hover,
        ))

    return traces


def build_layer4_traces(dust_df):
    """
    Build traces for Layer 4: 3D dust map.

    Renders dust as semi-transparent amber markers sized by density.
    """
    if dust_df is None or len(dust_df) == 0:
        return []

    # Normalize density for sizing and opacity
    d = dust_df["dust_density"].values
    d_norm = d / d.max()

    sizes = 2 + 6 * d_norm
    opacities = 0.05 + 0.35 * d_norm

    # Amber-brown color gradient based on density
    colors = []
    for dn in d_norm:
        r = int(180 + 60 * dn)
        g = int(120 + 50 * dn)
        b = int(40 + 20 * dn)
        colors.append(f"rgba({r},{g},{b},{0.05 + 0.35 * dn:.2f})")

    return [go.Scatter3d(
        x=dust_df["x_kpc"].values,
        y=dust_df["y_kpc"].values,
        z=dust_df["z_kpc"].values,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
        ),
        name=f"Dust ({len(dust_df)} pts)",
        legendgroup="layer4",
        legendgrouptitle_text="Layer 4: Interstellar Dust",
        hoverinfo="text",
        hovertext=[
            f"Dust density: {d:.3f}<br>"
            f"Position: ({row['x_kpc']:.1f}, {row['y_kpc']:.1f}, {row['z_kpc']:.2f}) kpc"
            for d, (_, row) in zip(dust_df["dust_density"],
                                    dust_df.iterrows())
        ],
    )]


def build_milkyway_scene(layer1_df, layer3_df, dust_df,
                          show_dust=True, show_arms=True,
                          show_bar=True, show_structure=True):
    """
    Assemble the full 4-layer Milky Way 3D scene.

    Returns a Plotly Figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Run: pip install plotly")

    fig = go.Figure()

    # Landmark labels (GC, Sun)
    for trace in _create_landmark_labels():
        fig.add_trace(trace)

    # Layer 1: Gaia parallax stars
    for trace in build_layer1_traces(layer1_df):
        fig.add_trace(trace)

    # Layer 2: Arm tracers
    for trace in build_layer2_traces():
        fig.add_trace(trace)

    # Layer 3: Distant giants
    for trace in build_layer3_traces(layer3_df):
        fig.add_trace(trace)

    # Layer 4: Dust
    if show_dust and dust_df is not None:
        for trace in build_layer4_traces(dust_df):
            fig.add_trace(trace)

    # Spiral arm curves
    if show_arms:
        for trace in build_spiral_arm_traces():
            fig.add_trace(trace)

    # Galactic bar
    if show_bar:
        for trace in _create_galactic_bar():
            fig.add_trace(trace)

    # Structure overlays
    if show_structure:
        for trace in _create_disk_outline():
            fig.add_trace(trace)
        for trace in _create_distance_rings():
            fig.add_trace(trace)

    # Count total points
    n_l1 = len(layer1_df) if layer1_df is not None else 0
    n_l3 = len(layer3_df) if layer3_df is not None else 0
    n_dust = len(dust_df) if dust_df is not None and show_dust else 0

    # Layout
    axis_base = dict(
        backgroundcolor="rgb(5, 5, 20)",
        gridcolor="rgba(50, 50, 80, 0.2)",
        showticklabels=True,
        tickfont=dict(size=9, color="rgba(150, 150, 200, 0.6)"),
        range=[-18, 18],
        dtick=5,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(**axis_base, title="x (kpc)"),
            yaxis=dict(**axis_base, title="y (kpc)"),
            zaxis=dict(backgroundcolor="rgb(5, 5, 20)",
                       gridcolor="rgba(50, 50, 80, 0.2)",
                       showticklabels=True,
                       tickfont=dict(size=9, color="rgba(150, 150, 200, 0.6)"),
                       title="z (kpc)", range=[-8, 8], dtick=2),
            bgcolor="rgb(5, 5, 20)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.3),
            camera=dict(
                eye=dict(x=0, y=0, z=2.5),
                up=dict(x=0, y=1, z=0),
            ),
        ),
        paper_bgcolor="rgb(5, 5, 20)",
        plot_bgcolor="rgb(5, 5, 20)",
        font=dict(color="white", family="Arial"),
        title=dict(
            text=(
                "The Milky Way Galaxy — 3D Model<br>"
                f"<sup style='color:gray'>{n_l1:,} Gaia stars + "
                f"{n_l3:,} distant giants + "
                f"arm tracers + dust | Galactocentric coordinates (kpc)</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="Layers", font=dict(size=12)),
            bgcolor="rgba(10, 10, 30, 0.8)",
            bordercolor="rgba(100, 100, 150, 0.3)",
            borderwidth=1,
            font=dict(size=9),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    # Camera presets
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
                        label="Face-on (Top Down)",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5},
                               "scene.camera.up": {"x": 0, "y": 1, "z": 0}}],
                    ),
                    dict(
                        label="Edge-on",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0.05},
                               "scene.camera.up": {"x": 0, "y": 0, "z": 1}}],
                    ),
                    dict(
                        label="Sun's View",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": -1.8, "y": 0.3, "z": 0.8},
                               "scene.camera.up": {"x": 0, "y": 0, "z": 1}}],
                    ),
                    dict(
                        label="Wide View",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.2},
                               "scene.camera.up": {"x": 0, "y": 0, "z": 1}}],
                    ),
                ],
            ),
        ],
    )

    return fig


# ============================================================================
# SECTION 9: MAIN ENTRY POINT & CLI
# ============================================================================

def render_milkyway_3d(output_html="milkyway_3d.html",
                        use_cache=True, show_dust=True, auto_open=True):
    """
    Build and save the 4-layer 3D Milky Way visualization.

    Parameters
    ----------
    output_html : str
        Output HTML file path.
    use_cache : bool
        Use cached Gaia query results.
    show_dust : bool
        Include the dust layer (adds ~2,000 points).
    auto_open : bool
        Open in browser when complete.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Run: pip install plotly")

    print("=" * 55)
    print("  3D Milky Way Model — 4-Layer Visualization")
    print("=" * 55)
    print()

    # Layer 1: Gaia parallax stars
    print("Layer 1: Querying Gaia DR3 for nearby stars (< 5 kpc)...")
    layer1_df = query_layer1_stars(use_cache)
    layer1_df = process_gaia_layer(layer1_df, "distance_pc")
    print(f"  Layer 1: {len(layer1_df)} stars\n")

    # Layer 2: Curated catalogs (no query needed)
    n_tracers = len(MASER_TRACERS) + len(HII_REGIONS) + len(OPEN_CLUSTERS) + len(GLOBULAR_CLUSTERS)
    print(f"Layer 2: {n_tracers} curated arm tracers "
          f"({len(MASER_TRACERS)} masers, {len(HII_REGIONS)} HII, "
          f"{len(OPEN_CLUSTERS)} clusters, {len(GLOBULAR_CLUSTERS)} GCs)\n")

    # Layer 3: Distant giants
    print("Layer 3: Querying Gaia DR3 for distant giants (5-15 kpc)...")
    layer3_df = query_layer3_giants(use_cache)
    layer3_df = process_gaia_layer(layer3_df, "distance_pc")
    print(f"  Layer 3: {len(layer3_df)} giants\n")

    # Layer 4: Dust grid
    dust_df = None
    if show_dust:
        print("Layer 4: Building analytical dust model...")
        dust_df = build_dust_grid()
        print()

    # Build scene
    total = len(layer1_df) + n_tracers + len(layer3_df) + (len(dust_df) if dust_df is not None else 0)
    print(f"Building 3D scene ({total:,} total points)...")

    fig = build_milkyway_scene(layer1_df, layer3_df, dust_df,
                                show_dust=show_dust)

    # Save
    print(f"Saving to {output_html}...")
    fig.write_html(output_html, include_plotlyjs=True)
    print(f"Milky Way 3D model saved to {output_html}")

    if auto_open:
        fig.show()

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="3D Milky Way Model — 4-Layer Visualization"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force fresh Gaia queries (ignore cache)"
    )
    parser.add_argument(
        "--no-dust", action="store_true",
        help="Skip the dust layer"
    )
    parser.add_argument(
        "--output", default="milkyway_3d.html",
        help="Output HTML filename (default: milkyway_3d.html)"
    )
    args = parser.parse_args()

    render_milkyway_3d(
        output_html=args.output,
        use_cache=not args.no_cache,
        show_dust=not args.no_dust,
    )
