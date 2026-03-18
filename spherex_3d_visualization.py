"""
SPHEREx 3D Milky Way Visualization
====================================

Interactive 3D visualization placing Earth at the center of a 360-degree
sphere of astronomical objects, from nearby stars to distant galaxy clusters.

Uses Plotly for browser-based interactivity (orbit, zoom, pan, hover tooltips).
Objects are positioned using RA/Dec coordinates with literature-sourced distances.
Log-scaled radial distances handle the 7-order-of-magnitude range (1 pc to 16 Mpc).

Usage:
    python spherex_3d_visualization.py

    Or from Python:
        from spherex_3d_visualization import render_3d_milky_way
        render_3d_milky_way()

Requirements:
    pip install numpy astropy plotly

Author: Generated for Fred
Date: 2026-03-17
"""

import numpy as np
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("plotly not installed. Run: pip install plotly")


# ============================================================================
# SECTION 1: CURATED OBJECT CATALOG
# ============================================================================

# Type -> visual style mapping
TYPE_STYLES = {
    "star": {
        "symbol": "circle",
        "size": 5,
        "color": "#FFFFAA",
        "label": "Stars",
    },
    "nebula": {
        "symbol": "diamond",
        "size": 8,
        "color": "#FF69B4",
        "label": "Nebulae & Star-Forming Regions",
    },
    "snr": {
        "symbol": "cross",
        "size": 8,
        "color": "#FFA500",
        "label": "Supernova Remnants",
    },
    "pn": {
        "symbol": "circle",
        "size": 7,
        "color": "#00CED1",
        "label": "Planetary Nebulae",
    },
    "cluster": {
        "symbol": "circle",
        "size": 7,
        "color": "#00FFFF",
        "label": "Star Clusters",
    },
    "structure": {
        "symbol": "diamond",
        "size": 10,
        "color": "#FFD700",
        "label": "Milky Way Structure",
    },
    "xray": {
        "symbol": "x",
        "size": 8,
        "color": "#FF4444",
        "label": "X-ray / Black Hole Sources",
    },
    "galaxy": {
        "symbol": "square",
        "size": 9,
        "color": "#6699FF",
        "label": "Galaxies",
    },
    "galaxy_cluster": {
        "symbol": "square",
        "size": 11,
        "color": "#3344AA",
        "label": "Galaxy Clusters",
    },
}

# Which objects are key SPHEREx science targets (all-sky survey covers
# everything, but these are highlighted for their infrared science value).
SPHEREX_TARGETS = {
    "Orion Nebula (M42)",       # PAH + ice features
    "Rho Ophiuchi Cloud",       # Ice features at 3 um
    "Galactic Center (Sgr A*)", # Dust, CO/CO2 absorption
    "Crab Nebula (M1)",         # Synchrotron + line emission
    "Large Magellanic Cloud",   # Varied stellar populations
    "Eagle Nebula (M16)",       # Active star formation
    "Eta Carinae Nebula",       # Massive star-forming complex
    "Pleiades (M45)",           # Reflection nebula, IR excess
    "Helix Nebula (NGC 7293)",  # Planetary nebula IR emission
    "Betelgeuse",               # Cool supergiant IR-bright
    "Andromeda (M31)",          # Galaxy redshift survey target
    "Virgo Cluster (M87)",      # Galaxy cluster, redshift mapping
}

# Curated catalog: ~25 objects with RA, Dec, distance, type, and description.
# Distances sourced from Hipparcos, Gaia DR3, NED, and standard references.
CATALOG_3D = {
    # --- Nearby Stars (< 100 pc) ---
    "Proxima Centauri": {
        "ra": 217.429, "dec": -62.680, "distance_pc": 1.30,
        "type": "star",
        "description": "Nearest star to the Sun",
    },
    "Alpha Centauri A": {
        "ra": 219.902, "dec": -60.834, "distance_pc": 1.34,
        "type": "star",
        "description": "Sun-like star in nearest system",
    },
    "Barnard's Star": {
        "ra": 269.452, "dec": 4.694, "distance_pc": 1.83,
        "type": "star",
        "description": "Fastest proper-motion star",
    },
    "Sirius": {
        "ra": 101.287, "dec": -16.716, "distance_pc": 2.64,
        "type": "star",
        "description": "Brightest star in the night sky",
    },
    "Vega": {
        "ra": 279.235, "dec": 38.784, "distance_pc": 7.68,
        "type": "star",
        "description": "Bright standard star with debris disk",
    },
    "Polaris": {
        "ra": 37.954, "dec": 89.264, "distance_pc": 132,
        "type": "star",
        "description": "North Star, Cepheid variable",
    },
    "Betelgeuse": {
        "ra": 88.793, "dec": 7.407, "distance_pc": 200,
        "type": "star",
        "description": "Red supergiant in Orion",
    },

    # --- Nebulae, Clusters, and Star-Forming Regions (100-2000 pc) ---
    "Pleiades (M45)": {
        "ra": 56.871, "dec": 24.105, "distance_pc": 136,
        "type": "cluster",
        "description": "Nearby open cluster with reflection nebula",
    },
    "Rho Ophiuchi Cloud": {
        "ra": 246.79, "dec": -24.54, "distance_pc": 139,
        "type": "nebula",
        "description": "Nearby molecular cloud with ice features at 3 um",
    },
    "Helix Nebula (NGC 7293)": {
        "ra": 337.411, "dec": -20.837, "distance_pc": 213,
        "type": "pn",
        "description": "Nearest bright planetary nebula",
    },
    "Orion Nebula (M42)": {
        "ra": 83.822, "dec": -5.391, "distance_pc": 412,
        "type": "nebula",
        "description": "Rich star-forming region with PAH + ice features",
    },
    "Ring Nebula (M57)": {
        "ra": 283.396, "dec": 33.029, "distance_pc": 790,
        "type": "pn",
        "description": "Classic planetary nebula in Lyra",
    },
    "Eagle Nebula (M16)": {
        "ra": 274.700, "dec": -13.807, "distance_pc": 1740,
        "type": "nebula",
        "description": "Pillars of Creation -- active star formation",
    },
    "Crab Nebula (M1)": {
        "ra": 83.633, "dec": 22.015, "distance_pc": 2000,
        "type": "snr",
        "description": "Supernova remnant with synchrotron + line emission",
    },
    "Eta Carinae Nebula": {
        "ra": 161.265, "dec": -59.684, "distance_pc": 2300,
        "type": "nebula",
        "description": "Massive star-forming complex in the southern sky",
    },

    # --- Milky Way Structure (2-10 kpc) ---
    "Cygnus X-1": {
        "ra": 299.590, "dec": 35.202, "distance_pc": 1860,
        "type": "xray",
        "description": "First confirmed stellar black hole",
    },
    "Galactic Center (Sgr A*)": {
        "ra": 266.417, "dec": -29.008, "distance_pc": 8178,
        "type": "structure",
        "description": "Supermassive black hole at the center of the Milky Way",
    },

    # --- Nearby Galaxies (50-900 kpc) ---
    "Large Magellanic Cloud": {
        "ra": 80.894, "dec": -69.756, "distance_pc": 49970,
        "type": "galaxy",
        "description": "Nearest major satellite galaxy",
    },
    "Small Magellanic Cloud": {
        "ra": 13.187, "dec": -72.829, "distance_pc": 61000,
        "type": "galaxy",
        "description": "Irregular dwarf galaxy companion",
    },
    "Andromeda (M31)": {
        "ra": 10.685, "dec": 41.269, "distance_pc": 778000,
        "type": "galaxy",
        "description": "Nearest major spiral galaxy",
    },
    "Triangulum (M33)": {
        "ra": 23.462, "dec": 30.660, "distance_pc": 840000,
        "type": "galaxy",
        "description": "Third-largest Local Group galaxy",
    },

    # --- Deep Field (> 1 Mpc) ---
    "Centaurus A (NGC 5128)": {
        "ra": 201.365, "dec": -43.019, "distance_pc": 3800000,
        "type": "galaxy",
        "description": "Nearest radio galaxy with active nucleus",
    },
    "Virgo Cluster (M87)": {
        "ra": 187.706, "dec": 12.391, "distance_pc": 16400000,
        "type": "galaxy_cluster",
        "description": "Nearest large galaxy cluster, home of first imaged black hole",
    },
}


# ============================================================================
# SECTION 2: COORDINATE CONVERSION
# ============================================================================

def catalog_to_cartesian(catalog, log_scale=True):
    """
    Convert catalog RA/Dec/distance to 3D Cartesian coordinates.

    Parameters
    ----------
    catalog : dict
        Object catalog with 'ra', 'dec', 'distance_pc' per entry.
    log_scale : bool
        If True, use log10(distance_pc) as the radial coordinate.
        Essential for visualizing objects spanning 1 pc to 16 Mpc.

    Returns
    -------
    result : list of dict
        Each entry has 'name', 'x', 'y', 'z' plus original metadata.
    """
    result = []

    for name, info in catalog.items():
        ra_rad = np.deg2rad(info["ra"])
        dec_rad = np.deg2rad(info["dec"])
        dist = info["distance_pc"]

        if log_scale:
            r = np.log10(dist)
        else:
            r = dist

        # Spherical to Cartesian (RA=longitude, Dec=latitude)
        x = r * np.cos(dec_rad) * np.cos(ra_rad)
        y = r * np.cos(dec_rad) * np.sin(ra_rad)
        z = r * np.sin(dec_rad)

        result.append({
            "name": name,
            "x": x, "y": y, "z": z,
            "distance_pc": dist,
            "log_distance": np.log10(dist),
            **info,
        })

    return result


def _format_distance(distance_pc):
    """Format a distance in parsecs to a human-readable string."""
    if distance_pc < 1000:
        return f"{distance_pc:.1f} pc"
    elif distance_pc < 1e6:
        return f"{distance_pc / 1000:.1f} kpc"
    else:
        return f"{distance_pc / 1e6:.1f} Mpc"


# ============================================================================
# SECTION 3: MILKY WAY STRUCTURAL OVERLAYS
# ============================================================================

def _create_milky_way_arms(log_scale=True):
    """
    Generate Milky Way spiral arm traces in Galactocentric coordinates,
    then transform to heliocentric for plotting.

    Uses a simple logarithmic spiral model with 4 major arms.
    """
    # Sun's position in Galactocentric frame
    sun_x_kpc = -8.122  # kpc from Galactic center
    sun_y_kpc = 0.0
    sun_z_kpc = 0.0208

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
        mask = r_kpc <= 18  # limit to 18 kpc radius
        r_kpc = r_kpc[mask]
        t = theta[mask] + offset

        # Galactocentric X, Y
        gx = r_kpc * np.cos(t)
        gy = r_kpc * np.sin(t)

        # Shift to heliocentric (Sun at origin)
        hx = gx - sun_x_kpc  # in kpc
        hy = gy - sun_y_kpc

        # Convert kpc to pc for log scaling
        dist_pc = np.sqrt(hx**2 + hy**2) * 1000  # pc
        dist_pc = np.maximum(dist_pc, 1)  # avoid log(0)

        if log_scale:
            scale = np.log10(dist_pc)
        else:
            scale = dist_pc

        # Preserve direction, apply scale
        angle = np.arctan2(hy, hx)
        px = scale * np.cos(angle)
        py = scale * np.sin(angle)
        pz = np.zeros_like(px)  # arms lie in the Galactic plane

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


def _create_distance_shells(log_scale=True):
    """
    Create wireframe spheres at key distances as spatial references.
    """
    shells = [
        ("10 pc", 10),
        ("1 kpc", 1000),
        ("100 kpc", 100000),
        ("10 Mpc", 10000000),
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

        # Label at the top of each shell
        traces.append(go.Scatter3d(
            x=[0], y=[0], z=[radius],
            mode="text",
            text=[label],
            textfont=dict(size=9, color="rgba(180, 180, 220, 0.6)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


# ============================================================================
# SECTION 4: GALACTIC PLANE REFERENCE
# ============================================================================

def _create_galactic_plane(log_scale=True, max_radius_pc=20000000):
    """
    Create a subtle grid on the Galactic plane for orientation.
    """
    radius = np.log10(max_radius_pc) if log_scale else max_radius_pc

    # Transform Galactic plane to ICRS-based coordinates
    # The Galactic plane is tilted ~63 degrees from the celestial equator.
    # For simplicity, we show it as a flat disk at z=0 in Galactic coords,
    # transformed to our plotting frame.

    n = 40
    t = np.linspace(0, 2 * np.pi, n)
    traces = []

    # Concentric rings on the plane
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
# SECTION 5: 3D SCENE CONSTRUCTION
# ============================================================================

def build_3d_scene(catalog=None, log_scale=True, show_mw_arms=True,
                   show_distance_shells=True, show_galactic_plane=True):
    """
    Construct the full interactive 3D Plotly figure.

    Parameters
    ----------
    catalog : dict, optional
        Object catalog. Defaults to CATALOG_3D.
    log_scale : bool
        Use log10(distance) for radial coordinates.
    show_mw_arms : bool
        Overlay schematic Milky Way spiral arms.
    show_distance_shells : bool
        Show wireframe distance reference spheres.
    show_galactic_plane : bool
        Show a subtle grid on the Galactic plane.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Run: pip install plotly")

    if catalog is None:
        catalog = CATALOG_3D

    fig = go.Figure()

    # --- Earth / Sun at origin ---
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(
            size=7,
            color="#FFE44D",
            symbol="diamond",
            line=dict(width=2, color="white"),
        ),
        text=["  Earth / Sun"],
        textposition="top center",
        textfont=dict(size=11, color="#FFE44D"),
        name="Earth / Sun",
        hoverinfo="text",
        hovertext=(
            "<b>You Are Here</b><br>"
            "The Sun and Earth<br>"
            "Origin of all observations"
        ),
    ))

    # --- Catalog objects grouped by type ---
    objects_3d = catalog_to_cartesian(catalog, log_scale=log_scale)

    # Group by type
    by_type = {}
    for obj in objects_3d:
        t = obj["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(obj)

    for obj_type, objs in by_type.items():
        style = TYPE_STYLES.get(obj_type, TYPE_STYLES["star"])

        xs = [o["x"] for o in objs]
        ys = [o["y"] for o in objs]
        zs = [o["z"] for o in objs]
        names = [o["name"] for o in objs]

        hover_texts = []
        for o in objs:
            dist_str = _format_distance(o["distance_pc"])
            spherex_badge = (" [SPHEREx Target]"
                             if o["name"] in SPHEREX_TARGETS else "")
            hover_texts.append(
                f"<b>{o['name']}</b>{spherex_badge}<br>"
                f"Type: {style['label']}<br>"
                f"Distance: {dist_str}<br>"
                f"RA: {o['ra']:.3f}  Dec: {o['dec']:.3f}<br>"
                f"{o['description']}"
            )

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(
                size=style["size"],
                color=style["color"],
                symbol=style["symbol"],
                line=dict(width=1, color="white"),
                opacity=0.9,
            ),
            text=names,
            textposition="top center",
            textfont=dict(size=8, color="rgba(255,255,255,0.7)"),
            name=style["label"],
            legendgroup=obj_type,
            hoverinfo="text",
            hovertext=hover_texts,
        ))

    # --- Connection lines from Earth to each object ---
    for obj in objects_3d:
        fig.add_trace(go.Scatter3d(
            x=[0, obj["x"]], y=[0, obj["y"]], z=[0, obj["z"]],
            mode="lines",
            line=dict(
                color="rgba(255, 255, 255, 0.06)",
                width=1,
            ),
            showlegend=False,
            hoverinfo="skip",
        ))

    # --- SPHEREx Target highlights ---
    # Glowing rings around objects that are key SPHEREx infrared science targets.
    # Toggleable via the legend.
    spherex_objs = [o for o in objects_3d if o["name"] in SPHEREX_TARGETS]
    if spherex_objs:
        n_ring = 30  # points per ring
        ring_xs, ring_ys, ring_zs = [], [], []
        ring_hovers = []
        for o in spherex_objs:
            # Small circle around the object position
            ring_r = 0.15  # radius in log-distance units
            theta = np.linspace(0, 2 * np.pi, n_ring)
            # Ring in a plane perpendicular to the radial direction
            ox, oy, oz = o["x"], o["y"], o["z"]
            dist = np.sqrt(ox**2 + oy**2 + oz**2)
            if dist < 1e-6:
                continue
            # Build a local coordinate frame: radial, and two perpendicular
            er = np.array([ox, oy, oz]) / dist
            # Pick an arbitrary non-parallel vector
            up = np.array([0, 0, 1]) if abs(er[2]) < 0.9 else np.array([1, 0, 0])
            e1 = np.cross(er, up)
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(er, e1)
            for t in theta:
                pt = np.array([ox, oy, oz]) + ring_r * (np.cos(t) * e1 + np.sin(t) * e2)
                ring_xs.append(pt[0])
                ring_ys.append(pt[1])
                ring_zs.append(pt[2])
            # None to break line between rings
            ring_xs.append(None)
            ring_ys.append(None)
            ring_zs.append(None)

        spherex_hover_xs, spherex_hover_ys, spherex_hover_zs = [], [], []
        spherex_hover_texts = []
        for o in spherex_objs:
            spherex_hover_xs.append(o["x"])
            spherex_hover_ys.append(o["y"])
            spherex_hover_zs.append(o["z"])
            dist_str = _format_distance(o["distance_pc"])
            n_imgs = o.get("spherex_n_images", "")
            coverage_line = (f"<br>SPHEREx images: {n_imgs}"
                             if n_imgs != "" else "")
            spherex_hover_texts.append(
                f"<b>{o['name']}</b> [SPHEREx Target]<br>"
                f"Distance: {dist_str}{coverage_line}<br>"
                f"{o['description']}"
            )

        # Ring trace (the glowing halo)
        fig.add_trace(go.Scatter3d(
            x=ring_xs, y=ring_ys, z=ring_zs,
            mode="lines",
            line=dict(color="rgba(0, 255, 200, 0.5)", width=3),
            name="SPHEREx Targets",
            legendgroup="spherex",
            legendgrouptitle_text="SPHEREx",
            hoverinfo="skip",
        ))
        # Invisible hover points at object positions (for tooltip)
        fig.add_trace(go.Scatter3d(
            x=spherex_hover_xs, y=spherex_hover_ys, z=spherex_hover_zs,
            mode="markers",
            marker=dict(size=12, color="rgba(0, 255, 200, 0.15)",
                        symbol="circle", line=dict(width=0)),
            name="SPHEREx Info",
            legendgroup="spherex",
            showlegend=False,
            hoverinfo="text",
            hovertext=spherex_hover_texts,
        ))

    # --- Structural overlays ---
    if show_distance_shells:
        for trace in _create_distance_shells(log_scale=log_scale):
            fig.add_trace(trace)

    if show_mw_arms:
        for trace in _create_milky_way_arms(log_scale=log_scale):
            fig.add_trace(trace)

    if show_galactic_plane:
        for trace in _create_galactic_plane(log_scale=log_scale):
            fig.add_trace(trace)

    # --- Layout ---
    axis_range = 8 if log_scale else None  # log10(16 Mpc) ~ 7.2

    axis_template = dict(
        backgroundcolor="rgb(5, 5, 20)",
        gridcolor="rgba(50, 50, 80, 0.3)",
        showticklabels=False,
        title="",
        range=[-axis_range, axis_range] if axis_range else None,
    )

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
                "Earth in the Cosmos: A 360-Degree View<br>"
                "<sup style='color:gray'>SPHEREx All-Sky Infrared Survey | "
                "Distances log-scaled from 1 pc to 16 Mpc</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="Object Types", font=dict(size=12)),
            bgcolor="rgba(10, 10, 30, 0.8)",
            bordercolor="rgba(100, 100, 150, 0.3)",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    # --- Camera preset buttons ---
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
                        label="Top Down (Galactic Pole)",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 3.0}}],
                    ),
                    dict(
                        label="Edge On (Disk View)",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": 3.0, "y": 0, "z": 0.01}}],
                    ),
                    dict(
                        label="From Andromeda",
                        method="relayout",
                        args=[{"scene.camera.eye": {"x": -2.0, "y": 1.5, "z": 0.5}}],
                    ),
                ],
            ),
        ],
    )

    return fig


# ============================================================================
# SECTION 6: SPHEREX DATA INTEGRATION
# ============================================================================

def check_spherex_coverage(catalog=None):
    """
    Query IRSA to determine which catalog objects have SPHEREx observations.

    Requires pyvo and network access. Updates each catalog entry with
    'has_spherex_data' and 'spherex_n_images' fields.

    Parameters
    ----------
    catalog : dict, optional
        Defaults to CATALOG_3D.

    Returns
    -------
    catalog : dict
        Updated catalog with SPHEREx coverage info.
    """
    try:
        from spherex_analysis import query_spherex_region
    except ImportError:
        print("Could not import spherex_analysis. Skipping SPHEREx coverage check.")
        return catalog or CATALOG_3D

    if catalog is None:
        catalog = dict(CATALOG_3D)

    for name, info in catalog.items():
        try:
            results = query_spherex_region(
                info["ra"], info["dec"], radius_deg=0.1
            )
            info["has_spherex_data"] = len(results) > 0
            info["spherex_n_images"] = len(results)
            status = f"{len(results)} images" if len(results) > 0 else "no data"
            print(f"  {name}: {status}")
        except Exception as e:
            info["has_spherex_data"] = False
            info["spherex_n_images"] = 0
            print(f"  {name}: query failed ({e})")

    return catalog


# ============================================================================
# SECTION 7: MAIN ENTRY POINT
# ============================================================================

def render_3d_milky_way(output_html="spherex_3d_milkyway.html",
                         check_spherex=False, log_scale=True,
                         show_mw_arms=True, auto_open=True):
    """
    Build and save the interactive 3D visualization.

    Parameters
    ----------
    output_html : str
        Path for the self-contained HTML file.
    check_spherex : bool
        If True, query IRSA to check which objects have SPHEREx data
        (requires network access and pyvo).
    log_scale : bool
        Use logarithmic distance scaling (recommended).
    show_mw_arms : bool
        Show schematic Milky Way spiral arms.
    auto_open : bool
        Automatically open the HTML file in the default browser.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required. Run: pip install plotly")

    catalog = dict(CATALOG_3D)

    if check_spherex:
        print("Checking SPHEREx data coverage for catalog objects...")
        catalog = check_spherex_coverage(catalog)

    print("Building 3D scene...")
    fig = build_3d_scene(
        catalog,
        log_scale=log_scale,
        show_mw_arms=show_mw_arms,
    )

    print(f"Saving to {output_html}...")
    fig.write_html(output_html, include_plotlyjs=True)
    print(f"3D visualization saved to {output_html}")

    if auto_open:
        fig.show()

    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("SPHEREx 3D Milky Way Visualization")
    print("=" * 40)
    print()
    print(f"Catalog contains {len(CATALOG_3D)} objects:")
    print(f"{'Object':<30s} {'Distance':>12s}  {'Type':<15s}")
    print("-" * 65)
    for name, info in sorted(CATALOG_3D.items(),
                              key=lambda x: x[1]["distance_pc"]):
        dist_str = _format_distance(info["distance_pc"])
        print(f"{name:<30s} {dist_str:>12s}  {info['type']:<15s}")
    print()

    fig = render_3d_milky_way()
