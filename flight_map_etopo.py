# -*- coding: utf-8 -*-
"""
Flight Map Generator — ETOPO Terrain Version (local use only)
Requires: ETOPO1_Bed_c_geotiff.tif on local disk
          pip install rasterio scipy
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import io
import os

try:
    import rasterio
    from rasterio.enums import Resampling
    from scipy.ndimage import distance_transform_edt
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

HALO_COLOR = np.array([170 / 255, 197 / 255, 251 / 255])

mpl.rcParams['font.family'] = 'Arial'

st.set_page_config(page_title="Flight Map (ETOPO)", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0e1621; }
.section-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: #4a6fa5;
    margin-top: 1.4rem; margin-bottom: 0.2rem;
    padding-bottom: 4px; border-bottom: 1px solid #1c2a3f;
}
</style>
""", unsafe_allow_html=True)


# ── ETOPO helpers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading terrain data…")
def load_dem(etopo_path, downsample=8):
    with rasterio.open(etopo_path) as src:
        data = src.read(
            1,
            out_shape=(src.height // downsample, src.width // downsample),
            resampling=Resampling.bilinear,
        )
        transform = src.transform * src.transform.scale(
            src.width / data.shape[1], src.height / data.shape[0]
        )
    nrows, _ = data.shape
    lat = np.linspace(transform[5] + transform[4] * nrows, transform[5], nrows)
    land_mask  = (data > 0).astype(float)
    ocean_mask = 1 - land_mask
    return data, lat, land_mask, ocean_mask, transform


def coastal_glow(ocean_mask, dist_factor=45):
    nrows, ncols = ocean_mask.shape
    ocean_wrap = np.concatenate([ocean_mask, ocean_mask, ocean_mask], axis=1)
    dist_wrap  = distance_transform_edt(ocean_wrap == 1)
    mid        = dist_wrap.shape[1] // 3
    dist       = dist_wrap[:, mid:mid + ncols]
    glow       = np.exp(-(dist / dist_factor) ** 2) * ocean_mask
    glow       = glow / np.nanmax(glow)
    rgba       = np.zeros((nrows, ncols, 4))
    rgba[..., 0:3] = HALO_COLOR
    rgba[..., 3]   = glow * 0.2
    return rgba


def mountain_layer(data, threshold=500):
    m = np.where(data > threshold, data, np.nan)
    if np.isfinite(m).any():
        m = (m - np.nanmin(m)) / (np.nanmax(m) - np.nanmin(m))
    return m * 0.3


# ── Flight data helpers ───────────────────────────────────────────────────────

def interpolate_great_circle(start, end, num_points=400):
    start_lat, start_lon = np.deg2rad(start)
    end_lat,   end_lon   = np.deg2rad(end)
    delta_lon = end_lon - start_lon
    d = np.arccos(np.clip(
        np.sin(start_lat) * np.sin(end_lat) +
        np.cos(start_lat) * np.cos(end_lat) * np.cos(delta_lon), -1, 1))
    lats, lons = [], []
    for i in range(num_points):
        f = i / (num_points - 1)
        if d < 1e-10:
            lats.append(np.rad2deg(start_lat))
            lons.append(np.rad2deg(start_lon))
            continue
        A = np.sin((1 - f) * d) / np.sin(d)
        B = np.sin(f * d)       / np.sin(d)
        x = A * np.cos(start_lat) * np.cos(start_lon) + B * np.cos(end_lat) * np.cos(end_lon)
        y = A * np.cos(start_lat) * np.sin(start_lon) + B * np.cos(end_lat) * np.sin(end_lon)
        z = A * np.sin(start_lat) + B * np.sin(end_lat)
        lats.append(np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2))))
        lons.append(np.rad2deg(np.arctan2(y, x)))
    return lats, lons


def draw_etopo_map(flight_df, cfg, dem_data, lat_arr, land_mask, ocean_mask):
    nrows, ncols = dem_data.shape
    extent = [-180, 180, lat_arr.min(), lat_arr.max()]

    proj = ccrs.PlateCarree(central_longitude=cfg["center_lon"])
    fig, ax = plt.subplots(
        figsize=(cfg["fig_w"], cfg["fig_h"]),
        dpi=cfg["dpi"],
        subplot_kw={"projection": proj},
    )
    fig.patch.set_facecolor("white")
    ax.set_global()

    # Ocean background
    from matplotlib.colors import ListedColormap
    ax.imshow(np.ones_like(dem_data), extent=extent, transform=ccrs.PlateCarree(),
              cmap=ListedColormap(["white"]), origin="upper", zorder=0)

    # Land
    def hex_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

    r, g, b = hex_rgb(cfg["land_color"])
    land_rgba = np.zeros((nrows, ncols, 4))
    land_rgba[..., 0] = r
    land_rgba[..., 1] = g
    land_rgba[..., 2] = b
    land_rgba[..., 3] = land_mask
    ax.imshow(land_rgba, extent=extent, transform=ccrs.PlateCarree(),
              origin="upper", zorder=1)

    # Coastal glow
    if cfg["show_glow"]:
        glow = coastal_glow(ocean_mask, cfg["glow_factor"])
        ax.imshow(glow, extent=extent, transform=ccrs.PlateCarree(),
                  origin="upper", zorder=1.5)

    # Mountains
    if cfg["show_mountains"]:
        mountains = mountain_layer(dem_data, cfg["mountain_threshold"])
        mountain_rgba = np.zeros((nrows, ncols, 4))
        mountain_rgba[..., 0:3] = 0.4
        mountain_rgba[..., 3]   = np.where(np.isfinite(mountains), mountains, 0)
        ax.imshow(mountain_rgba, extent=extent, transform=ccrs.PlateCarree(),
                  origin="upper", zorder=2)

    # Borders
    if cfg["show_borders"]:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor=cfg["border_color"], linewidth=cfg["border_w"],
                       alpha=0.8, zorder=3)
    if cfg["show_states"]:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor=cfg["state_color"], linewidth=cfg["state_w"],
                       alpha=0.7, zorder=3)

    # Routes + airports
    labeled = {}
    v_offset = cfg["label_offset"]

    for _, row in flight_df.iterrows():
        slat = row["Origin_Lat"];    slon = row["Origin_Lon"]
        elat = row["Destination_Lat"]; elon = row["Destination_Lon"]
        color = str(row.get("Color", cfg["route_color"]))
        lw    = float(row.get("line_thickness", cfg["route_lw"]))

        lats, lons = interpolate_great_circle((slat, slon), (elat, elon))
        ax.plot(lons, lats, transform=ccrs.Geodetic(),
                color=color, linewidth=lw, linestyle="-",
                alpha=cfg["route_alpha"], zorder=4)

        # Origin dot + label
        ops = row.get("Origin_Point_Size", np.nan)
        ofs = row.get("origin_font_size",  np.nan)
        ol  = str(row.get("Origin", "")).strip()
        if pd.notna(ops) and ops > 0 and ol and pd.notna(ofs):
            ax.scatter(slon, slat, transform=ccrs.PlateCarree(),
                       s=ops, color=cfg["dot_color"], zorder=5)
            key = (slat, slon)
            if key not in labeled:
                ax.text(slon, slat + v_offset, ol,
                        transform=ccrs.PlateCarree(), fontsize=ofs,
                        color=cfg["label_color"], zorder=6,
                        ha="center", va="bottom")
                labeled[key] = ol

        # Destination dot + label
        dps = row.get("Destination_Point_Size", np.nan)
        dfs = row.get("destination_font_size",  np.nan)
        dl  = str(row.get("Destination", "")).strip()
        if pd.notna(dps) and dps > 0 and dl and pd.notna(dfs):
            ax.scatter(elon, elat, transform=ccrs.PlateCarree(),
                       s=dps, color=cfg["dot_color"], zorder=5)
            key = (elat, elon)
            if key not in labeled:
                ax.text(elon, elat + v_offset, dl,
                        transform=ccrs.PlateCarree(), fontsize=dfs,
                        color=cfg["label_color"], zorder=6,
                        ha="center", va="bottom")
                labeled[key] = dl

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format=cfg["fmt"].lower(), bbox_inches="tight",
                facecolor="white", dpi=cfg["dpi"])
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ Flight Map (ETOPO)")
    st.markdown("---")

    def section(label):
        st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

    section("Data")
    flights_file = st.file_uploader("Flight data CSV", type="csv")
    etopo_path   = st.text_input("ETOPO1 .tif path",
                                  placeholder=r"C:\...\ETOPO1_Bed_c_geotiff.tif")

    section("Map Layout")
    center_lon = st.slider("Center longitude", -180, 180, -90, 5)
    c1, c2 = st.columns(2)
    fig_w = c1.number_input("Width (in)",  8, 36, 12)
    fig_h = c2.number_input("Height (in)", 4, 20,  8)
    downsample = st.slider("Terrain downsample", 4, 16, 8, 2,
                            help="Higher = faster but less detail")
    export_dpi = st.select_slider("Export DPI", [150, 300, 600, 800], value=300)

    section("Terrain")
    show_glow        = st.checkbox("Coastal glow", True)
    glow_factor      = st.slider("Glow spread", 10, 100, 45, 5)
    show_mountains   = st.checkbox("Mountain shading", True)
    mountain_thresh  = st.slider("Mountain threshold (m)", 100, 2000, 500, 50)

    section("Map Style")
    land_color = st.color_picker("Land color", "#bfbfbf")
    c3, c4 = st.columns(2)
    show_borders = c3.checkbox("Country borders", True)
    show_states  = c4.checkbox("State borders",   True)
    border_color = st.color_picker("Border color", "#ffffff")
    c5, c6 = st.columns(2)
    border_w    = c5.slider("Border width", 0.1, 2.0, 0.5, 0.1)
    state_color = c6.color_picker("State color", "#ffffff")
    state_w     = st.slider("State border width", 0.1, 2.0, 0.3, 0.1)

    section("Routes & Airports")
    route_color  = st.color_picker("Default route color", "#005DAA")
    route_lw     = st.slider("Default line width", 0.1, 3.0, 0.4, 0.1)
    route_alpha  = st.slider("Route opacity", 0.1, 1.0, 1.0, 0.05)
    dot_color    = st.color_picker("Dot color",   "#000000")
    label_color  = st.color_picker("Label color", "#000000")
    label_offset = st.slider("Label vertical offset", 0.0, 5.0, 0.3, 0.1)

    section("Output")
    fmt = st.selectbox("Format", ["PNG", "PDF", "SVG"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🗺️ Flight Map Generator — Terrain Edition")
st.caption("Run locally with your ETOPO1 GeoTIFF. Enter the file path in the sidebar.")

flight_df = None
if flights_file:
    flight_df = pd.read_csv(flights_file)
    with st.expander("Preview data", expanded=False):
        st.dataframe(flight_df.head(10), use_container_width=True)

if not RASTERIO_OK:
    st.error("`rasterio` and `scipy` are required for terrain mode. "
             "Run: `pip install rasterio scipy`")
    st.stop()

def make_cfg(dpi):
    return dict(
        center_lon=center_lon, fig_w=fig_w, fig_h=fig_h, dpi=dpi,
        show_glow=show_glow, glow_factor=glow_factor,
        show_mountains=show_mountains, mountain_threshold=mountain_thresh,
        land_color=land_color, show_borders=show_borders, show_states=show_states,
        border_color=border_color, border_w=border_w,
        state_color=state_color, state_w=state_w,
        route_color=route_color, route_lw=route_lw, route_alpha=route_alpha,
        dot_color=dot_color, label_color=label_color, label_offset=label_offset,
        fmt="PNG",
    )

if flight_df is not None and etopo_path and os.path.exists(etopo_path.strip()):
    dem_data, lat_arr, land_mask, ocean_mask, _ = load_dem(etopo_path.strip(), downsample)

    with st.spinner("Rendering preview…"):
        try:
            prev_buf = draw_etopo_map(flight_df, make_cfg(dpi=72),
                                       dem_data, lat_arr, land_mask, ocean_mask)
            st.image(prev_buf, use_container_width=True,
                     caption="Live preview (72 DPI)")
        except Exception as e:
            st.error(f"Preview error: {e}")
            st.exception(e)

    col_btn, col_dl = st.columns([1, 3])
    if col_btn.button("⬇️ Export full-res", type="primary", use_container_width=True):
        with st.spinner(f"Rendering at {export_dpi} DPI…"):
            try:
                export_cfg = make_cfg(dpi=export_dpi)
                export_cfg["fmt"] = fmt
                exp_buf = draw_etopo_map(flight_df, export_cfg,
                                          dem_data, lat_arr, land_mask, ocean_mask)
                mime = {"PNG": "image/png", "PDF": "application/pdf",
                        "SVG": "image/svg+xml"}[fmt]
                col_dl.download_button(
                    f"Click to download {fmt}", exp_buf,
                    file_name=f"flight_map_terrain.{fmt.lower()}",
                    mime=mime, use_container_width=True,
                )
            except Exception as e:
                st.error(f"Export error: {e}")
                st.exception(e)
elif flight_df is None:
    st.info("Upload your flight CSV in the sidebar.")
elif not etopo_path:
    st.info("Enter the path to your ETOPO1 .tif file in the sidebar.")
elif not os.path.exists(etopo_path.strip()):
    st.error(f"File not found: `{etopo_path.strip()}`")
