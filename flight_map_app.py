# -*- coding: utf-8 -*-
"""
Flight Map Generator — Streamlit Web App
Modes: Standard (Cartopy shapes) and Terrain (ETOPO1 GeoTIFF, auto-downloaded)
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import io
import os
from pyproj import Geod
from matplotlib.colors import ListedColormap

try:
    from adjustText import adjust_text
    ADJUSTTEXT_OK = True
except ImportError:
    ADJUSTTEXT_OK = False

try:
    import rasterio
    from rasterio.enums import Resampling
    from scipy.ndimage import distance_transform_edt
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False

GEOD            = Geod(ellps="WGS84")
ETOPO_GDRIVE_ID = "1WhX46wNCgk7WufVkpk9KO67gtSDSZWjv"
ETOPO_LOCAL     = "/tmp/ETOPO1_Bed_c_geotiff.tif"
HALO_COLOR      = np.array([170/255, 197/255, 251/255])

mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans', 'Arial', 'Helvetica']

@st.cache_data
def load_airports():
    return pd.read_csv("airport_coords.csv")

@st.cache_resource(show_spinner="Downloading terrain file (one-time, ~1 GB)…")
def ensure_etopo():
    if os.path.exists(ETOPO_LOCAL):
        return ETOPO_LOCAL
    try:
        import gdown
        gdown.download(id=ETOPO_GDRIVE_ID, output=ETOPO_LOCAL, quiet=False)
        return ETOPO_LOCAL
    except Exception:
        return None

@st.cache_data
def country_names():
    """Pull the canonical list of country names straight from Natural Earth."""
    shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                   name="admin_0_countries")
    names = set()
    for rec in shpreader.Reader(shp).records():
        n = rec.attributes.get("NAME", "") or rec.attributes.get("ADMIN", "")
        if n:
            names.add(n)
    return sorted(names)


# ═══════════════════════════════════════════════════════════════════════════════
# Page config & global styles
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Flight Map Generator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1420;
        border-right: 1px solid #1a2638;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

    /* Section headers in sidebar */
    .section-label {
        font-size: 10px; font-weight: 700; letter-spacing: 1.8px;
        text-transform: uppercase; color: #5d82c4;
        margin-top: 1.5rem; margin-bottom: 0.35rem;
        padding-bottom: 5px; border-bottom: 1px solid #1c2a3f;
    }

    /* App title */
    .app-title {
        font-size: 30px; font-weight: 800; letter-spacing: -0.8px;
        color: #e8eef9; margin-bottom: 2px;
    }
    .app-sub {
        font-size: 14px; color: #7f93b3; margin-bottom: 18px;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #005DAA 0%, #0078d4 100%);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; letter-spacing: 0.3px; padding: 0.55rem 1rem;
        box-shadow: 0 2px 8px rgba(0,93,170,0.35);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0068bd 0%, #0a85e6 100%);
        box-shadow: 0 4px 14px rgba(0,93,170,0.5);
    }
    .stDownloadButton > button {
        background-color: #1f8a4c !important; color: white !important;
        border: none; border-radius: 8px; font-weight: 600; width: 100%;
    }
    .stDownloadButton > button:hover { background-color: #25a85d !important; }

    /* Preview card */
    .preview-card {
        background: #0f1726; border: 1px solid #1c2a3f;
        border-radius: 12px; padding: 10px; margin-top: 4px;
    }

    /* Empty state */
    .empty-state {
        text-align: center; padding: 64px 24px; color: #6b7f9e;
        border: 2px dashed #243348; border-radius: 14px; background: #0d1420;
    }
    .empty-state .icon { font-size: 46px; margin-bottom: 12px; }
    .empty-state .title { font-size: 18px; font-weight: 700; color: #aebfe0; margin-bottom: 6px; }
    .empty-state .desc { font-size: 14px; color: #6b7f9e; }
</style>
""", unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def great_circle_pyproj(lon0, lat0, lon1, lat1, npts=200):
    lon0 = ((lon0+180)%360)-180
    lon1 = ((lon1+180)%360)-180
    pts  = GEOD.npts(lon0, lat0, lon1, lat1, npts)
    lons = [lon0] + [p[0] for p in pts] + [lon1]
    lats = [lat0] + [p[1] for p in pts] + [lat1]
    return lons, lats


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_merge(routes_df, airports_df):
    r = routes_df.copy()
    a = airports_df.copy()
    r.columns = r.columns.str.strip().str.lower()
    a.columns = a.columns.str.strip().str.lower()

    a_orig = a.rename(columns={"longitude":"lon_orig","latitude":"lat_orig","region":"region_orig"})
    r = r.merge(a_orig, left_on="origin", right_on="iata", how="left").drop(columns=["iata"])

    a_dest = a.rename(columns={"longitude":"lon_dest","latitude":"lat_dest","region":"region_dest"})
    r = r.merge(a_dest, left_on="dest", right_on="iata", how="left").drop(columns=["iata"])

    return r.drop_duplicates(
        subset=["origin","dest","lon_orig","lat_orig","lon_dest","lat_dest"]
    ).reset_index(drop=True)


def get_unique_airports(routes):
    unique = {}
    for _, row in routes.iterrows():
        for code, lon, lat, region in [
            (row["origin"], row["lon_orig"], row["lat_orig"], row.get("region_orig")),
            (row["dest"],   row["lon_dest"], row["lat_dest"], row.get("region_dest")),
        ]:
            if pd.notna(lon) and pd.notna(lat):
                dot = float(row.get("dotsize", 3.0)); lbl = float(row.get("labelsize", 9.0))
                if code not in unique:
                    unique[code] = {"code":code,"lon":lon,"lat":lat,
                                    "dot_size":dot,"label_size":lbl,"region":region}
                else:
                    unique[code]["dot_size"]   = max(unique[code]["dot_size"],   dot)
                    unique[code]["label_size"] = max(unique[code]["label_size"], lbl)
    return unique


def unmatched_codes(routes):
    """IATA codes in routes that didn't find coords in the airport table."""
    bad = set()
    for _, row in routes.iterrows():
        if pd.isna(row["lon_orig"]): bad.add(row["origin"])
        if pd.isna(row["lon_dest"]): bad.add(row["dest"])
    return sorted(bad)


# ═══════════════════════════════════════════════════════════════════════════════
# Land base layer — shared between modes
# ═══════════════════════════════════════════════════════════════════════════════

def draw_land_shapes(ax, cfg):
    """Cartopy shapefile land fill with per-country exclusion."""
    excluded = set(cfg["excluded_countries"])
    cshp = shpreader.natural_earth(resolution="50m", category="cultural",
                                    name="admin_0_countries")
    for rec in shpreader.Reader(cshp).records():
        name  = rec.attributes.get("NAME","") or rec.attributes.get("ADMIN","")
        if name in excluded:
            continue  # don't draw at all → stays ocean/white
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                          facecolor=cfg["land_color"], edgecolor="none", zorder=2)

    if cfg["great_lakes"]:
        lshp = shpreader.natural_earth(resolution="50m", category="physical", name="lakes")
        GL   = {"Lake Superior","Lake Michigan","Lake Huron","Lake Erie","Lake Ontario"}
        for rec in shpreader.Reader(lshp).records():
            if rec.attributes.get("name","") in GL:
                ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                                  facecolor="white", edgecolor="none", zorder=2.5)


def draw_borders(ax, cfg):
    if cfg["show_borders"]:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor=cfg["border_color"], linewidth=cfg["border_w"],
                       alpha=0.8, zorder=3.7)
    if cfg["show_states"]:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor=cfg["state_color"], linewidth=cfg["state_w"],
                       alpha=0.7, zorder=3.6)


def draw_routes_and_airports(ax, routes, cfg):
    """Draw all routes + airport dots/labels. No region filtering."""
    routes = routes.sort_values("zorder", ascending=True) if "zorder" in routes.columns else routes

    for _, row in routes.iterrows():
        if pd.isna(row["lon_orig"]) or pd.isna(row["lon_dest"]):
            continue
        color = str(row.get("color", cfg["route_color"])) if cfg["use_csv_colors"] else cfg["route_color"]
        lw    = float(row.get("linewidth", cfg["route_lw"]))
        style = "-" if str(row.get("style","solid"))=="solid" else "--"
        z     = float(row.get("zorder", 4))
        lons, lats = great_circle_pyproj(row["lon_orig"],row["lat_orig"],
                                          row["lon_dest"],row["lat_dest"])
        ax.plot(lons, lats, transform=ccrs.Geodetic(), color=color,
                linewidth=lw, linestyle=style, alpha=cfg["route_alpha"], zorder=z)

    airports = get_unique_airports(routes)
    shadow   = [PathEffects.withStroke(linewidth=2.5, foreground="white", alpha=0.8)]
    all_lbl  = {}; texts = []; pts = []

    for code, info in airports.items():
        lon, lat = info["lon"], info["lat"]
        pt = ax.plot(lon, lat, "o", markersize=info["dot_size"], color=cfg["dot_color"],
                     transform=ccrs.PlateCarree(), alpha=0.85, zorder=5)
        if info["label_size"] > 0:
            txt = ax.text(lon, lat, code, fontsize=info["label_size"], fontweight="bold",
                          color=cfg["label_color"], transform=ccrs.PlateCarree(),
                          ha="left" if lon < -20 else "right", va="bottom",
                          zorder=7, path_effects=shadow)
            texts.append(txt); pts.append(pt[0]); all_lbl[txt] = (lon, lat)

    if texts and ADJUSTTEXT_OK and not cfg["is_preview"]:
        adjust_text(texts, add_objects=pts, ax=ax, expand_points=(2,3), expand_text=(2.5,3),
                    force_points=2.5, force_text=3, add_step_breaks=True,
                    only_move={"points":"xy","text":"xy"}, arrowprops=None)

    for txt,(ox,oy) in all_lbl.items():
        tx,ty = txt.get_position()
        ax.plot([ox,tx],[oy,ty], color="lightgray", linewidth=0.4, alpha=0.8, zorder=6,
                transform=ccrs.PlateCarree(),
                path_effects=[PathEffects.withStroke(linewidth=0.8, foreground="white", alpha=0.8)])


def finish(fig, ax, cfg):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format=cfg["fmt"].lower(), bbox_inches="tight",
                facecolor="white", dpi=cfg["dpi"])
    buf.seek(0); plt.close(fig); return buf


# ═══════════════════════════════════════════════════════════════════════════════
# Standard mode
# ═══════════════════════════════════════════════════════════════════════════════

def draw_standard(routes_df, airports_df, cfg):
    routes = load_and_merge(routes_df, airports_df)
    proj   = ccrs.PlateCarree(central_longitude=cfg["center_lon"])
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]), dpi=cfg["dpi"],
                            subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); ax.set_global()

    draw_land_shapes(ax, cfg)
    draw_borders(ax, cfg)
    draw_routes_and_airports(ax, routes, cfg)
    return finish(fig, ax, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Terrain mode (ETOPO)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading terrain (one-time)…")
def load_dem(etopo_path, downsample):
    with rasterio.open(etopo_path) as src:
        data = src.read(1,
            out_shape=(src.height//downsample, src.width//downsample),
            resampling=Resampling.bilinear)
        transform = src.transform * src.transform.scale(
            src.width/data.shape[1], src.height/data.shape[0])
    nrows,_ = data.shape
    lat = np.linspace(transform[5]+transform[4]*nrows, transform[5], nrows)
    land  = (data>0).astype(float)
    ocean = 1-land
    return data, lat, land, ocean, transform

def coastal_glow(ocean_mask, dist_factor=45):
    nrows,ncols = ocean_mask.shape
    wrap  = np.concatenate([ocean_mask]*3, axis=1)
    dist  = distance_transform_edt(wrap==1)
    mid   = dist.shape[1]//3
    d     = dist[:,mid:mid+ncols]
    glow  = np.exp(-(d/dist_factor)**2)*ocean_mask
    glow  = glow/np.nanmax(glow)
    rgba  = np.zeros((nrows,ncols,4))
    rgba[...,0:3] = HALO_COLOR
    rgba[...,3]   = glow*0.2
    return rgba

def mountain_layer(data, threshold=500):
    m = np.where(data>threshold, data, np.nan)
    if np.isfinite(m).any():
        m = (m-np.nanmin(m))/(np.nanmax(m)-np.nanmin(m))
    return m*0.3

def draw_terrain(routes_df, airports_df, dem_data, lat_arr, land_mask, ocean_mask, cfg):
    routes = load_and_merge(routes_df, airports_df)
    nrows,ncols = dem_data.shape
    extent = [-180,180,lat_arr.min(),lat_arr.max()]
    proj   = ccrs.PlateCarree(central_longitude=cfg["center_lon"])
    fig,ax = plt.subplots(figsize=(cfg["fig_w"],cfg["fig_h"]),dpi=cfg["dpi"],
                           subplot_kw={"projection":proj})
    fig.patch.set_facecolor("white"); ax.set_global()

    # White ocean
    ax.imshow(np.ones_like(dem_data),extent=extent,transform=ccrs.PlateCarree(),
              cmap=ListedColormap(["white"]),origin="upper",zorder=0)

    # Terrain land fill (DEM-based, no per-country exclusion in terrain mode)
    def hex_rgb(h):
        h=h.lstrip("#"); return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))
    r,g,b = hex_rgb(cfg["land_color"])
    land_rgba = np.zeros((nrows,ncols,4))
    land_rgba[...,0]=r; land_rgba[...,1]=g; land_rgba[...,2]=b; land_rgba[...,3]=land_mask
    ax.imshow(land_rgba,extent=extent,transform=ccrs.PlateCarree(),origin="upper",zorder=1)

    if cfg["show_glow"]:
        ax.imshow(coastal_glow(ocean_mask,cfg["glow_factor"]),extent=extent,
                  transform=ccrs.PlateCarree(),origin="upper",zorder=1.5)
    if cfg["show_mountains"]:
        mtn = mountain_layer(dem_data,cfg["mountain_thresh"])
        mr  = np.zeros((nrows,ncols,4))
        mr[...,0:3]=0.4; mr[...,3]=np.where(np.isfinite(mtn),mtn,0)
        ax.imshow(mr,extent=extent,transform=ccrs.PlateCarree(),origin="upper",zorder=2)

    draw_borders(ax, cfg)
    draw_routes_and_airports(ax, routes, cfg)

    ax.set_axis_off()
    return finish(fig, ax, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ✈️ Flight Map")
    if hasattr(st, "segmented_control"):
        mode = st.segmented_control(
            "Map style", ["Standard", "Terrain"],
            default="Standard",
            help="Standard = clean vector land. Terrain = ETOPO elevation shading."
        ) or "Standard"
    else:
        mode = st.radio("Map style", ["Standard", "Terrain"], horizontal=True)

    section("Data")
    if mode == "Standard":
        routes_file = st.file_uploader("Routes CSV", type="csv", key="routes")
    else:
        routes_file = st.file_uploader("Routes CSV", type="csv", key="routes_t")
        st.caption("🌍 Terrain file auto-downloads on first use.")

    with st.expander("CSV format help"):
        st.markdown(
            "**Routes CSV columns:** `origin, dest, color, style, "
            "linewidth, dotsize, labelsize, zorder`\n\n"
            "Airport coordinates are built in — no need to upload them."
        )

    section("Projection & Size")
    center_lon = st.slider("Center longitude", -180, 180, -90, 5)
    c1,c2 = st.columns(2)
    fig_w = c1.number_input("Width (in)",  8, 36, 18)
    fig_h = c2.number_input("Height (in)", 4, 20,  9)
    export_dpi = st.select_slider("Export DPI", [150,300,600], value=300,
                                   help="Preview is always 72 DPI for speed")

    if mode == "Terrain":
        section("Terrain")
        downsample      = st.slider("Detail (downsample)", 4, 16, 8, 2,
                                     help="Lower = sharper but slower")
        show_glow       = st.checkbox("Coastal glow", True)
        glow_factor     = st.slider("Glow spread", 10, 100, 45, 5,
                                     disabled=not show_glow)
        show_mountains  = st.checkbox("Mountain shading", True)
        mountain_thresh = st.slider("Mountain threshold (m)", 100, 2000, 500, 50,
                                     disabled=not show_mountains)

    section("Routes")
    use_csv_colors = st.checkbox("Use colors from CSV", True,
                                  help="Off = apply the single color below to every route")
    route_color = st.color_picker("Route color", "#005DAA", disabled=use_csv_colors)
    c3,c4 = st.columns(2)
    route_lw    = c3.slider("Line width", 0.1, 3.0, 1.0, 0.1)
    route_alpha = c4.slider("Opacity",    0.1, 1.0, 1.0, 0.05)

    section("Airports")
    c5,c6 = st.columns(2)
    dot_color   = c5.color_picker("Dot",   "#000000")
    label_color = c6.color_picker("Label", "#000000")

    section("Land & Borders")
    land_color = st.color_picker("Land color", "#bfbfbf")
    c7,c8 = st.columns(2)
    show_borders = c7.toggle("Country borders", True)
    show_states  = c8.toggle("State borders",   True)
    great_lakes  = st.toggle("Great Lakes", True) if mode == "Standard" else False
    cc1, cc2 = st.columns(2)
    border_color = cc1.color_picker("Border", "#ffffff")
    state_color  = cc2.color_picker("State",  "#ffffff")
    border_w     = st.slider("Border width", 0.1, 2.0, 0.5, 0.1)
    state_w      = st.slider("State width",  0.1, 2.0, 0.3, 0.1)

    if mode == "Standard":
        section("Hide Countries")
        excluded_countries = st.multiselect(
            "Render as ocean",
            options=country_names(),
            default=[],
            help="Selected countries are removed from the land layer"
        )
    else:
        excluded_countries = []

    section("Export")
    fmt = st.selectbox("File format", ["PNG","PDF","SVG"])


# ═══════════════════════════════════════════════════════════════════════════════
# Main panel
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="app-title">Flight Map Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Upload a routes CSV and the map updates live. '
            'Tune it in the sidebar, then export at full resolution.</div>',
            unsafe_allow_html=True)


def build_cfg(dpi, is_preview):
    cfg = dict(
        center_lon=center_lon, fig_w=fig_w, fig_h=fig_h, dpi=dpi,
        use_csv_colors=use_csv_colors, route_color=route_color,
        route_lw=route_lw, route_alpha=route_alpha,
        dot_color=dot_color, label_color=label_color, land_color=land_color,
        show_borders=show_borders, show_states=show_states, great_lakes=great_lakes,
        border_color=border_color, border_w=border_w,
        state_color=state_color, state_w=state_w,
        excluded_countries=excluded_countries,
        is_preview=is_preview,
        show_glow=False, glow_factor=45, show_mountains=False, mountain_thresh=500,
        fmt="PNG",
    )
    if mode == "Terrain":
        cfg.update(show_glow=show_glow, glow_factor=glow_factor,
                   show_mountains=show_mountains, mountain_thresh=mountain_thresh)
    return cfg


def empty_state():
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">🗺️</div>'
        '<div class="title">No routes loaded yet</div>'
        '<div class="desc">Upload a routes CSV in the sidebar to generate your map.</div>'
        '</div>',
        unsafe_allow_html=True
    )


def show_preview_and_export(render_fn, render_args, base_cfg):
    # Stats row
    routes_df = render_args[0]
    airports_df = render_args[1]
    merged = load_and_merge(routes_df, airports_df)
    n_routes = len(merged)
    n_airports = len(get_unique_airports(merged))
    missing = unmatched_codes(merged)

    m1, m2, m3 = st.columns(3)
    m1.metric("Routes", n_routes)
    m2.metric("Airports", n_airports)
    m3.metric("Unmatched codes", len(missing))

    if missing:
        st.warning(f"These codes weren't found in the airport table and will be skipped: "
                   f"{', '.join(missing[:15])}{'…' if len(missing) > 15 else ''}")

    # Live preview
    st.markdown("#### Preview")
    with st.spinner("Rendering preview…"):
        try:
            prev = render_fn(*render_args, {**base_cfg, "dpi":72, "fmt":"PNG", "is_preview":True})
            st.image(prev, width="stretch")
            st.caption("Live preview at 72 DPI. Labels are auto-spaced only in the full-res export.")
        except Exception as e:
            st.error(f"Couldn't render preview: {e}")
            with st.expander("Details"):
                st.exception(e)
            return

    # Export
    st.markdown("#### Export")
    cexp, cdl = st.columns([1, 2])
    if cexp.button(f"Render {export_dpi} DPI", type="primary", width="stretch"):
        with st.spinner(f"Rendering at {export_dpi} DPI…"):
            try:
                full = render_fn(*render_args, {**base_cfg, "dpi":export_dpi, "fmt":fmt, "is_preview":False})
                mime = {"PNG":"image/png","PDF":"application/pdf","SVG":"image/svg+xml"}[fmt]
                cdl.download_button(f"⬇ Download {fmt}", full,
                                     file_name=f"flight_map.{fmt.lower()}",
                                     mime=mime, width="stretch")
            except Exception as e:
                st.error(f"Export failed: {e}")
                with st.expander("Details"):
                    st.exception(e)


# ── Standard mode ─────────────────────────────────────────────────────────────
if mode == "Standard":
    airports_df = load_airports()
    routes_df   = pd.read_csv(routes_file) if routes_file else None

    if routes_df is None:
        empty_state()
    else:
        show_preview_and_export(draw_standard, (routes_df, airports_df), build_cfg(72, True))

# ── Terrain mode ──────────────────────────────────────────────────────────────
else:
    if not RASTERIO_OK:
        st.error("`rasterio` and `scipy` aren't installed in this environment.")
        st.stop()

    airports_df = load_airports()
    routes_df   = pd.read_csv(routes_file) if routes_file else None

    if routes_df is None:
        empty_state()
    else:
        etopo_path = ensure_etopo()
        if etopo_path is None:
            st.error("Couldn't download the ETOPO terrain file. Make sure the Google Drive "
                     "file is shared as 'Anyone with the link'.")
        else:
            dem_data, lat_arr, land_mask, ocean_mask, _ = load_dem(etopo_path, downsample)
            show_preview_and_export(
                draw_terrain,
                (routes_df, airports_df, dem_data, lat_arr, land_mask, ocean_mask),
                build_cfg(72, True),
            )
