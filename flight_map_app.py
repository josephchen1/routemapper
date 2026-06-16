# -*- coding: utf-8 -*-
"""
Flight Map Generator — Streamlit Web App
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import pandas as pd
import io
from pyproj import Geod

try:
    from adjustText import adjust_text
    ADJUSTTEXT_OK = True
except ImportError:
    ADJUSTTEXT_OK = False

GEOD = Geod(ellps="WGS84")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Map Generator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0e1621; }
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #4a6fa5;
    margin-top: 1.4rem;
    margin-bottom: 0.2rem;
    padding-bottom: 4px;
    border-bottom: 1px solid #1c2a3f;
}
.stDownloadButton > button {
    background-color: #005DAA !important;
    color: white !important;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ── Core functions ────────────────────────────────────────────────────────────

def load_and_merge(routes_df, airports_df):
    routes = routes_df.copy()
    airports = airports_df.copy()
    routes.columns = routes.columns.str.strip().str.lower()
    airports.columns = airports.columns.str.strip().str.lower()
    routes = routes.merge(
        airports.rename(columns={"iata": "origin", "longitude": "lon_orig",
                                  "latitude": "lat_orig", "region": "region_orig"}),
        on="origin", how="left"
    )
    routes = routes.merge(
        airports.rename(columns={"iata": "dest", "longitude": "lon_dest",
                                  "latitude": "lat_dest", "region": "region_dest"}),
        on="dest", how="left"
    )
    return routes.drop_duplicates(
        subset=["origin", "dest", "lon_orig", "lat_orig", "lon_dest", "lat_dest"]
    ).reset_index(drop=True)


def great_circle_curve(lon0, lat0, lon1, lat1, npts=200):
    lon0 = ((lon0 + 180) % 360) - 180
    lon1 = ((lon1 + 180) % 360) - 180
    pts = GEOD.npts(lon0, lat0, lon1, lat1, npts)
    lons = [lon0] + [pt[0] for pt in pts] + [lon1]
    lats = [lat0] + [pt[1] for pt in pts] + [lat1]
    return lons, lats


def get_unique_airports(routes):
    unique = {}
    for _, row in routes.iterrows():
        for code, lon, lat, region in [
            (row["origin"], row["lon_orig"], row["lat_orig"], row.get("region_orig")),
            (row["dest"],   row["lon_dest"], row["lat_dest"], row.get("region_dest")),
        ]:
            if pd.notna(lon) and pd.notna(lat):
                dot   = float(row.get("dotsize",   3.0))
                label = float(row.get("labelsize", 9.0))
                if code not in unique:
                    unique[code] = {"code": code, "lon": lon, "lat": lat,
                                    "dot_size": dot, "label_size": label, "region": region}
                else:
                    unique[code]["dot_size"]   = max(unique[code]["dot_size"],   dot)
                    unique[code]["label_size"] = max(unique[code]["label_size"], label)
    return unique


def draw_map(routes_df, airports_df, cfg):
    routes   = load_and_merge(routes_df, airports_df)
    airports = get_unique_airports(routes)

    proj = ccrs.PlateCarree(central_longitude=cfg["center_lon"])
    fig, ax = plt.subplots(
        figsize=(cfg["fig_w"], cfg["fig_h"]),
        dpi=cfg["dpi"],
        subplot_kw={"projection": proj},
    )
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_global()

    # ── Land fill ─────────────────────────────────────────────────────────────
    countries_shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                             name="admin_0_countries")
    for rec in shpreader.Reader(countries_shp).records():
        if rec.attributes.get("NAME", "") not in cfg["excluded_countries"]:
            ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                              facecolor=cfg["land_color"], edgecolor="none", zorder=2)

    # ── Great Lakes ───────────────────────────────────────────────────────────
    if cfg["great_lakes"]:
        lakes_shp = shpreader.natural_earth(resolution="50m", category="physical", name="lakes")
        gl = {"Lake Superior", "Lake Michigan", "Lake Huron", "Lake Erie", "Lake Ontario"}
        for rec in shpreader.Reader(lakes_shp).records():
            if rec.attributes.get("name", "") in gl:
                ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                                  facecolor="white", edgecolor="none", zorder=2.5)

    # ── Borders ───────────────────────────────────────────────────────────────
    if cfg["show_borders"]:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor=cfg["border_color"], linewidth=cfg["border_w"],
                       alpha=0.8, zorder=3.7)
    if cfg["show_states"]:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor=cfg["state_color"], linewidth=cfg["state_w"],
                       alpha=0.7, zorder=3.6)

    # ── Routes ────────────────────────────────────────────────────────────────
    valid = set(cfg["regions"])
    filtered = routes[routes["region_orig"].isin(valid) & routes["region_dest"].isin(valid)]
    filtered = filtered.sort_values("zorder", ascending=True)

    for _, row in filtered.iterrows():
        color = str(row.get("color", cfg["route_color"])) if cfg["use_csv_colors"] else cfg["route_color"]
        lw    = float(row.get("linewidth", cfg["route_lw"]))
        style = "-" if str(row.get("style", "solid")) == "solid" else "--"
        z     = float(row.get("zorder", 4))
        lons, lats = great_circle_curve(row["lon_orig"], row["lat_orig"],
                                         row["lon_dest"], row["lat_dest"])
        ax.plot(lons, lats, transform=ccrs.Geodetic(),
                color=color, linewidth=lw, linestyle=style,
                alpha=cfg["route_alpha"], zorder=z)

    # ── Airports ──────────────────────────────────────────────────────────────
    shadow = [PathEffects.withStroke(linewidth=2.5, foreground="white", alpha=0.8)]
    filtered_airports = {c: i for c, i in airports.items() if i.get("region") in valid}

    col_regions  = [r for r in ["Mexico", "Central", "Caribbean"] if r in valid]
    col_params   = {
        "Mexico":    (-135, 33,  -2.5, 1.8),
        "Central":   (-106, 14,  -2.5, 1.8),
        "Caribbean": (-61,  35,  -2.5, 2.2),
    }
    col_airports = {r: sorted([i for i in filtered_airports.values() if i["region"] == r],
                               key=lambda x: x["lat"], reverse=True)
                    for r in col_regions}

    all_labels   = {}
    texts        = []
    pts          = []

    for region in col_regions:
        sl, slat, lsp, lsl = col_params[region]
        for i, info in enumerate(col_airports[region]):
            ax.plot(info["lon"], info["lat"], "o", markersize=info["dot_size"],
                    color=cfg["dot_color"], transform=ccrs.PlateCarree(), alpha=0.8, zorder=6)
            if info["label_size"] > 0:
                txt = ax.text(sl + i * lsl, slat + i * lsp, info["code"],
                              fontsize=info["label_size"], fontweight="bold",
                              color=cfg["label_color"], transform=ccrs.PlateCarree(),
                              ha="center", va="center", zorder=7, path_effects=shadow)
                all_labels[txt] = (info["lon"], info["lat"])

    for code, info in filtered_airports.items():
        if info["region"] in col_regions:
            continue
        lon, lat = info["lon"], info["lat"]
        pt = ax.plot(lon, lat, "o", markersize=info["dot_size"],
                     color=cfg["dot_color"], transform=ccrs.PlateCarree(), alpha=0.8, zorder=5)
        if info["label_size"] > 0:
            txt = ax.text(lon, lat, code,
                          fontsize=info["label_size"], fontweight="bold",
                          color=cfg["label_color"], transform=ccrs.PlateCarree(),
                          ha="left" if lon < -20 else "right", va="bottom",
                          zorder=7, path_effects=shadow)
            texts.append(txt)
            pts.append(pt[0])
            all_labels[txt] = (lon, lat)

    if texts and ADJUSTTEXT_OK:
        adjust_text(texts, add_objects=pts, ax=ax,
                    expand_points=(2.0, 3.0), expand_text=(2.5, 3.0),
                    force_points=2.5, force_text=3.0, add_step_breaks=True,
                    only_move={"points": "xy", "text": "xy"}, arrowprops=None)

    for txt, (ox, oy) in all_labels.items():
        tx, ty = txt.get_position()
        ax.plot([ox, tx], [oy, ty], color="lightgray", linewidth=0.4, alpha=0.8,
                zorder=6, transform=ccrs.PlateCarree(),
                path_effects=[PathEffects.withStroke(linewidth=0.8, foreground="white", alpha=0.8)])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format=cfg["fmt"].lower(), bbox_inches="tight",
                facecolor="white", dpi=cfg["dpi"])
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Flight Map")
    st.markdown("---")

    def section(label):
        st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

    section("Data")
    routes_file   = st.file_uploader("Routes CSV",   type="csv")
    airports_file = st.file_uploader("Airports CSV", type="csv")

    section("Map Layout")
    center_lon = st.slider("Center longitude", -180, 180, -90, 5)
    c1, c2 = st.columns(2)
    fig_w = c1.number_input("Width (in)",  8, 36, 18)
    fig_h = c2.number_input("Height (in)", 4, 20,  9)
    dpi   = st.select_slider("DPI", [72, 150, 300, 600], value=150,
                              help="72–150 for preview, 300–600 for final export")

    section("Regions")
    all_regions = ["Domestic","Pacific","Atlantic","South","Mexico","Central","Caribbean","Other"]
    regions = st.multiselect("Include", all_regions,
                              default=["Domestic","Pacific","Atlantic","South",
                                       "Mexico","Central","Caribbean"])

    section("Routes")
    use_csv_colors = st.checkbox("Use colors from CSV", value=True)
    route_color    = st.color_picker("Default color", "#005DAA")
    c3, c4 = st.columns(2)
    route_lw    = c3.slider("Line width",  0.3, 3.0, 1.0, 0.1)
    route_alpha = c4.slider("Opacity",     0.1, 1.0, 1.0, 0.05)

    section("Airports")
    c5, c6 = st.columns(2)
    dot_color   = c5.color_picker("Dot",   "#000000")
    label_color = c6.color_picker("Label", "#000000")

    section("Map Style")
    land_color = st.color_picker("Land color", "#bfbfbf")
    c7, c8 = st.columns(2)
    show_borders = c7.checkbox("Country borders", True)
    show_states  = c8.checkbox("State borders",   True)
    great_lakes  = st.checkbox("Great Lakes", True)
    border_color = st.color_picker("Border color", "#ffffff")
    c9, c10 = st.columns(2)
    border_w    = c9.slider("Border width",       0.1, 2.0, 0.5, 0.1)
    state_color = c10.color_picker("State color", "#ffffff")
    state_w     = st.slider("State border width", 0.1, 2.0, 0.3, 0.1)

    section("Output")
    fmt = st.selectbox("Format", ["PNG", "PDF", "SVG"])

    excluded_countries: list[str] = []   # extend via future UI if needed


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("✈️ Flight Map Generator")
st.caption("Upload your CSVs, adjust settings in the sidebar, then generate.")

col_gen, _ = st.columns([1, 3])
generate = col_gen.button("🗺️ Generate Map", type="primary", use_container_width=True)

with st.expander("Preview data", expanded=False):
    pc1, pc2 = st.columns(2)

routes_df = airports_df = None

if routes_file:
    routes_df = pd.read_csv(routes_file)
if airports_file:
    airports_df = pd.read_csv(airports_file)

with st.expander("Preview data", expanded=False):
    pass   # placeholder — real preview below

if routes_df is not None and airports_df is not None:
    with st.expander("Preview loaded data", expanded=False):
        p1, p2 = st.columns(2)
        p1.markdown(f"**Routes** — {len(routes_df)} rows")
        p1.dataframe(routes_df.head(8), use_container_width=True)
        p2.markdown(f"**Airports** — {len(airports_df)} rows")
        p2.dataframe(airports_df.head(8), use_container_width=True)

if generate:
    if routes_df is None or airports_df is None:
        st.error("Upload both a Routes CSV and an Airports CSV first.")
    elif not regions:
        st.error("Select at least one region.")
    else:
        cfg = dict(
            center_lon=center_lon, fig_w=fig_w, fig_h=fig_h, dpi=dpi,
            regions=regions, use_csv_colors=use_csv_colors,
            route_color=route_color, route_lw=route_lw, route_alpha=route_alpha,
            dot_color=dot_color, label_color=label_color, land_color=land_color,
            show_borders=show_borders, show_states=show_states, great_lakes=great_lakes,
            border_color=border_color, border_w=border_w,
            state_color=state_color, state_w=state_w,
            fmt=fmt, excluded_countries=excluded_countries,
        )
        with st.spinner("Rendering… (30–60 sec at 300+ DPI)"):
            try:
                buf = draw_map(routes_df, airports_df, cfg)
                st.success("Done!")
                if fmt == "PNG":
                    st.image(buf, use_container_width=True)
                    buf.seek(0)
                mime = {"PNG": "image/png", "PDF": "application/pdf", "SVG": "image/svg+xml"}[fmt]
                st.download_button(f"⬇️ Download {fmt}", buf,
                                   file_name=f"flight_map.{fmt.lower()}", mime=mime,
                                   use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
