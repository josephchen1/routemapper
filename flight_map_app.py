# -*- coding: utf-8 -*-
"""
Flight Map Generator — Streamlit Web App (no ETOPO)
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

ALL_COUNTRIES = [
    "Afghanistan","Albania","Algeria","Angola","Argentina","Armenia","Australia",
    "Austria","Azerbaijan","Bahrain","Bangladesh","Belarus","Belgium","Belize",
    "Benin","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana","Brazil",
    "Brunei","Bulgaria","Burkina Faso","Burundi","Cambodia","Cameroon","Canada",
    "Central African Republic","Chad","Chile","China","Colombia","Costa Rica",
    "Croatia","Cuba","Czech Republic","Democratic Republic of the Congo",
    "Denmark","Dominican Republic","Ecuador","Egypt","El Salvador","Estonia",
    "Ethiopia","Finland","France","Gabon","Germany","Ghana","Greece","Guatemala",
    "Guinea","Haiti","Honduras","Hungary","Iceland","India","Indonesia","Iran",
    "Iraq","Ireland","Israel","Italy","Jamaica","Japan","Jordan","Kazakhstan",
    "Kenya","Kosovo","Kuwait","Kyrgyzstan","Laos","Latvia","Lebanon","Lesotho",
    "Libya","Lithuania","Luxembourg","Madagascar","Malawi","Malaysia","Mali",
    "Mauritania","Mexico","Moldova","Mongolia","Montenegro","Morocco","Mozambique",
    "Myanmar","Namibia","Nepal","Netherlands","New Zealand","Nicaragua","Niger",
    "Nigeria","North Korea","North Macedonia","Norway","Oman","Pakistan","Panama",
    "Papua New Guinea","Paraguay","Peru","Philippines","Poland","Portugal",
    "Qatar","Republic of the Congo","Romania","Russia","Rwanda","Saudi Arabia",
    "Senegal","Serbia","Sierra Leone","Slovakia","Slovenia","Somalia","South Africa",
    "South Korea","South Sudan","Spain","Sri Lanka","Sudan","Sweden","Switzerland",
    "Syria","Taiwan","Tajikistan","Tanzania","Thailand","Togo","Tunisia","Turkey",
    "Turkmenistan","Uganda","Ukraine","United Arab Emirates","United Kingdom",
    "United States of America","Uruguay","Uzbekistan","Venezuela","Vietnam",
    "Yemen","Zambia","Zimbabwe","Greenland","Puerto Rico","Kosovo"
]

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

    # ── Land fill — excluded countries get ocean (white) fill ─────────────────
    excluded = set(cfg["excluded_countries"])
    countries_shp = shpreader.natural_earth(resolution="50m", category="cultural",
                                             name="admin_0_countries")
    for rec in shpreader.Reader(countries_shp).records():
        name = rec.attributes.get("NAME", "") or rec.attributes.get("ADMIN", "")
        color = "white" if name in excluded else cfg["land_color"]
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                          facecolor=color, edgecolor="none", zorder=2)

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
    valid    = set(cfg["regions"])
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
    shadow           = [PathEffects.withStroke(linewidth=2.5, foreground="white", alpha=0.8)]
    filtered_airports = {c: i for c, i in airports.items() if i.get("region") in valid}

    col_regions = [r for r in ["Mexico", "Central", "Caribbean"] if r in valid]
    col_params  = {
        "Mexico":    (-135, 33,  -2.5, 1.8),
        "Central":   (-106, 14,  -2.5, 1.8),
        "Caribbean": (-61,  35,  -2.5, 2.2),
    }
    col_airports = {r: sorted([i for i in filtered_airports.values() if i["region"] == r],
                               key=lambda x: x["lat"], reverse=True)
                    for r in col_regions}

    all_labels = {}
    texts      = []
    pts        = []

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

    if texts and ADJUSTTEXT_OK and not cfg["is_preview"]:
        adjust_text(texts, add_objects=pts, ax=ax,
                    expand_points=(2.0, 3.0), expand_text=(2.5, 3.0),
                    force_points=2.5, force_text=3.0, add_step_breaks=True,
                    only_move={"points": "xy", "text": "xy"}, arrowprops=None)

    for txt, (ox, oy) in all_labels.items():
        tx, ty = txt.get_position()
        ax.plot([ox, tx], [oy, ty], color="lightgray", linewidth=0.4, alpha=0.8,
                zorder=6, transform=ccrs.PlateCarree(),
                path_effects=[PathEffects.withStroke(linewidth=0.8,
                                                      foreground="white", alpha=0.8)])

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
    export_dpi = st.select_slider("Export DPI", [150, 300, 600], value=300,
                                   help="Preview always uses 72 DPI. This only affects the download.")

    section("Regions")
    all_regions = ["Domestic","Pacific","Atlantic","South","Mexico","Central","Caribbean","Other"]
    regions = st.multiselect("Include", all_regions,
                              default=["Domestic","Pacific","Atlantic","South",
                                       "Mexico","Central","Caribbean"])

    section("Routes")
    use_csv_colors = st.checkbox("Use colors from CSV", value=True)
    route_color    = st.color_picker("Default color", "#005DAA")
    c3, c4 = st.columns(2)
    route_lw    = c3.slider("Line width", 0.3, 3.0, 1.0, 0.1)
    route_alpha = c4.slider("Opacity",    0.1, 1.0, 1.0, 0.05)

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

    section("Exclude Countries")
    excluded_countries = st.multiselect(
        "Hide these countries (show as ocean)",
        options=sorted(ALL_COUNTRIES),
        default=[],
        help="Selected countries will appear as white (ocean color) instead of land color"
    )

    section("Output")
    fmt = st.selectbox("Format", ["PNG", "PDF", "SVG"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("✈️ Flight Map Generator")
st.caption("Upload your CSVs, tweak settings, and the preview updates automatically. Hit **Export** for full-res download.")

routes_df = airports_df = None
if routes_file:
    routes_df = pd.read_csv(routes_file)
if airports_file:
    airports_df = pd.read_csv(airports_file)

if routes_df is not None and airports_df is not None:
    with st.expander("Preview loaded data", expanded=False):
        p1, p2 = st.columns(2)
        p1.markdown(f"**Routes** — {len(routes_df)} rows")
        p1.dataframe(routes_df.head(8), use_container_width=True)
        p2.markdown(f"**Airports** — {len(airports_df)} rows")
        p2.dataframe(airports_df.head(8), use_container_width=True)

# ── Build shared cfg (minus dpi — differs between preview and export) ─────────
def make_cfg(dpi, is_preview):
    return dict(
        center_lon=center_lon, fig_w=fig_w, fig_h=fig_h, dpi=dpi,
        regions=regions, use_csv_colors=use_csv_colors,
        route_color=route_color, route_lw=route_lw, route_alpha=route_alpha,
        dot_color=dot_color, label_color=label_color, land_color=land_color,
        show_borders=show_borders, show_states=show_states, great_lakes=great_lakes,
        border_color=border_color, border_w=border_w,
        state_color=state_color, state_w=state_w,
        fmt="PNG", excluded_countries=excluded_countries,
        is_preview=is_preview,
    )

preview_placeholder = st.empty()
export_placeholder  = st.empty()

if routes_df is not None and airports_df is not None and regions:
    # ── Live preview (72 DPI, no adjustText) ──────────────────────────────────
    with preview_placeholder.container():
        with st.spinner("Updating preview…"):
            try:
                prev_buf = draw_map(routes_df, airports_df, make_cfg(dpi=72, is_preview=True))
                st.image(prev_buf, use_container_width=True,
                         caption="Live preview (72 DPI) — export below for full resolution")
            except Exception as e:
                st.error(f"Preview error: {e}")
                st.exception(e)

    # ── Export button ──────────────────────────────────────────────────────────
    with export_placeholder.container():
        col_btn, col_dl = st.columns([1, 3])
        if col_btn.button("⬇️ Export full-res", type="primary", use_container_width=True):
            with st.spinner(f"Rendering at {export_dpi} DPI…"):
                try:
                    export_cfg = make_cfg(dpi=export_dpi, is_preview=False)
                    export_cfg["fmt"] = fmt
                    exp_buf = draw_map(routes_df, airports_df, export_cfg)
                    mime = {"PNG": "image/png", "PDF": "application/pdf",
                            "SVG": "image/svg+xml"}[fmt]
                    col_dl.download_button(
                        f"Click to download {fmt}",
                        exp_buf,
                        file_name=f"flight_map.{fmt.lower()}",
                        mime=mime,
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")
                    st.exception(e)
elif routes_df is None or airports_df is None:
    st.info("Upload both CSVs in the sidebar to get started.")
elif not regions:
    st.warning("Select at least one region in the sidebar.")
