# -*- coding: utf-8 -*-
"""
Flight Map Generator — Streamlit Web App
Supports two modes: Standard (Cartopy shapes) and Terrain (ETOPO1 GeoTIFF, auto-downloaded)
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

GEOD       = Geod(ellps="WGS84")
ETOPO_GDRIVE_ID = "1WhX46wNCgk7WufVkpk9KO67gtSDSZWjv"
ETOPO_LOCAL     = "/tmp/ETOPO1_Bed_c_geotiff.tif"

@st.cache_data
def load_airports():
    """Load bundled airport coords from the repo."""
    return pd.read_csv("airport_coords.csv")

@st.cache_resource(show_spinner="Downloading terrain file (one-time, ~1 GB)…")
def ensure_etopo():
    """Download ETOPO1 from Google Drive if not already cached on disk."""
    if os.path.exists(ETOPO_LOCAL):
        return ETOPO_LOCAL
    try:
        import gdown
        gdown.download(id=ETOPO_GDRIVE_ID, output=ETOPO_LOCAL, quiet=False)
        return ETOPO_LOCAL
    except Exception as e:
        return None
HALO_COLOR = np.array([170/255, 197/255, 251/255])
mpl.rcParams['font.family'] = 'Arial'

ALL_COUNTRIES = sorted([
    "Afghanistan","Albania","Algeria","Angola","Argentina","Armenia","Australia",
    "Austria","Azerbaijan","Bahrain","Bangladesh","Belarus","Belgium","Belize",
    "Benin","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana","Brazil",
    "Brunei","Bulgaria","Burkina Faso","Burundi","Cambodia","Cameroon","Canada",
    "Central African Republic","Chad","Chile","China","Colombia","Costa Rica",
    "Croatia","Cuba","Czech Republic","Democratic Republic of the Congo",
    "Denmark","Dominican Republic","Ecuador","Egypt","El Salvador","Estonia",
    "Ethiopia","Finland","France","Gabon","Germany","Ghana","Greece","Greenland",
    "Guatemala","Guinea","Haiti","Honduras","Hungary","Iceland","India",
    "Indonesia","Iran","Iraq","Ireland","Israel","Italy","Jamaica","Japan",
    "Jordan","Kazakhstan","Kenya","Kuwait","Kyrgyzstan","Laos","Latvia",
    "Lebanon","Lesotho","Libya","Lithuania","Luxembourg","Madagascar","Malawi",
    "Malaysia","Mali","Mauritania","Mexico","Moldova","Mongolia","Montenegro",
    "Morocco","Mozambique","Myanmar","Namibia","Nepal","Netherlands",
    "New Zealand","Nicaragua","Niger","Nigeria","North Korea","North Macedonia",
    "Norway","Oman","Pakistan","Panama","Papua New Guinea","Paraguay","Peru",
    "Philippines","Poland","Portugal","Qatar","Republic of the Congo","Romania",
    "Russia","Rwanda","Saudi Arabia","Senegal","Serbia","Sierra Leone",
    "Slovakia","Slovenia","Somalia","South Africa","South Korea","South Sudan",
    "Spain","Sri Lanka","Sudan","Sweden","Switzerland","Syria","Tajikistan",
    "Tanzania","Thailand","Togo","Tunisia","Turkey","Turkmenistan","Uganda",
    "Ukraine","United Arab Emirates","United Kingdom","United States of America",
    "Uruguay","Uzbekistan","Venezuela","Vietnam","Yemen","Zambia","Zimbabwe",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Page config & styles
# ═══════════════════════════════════════════════════════════════════════════════
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
    font-size: 10px; font-weight: 700; letter-spacing: 1.8px;
    text-transform: uppercase; color: #4a6fa5;
    margin-top: 1.4rem; margin-bottom: 0.2rem;
    padding-bottom: 4px; border-bottom: 1px solid #1c2a3f;
}
</style>
""", unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def great_circle_pyproj(lon0, lat0, lon1, lat1, npts=200):
    lon0 = ((lon0+180)%360)-180
    lon1 = ((lon1+180)%360)-180
    pts  = GEOD.npts(lon0, lat0, lon1, lat1, npts)
    lons = [lon0] + [p[0] for p in pts] + [lon1]
    lats = [lat0] + [p[1] for p in pts] + [lat1]
    return lons, lats

def great_circle_slerp(start, end, num_points=400):
    slat, slon = np.deg2rad(start)
    elat, elon = np.deg2rad(end)
    dl = elon - slon
    d  = np.arccos(np.clip(
        np.sin(slat)*np.sin(elat) + np.cos(slat)*np.cos(elat)*np.cos(dl), -1, 1))
    lats, lons = [], []
    for i in range(num_points):
        f = i / (num_points - 1)
        if d < 1e-10:
            lats.append(np.rad2deg(slat)); lons.append(np.rad2deg(slon)); continue
        A = np.sin((1-f)*d)/np.sin(d); B = np.sin(f*d)/np.sin(d)
        x = A*np.cos(slat)*np.cos(slon) + B*np.cos(elat)*np.cos(elon)
        y = A*np.cos(slat)*np.sin(slon) + B*np.cos(elat)*np.sin(elon)
        z = A*np.sin(slat) + B*np.sin(elat)
        lats.append(np.rad2deg(np.arctan2(z, np.sqrt(x**2+y**2))))
        lons.append(np.rad2deg(np.arctan2(y, x)))
    return lats, lons


# ═══════════════════════════════════════════════════════════════════════════════
# Standard mode (Cartopy shapefiles)
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

def draw_standard(routes_df, airports_df, cfg):
    routes   = load_and_merge(routes_df, airports_df)
    airports = get_unique_airports(routes)

    proj = ccrs.PlateCarree(central_longitude=cfg["center_lon"])
    fig, ax = plt.subplots(figsize=(cfg["fig_w"], cfg["fig_h"]), dpi=cfg["dpi"],
                            subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_global()

    excluded = set(cfg["excluded_countries"])
    cshp = shpreader.natural_earth(resolution="50m", category="cultural",
                                    name="admin_0_countries")
    for rec in shpreader.Reader(cshp).records():
        name  = rec.attributes.get("NAME","") or rec.attributes.get("ADMIN","")
        color = "white" if name in excluded else cfg["land_color"]
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                          facecolor=color, edgecolor="none", zorder=2)

    if cfg["great_lakes"]:
        lshp = shpreader.natural_earth(resolution="50m", category="physical", name="lakes")
        GL   = {"Lake Superior","Lake Michigan","Lake Huron","Lake Erie","Lake Ontario"}
        for rec in shpreader.Reader(lshp).records():
            if rec.attributes.get("name","") in GL:
                ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                                  facecolor="white", edgecolor="none", zorder=2.5)

    if cfg["show_borders"]:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                       edgecolor=cfg["border_color"], linewidth=cfg["border_w"],
                       alpha=0.8, zorder=3.7)
    if cfg["show_states"]:
        ax.add_feature(cfeature.STATES.with_scale("50m"),
                       edgecolor=cfg["state_color"], linewidth=cfg["state_w"],
                       alpha=0.7, zorder=3.6)

    valid    = set(cfg["regions"])
    filtered = routes[routes["region_orig"].isin(valid) & routes["region_dest"].isin(valid)]
    filtered = filtered.sort_values("zorder", ascending=True)
    for _, row in filtered.iterrows():
        color = str(row.get("color", cfg["route_color"])) if cfg["use_csv_colors"] else cfg["route_color"]
        lw    = float(row.get("linewidth", cfg["route_lw"]))
        style = "-" if str(row.get("style","solid"))=="solid" else "--"
        lons, lats = great_circle_pyproj(row["lon_orig"],row["lat_orig"],
                                          row["lon_dest"],row["lat_dest"])
        ax.plot(lons, lats, transform=ccrs.Geodetic(), color=color,
                linewidth=lw, linestyle=style, alpha=cfg["route_alpha"], zorder=float(row.get("zorder",4)))

    shadow  = [PathEffects.withStroke(linewidth=2.5, foreground="white", alpha=0.8)]
    fa      = {c:i for c,i in airports.items() if i.get("region") in valid}
    col_reg = [r for r in ["Mexico","Central","Caribbean"] if r in valid]
    col_par = {"Mexico":(-135,33,-2.5,1.8),"Central":(-106,14,-2.5,1.8),"Caribbean":(-61,35,-2.5,2.2)}
    col_apt = {r:sorted([i for i in fa.values() if i["region"]==r],key=lambda x:x["lat"],reverse=True)
               for r in col_reg}
    all_lbl = {}; texts = []; pts = []

    for region in col_reg:
        sl,slat,lsp,lsl = col_par[region]
        for i,info in enumerate(col_apt[region]):
            ax.plot(info["lon"],info["lat"],"o",markersize=info["dot_size"],
                    color=cfg["dot_color"],transform=ccrs.PlateCarree(),alpha=0.8,zorder=6)
            if info["label_size"]>0:
                txt=ax.text(sl+i*lsl,slat+i*lsp,info["code"],fontsize=info["label_size"],
                            fontweight="bold",color=cfg["label_color"],
                            transform=ccrs.PlateCarree(),ha="center",va="center",
                            zorder=7,path_effects=shadow)
                all_lbl[txt]=(info["lon"],info["lat"])

    for code,info in fa.items():
        if info["region"] in col_reg: continue
        lon,lat=info["lon"],info["lat"]
        pt=ax.plot(lon,lat,"o",markersize=info["dot_size"],color=cfg["dot_color"],
                   transform=ccrs.PlateCarree(),alpha=0.8,zorder=5)
        if info["label_size"]>0:
            txt=ax.text(lon,lat,code,fontsize=info["label_size"],fontweight="bold",
                        color=cfg["label_color"],transform=ccrs.PlateCarree(),
                        ha="left" if lon<-20 else "right",va="bottom",
                        zorder=7,path_effects=shadow)
            texts.append(txt); pts.append(pt[0]); all_lbl[txt]=(lon,lat)

    if texts and ADJUSTTEXT_OK and not cfg["is_preview"]:
        adjust_text(texts,add_objects=pts,ax=ax,expand_points=(2,3),expand_text=(2.5,3),
                    force_points=2.5,force_text=3,add_step_breaks=True,
                    only_move={"points":"xy","text":"xy"},arrowprops=None)

    for txt,(ox,oy) in all_lbl.items():
        tx,ty=txt.get_position()
        ax.plot([ox,tx],[oy,ty],color="lightgray",linewidth=0.4,alpha=0.8,zorder=6,
                transform=ccrs.PlateCarree(),
                path_effects=[PathEffects.withStroke(linewidth=0.8,foreground="white",alpha=0.8)])

    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    buf=io.BytesIO()
    fig.savefig(buf,format=cfg["fmt"].lower(),bbox_inches="tight",facecolor="white",dpi=cfg["dpi"])
    buf.seek(0); plt.close(fig); return buf


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
    routes  = load_and_merge(routes_df, airports_df)
    airports = get_unique_airports(routes)

    nrows,ncols = dem_data.shape
    extent = [-180,180,lat_arr.min(),lat_arr.max()]
    proj   = ccrs.PlateCarree(central_longitude=cfg["center_lon"])
    fig,ax = plt.subplots(figsize=(cfg["fig_w"],cfg["fig_h"]),dpi=cfg["dpi"],
                           subplot_kw={"projection":proj})
    fig.patch.set_facecolor("white"); ax.set_global()

    ax.imshow(np.ones_like(dem_data),extent=extent,transform=ccrs.PlateCarree(),
              cmap=ListedColormap(["white"]),origin="upper",zorder=0)

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

    if cfg["show_borders"]:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),edgecolor=cfg["border_color"],
                       linewidth=cfg["border_w"],alpha=0.8,zorder=3)
    if cfg["show_states"]:
        ax.add_feature(cfeature.STATES.with_scale("50m"),edgecolor=cfg["state_color"],
                       linewidth=cfg["state_w"],alpha=0.7,zorder=3)

    # Routes — same merged format as standard mode
    valid    = set(cfg["regions"])
    filtered = routes[routes["region_orig"].isin(valid) & routes["region_dest"].isin(valid)]
    filtered = filtered.sort_values("zorder", ascending=True)

    for _, row in filtered.iterrows():
        color = str(row.get("color", cfg["route_color"])) if cfg.get("use_csv_colors", True) else cfg["route_color"]
        lw    = float(row.get("linewidth", cfg["route_lw"]))
        style = "-" if str(row.get("style","solid"))=="solid" else "--"
        lats,lons = great_circle_slerp(
            (row["lat_orig"], row["lon_orig"]),
            (row["lat_dest"], row["lon_dest"])
        )
        ax.plot(lons,lats,transform=ccrs.Geodetic(),color=color,linewidth=lw,
                linestyle=style,alpha=cfg["route_alpha"],zorder=4)

    # Airport dots + labels
    shadow  = [PathEffects.withStroke(linewidth=2.5, foreground="white", alpha=0.8)]
    fa      = {c:i for c,i in airports.items() if i.get("region") in valid}
    labeled = {}
    for code, info in fa.items():
        lon, lat = info["lon"], info["lat"]
        ax.scatter(lon, lat, transform=ccrs.PlateCarree(),
                   s=info["dot_size"]*10, color=cfg["dot_color"], zorder=5)
        if info["label_size"] > 0 and (lat,lon) not in labeled:
            ax.text(lon, lat+cfg["label_offset"], code,
                    transform=ccrs.PlateCarree(), fontsize=info["label_size"],
                    color=cfg["label_color"], fontweight="bold",
                    zorder=6, ha="center", va="bottom", path_effects=shadow)
            labeled[(lat,lon)] = code

    ax.set_axis_off(); plt.tight_layout(pad=0)
    buf=io.BytesIO()
    fig.savefig(buf,format=cfg["fmt"].lower(),bbox_inches="tight",facecolor="white",dpi=cfg["dpi"])
    buf.seek(0); plt.close(fig); return buf


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — mode toggle at the top, then conditional settings
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✈️ Flight Map")
    mode = st.radio("Mode", ["Standard", "Terrain (ETOPO)"],
                    horizontal=True,
                    help="Standard works on Streamlit Cloud. Terrain requires local ETOPO1 file + rasterio.")
    st.markdown("---")

    # ── Data ──────────────────────────────────────────────────────────────────
    section("Data")
    if mode == "Standard":
        routes_file = st.file_uploader("Routes CSV", type="csv")
    else:
        routes_file_t = st.file_uploader("Routes CSV", type="csv", key="routes_t")
        st.caption("🌍 ETOPO1 terrain file downloads automatically from Google Drive on first use.")

    # ── Map layout ─────────────────────────────────────────────────────────────
    section("Map Layout")
    center_lon = st.slider("Center longitude", -180, 180, -90, 5)
    c1,c2 = st.columns(2)
    fig_w = c1.number_input("Width (in)",  8, 36, 18)
    fig_h = c2.number_input("Height (in)", 4, 20,  9)
    export_dpi = st.select_slider("Export DPI", [150,300,600], value=300,
                                   help="Preview always uses 72 DPI")

    if mode == "Terrain (ETOPO)":
        downsample = st.slider("Terrain downsample", 4, 16, 8, 2,
                                help="Higher = faster, less detail")

    # ── Regions (standard only) ───────────────────────────────────────────────
    if mode == "Standard":
        section("Regions")
        all_regions = ["Domestic","Pacific","Atlantic","South","Mexico","Central","Caribbean","Other"]
        regions = st.multiselect("Include", all_regions,
                                  default=["Domestic","Pacific","Atlantic","South",
                                           "Mexico","Central","Caribbean"])

    if mode == "Terrain (ETOPO)":
        section("Regions")
        all_regions_t = ["Domestic","Pacific","Atlantic","South","Mexico","Central","Caribbean","Other"]
        regions_t = st.multiselect("Include", all_regions_t,
                                    default=["Domestic","Pacific","Atlantic","South",
                                             "Mexico","Central","Caribbean"],
                                    key="regions_t")

    # ── Terrain options ───────────────────────────────────────────────────────
    if mode == "Terrain (ETOPO)":
        section("Terrain")
        show_glow       = st.checkbox("Coastal glow", True)
        glow_factor     = st.slider("Glow spread", 10, 100, 45, 5)
        show_mountains  = st.checkbox("Mountain shading", True)
        mountain_thresh = st.slider("Mountain threshold (m)", 100, 2000, 500, 50)

    # ── Routes ────────────────────────────────────────────────────────────────
    section("Routes")
    if mode == "Standard":
        use_csv_colors = st.checkbox("Use colors from CSV", True)
    route_color = st.color_picker("Default color", "#005DAA")
    c3,c4 = st.columns(2)
    route_lw    = c3.slider("Line width", 0.1, 3.0, 1.0, 0.1)
    route_alpha = c4.slider("Opacity",    0.1, 1.0, 1.0, 0.05)

    # ── Airports ──────────────────────────────────────────────────────────────
    section("Airports")
    c5,c6 = st.columns(2)
    dot_color   = c5.color_picker("Dot",   "#000000")
    label_color = c6.color_picker("Label", "#000000")
    if mode == "Terrain (ETOPO)":
        label_offset = st.slider("Label vertical offset", 0.0, 5.0, 0.3, 0.1)

    # ── Map style ─────────────────────────────────────────────────────────────
    section("Map Style")
    land_color = st.color_picker("Land color", "#bfbfbf")
    c7,c8 = st.columns(2)
    show_borders = c7.checkbox("Country borders", True)
    show_states  = c8.checkbox("State borders",   True)
    if mode == "Standard":
        great_lakes = st.checkbox("Great Lakes", True)
    border_color = st.color_picker("Border color", "#ffffff")
    c9,c10 = st.columns(2)
    border_w    = c9.slider("Border width",       0.1, 2.0, 0.5, 0.1)
    state_color = c10.color_picker("State color", "#ffffff")
    state_w     = st.slider("State border width", 0.1, 2.0, 0.3, 0.1)

    if mode == "Standard":
        section("Exclude Countries")
        excluded_countries = st.multiselect(
            "Show as ocean",
            options=ALL_COUNTRIES,
            default=[],
            help="These countries will appear white instead of land color"
        )

    # ── Output ────────────────────────────────────────────────────────────────
    section("Output")
    fmt = st.selectbox("Format", ["PNG","PDF","SVG"])


# ═══════════════════════════════════════════════════════════════════════════════
# Main panel
# ═══════════════════════════════════════════════════════════════════════════════
st.title("✈️ Flight Map Generator")

def render_and_show(draw_fn, *args, base_cfg):
    # Live preview
    with st.spinner("Updating preview…"):
        try:
            prev = draw_fn(*args, {**base_cfg, "dpi":72, "fmt":"PNG", "is_preview":True})
            st.image(prev, use_container_width=True,
                     caption="Live preview (72 DPI) — use Export for full resolution")
        except Exception as e:
            st.error(f"Preview error: {e}"); st.exception(e); return

    # Export
    col_btn, col_dl = st.columns([1,3])
    if col_btn.button("⬇️ Export full-res", type="primary", use_container_width=True):
        with st.spinner(f"Rendering at {export_dpi} DPI…"):
            try:
                exp = draw_fn(*args, {**base_cfg, "dpi":export_dpi, "fmt":fmt, "is_preview":False})
                mime = {"PNG":"image/png","PDF":"application/pdf","SVG":"image/svg+xml"}[fmt]
                col_dl.download_button(f"Download {fmt}", exp,
                                        file_name=f"flight_map.{fmt.lower()}",
                                        mime=mime, use_container_width=True)
            except Exception as e:
                st.error(f"Export error: {e}"); st.exception(e)


# ── Standard mode ─────────────────────────────────────────────────────────────
if mode == "Standard":
    st.caption("Upload your routes CSV — preview updates automatically as you change settings.")

    airports_df = load_airports()
    routes_df   = None
    if routes_file: routes_df = pd.read_csv(routes_file)

    if routes_df is not None:
        with st.expander("Preview data", expanded=False):
            p1,p2 = st.columns(2)
            p1.markdown(f"**Routes** — {len(routes_df)} rows")
            p1.dataframe(routes_df.head(8), use_container_width=True)
            p2.markdown(f"**Airports** — {len(airports_df)} rows")
            p2.dataframe(airports_df.head(8), use_container_width=True)

        if not regions:
            st.warning("Select at least one region in the sidebar.")
        else:
            base = dict(center_lon=center_lon, fig_w=fig_w, fig_h=fig_h,
                        regions=regions, use_csv_colors=use_csv_colors,
                        route_color=route_color, route_lw=route_lw, route_alpha=route_alpha,
                        dot_color=dot_color, label_color=label_color, land_color=land_color,
                        show_borders=show_borders, show_states=show_states,
                        great_lakes=great_lakes, border_color=border_color, border_w=border_w,
                        state_color=state_color, state_w=state_w,
                        excluded_countries=excluded_countries)
            render_and_show(draw_standard, routes_df, airports_df, base_cfg=base)
    else:
        st.info("Upload a routes CSV in the sidebar to get started.")


# ── Terrain mode ──────────────────────────────────────────────────────────────
else:
    st.caption("ETOPO1 terrain file downloads automatically from Google Drive on first use (~1 GB, cached for the session).")

    if not RASTERIO_OK:
        st.error("`rasterio` and `scipy` are not installed in this environment.")
        st.stop()

    routes_df_t = None
    airports_df_t = load_airports()
    if routes_file_t: routes_df_t = pd.read_csv(routes_file_t)

    etopo_path = ensure_etopo()

    if etopo_path is None:
        st.error("Failed to download ETOPO1 terrain file from Google Drive. Check that the file is publicly shared.")
    elif routes_df_t is not None:
        with st.expander("Preview data", expanded=False):
            p1,p2 = st.columns(2)
            p1.markdown(f"**Routes** — {len(routes_df_t)} rows")
            p1.dataframe(routes_df_t.head(8), use_container_width=True)
            p2.markdown(f"**Airports** — {len(airports_df_t)} rows")
            p2.dataframe(airports_df_t.head(8), use_container_width=True)

        dem_data, lat_arr, land_mask, ocean_mask, _ = load_dem(etopo_path, downsample)

        base = dict(center_lon=center_lon, fig_w=fig_w, fig_h=fig_h,
                    regions=regions_t, use_csv_colors=True,
                    show_glow=show_glow, glow_factor=glow_factor,
                    show_mountains=show_mountains, mountain_thresh=mountain_thresh,
                    land_color=land_color, show_borders=show_borders, show_states=show_states,
                    border_color=border_color, border_w=border_w,
                    state_color=state_color, state_w=state_w,
                    route_color=route_color, route_lw=route_lw, route_alpha=route_alpha,
                    dot_color=dot_color, label_color=label_color, label_offset=label_offset)

        render_and_show(draw_terrain, routes_df_t, airports_df_t,
                        dem_data, lat_arr, land_mask, ocean_mask, base_cfg=base)
    else:
        st.info("Upload both CSVs in the sidebar to get started.")
