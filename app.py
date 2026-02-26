# app.py
# AI-Powered Coral Health Dashboard – Florida Pilot
# Presentation-ready UI upgrade (ocean glass theme + premium header + polished Plotly)
# Paste this entire file as your new app.py

from pathlib import Path
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# Optional raster / NetCDF helpers
try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import xarray as xr
except ImportError:
    xr = None


# ===================== STREAMLIT PAGE CONFIG =====================
st.set_page_config(
    page_title="AI-Powered Coral Health Dashboard – Florida Pilot",
    layout="wide",
    page_icon="🌊",
)

# ===================== PATHS =====================
ROOT = Path(__file__).parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"

PATHS = {
    "crw_sst": DATA / "noaa_crw" / "crw_sst_20240101.tif",
    "aoml_chl": DATA / "noaa_aoml" / "aoml_chlorophyll_20240101.tif",
    "cremp_cover": DATA / "cremp" / "cremp_coral_cover_latest.tif",
    "cremp_monitor": DATA / "cremp" / "cremp_monitoring_data.csv",
    "integrated_raster": DATA / "integrated" / "coral_reef_integrated.tif",
    "ml_sample": DATA / "integrated" / "ml_ready_data_sample.csv",
    "metadata": DATA / "integrated" / "metadata.json",
}

# ===================== UI: BACKGROUND IMAGE INJECTION =====================
def set_background_image(image_path: Path):
    """Inject a full-page background image into the Streamlit app."""
    if not image_path.exists():
        # Keep this calm for sponsor demos
        st.info(f"Background image not found: {image_path.name}. (Add it in /assets)")
        return

    img_bytes = image_path.read_bytes()
    encoded = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background: linear-gradient(
                to bottom,
                rgba(0, 10, 30, 0.70),
                rgba(0, 20, 40, 0.88)
            );
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ===================== UI: PREMIUM THEME (INLINE CSS) =====================
THEME_CSS = """
<style>
/* ===== Base page styling ===== */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #042f4b 0%, #010816 55%, #000000 100%);
    color: #f6f7fb;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* remove extra whitespace */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(6,16,32,0.92) 0%, rgba(2,6,18,0.95) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.10);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #f6f7fb !important;
}

/* ===== Glass container wrapping the whole page ===== */
.glass {
  background: rgba(3, 10, 25, 0.70);
  border: 1px solid rgba(160, 211, 255, 0.18);
  border-radius: 18px;
  padding: 18px 18px;
  backdrop-filter: blur(10px);
}

/* ===== Top strip once logged in ===== */
.top-strip {
    width: 100%;
    padding: 0.40rem 0.95rem;
    margin: 0 0 0.75rem 0;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(15, 23, 42, 0.92), rgba(56, 189, 248, 0.18));
    font-size: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #e5f4ff;
    border: 1px solid rgba(255,255,255,0.10);
}
.top-strip-right { opacity: 0.85; font-size: 0.82rem; }

/* ===== Hero header ===== */
.app-hero{
  display:flex;
  justify-content:space-between;
  gap:14px;
  padding:16px 18px;
  border-radius:18px;
  border:1px solid rgba(255,255,255,0.12);
  background: radial-gradient(circle at top left, rgba(0, 209, 255, 0.25), transparent 55%),
              radial-gradient(circle at bottom right, rgba(111, 66, 193, 0.22), transparent 60%),
              rgba(6, 16, 32, 0.75);
  margin-bottom: 14px;
}
.hero-pill{
  display:inline-flex;
  padding:4px 10px;
  border-radius:999px;
  font-size:12px;
  letter-spacing:0.08em;
  text-transform:uppercase;
  border:1px solid rgba(255,255,255,0.18);
  background: rgba(0,0,0,0.25);
  color:#c5d5ff;
}
.hero-title{
  font-size:26px;
  font-weight:850;
  margin-top:8px;
  color:#ffffff;
  text-shadow: 0 0 22px rgba(0,214,255,0.55);
}
.hero-sub{
  margin-top:6px;
  color:#cfd5ff;
  font-size:13px;
}
.hero-right{
  display:flex;
  gap:10px;
  align-items:flex-end;
}
.hero-kpi{
  background: rgba(0,0,0,0.22);
  border:1px solid rgba(255,255,255,0.12);
  border-radius:14px;
  padding:10px 12px;
  min-width:92px;
  text-align:center;
}
.kpi-num{
  font-size:16px;
  font-weight:850;
  color:#ffffff;
}
.kpi-lab{
  font-size:11px;
  color:rgba(255,255,255,0.72);
}

/* ===== Panels/Cards ===== */
.panel {
  background: rgba(3, 10, 25, 0.68);
  border: 1px solid rgba(160, 211, 255, 0.16);
  border-radius: 16px;
  padding: 14px 14px;
  margin-bottom: 12px;
}
.card{
  background: rgba(3, 10, 25, 0.62);
  border: 1px solid rgba(160, 211, 255, 0.14);
  border-radius: 16px;
  padding: 14px 14px;
  margin-bottom: 12px;
}
.card-title{
  font-weight: 750;
  color:#ffffff;
  margin-bottom: 6px;
}
.card-body{
  color: rgba(255,255,255,0.82);
  font-size: 13px;
  line-height: 1.55;
}

/* ===== Headings: make them pop ===== */
[data-testid="stAppViewContainer"] .main h1 {
    font-size: 1.9rem;
    font-weight: 850;
    letter-spacing: 0.03em;
    color: #ffffff;
    text-shadow: 0 0 18px rgba(0, 214, 255, 0.55);
    margin-bottom: 0.35rem;
}
[data-testid="stAppViewContainer"] .main h2,
[data-testid="stAppViewContainer"] .main h3 {
    color: #f3f6ff;
    font-weight: 750;
}

/* ===== Tabs: subtle styling ===== */
.stTabs [data-baseweb="tab"] {
    font-weight: 650;
    color: rgba(255,255,255,0.82);
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
}

/* ===== Login hero + panels ===== */
.login-hero {
    max-width: 980px;
    margin: 1.2rem auto 0.8rem auto;
    padding: 1.5rem 2rem;
    border-radius: 18px;
    background: radial-gradient(circle at top left, rgba(0, 209, 255, 0.35), transparent 55%),
                radial-gradient(circle at bottom right, rgba(111, 66, 193, 0.32), transparent 60%),
                rgba(6, 16, 32, 0.92);
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.12);
}
.login-hero-pill {
    display: inline-flex;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.25);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #c5d5ff;
    margin-bottom: 0.45rem;
}
.login-hero-title {
    margin: 0;
    font-size: 2.0rem;
    font-weight: 850;
    letter-spacing: 0.02em;
    color: #ffffff;
    text-shadow: 0 0 18px rgba(0, 214, 255, 0.55);
}
.login-hero-subtitle {
    margin-top: 0.55rem;
    max-width: 720px;
    color: #cfd5ff;
    font-size: 0.95rem;
}

.login-panel, .team-panel {
    margin-top: 1.2rem;
    padding: 1.2rem 1.3rem 1.1rem;
    border-radius: 16px;
    background: rgba(3, 10, 25, 0.88);
    border: 1px solid rgba(160, 211, 255, 0.22);
    box-shadow: 0 16px 50px rgba(0, 0, 0, 0.55);
}
.login-panel label { color: #e7ecff !important; font-weight: 600; }
.login-panel input[type="text"], .login-panel input[type="password"] {
    background-color: rgba(5, 18, 38, 0.95) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(164, 208, 255, 0.45) !important;
    color: #f7fbff !important;
}
.login-panel button[kind="primary"] {
    border-radius: 999px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #00d4ff, #3b82f6) !important;
    border: none !important;
}
.login-hint { margin-top: 0.7rem; font-size: 0.82rem; color: #bcd4ff; }

.team-subtitle { font-size: 0.88rem; color: #cdd8ff; margin-bottom: 0.65rem; }
.team-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.45rem 0.45rem;
    border-radius: 12px;
    transition: background 0.18s ease, transform 0.18s ease;
}
.team-row:hover { background: rgba(42, 112, 255, 0.16); transform: translateY(-1px); }
.team-name { font-size: 0.95rem; font-weight: 750; color: #ffffff; }
.team-role { font-size: 0.80rem; color: #a0b5ff; }

/* Reduce noisy Streamlit default borders */
[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 10px 12px; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# Apply ocean background (put coral_bg.jpg into ./assets)
set_background_image(ASSETS / "coral_bg.jpg")


# ===================== HELPERS =====================
def normalize_to_01(arr: np.ndarray):
    arr = arr.astype("float32")
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if vmax - vmin == 0:
        return np.zeros_like(arr)
    norm = (arr - vmin) / (vmax - vmin + 1e-8)
    return np.nan_to_num(norm)


def polish_plotly(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        title=dict(font=dict(size=18)),
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def load_tif(path: Path):
    if rasterio is None:
        st.info("Raster layer requires rasterio. Install: pip install rasterio")
        return None, None
    if not path.exists():
        st.info(f"Layer unavailable: {path.name}. Waiting for final export.")
        return None, None
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile
    return arr, profile


def plot_raster_heatmap(arr: np.ndarray, title: str, units: str = ""):
    if arr is None:
        st.info("No raster loaded for this layer yet.")
        return
    norm = normalize_to_01(arr)
    fig = px.imshow(norm, color_continuous_scale="Turbo", origin="upper", aspect="auto")
    fig.update_layout(title=title, coloraxis_colorbar=dict(title=units), margin=dict(l=10, r=10, t=40, b=10))
    fig = polish_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)


def safe_read_csv(path: Path):
    if not path.exists():
        st.info(f"Table unavailable: {path.name}. Waiting for final export.")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.info(f"Could not read {path.name}: {e}")
        return None


def metric_row(cols, labels, values, deltas=None):
    deltas = deltas or [""] * len(labels)
    for col, label, val, delt in zip(cols, labels, values, deltas):
        col.metric(label, val, delt)


# ===================== SECTIONS =====================
def section_overview():
    st.markdown("""
    <div class="card">
      <div class="card-title">Pilot Overview</div>
      <div class="card-body">
        This dashboard integrates satellite and in-situ layers for the Florida Keys pilot to support
        rapid spatial assessment of coral condition, ocean chemistry drivers, and readiness for ML + policy scoring.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Data coverage (connected sources)")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    n_sst = len(list((DATA / "noaa_crw").glob("crw_sst_*.tif"))) if (DATA / "noaa_crw").exists() else 0
    n_aoml = len(list((DATA / "noaa_aoml").glob("*.tif"))) if (DATA / "noaa_aoml").exists() else 0
    n_cremp = 1 if PATHS["cremp_monitor"].exists() else 0

    metric_row(
        (col1, col2, col3),
        ["NOAA CRW SST Scenes", "AOML Chemistry Layers", "CREMP Survey Tables"],
        [n_sst, n_aoml, n_cremp],
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Live preview")
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🧭 Sea Surface Temperature hotspots")
        st.caption("NOAA Coral Reef Watch sample — Jan 1, 2024 (normalized preview).")
        sst, _ = load_tif(PATHS["crw_sst"])
        plot_raster_heatmap(sst, "SST – Florida Keys window", "normalized")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### 🧠 ML-ready feature snapshot")
        ml_df = safe_read_csv(PATHS["ml_sample"])
        if ml_df is not None:
            st.dataframe(ml_df.head(8), use_container_width=True)
            num_cols = ml_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                fig = px.scatter(
                    ml_df,
                    x=num_cols[0],
                    y=num_cols[1],
                    color=num_cols[2] if len(num_cols) > 2 else None,
                    title=f"{num_cols[0]} vs {num_cols[1]}",
                )
                fig = polish_plotly(fig)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def section_coral_health():
    st.markdown("### 🪸 Coral health & benthic structure")

    tab1, tab2, tab3 = st.tabs(["Integrated Health Layer", "Allen Benthic Map", "CREMP Monitoring Table"])

    with tab1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        arr, _ = load_tif(PATHS["integrated_raster"])
        plot_raster_heatmap(arr, "Integrated Coral Health Index", "index")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        arr, _ = load_tif(DATA / "allen_coral" / "allen_benthic_map.tif")
        plot_raster_heatmap(arr, "Allen Coral Atlas – Benthic Habitat", "class")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        df = safe_read_csv(PATHS["cremp_monitor"])
        if df is not None:
            st.dataframe(df.head(25), use_container_width=True)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                colname = num_cols[0]
                fig = px.histogram(df, x=colname, nbins=30, title=f"Distribution: {colname}")
                fig = polish_plotly(fig)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def section_ocean_chemistry():
    st.markdown("### 🌡️ Ocean chemistry – temperature & chlorophyll")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Sea Surface Temperature (NOAA CRW)")
        arr, _ = load_tif(PATHS["crw_sst"])
        plot_raster_heatmap(arr, "SST – Jan 1, 2024", "normalized")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("#### Chlorophyll (NOAA AOML)")
        arr, _ = load_tif(PATHS["aoml_chl"])
        plot_raster_heatmap(arr, "Chlorophyll – Jan 1, 2024", "normalized")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Time-series view (if NetCDF exists)")
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    nc_path = DATA / "noaa_crw" / "processed_crw_data.nc"
    if xr is not None and nc_path.exists():
        try:
            ds = xr.open_dataset(nc_path)
            st.caption("Auto-detected variables: " + ", ".join(list(ds.data_vars)[:8]))
            var_name = list(ds.data_vars)[0]
            da = ds[var_name]

            dims = list(da.dims)
            time_dim = next((d for d in dims if "time" in d.lower()), None)

            if time_dim is not None:
                ts = da.mean([d for d in dims if d != time_dim]).to_series().dropna()
                fig = px.line(ts, labels={"value": var_name, "index": "Time"}, title=f"{var_name} (spatial mean)")
                fig = polish_plotly(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time dimension detected in this NetCDF.")
        except Exception as e:
            st.info(f"Could not read NetCDF: {e}")
    else:
        st.info("NetCDF not available yet (or xarray not installed).")

    st.markdown("</div>", unsafe_allow_html=True)


def section_policy_insights():
    st.markdown("### 📑 Policy insights (placeholder)")

    st.markdown("""
    <div class="card">
      <div class="card-title">Planned integration</div>
      <div class="card-body">
        This panel will connect policy documents and enforcement metrics to spatial reef outcomes using NLP + ML scoring.
        For the sponsor demo, we show a mock scorecard to demonstrate the final UX.
      </div>
    </div>
    """, unsafe_allow_html=True)

    policy_df = pd.DataFrame(
        {
            "Policy": ["Marine Protected Area", "Fishing Regulation", "Water Quality Act", "Tourism Control"],
            "Effectiveness": [0.82, 0.73, 0.65, 0.59],
            "Compliance": [0.90, 0.70, 0.60, 0.55],
        }
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    fig = px.bar(policy_df, x="Policy", y="Effectiveness", color="Compliance", range_y=[0, 1],
                 title="Policy effectiveness (demo visualization)")
    fig = polish_plotly(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def section_map_explorer():
    st.markdown("### 🗺️ Map explorer – compare layers")

    layer = st.selectbox(
        "Choose a layer",
        [
            "Integrated Coral Health Layer",
            "CRW – SST (Jan 1, 2024)",
            "AOML – Chlorophyll (Jan 1, 2024)",
            "Allen Coral – Benthic Map",
            "Florida Keys – Bathymetry",
            "Florida Keys – Habitat",
        ],
    )

    if layer.startswith("CRW"):
        path = PATHS["crw_sst"]; title = "CRW Sea Surface Temperature"; units = "normalized"
    elif layer.startswith("AOML"):
        path = PATHS["aoml_chl"]; title = "AOML Chlorophyll"; units = "normalized"
    elif "Benthic" in layer:
        path = DATA / "allen_coral" / "allen_benthic_map.tif"; title = "Allen Coral Benthic Map"; units = "class"
    elif "Bathymetry" in layer:
        path = DATA / "florida_keys" / "florida_keys_bathymetry.tif"; title = "Florida Keys Bathymetry"; units = "depth"
    elif "Habitat" in layer:
        path = DATA / "florida_keys" / "florida_keys_habitat.tif"; title = "Florida Keys Habitat"; units = "class"
    else:
        path = PATHS["integrated_raster"]; title = "Integrated Coral Health Index"; units = "index"

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    arr, _ = load_tif(path)
    plot_raster_heatmap(arr, title, units)
    st.caption("Next: replace this heatmap view with an interactive map (Leaflet/Mapbox) + ML overlay layer.")
    st.markdown("</div>", unsafe_allow_html=True)


# ===================== LOGIN & ROUTING =====================
ACCESS_CODE = "FLORIDA2026"

TEAM = [
    {"name": "Anitha", "role": "Data Pipeline Engineer"},
    {"name": "Nikita", "role": "ML Model Engineer"},
    {"name": "Manoj", "role": "Frontend Engineer"},
    {"name": "Meet", "role": "Backend Engineer"},
]


def show_login():
    st.markdown(
        """
        <div class="login-hero">
            <div class="login-hero-pill">Florida Pilot • Internal Prototype</div>
            <h1 class="login-hero-title">🌊 AI-Powered Coral Health Dashboard</h1>
            <p class="login-hero-subtitle">
                Secure access for the project team monitoring coral reef health,
                ocean chemistry, and policy readiness in the Florida Keys.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_form, col_team = st.columns([1.35, 1])

    with col_form:
        st.markdown('<div class="login-panel">', unsafe_allow_html=True)

        st.markdown("#### Sign in to continue")
        name = st.text_input("Name (for display only)", key="login_name")
        code = st.text_input("Access code", type="password", key="login_code")

        login_clicked = st.button("Enter dashboard", type="primary", use_container_width=True)

        if login_clicked:
            if code == ACCESS_CODE:
                st.session_state["logged_in"] = True
                st.session_state["user_name"] = name or "Guest"
                st.success("✅ Login successful – loading dashboard…")
                st.rerun()
            else:
                st.error("❌ Incorrect access code. Please try again.")

        st.markdown(
            """
            <p class="login-hint">
                Demo code: <code>FLORIDA2026</code>
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_team:
        st.markdown('<div class="team-panel">', unsafe_allow_html=True)
        st.markdown("#### Project team")
        st.markdown(
            """
            <p class="team-subtitle">
                Cross-functional team collaborating on AI-driven ocean health monitoring.
            </p>
            """,
            unsafe_allow_html=True,
        )

        for member in TEAM:
            st.markdown(
                f"""
                <div class="team-row">
                    <div class="team-text">
                        <div class="team-name">{member['name']}</div>
                        <div class="team-role">{member['role']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


def main():
    if not st.session_state["logged_in"]:
        show_login()
        return

    user = st.session_state.get("user_name", "Guest")

    st.markdown(
        f"""
        <div class="top-strip">
            <span>👋 Welcome, <b>{user}</b></span>
            <span class="top-strip-right">Florida Pilot • Live prototype</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-hero">
          <div class="hero-left">
            <div class="hero-pill">Florida Pilot • Coral Reef Health</div>
            <div class="hero-title">AI-Powered Coral Health Dashboard</div>
            <div class="hero-sub">Map-first monitoring of reef condition, ocean chemistry & risk signals</div>
          </div>
          <div class="hero-right">
            <div class="hero-kpi">
              <div class="kpi-num">LIVE</div>
              <div class="kpi-lab">Mode</div>
            </div>
            <div class="hero-kpi">
              <div class="kpi-num">FL Keys</div>
              <div class="kpi-lab">Region</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation (Atlas-like: map-first)
    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Section",
        [
            "Overview",
            "Coral Health",
            "Ocean Chemistry",
            "Map Explorer",
            "Policy Insights (Placeholder)",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Florida Pilot Project**  
        Coral reef health 🪸  
        Ocean chemistry 🌡️  
        Policy readiness 📑  
        """
    )

    # Wrap all content in glass panel for a premium look
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if section == "Overview":
        section_overview()
    elif section == "Coral Health":
        section_coral_health()
    elif section == "Ocean Chemistry":
        section_ocean_chemistry()
    elif section.startswith("Policy Insights"):
        section_policy_insights()
    elif section == "Map Explorer":
        section_map_explorer()

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.caption("AI Ocean Data | Florida Pilot | NOAA + AOML + CREMP + Integrated Layers")


if __name__ == "__main__":
    main()