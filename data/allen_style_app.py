
# allen_style_points_v1.py
# V1: Allen Coral Atlas–inspired, point-based Streamlit dashboard (Florida Keys)

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------- Page config ----------------------
st.set_page_config(
    page_title="Florida Keys Coral Explorer (V1)",
    page_icon="🗺️",
    layout="wide",
)

# ---------------------- CSS ----------------------
ATLAS_CSS = """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(15, 23, 42, 0.12); }
h1, h2, h3 { letter-spacing: 0.2px; }
.small-muted { color: rgba(15,23,42,0.72); font-size: 0.9rem; }

.atlas-card {
  background: #ffffff;
  border: 1px solid rgba(15, 23, 42, 0.10);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 6px 24px rgba(15, 23, 42, 0.04);
}
.kpi-grid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.kpi-tile{
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(248, 250, 252, 1);
}
.kpi-val{ font-size: 18px; font-weight: 750; color:#0f172a;}
.kpi-lab{ font-size: 12px; color: rgba(15,23,42,0.65); }
.atlas-note { font-size: 12px; color: rgba(15,23,42,0.62); margin-top: 8px; }
</style>
"""
st.markdown(ATLAS_CSS, unsafe_allow_html=True)

# ---------------------- Data loading ----------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT 

# ---------------------- CSV candidates ----------------------
CSV_CANDIDATES = [
    DATA_DIR / "cremp" / "cremp_monitoring_data.csv",
    DATA_DIR / "integrated" / "ml_ready_data_sample.csv",
    DATA_DIR / "florida_keys_coral_reef_data (1).csv",
    DATA_DIR / "florida_keys_coral_reef_data.csv",
]

csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)

st.title("Florida Keys Coral Explorer (V1)")
st.caption("Atlas-style • Point-based • Florida Keys pilot region")

if not csv_path:
    st.error("No CSV found. Put your CSV in project root or ./data/")
    st.info("Looked for: " + ", ".join([p.name for p in CSV_CANDIDATES]))
    st.stop()

@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

df_raw = load_df(csv_path)

# ---------------------- Helpers ----------------------
def find_col(df: pd.DataFrame, possible_names):
    possible = {x.lower().strip() for x in possible_names}
    for c in df.columns:
        if c.lower().strip() in possible:
            return c
    return None

lat_col = find_col(df_raw, {"latitude", "lat"})
lon_col = find_col(df_raw, {"longitude", "lon", "lng"})

if not lat_col or not lon_col:
    st.error("CSV must contain latitude and longitude columns.")
    st.write("Columns found:", list(df_raw.columns))
    st.stop()

# Optional label column
site_col = None
for name in ["site", "site_name", "reef", "reef_name", "station", "station_name", "location", "region"]:
    c = find_col(df_raw, {name})
    if c:
        site_col = c
        break

numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in [lat_col, lon_col]]

# Metric preference order
preferred = ["predicted_health_score", "health_score", "coral_cover_percent", "bleaching_percent", "species_diversity",
             "effectiveness_score", "compliance_rate"]
preferred_metrics = [c for c in numeric_cols if c.lower() in {x.lower() for x in preferred}]
default_metric = preferred_metrics[0] if preferred_metrics else (numeric_cols[0] if numeric_cols else lat_col)

def safe_stats(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanmean(s)), float(np.nanmin(s)), float(np.nanmax(s))

# ---------------------- Sidebar controls ----------------------
st.sidebar.markdown("### Layers & Filters")
st.sidebar.write(f"**Data file:** {csv_path.name}")

metric = st.sidebar.selectbox(
    "Map layer (color by)",
    options=[default_metric] + [c for c in numeric_cols if c != default_metric]
)

basemap = st.sidebar.selectbox("Basemap", ["Light (Atlas)", "Satellite", "Dark"])
opacity = st.sidebar.slider("Point opacity", 0.10, 1.00, 0.85, 0.05)
size = st.sidebar.slider("Point size", 5, 18, 10)

st.sidebar.markdown("### Quick filters")

df = df_raw.copy()

# Site filter
if site_col:
    sites = ["All"] + sorted(df[site_col].dropna().astype(str).unique().tolist())
    chosen_site = st.sidebar.selectbox("Site / Reef", sites)
    if chosen_site != "All":
        df = df[df[site_col].astype(str) == chosen_site]

# Metric range filter
if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
    lo, hi = float(df[metric].min()), float(df[metric].max())
    if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
        rng = st.sidebar.slider("Metric range", lo, hi, (lo, hi))
        df = df[(df[metric] >= rng[0]) & (df[metric] <= rng[1])]

# Hotspot definition (simple sponsor-friendly)
bleach_col = find_col(df, {"bleaching_percent", "bleaching"})
hotspot_threshold = st.sidebar.slider("Hotspot threshold (bleaching %)", 0.0, 50.0, 15.0, 0.5) if bleach_col else None

# ---------------------- Layout ----------------------
left, right = st.columns([2.35, 1])

# --- Map ---
with left:
    st.subheader("Map")
    st.caption("Use the sidebar to switch layers and filter the pilot region.")

    mapbox_style = "carto-positron"
    if basemap == "Satellite":
        mapbox_style = "satellite-streets"
    elif basemap == "Dark":
        mapbox_style = "carto-darkmatter"

    hover_cols = []
    if site_col:
        hover_cols.append(site_col)
    # Add key metrics for hover if present
    for c in ["health_score", "predicted_health_score", "coral_cover_percent", "bleaching_percent",
              "species_diversity", "effectiveness_score", "compliance_rate"]:
        cc = find_col(df, {c})
        if cc and cc not in hover_cols:
            hover_cols.append(cc)

    center = {"lat": float(df[lat_col].mean()), "lon": float(df[lon_col].mean())}

    fig = px.scatter_mapbox(
        df,
        lat=lat_col,
        lon=lon_col,
        color=metric if metric in df.columns else None,
        hover_name=site_col if site_col else None,
        hover_data=hover_cols if hover_cols else df.columns,
        zoom=6.8,
        opacity=opacity,
        height=680,
    )

    fig.update_traces(marker={"size": size})
    fig.update_layout(
        mapbox_style=mapbox_style,
        mapbox_center=center,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title=metric),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='atlas-note'>V1 output: map-driven exploration + KPIs + simple hotspot logic.</div>",
                unsafe_allow_html=True)

# --- Right panel KPIs + ranked outputs ---
with right:
    st.subheader("Outputs")
    st.markdown("<div class='atlas-card'>", unsafe_allow_html=True)

    st.markdown(f"**Pilot Summary**")
    st.markdown(f"<div class='small-muted'>{len(df):,} records in view • {csv_path.name}</div>",
                unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)

    # KPI 1: records
    st.markdown(
        f"<div class='kpi-tile'><div class='kpi-val'>{len(df):,}</div><div class='kpi-lab'>Records in view</div></div>",
        unsafe_allow_html=True
    )

    # KPI 2: metric mean
    if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
        m_mean, m_min, m_max = safe_stats(df[metric])
        st.markdown(
            f"<div class='kpi-tile'><div class='kpi-val'>{m_mean:.2f}</div><div class='kpi-lab'>Mean {metric}</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='kpi-tile'><div class='kpi-val'>{m_min:.2f} – {m_max:.2f}</div><div class='kpi-lab'>Range {metric}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='kpi-tile'><div class='kpi-val'>—</div><div class='kpi-lab'>Mean (not numeric)</div></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='kpi-tile'><div class='kpi-val'>—</div><div class='kpi-lab'>Range (not numeric)</div></div>",
            unsafe_allow_html=True
        )

    # KPI 3: hotspot count if bleaching exists
    if bleach_col:
        hotspots = df[pd.to_numeric(df[bleach_col], errors="coerce") >= hotspot_threshold]
        st.markdown(
            f"<div class='kpi-tile'><div class='kpi-val'>{len(hotspots):,}</div><div class='kpi-lab'>Hotspots (bleaching ≥ {hotspot_threshold}%)</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='kpi-tile'><div class='kpi-val'>—</div><div class='kpi-lab'>Hotspots (no bleaching column)</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("**Ranked output (quick story)**")
    if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
        # Top and bottom 5
        top5 = df.sort_values(metric, ascending=False).head(5)
        bot5 = df.sort_values(metric, ascending=True).head(5)

        st.caption("Top 5 sites (highest metric)")
        st.dataframe(top5[[c for c in [site_col, lat_col, lon_col, metric] if c]].reset_index(drop=True),
                     use_container_width=True, height=160)

        st.caption("Bottom 5 sites (lowest metric)")
        st.dataframe(bot5[[c for c in [site_col, lat_col, lon_col, metric] if c]].reset_index(drop=True),
                     use_container_width=True, height=160)
    else:
        st.info("Select a numeric layer to see ranked outputs.")

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ---------------------- Analysis tabs ----------------------
st.subheader("Analysis")
t1, t2, t3 = st.tabs(["Distribution", "Relationships", "Table"])

with t1:
    if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
        fig_h = px.histogram(df, x=metric, nbins=28, title=f"Distribution of {metric}")
        fig_h.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Selected layer is not numeric, so distribution is not available.")

with t2:
    if len(numeric_cols) >= 2:
        xcol = st.selectbox("X", numeric_cols, index=0)
        ycol = st.selectbox("Y", numeric_cols, index=1)
        fig_sc = px.scatter(df, x=xcol, y=ycol, color=metric if metric in numeric_cols else None,
                            title=f"{xcol} vs {ycol}")
        fig_sc.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Explain drivers: bleaching vs health, coral cover vs health, etc.")
    else:
        st.info("Not enough numeric columns for a relationship view.")

with t3:
    st.dataframe(df.head(500), use_container_width=True)
    st.caption("First 500 rows of your filtered view.")

