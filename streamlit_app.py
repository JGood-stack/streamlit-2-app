import io
import re
import streamlit as st
import pandas as pd
import altair as alt

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(
    page_title="PFAS Ridgeline Tool — Expanded Loader",
    page_icon="💧",
    layout="wide"
)

# ------------------------------
# Altair Global Font & Chart Size Theme
# ------------------------------
alt.themes.enable('none')  # remove Streamlit defaults

TITLE_FONT = 20
LABEL_FONT = 14

alt.themes.register(
    'large_theme',
    lambda: {
        "config": {
            "title": {"fontSize": TITLE_FONT},
            "axis": {"labelFontSize": LABEL_FONT, "titleFontSize": LABEL_FONT},
            "legend": {"labelFontSize": LABEL_FONT, "titleFontSize": LABEL_FONT},
            "view": {"continuousWidth": 850, "continuousHeight": 400}
        }
    }
)
alt.themes.enable('large_theme')

# ------------------------------
# Title
# ------------------------------
st.title("💧 PFAS Treatment Technologies — Ridgeline Visualization Tool (Prototype)")
st.markdown(
    "Upload your dataset and adjust the assumptions. "
    "This tool analyzes 17 distinct metrics across environmental, operational, and financial categories."
)
st.divider()

# =====================================================
# Robust Loader + Header Normalization & Mapping
# =====================================================

# Required columns for the 17 tabs and core logic
REQUIRED = {
    "tech", "score", "ghg", "affordability", "gehh", "lifecycle_cost",
    "GEHH_Acid_CIIX", "Afford_CapCost_CIIX", "GEHH_Ecot_CIIX",
    "GEHH_Eutr_CIIX", "FacOp_Foot_CIIX", "GWP_GHG_CIIX",
    "Afford_LifecycleCost_CIIX", "FacOp_Maint_CIIX", "GEHH_ODP_CIIX", 
    "GEHH_Smog_CIIX", "FacOp_Waste_CIIX", "Afford_OMCost_CIIX"
}

# Optional fields for sidebar filtering
OPTIONAL = {
    "weighting_scheme", "region", "pump_efficiency", "media_usage", 
    "gac_disposal", "booster_pumps", "ebct", "vessel_diameter", 
    "fouling", "redundant_filter", "backwash_interval", 
    "redundant_trains", "escalation_rate", "cleaning_chemicals", "redundant_pumps"
}

HEADER_MAP = {
    "tech": "tech", "score": "score", "ghg": "ghg", "affordability": "affordability",
    "gehh": "gehh", "lifecycle cost (capital + o&m)": "lifecycle_cost_(Capital_+_O&M)",
    # Specific metric mappings
    "gehh acid ciix": "GEHH_Acid_CIIX", "afford capcost ciix": "Afford_CapCost_CIIX",
    "gehh ecot ciix": "GEHH_Ecot_CIIX", "gehh eutr ciix": "GEHH_Eutr_CIIX", 
    "facop foot ciix": "FacOp_Foot_CIIX", "gwp ghg ciix": "GWP_GHG_CIIX",
    "afford lifecyclecost ciix": "Afford_LifecycleCost_CIIX", "facop maint ciix": "FacOp_Maint_CIIX",
    "gehh odp ciix": "GEHH_ODP_CIIX", "gehh smog ciix": "GEHH_Smog_CIIX",
    "facop waste ciix": "FacOp_Waste_CIIX", "afford omcost ciix": "Afford_OMCost_CIIX",
    # Sidebar parameter mappings
    "pump efficiency": "pump_efficiency", "booster pumps": "booster_pumps",
    "ebct": "ebct", "vessel diameter": "vessel_diameter", "fouling": "fouling",
    "escalation rate": "escalation_rate", "redundant pumps": "redundant_pumps",
    "media usage": "media_usage", "gac disposal": "gac_disposal",
    "redundant filter": "redundant_filter", "backwash interval": "backwash_interval",
    "redundant trains": "redundant_trains", "cleaning chemicals": "cleaning_chemicals"
}

def _norm_header_key(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.replace("\ufeff", "").strip().lower()
    s = s.replace("&amp;", "&").replace("_", " ").replace("(", " ").replace(")", " ")
    s = re.sub(r"[^a-z0-9+& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = list(df.columns)
    new_cols = []
    for c in original_cols:
        key = _norm_header_key(c)
        if key in HEADER_MAP:
            new_cols.append(HEADER_MAP[key])
        else:
            snake = re.sub(r"[+&]", " ", key)
            snake = re.sub(r"\s+", "_", snake).strip("_")
            new_cols.append(snake)
    df.columns = new_cols
    return df.loc[:, ~df.columns.isna()]

@st.cache_data(show_spinner=False)
def load_table(file) -> pd.DataFrame:
    raw = file.read()
    try: file.seek(0)
    except: pass
    sample = raw[:4096].decode("utf-8", errors="replace")
    buf = io.BytesIO(raw)
    sep = "\t" if sample.count("\t") > sample.count(",") else ","
    try:
        df = pd.read_csv(buf, sep=sep, header=0, dtype=str, encoding="utf-8-sig")
    except:
        buf.seek(0)
        df = pd.read_excel(buf, sheet_name=0, header=0, engine="openpyxl", dtype=str)
    
    df = canonicalize_columns(df)
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    return df

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------------
# File Upload & Pre-processing
# ------------------------------
uploaded = st.file_uploader("📤 Upload PFAS CSV/TSV/Excel", type=["csv", "tsv", "txt", "xlsx"])
if not uploaded:
    st.warning("Awaiting file upload...")
    st.stop()

try:
    df = load_table(uploaded)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Convert all metrics and parameters to numeric
num_cols = list(REQUIRED - {"tech"}) + list(OPTIONAL - {"weighting_scheme", "region", "gac_disposal"})
df = coerce_numeric(df, num_cols)

# ------------------------------
# Helper masks by technology
# ------------------------------
def make_masks(frame: pd.DataFrame):
    ts = frame["tech"].astype(str)
    return (
        ts.str.contains("GAC", case=False, na=False),
        ts.str.fullmatch("IX", case=False, na=False),
        ts.str.fullmatch("RO", case=False, na=False),
        ts.str.fullmatch("NF", case=False, na=False)
    )

# ------------------------------
# SIDEBAR — Filters
# ------------------------------
st.sidebar.header("Overall Assumptions")
flow_rate = st.sidebar.slider("Flowrate (MGD)", 0.01, 10.0, 1.00, 0.01)

# Pump Efficiency
if "pump_efficiency" in df.columns:
    pe_min = float(df["pump_efficiency"].min())
    pe_max = float(df["pump_efficiency"].max())
    pump_eff = st.sidebar.slider("Pump Efficiency (%)", pe_min, pe_max, (pe_min, pe_max))
else: pump_eff = None

if "weighting_scheme" in df.columns:
    ws_opts = ["All"] + sorted(df["weighting_scheme"].dropna().unique().tolist())
    weighting_scheme = st.sidebar.selectbox("Score Weighting Scheme", ws_opts)
else: weighting_scheme = "All"

if "region" in df.columns:
    reg_opts = ["All"] + sorted(df["region"].dropna().unique().tolist())
    region_select = st.sidebar.selectbox("Electrical Grid Region", reg_opts)
else: region_select = "All"

st.sidebar.divider()

# MEDIA TREATMENT (GAC & IX)
st.sidebar.subheader("🌀 Media Treatment")
m_gac, m_ix, m_ro, m_nf = make_masks(df)

# GAC Section
st.sidebar.markdown("#### GAC Assumptions")
if "booster_pumps" in df.columns and m_gac.any():
    bp_min, bp_max = int(df.loc[m_gac, "booster_pumps"].min()), int(df.loc[m_gac, "booster_pumps"].max())
    gac_booster = st.sidebar.slider("Booster Pumps — GAC", bp_min, bp_max, (bp_min, bp_max))
else: gac_booster = None

if "media_usage" in df.columns and m_gac.any():
    mu_min, mu_max = float(df.loc[m_gac, "media_usage"].min()), float(df.loc[m_gac, "media_usage"].max())
    gac_media = st.sidebar.slider("Media Usage (lb/1000gal) — GAC", mu_min, mu_max, (mu_min, mu_max))
else: gac_media = None

if "redundant_filter" in df.columns and m_gac.any():
    rf_min, rf_max = int(df.loc[m_gac, "redundant_filter"].min()), int(df.loc[m_gac, "redundant_filter"].max())
    gac_rf = st.sidebar.slider("Redundant Filters — GAC", rf_min, rf_max, (rf_min, rf_max))
else: gac_rf = None

if "gac_disposal" in df.columns:
    dispo_opts = ["All"] + sorted(df["gac_disposal"].dropna().astype(str).unique().tolist())
    gac_disposal_select = st.sidebar.selectbox("GAC Disposal", dispo_opts)
else: gac_disposal_select = "All"

# IX Section
st.sidebar.markdown("#### IX Assumptions")
if "booster_pumps" in df.columns and m_ix.any():
    bp_min, bp_max = int(df.loc[m_ix, "booster_pumps"].min()), int(df.loc[m_ix, "booster_pumps"].max())
    ix_booster = st.sidebar.slider("Booster Pumps — IX", bp_min, bp_max, (bp_min, bp_max))
else: ix_booster = None

for param in ["ebct", "vessel_diameter", "fouling"]:
    if param in df.columns and m_ix.any():
        p_min, p_max = float(df.loc[m_ix, param].min()), float(df.loc[m_ix, param].max())
        st.sidebar.slider(f"{param.replace('_', ' ').upper()} — IX", p_min, p_max, (p_min, p_max), key=f"ix_{param}")

st.sidebar.divider()

# MEMBRANES (RO & NF)
st.sidebar.subheader("🧪 Membrane Separation")
for m_name, mask in [("RO", m_ro), ("NF", m_nf)]:
    st.sidebar.markdown(f"#### {m_name} Assumptions")
    if "escalation_rate" in df.columns and mask.any():
        er_min, er_max = float(df.loc[mask, "escalation_rate"].min()), float(df.loc[mask, "escalation_rate"].max())
        st.sidebar.slider(f"Escalation Rate — {m_name}", er_min, er_max, (er_min, er_max), key=f"{m_name}_er")
    
    if "redundant_pumps" in df.columns and mask.any():
        rp_min, rp_max = int(df.loc[mask, "redundant_pumps"].min()), int(df.loc[mask, "redundant_pumps"].max())
        st.sidebar.slider(f"Redundant Pumps — {m_name}", rp_min, rp_max, (rp_min, rp_max), key=f"{m_name}_rp")

# ------------------------------
# Data Filtering Logic
# ------------------------------
filtered = df.copy()

if weighting_scheme != "All" and "weighting_scheme" in filtered.columns:
    filtered = filtered[filtered["weighting_scheme"] == weighting_scheme]
if region_select != "All" and "region" in filtered.columns:
    filtered = filtered[filtered["region"] == region_select]
if pump_eff and "pump_efficiency" in filtered.columns:
    filtered = filtered[filtered["pump_efficiency"].between(*pump_eff)]

def apply_tech_filter(df_in, mask, col, val_range):
    if val_range and col in df_in.columns:
        return df_in[(~mask) | (df_in[col].between(*val_range))]
    return df_in

m_gac, m_ix, m_ro, m_nf = make_masks(filtered)

filtered = apply_tech_filter(filtered, m_gac, "booster_pumps", gac_booster)
filtered = apply_tech_filter(filtered, m_gac, "media_usage", gac_media)
filtered = apply_tech_filter(filtered, m_gac, "redundant_filter", gac_rf)
if gac_disposal_select != "All" and "gac_disposal" in filtered.columns:
    filtered = filtered[(~m_gac) | (filtered["gac_disposal"] == gac_disposal_select)]

filtered = apply_tech_filter(filtered, m_ix, "booster_pumps", ix_booster)
for p in ["ebct", "vessel_diameter", "fouling"]:
    v = st.session_state.get(f"ix_{p}")
    filtered = apply_tech_filter(filtered, m_ix, p, v)

for m in ["RO", "NF"]:
    mask = m_ro if m == "RO" else m_nf
    filtered = apply_tech_filter(filtered, mask, "escalation_rate", st.session_state.get(f"{m}_er"))
    filtered = apply_tech_filter(filtered, mask, "redundant_pumps", st.session_state.get(f"{m}_rp"))

# ------------------------------
# Ridgeline Chart Function
# ------------------------------
def ridgeline(df_in, xcol, title, xaxis_label):
    if df_in.empty or xcol not in df_in.columns:
        return alt.Chart().mark_text(text="Insufficient data").properties(height=100)
    
    tech_order = sorted(df_in["tech"].dropna().unique().tolist())
    
    return (
        alt.Chart(df_in)
        .transform_density(xcol, groupby=["tech"], as_=[xcol, "density"], steps=50)
        .mark_area(opacity=0.7, stroke='black', strokeWidth=0.5)
        .encode(
            x=alt.X(f"{xcol}:Q", title=xaxis_label),
            y=alt.Y("density:Q", axis=None),
            color=alt.Color("tech:N", legend=None, scale=alt.Scale(scheme='tableau10'))
        )
        .properties(width=900, height=120)
        .facet(
            row=alt.Row("tech:N", sort=tech_order, title=None, 
                        header=alt.Header(labelFontSize=13, labelAngle=0))
        )
        .properties(title=title)
    )

# ------------------------------
# TABS — Ridgeline Charts (Explicit Definitions)
# ------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17 = st.tabs([
    "📈Overall Score", "🌍Global Environment & Human Health", "💰Affordability", "📊Global Warming Potential", "💰Lifecycle Cost (Capital + O&M)",
    "Acidification", "Capital Cost", "Ecotoxicity", "Eutrophication", "Facility Footprint", "GWP GHG", "Lifecycle Cost (CIIX)", 
    "Maintenance Effort", "Ozone Depletion", "Smog Formation", "Waste Generation", "O&M Cost"
])

with tab1:
    st.subheader("Overall Score (lower is better)")
    st.altair_chart(ridgeline(filtered, "score", "Overall Score by Technology", "Overall Score (lower is better)"), use_container_width=True)

with tab2:
    st.subheader("Global Environment & Human Health (lower is better)")
    st.altair_chart(ridgeline(filtered, "gehh", "GEHH by Technology", "Score: GEHH (lower is better)"), use_container_width=True)

with tab3:
    st.subheader("Affordability ($$) (lower is better)")
    st.altair_chart(ridgeline(filtered, "affordability", "Affordability by Technology", "Score: Affordability ($$) (lower is better)"), use_container_width=True)

with tab4:
    st.subheader("Global Warming Potential (kgCO2e) (lower is better)")
    st.altair_chart(ridgeline(filtered, "ghg", "GHG by Technology", "Score: GHG (kgCO2e) (lower is better)"), use_container_width=True)

with tab5:
    st.subheader("Lifecycle Cost (Capital + O&M) ($$) (lower is better)")
    st.altair_chart(ridgeline(filtered, "lifecycle_cost", "Lifecycle Cost by Technology", "Lifecycle Cost ($$) (lower is better)"), use_container_width=True)

with tab6:
    st.subheader("Acidification (kgSO2eq) (lower is better)")
    st.altair_chart(ridgeline(filtered, "GEHH_Acid_CIIX", "Acidification Comparison to IX (kgSO2eq technology/kgSO2eq IX)", "Ratio: Acidification (kgSO2eq technology/kgSO2eq IX) (lower is better)"), use_container_width=True)

with tab7:
    st.subheader("Capital Cost (NPV($))(lower is better)")
    st.altair_chart(ridgeline(filtered, "Afford_CapCost_CIIX", "Capital Cost Comparison to IX (NPV($) technology/NPV($) IX)", "Ratio: Cost (NPV($) technology/NPV($) IX) (lower is better)"), use_container_width=True)

with tab8:
    st.subheader("Ecotoxicity (CTUe) (lower is better)")
    st.altair_chart(ridgeline(filtered, "GEHH_Ecot_CIIX", "Ecotoxity Comparison to IX (CTUe technology/CTUe IX)", "Ratio: Ecotoxicity (CTUe technology/CTUe IX) (lower is better)"), use_container_width=True)

with tab9:
    st.subheader("Eutrophication (kgNeq) (lower is better)")
    st.altair_chart(ridgeline(filtered, "GEHH_Eutr_CIIX", "Eutrophication Comparison to IX (kgNeq technology/kgNeq IX)", "Ratio: Eutrophication (kgNeq technology/kgNeq IX) (lower is better)"), use_container_width=True)

with tab10:
    st.subheader("Facility Footprint (sqft) (lower is better)")
    st.altair_chart(ridgeline(filtered, "FacOp_Foot_CIIX", "Facility Footprint Comparison to IX (sqft technology/sqft IX)", "Ratio: Area (sqft technology/sqft IX) (lower is better)"), use_container_width=True)

with tab11:
    st.subheader("GWP GHG (kgCO2eq) (lower is better)")
    st.altair_chart(ridgeline(filtered, "GWP_GHG_CIIX", "GWP GHG Comparison to IX (kgCO2e technology/kgCO2e IX)", "Ratio:(kgCO2e technology/kgCO2e IX) (lower is better)"), use_container_width=True)

with tab12:
    st.subheader("Lifecycle Cost (CIIX) (NPV($)) (lower is better)")
    st.altair_chart(ridgeline(filtered, "Afford_LifecycleCost_CIIX", "Lifecycle Cost (CIIX) Comparison to IX (NPV($) technology/NPV($) IX)", "Ratio:Cost (NPV($) technology/NPV($) IX) (lower is better)"), use_container_width=True)

with tab13:
    st.subheader("Maintenance Effort (Hours/Year) (lower is better)")
    st.altair_chart(ridgeline(filtered, "FacOp_Maint_CIIX", "Maintenance Effort Comparison to IX (Hours/Year technology/Hours/Year IX)", "Ratio: Annual Hours technology/Annual Hours IX (lower is better)"), use_container_width=True)

with tab14:
    st.subheader("Ozone Depletion (kgCFC-11eq) (lower is better)")
    st.altair_chart(ridgeline(filtered, "GEHH_ODP_CIIX", "ODP Comparison to IX (kgCFC-11eq technology/kgCFC-11eq IX)", "Ratio: ODP (kgCFC-11eq technology/kgCFC-11eq IX) (lower is better)"), use_container_width=True)

with tab15:
    st.subheader("Smog Formation (kgO3eq) (lower is better)")
    st.altair_chart(ridgeline(filtered, "GEHH_Smog_CIIX", "Smog Formation Comparison to IX (kgO3eq technology/kgO3eq IX)", "Ratio: Smog (kgO3eq technology/kgO3eq IX) (lower is better)"), use_container_width=True)

with tab16:
    st.subheader("Waste Generation (Galloon/Year)(lower is better)")
    st.altair_chart(ridgeline(filtered, "FacOp_Waste_CIIX", "Waste Generation Comparison to IX (Gallon/Year technology/Gallon/Year IX)", "Ratio:Waste Generation (Gallon/Year technology/Gallon/Year IX) (lower is better)"), use_container_width=True)

with tab17:
    st.subheader("O&M Cost (NPV($)) (lower is better)")
    st.altair_chart(ridgeline(filtered, "Afford_OMCost_CIIX", "O&M Cost Comparison to IX (NPV($) technology/NPV($) IX", "Ratio: O&M Cost (NPV($) technology/NPV($) IX) (lower is better)"), use_container_width=True)

st.divider()
st.caption(f"PFAS Ridgeline Tool — Showing {len(filtered)} rows. University of Maine (2025)")
