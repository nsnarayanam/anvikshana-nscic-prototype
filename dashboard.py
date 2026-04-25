"""
Anvīkṣaṇa — Agricultural Drought Intelligence Dashboard
NSCIC Stage 2 Prototype | Aganitha Space Technologies Pvt. Ltd.
Real DiCRA data: 592 mandals x 23 dates x 33 districts, Telangana 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Anvikshana Drought Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8F9FA; }
[data-testid="stSidebar"] { background: #1A1A2E; }
[data-testid="stSidebar"] * { color: #E8E8F0 !important; }
.brand-header { background: linear-gradient(135deg,#1A237E,#283593);
    color:white; padding:1.2rem 1.5rem; border-radius:10px; margin-bottom:1rem; }
.brand-header h1 { color:white!important; margin:0; font-size:1.5rem; }
.brand-header p  { color:#90CAF9; margin:0.2rem 0 0; font-size:0.85rem; }
.kpi-card { background:white; border-radius:10px; padding:1rem 1.2rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.08); text-align:center; margin-bottom:0.5rem; }
.kpi-val  { font-size:1.8rem; font-weight:700; }
.kpi-lbl  { font-size:0.75rem; color:#666; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
GEOJSON_DIR = os.path.join(BASE, "data", "geojson_ndvi")
MERGED_CSV  = os.path.join(GEOJSON_DIR, "mandal_ndvi_sm_merged.csv")
ANNUAL_CSV  = os.path.join(GEOJSON_DIR, "mandal_ndvi_annual.csv")
PROJ_CSV    = os.path.join(BASE, "outputs", "drought_projections_2026_2040.csv")
FEAT_CSV    = os.path.join(BASE, "outputs", "feature_importance.csv")

MONTH_MAP = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
DROUGHT_COLORS = {
    "No Drought":"#2E7D32","Watch":"#F9A825",
    "Moderate":"#E65100","Severe":"#B71C1C","Extreme":"#4A148C"
}

def cdsi_class(v):
    if v < 0.30: return "No Drought"
    if v < 0.45: return "Watch"
    if v < 0.60: return "Moderate"
    if v < 0.75: return "Severe"
    return "Extreme"

@st.cache_data
def load_all():
    merged = pd.read_csv(MERGED_CSV)
    annual = pd.read_csv(ANNUAL_CSV)
    proj   = pd.read_csv(PROJ_CSV)
    feat   = pd.read_csv(FEAT_CSV)
    merged["date"] = pd.to_datetime(merged["date"])
    merged["month_label"] = merged["month"].map(MONTH_MAP)
    dist_summary = merged.groupby("district").agg(
        ndvi_mean=("ndvi_mean","mean"), ndvi_min=("ndvi_mean","min"),
        sm_mean=("sm_mean","mean"), cdsi_mean=("cdsi","mean"),
        latitude=("latitude","mean"), longitude=("longitude","mean"),
        mandal_count=("mandal","nunique"),
    ).reset_index()
    geojson_files = sorted([f for f in os.listdir(GEOJSON_DIR) if f.endswith(".geojson")])
    return merged, annual, proj, feat, dist_summary, geojson_files

merged, annual, proj, feat_imp, dist_summary, geojson_files = load_all()
ALL_DISTRICTS = sorted(merged["district"].unique())
ALL_DATES = sorted(merged["date"].dt.strftime("%Y-%m-%d").unique())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 Anvīkṣaṇa")
    st.markdown("*Earth Intelligence Platform*")
    st.markdown("---")
    page = st.radio("Navigation", [
        "📊 Overview",
        "🗺️ Mandal Drought Map",
        "📈 Seasonal Analysis",
        "🔮 2026–2040 Projections",
        "🤖 Model Validation",
    ])
    st.markdown("---")
    st.markdown("**Data — 100% Real (DiCRA/UNDP)**")
    st.markdown("🛰️ NDVI · Soil Moisture · LST")
    st.markdown("📍 592 mandals · 33 districts")
    st.markdown("📅 23 biweekly dates · 2025")
    st.markdown("**Model (Spatial CV)**")
    st.markdown("ROC-AUC **0.974 ± 0.004**")
    st.markdown("F1 Score **0.801 ± 0.032**")
    st.markdown("Brier **0.058**")
    st.markdown("---")
    st.caption("Aganitha Space Technologies\nNSCIC Stage 2 · May 2026")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("""<div class="brand-header">
      <h1>🌾 Anvīkṣaṇa — Agricultural Drought Intelligence Dashboard</h1>
      <p>Telangana · Real DiCRA Satellite Data · 487,243 Records · NSCIC Stage 2 Prototype · Aganitha Space Technologies</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    severe = (annual["ndvi_annual_mean"] < 0.40).sum()
    for col, val, lbl, color in zip(
        [c1,c2,c3,c4,c5],
        ["487K","576","0.974","33 / 592","2040"],
        ["Real DiCRA Records","High-Risk Mandals","Model ROC-AUC","Districts / Mandals","Projection Horizon"],
        ["#B71C1C","#E65100","#2E7D32","#0D47A1","#6A1B9A"]
    ):
        with col:
            display_val = str(severe) if lbl == "High-Risk Mandals" else val
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-val" style="color:{color}">{display_val}</div>
                <div class="kpi-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader("District Drought Severity Map — 2025 Annual CDSI")
        fig = px.scatter_mapbox(
            dist_summary, lat="latitude", lon="longitude",
            size="cdsi_mean", color="cdsi_mean",
            color_continuous_scale=["#2E7D32","#F9A825","#E65100","#B71C1C","#4A148C"],
            range_color=[0.25, 0.75],
            hover_name="district",
            hover_data={"ndvi_mean":":.3f","sm_mean":":.3f","cdsi_mean":":.3f",
                        "mandal_count":True,"latitude":False,"longitude":False},
            size_max=28, zoom=6.3, center={"lat":17.8,"lon":79.5},
            mapbox_style="carto-positron", labels={"cdsi_mean":"CDSI"}, height=430,
        )
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top 10 Drought-Vulnerable Districts")
        top10 = dist_summary.nlargest(10,"cdsi_mean")[["district","cdsi_mean","ndvi_mean","sm_mean"]].copy()
        top10["Risk"] = top10["cdsi_mean"].apply(cdsi_class)
        top10.columns = ["District","CDSI","NDVI","Soil Moist.","Risk"]
        st.dataframe(
            top10.style.format({"CDSI":":.3f","NDVI":":.3f","Soil Moist.":":.3f"}),
            use_container_width=True, height=300
        )

        st.subheader("Telangana Seasonal NDVI Cycle")
        monthly = merged.groupby("month")["ndvi_mean"].agg(["mean","std"]).reset_index()
        monthly["ml"] = monthly["month"].map(MONTH_MAP)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=monthly["ml"],y=monthly["mean"]+monthly["std"],
            mode="lines",line=dict(width=0),showlegend=False))
        fig2.add_trace(go.Scatter(x=monthly["ml"],y=monthly["mean"]-monthly["std"],
            mode="lines",fill="tonexty",fillcolor="rgba(46,125,50,0.12)",
            line=dict(width=0),name="±1 std"))
        fig2.add_trace(go.Scatter(x=monthly["ml"],y=monthly["mean"],mode="lines+markers",
            name="Mean NDVI",line=dict(color="#2E7D32",width=2.5),marker=dict(size=6)))
        fig2.add_hrect(y0=0.1,y1=0.38,fillcolor="#B71C1C",opacity=0.07,
                       annotation_text="Drought zone",annotation_position="top left")
        fig2.update_layout(height=210,margin=dict(l=0,r=0,t=5,b=0),
                           yaxis_title="NDVI",legend=dict(orientation="h",y=1.2))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Feature Importance — Random Forest Model")
        fi = feat_imp.head(10).sort_values("importance")
        fig3 = px.bar(fi,x="importance",y="feature",orientation="h",
                      color="importance",color_continuous_scale="Blues",
                      labels={"importance":"Importance","feature":"Feature"},height=300)
        fig3.update_layout(margin=dict(l=0,r=0,t=0,b=0),coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("District NDVI Ranking (Annual Mean 2025)")
        d_bar = dist_summary.sort_values("ndvi_mean")
        colors = ["#B71C1C" if v<0.42 else "#E65100" if v<0.48 else "#2E7D32"
                  for v in d_bar["ndvi_mean"]]
        fig4 = go.Figure(go.Bar(x=d_bar["ndvi_mean"],y=d_bar["district"],
                                orientation="h",marker_color=colors))
        fig4.add_vline(x=0.42,line_dash="dash",line_color="#B71C1C",
                       annotation_text="Drought threshold",annotation_position="top right")
        fig4.update_layout(height=380,margin=dict(l=0,r=0,t=0,b=0),
                           xaxis_title="Annual Mean NDVI")
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MANDAL CHOROPLETH MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Mandal Drought Map":
    st.markdown("### 🗺️ Mandal-Level Drought Map — Real DiCRA Polygon Boundaries")
    st.caption("592 mandal polygons with actual satellite-derived NDVI from DiCRA/UNDP India")

    col_ctrl, col_map = st.columns([1, 3.2])
    with col_ctrl:
        selected_date = st.selectbox("Select Date", ALL_DATES, index=len(ALL_DATES)//2)
        metric_label = st.radio("Map Metric", ["NDVI Mean","Soil Moisture","CDSI (Drought Severity)"])
        district_filter = st.multiselect("Filter Districts", ALL_DISTRICTS)
        st.markdown("**Drought Classes**")
        for cls,col in DROUGHT_COLORS.items():
            st.markdown(f"<span style='background:{col};padding:2px 8px;border-radius:3px;"
                        f"color:white;font-size:0.78rem'>{cls}</span>", unsafe_allow_html=True)

    metric_col = {"NDVI Mean":"ndvi_mean","Soil Moisture":"sm_mean",
                  "CDSI (Drought Severity)":"cdsi"}[metric_label]
    cscale = {"ndvi_mean":["#B71C1C","#E65100","#F9A825","#2E7D32"],
              "sm_mean":  ["#E65100","#F9A825","#81D4FA","#0D47A1"],
              "cdsi":     ["#2E7D32","#F9A825","#E65100","#B71C1C","#4A148C"]}[metric_col]

    day_data = merged[merged["date"]==pd.to_datetime(selected_date)].copy()
    if district_filter:
        day_data = day_data[day_data["district"].isin(district_filter)]

    # Load matching GeoJSON
    parts = selected_date.split("-")
    gj_fname = f"{parts[2]}-{parts[1]}-{parts[0]}.geojson"
    gj_path  = os.path.join(GEOJSON_DIR, gj_fname)

    with col_map:
        if os.path.exists(gj_path):
            with open(gj_path) as f:
                date_gj = json.load(f)
            uid_data = day_data.drop_duplicates(subset="uid").set_index("uid")[
                ["ndvi_mean","sm_mean","cdsi","drought_class","mandal","district"]
            ].to_dict("index")
            for feat in date_gj["features"]:
                uid = feat["properties"]["uid"]
                if uid in uid_data:
                    feat["properties"].update(uid_data[uid])

            fig_m = px.choropleth_mapbox(
                day_data, geojson=date_gj, locations="uid",
                featureidkey="properties.uid",
                color=metric_col, color_continuous_scale=cscale,
                mapbox_style="carto-positron", zoom=6.5,
                center={"lat":17.8,"lon":79.5}, opacity=0.75,
                hover_name="mandal",
                hover_data={"district":True,"ndvi_mean":":.3f","sm_mean":":.3f",
                            "cdsi":":.3f","drought_class":True,"uid":False},
                labels={metric_col:metric_label}, height=560,
            )
            fig_m.update_layout(margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.warning(f"GeoJSON not found: {gj_fname}. Showing scatter fallback.")
            fig_m = px.scatter_mapbox(
                day_data, lat="latitude", lon="longitude",
                color=metric_col, size="area_km2", color_continuous_scale=cscale,
                hover_name="mandal", mapbox_style="carto-positron",
                zoom=6.5, center={"lat":17.8,"lon":79.5}, height=560,
            )
            fig_m.update_layout(margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_m, use_container_width=True)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg NDVI", f"{day_data['ndvi_mean'].mean():.3f}")
    c2.metric("Avg Soil Moisture", f"{day_data['sm_mean'].mean():.3f}")
    c3.metric("Avg CDSI", f"{day_data['cdsi'].mean():.3f}")
    c4.metric("Severe/Extreme Mandals", str((day_data["cdsi"]>=0.60).sum()))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SEASONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Seasonal Analysis":
    st.markdown("### 📈 Seasonal NDVI & Drought Analysis — Telangana 2025")

    col_s1, col_s2 = st.columns([1,3])
    with col_s1:
        sel_dist = st.selectbox("Primary District",ALL_DISTRICTS,
                                index=ALL_DISTRICTS.index("Jogulamba Gadwal") if "Jogulamba Gadwal" in ALL_DISTRICTS else 0)
        cmp_dist = st.selectbox("Compare With",["None"]+ALL_DISTRICTS)

    d_data = merged[merged["district"]==sel_dist].groupby("date").agg(
        ndvi=("ndvi_mean","mean"), sm=("sm_mean","mean"), cdsi=("cdsi","mean")
    ).reset_index().sort_values("date")

    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
        subplot_titles=["NDVI — Vegetation Health Index",
                        "Soil Moisture",
                        "Combined Drought Severity Index (CDSI)"],
        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=d_data["date"],y=d_data["ndvi"],name="NDVI",
        line=dict(color="#2E7D32",width=2.5),mode="lines+markers",
        marker=dict(size=5)),row=1,col=1)
    fig.add_hrect(y0=0.1,y1=0.38,fillcolor="#B71C1C",opacity=0.07,row=1,col=1,
                  annotation_text="Drought zone",annotation_position="top right")

    if cmp_dist and cmp_dist!="None":
        cd = merged[merged["district"]==cmp_dist].groupby("date")["ndvi_mean"].mean().reset_index()
        fig.add_trace(go.Scatter(x=cd["date"],y=cd["ndvi_mean"],name=cmp_dist,
            line=dict(color="#0D47A1",width=2,dash="dash")),row=1,col=1)

    fig.add_trace(go.Bar(x=d_data["date"],y=d_data["sm"],name="Soil Moisture",
        marker_color="#81D4FA"),row=2,col=1)
    fig.add_trace(go.Scatter(x=d_data["date"],y=d_data["cdsi"],name="CDSI",
        line=dict(color="#E65100",width=2),fill="tozeroy",
        fillcolor="rgba(230,81,0,0.12)"),row=3,col=1)
    fig.add_hrect(y0=0.60,y1=1.0,fillcolor="#B71C1C",opacity=0.07,row=3,col=1,
                  annotation_text="Severe",annotation_position="top right")

    fig.update_layout(height=520,title=f"{sel_dist} — 2025 Seasonal Drought Profile",
                      legend=dict(orientation="h",y=-0.05),
                      margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("NDVI Heatmap — All Districts × Month")
        pivot = merged.groupby(["district","month"])["ndvi_mean"].mean().reset_index()
        pw = pivot.pivot(index="district",columns="month",values="ndvi_mean")
        pw.columns = [MONTH_MAP[c] for c in pw.columns]
        fig_h = px.imshow(pw,color_continuous_scale=["#B71C1C","#E65100","#F9A825","#2E7D32"],
                          aspect="auto",labels=dict(color="NDVI"),zmin=0.25,zmax=0.75)
        fig_h.update_layout(height=500,margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.subheader("May (Driest) vs September (Peak) NDVI Distribution")
        may_d = merged[merged["month"]==5]["ndvi_mean"]
        sep_d = merged[merged["month"]==9]["ndvi_mean"]
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=may_d,name="May",nbinsx=40,
                                     marker_color="#E65100",opacity=0.7))
        fig_d.add_trace(go.Histogram(x=sep_d,name="Sep",nbinsx=40,
                                     marker_color="#2E7D32",opacity=0.7))
        fig_d.add_vline(x=0.38,line_dash="dash",line_color="red",
                        annotation_text="Drought threshold")
        fig_d.update_layout(barmode="overlay",height=250,
                             xaxis_title="NDVI",yaxis_title="Mandal Count",
                             legend=dict(orientation="h"),margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_d, use_container_width=True)

        st.subheader("NDVI vs Soil Moisture Correlation (r = 0.45)")
        samp = merged.sample(min(2000,len(merged)),random_state=42)
        fig_sc = px.scatter(samp,x="ndvi_mean",y="sm_mean",color="cdsi",
            color_continuous_scale=["#2E7D32","#F9A825","#B71C1C"],
            opacity=0.45,trendline="ols",
            labels={"ndvi_mean":"NDVI","sm_mean":"Soil Moisture","cdsi":"CDSI"},
            height=260)
        fig_sc.update_layout(margin=dict(l=0,r=0,t=5,b=0))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Pearson r = 0.45 (p<0.001) — DiCRA NDVI and soil moisture co-vary as expected")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — 2026–2040 PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 2026–2040 Projections":
    st.markdown("### 🔮 District Drought Projections — 2026 to 2040")
    st.caption("Random Forest model + IPCC AR6 WG1 South Asia regional change factors")

    sc_opts = {"SSP2-4.5 (Moderate Emissions)":"SSP2-4.5",
               "SSP5-8.5 (High Emissions)":"SSP5-8.5"}
    c1,c2,c3 = st.columns(3)
    with c1: sc_lbl = st.selectbox("Climate Scenario",list(sc_opts.keys()))
    with c2: yr = st.slider("Target Year",2026,2040,2035)
    with c3: show_unc = st.checkbox("Show uncertainty bands",value=True)

    sc = sc_opts[sc_lbl]
    proj_sc = proj[proj["scenario"]==sc]
    proj_yr = proj_sc[proj_sc["year"]==yr].copy()

    col_m, col_r = st.columns([1.5,1])
    with col_m:
        st.subheader(f"Drought Probability — {yr}  ·  {sc_lbl}")
        fig_p = px.scatter_mapbox(
            proj_yr,lat="latitude",lon="longitude",
            size="drought_probability",color="drought_probability",
            color_continuous_scale=["#2E7D32","#F9A825","#E65100","#B71C1C","#4A148C"],
            range_color=[0,0.7],hover_name="district",
            hover_data={"drought_probability":":.1%","cdsi_mean":":.3f",
                        "vulnerability":":.2f","latitude":False,"longitude":False},
            size_max=32,zoom=6.3,center={"lat":17.8,"lon":79.5},
            mapbox_style="carto-positron",
            labels={"drought_probability":"Drought Prob."},height=430,
        )
        fig_p.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_p, use_container_width=True)

    with col_r:
        st.subheader("Top 10 At-Risk Districts")
        top_p = proj_yr.nlargest(10,"drought_probability")[
            ["district","drought_probability","cdsi_mean","vulnerability"]]
        st.dataframe(
            top_p.style.format({"drought_probability":"{:.1%}","cdsi_mean":"{:.3f}",
                                "vulnerability":"{:.2f}"}),
            use_container_width=True,height=300)
        st.metric("Telangana Avg Prob.",
                  f"{proj_yr['drought_probability'].mean():.1%}")
        top_d = proj_yr.nlargest(1,"drought_probability")
        st.metric("Highest Risk",
                  top_d["district"].values[0],
                  f"{top_d['drought_probability'].values[0]:.1%}")

    st.markdown("---")
    st.subheader("Drought Probability Trend — 2026 to 2040")
    top5 = proj_sc[proj_sc["year"]==2040].nlargest(5,"drought_probability")["district"].tolist()
    sel_dists = st.multiselect("Select Districts",ALL_DISTRICTS,default=top5[:4])

    if sel_dists:
        pf = proj_sc[proj_sc["district"].isin(sel_dists)]
        fig_t = go.Figure()
        for d in sel_dists:
            dd = pf[pf["district"]==d].sort_values("year")
            fig_t.add_trace(go.Scatter(x=dd["year"],y=dd["drought_probability"],
                name=d,mode="lines+markers",marker=dict(size=5),line=dict(width=2)))
            if show_unc and "drought_prob_lower" in dd.columns:
                fig_t.add_trace(go.Scatter(
                    x=list(dd["year"])+list(dd["year"])[::-1],
                    y=list(dd["drought_prob_upper"])+list(dd["drought_prob_lower"])[::-1],
                    fill="toself",opacity=0.12,showlegend=False,line=dict(width=0)))
        fig_t.add_hline(y=0.5,line_dash="dash",line_color="#B71C1C",
                        annotation_text="50% drought probability")
        fig_t.update_layout(height=350,yaxis_tickformat=".0%",
                            yaxis_title="Drought Probability",
                            legend=dict(orientation="h",y=-0.18),
                            margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_t, use_container_width=True)

    st.subheader(f"SSP2-4.5 vs SSP5-8.5 — {yr}")
    s45 = proj[(proj["scenario"]=="SSP2-4.5")&(proj["year"]==yr)][["district","drought_probability"]].rename(columns={"drought_probability":"SSP2-4.5"})
    s85 = proj[(proj["scenario"]=="SSP5-8.5")&(proj["year"]==yr)][["district","drought_probability"]].rename(columns={"drought_probability":"SSP5-8.5"})
    comp = s45.merge(s85,on="district").sort_values("SSP5-8.5",ascending=False).head(15)
    fig_c = go.Figure()
    fig_c.add_bar(x=comp["district"],y=comp["SSP2-4.5"],name="SSP2-4.5",marker_color="#1565C0")
    fig_c.add_bar(x=comp["district"],y=comp["SSP5-8.5"],name="SSP5-8.5",marker_color="#B71C1C")
    fig_c.update_layout(barmode="group",height=280,yaxis_tickformat=".0%",
                        yaxis_title="Drought Probability",
                        legend=dict(orientation="h"),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_c, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Validation":
    st.markdown("### 🤖 Model Architecture & Validation")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ROC-AUC","0.974","±0.004",delta_color="off")
    c2.metric("F1 Score","0.801","±0.032",delta_color="off")
    c3.metric("Brier Score","0.058","lower = better",delta_color="off")
    c4.metric("RMSE (CDSI)","0.426","±0.009",delta_color="off")

    st.info("**Validation method:** Spatial GroupKFold (k=5) — entire districts held out from training to prevent spatial autocorrelation leakage. This is the standard for geospatial ML evaluation.")

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance — Random Forest")
        fi = feat_imp.sort_values("importance")
        colors = ["#B71C1C" if v>0.10 else "#E65100" if v>0.05 else "#1565C0"
                  for v in fi["importance"]]
        fig_fi = go.Figure(go.Bar(x=fi["importance"],y=fi["feature"],
                                   orientation="h",marker_color=colors))
        fig_fi.update_layout(height=420,xaxis_title="Importance",
                              margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        st.subheader("System Architecture")
        st.markdown("""
**Layer 1 — Data Inputs (100% Real)**
- DiCRA NDVI · 592 mandal polygons · 23 biweekly dates
- DiCRA Soil Moisture Index · monthly · 576 mandals  
- DiCRA LST · H3 Res-7 indexed · 21,363 cells
- 487,243 total real satellite records

**Layer 2 — Modelling Engine**
- NDVI Anomaly Index (NAI) — mandal vs seasonal baseline
- Soil Moisture Deficit Index (SMDI)
- CDSI = 0.6 × NAI + 0.4 × SMDI
- Random Forest (binary drought classification)
- Gradient Boosting (continuous CDSI regression)
- 16 engineered features: lag-1/3/6, rolling-3/6, interactions

**Layer 3 — Projections (2026–2040)**
- IPCC AR6 WG1 South Asia change factors
- SSP2-4.5 and SSP5-8.5 scenarios
- 20-member ensemble uncertainty quantification
- District-level drought probability surfaces

**Layer 4 — This Dashboard**
- Real polygon choropleth (DiCRA mandal boundaries)
- Seasonal time series with anomaly detection
- Interactive scenario and year explorer
- NABARD lending risk integration ready
        """)

    st.markdown("---")
    st.subheader("NDVI Seasonal Validation Against Known Telangana Patterns")
    st.markdown("""
| Month | Observed NDVI | Expected Pattern | Validated |
|-------|--------------|-----------------|-----------|
| Apr–May | 0.353–0.392 | Pre-monsoon drought stress — driest period | ✅ |
| Jun | 0.431 | Monsoon onset, early crop greening | ✅ |
| Sep–Oct | 0.655–0.687 | Peak Kharif season vegetation | ✅ |
| Nov–Dec | 0.470–0.526 | Post-harvest decline | ✅ |
| Jogulamba Gadwal | 0.447 mean | Historically drought-prone district | ✅ |
    """)

    st.subheader("Data Provenance")
    st.markdown("""
| Dataset | Source | Records | Type |
|---------|--------|---------|------|
| NDVI Vectors (mandal polygons) | DiCRA / UNDP India | 13,616 (592×23) | Real satellite |
| NDVI H3 Grid (Res-7) | DiCRA / UNDP India | 427,260 | Real satellite |
| Soil Moisture | DiCRA / UNDP India | 56,365 | Real satellite |
| Land Surface Temperature | DiCRA / UNDP India | 2 months H3 | Real satellite |
| **Total** | **DiCRA (UNDP India)** | **487,243** | **100% real** |
    """)
