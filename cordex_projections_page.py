# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CORDEX-SA CLIMATE PROJECTIONS (real downscaled RCM data)
# Add to sidebar nav:  "🌡️ CORDEX-SA Projections",
# Place this block in dashboard.py
#
# REQUIRED: upload cordex_telangana_monthly.csv to your repo root (or /data)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌡️ CORDEX-SA Projections":
    st.markdown("### 🌡️ CORDEX-SA Climate Projections — Real Downscaled RCM Data")
    st.caption("IITM-RegCM4-4 · MPI-M-MPI-ESM-MR · Copernicus CDS · 0.44° (~50km) · 91 grid cells over Telangana")

    st.success("""
**Methodology upgrade:** These projections use **actual CORDEX-SA regional climate model output** —
not IPCC scaling factors. Data is dynamically downscaled by **IITM-RegCM4-4** (the Indian regional
climate model from IITM Pune) and sourced from the **Copernicus Climate Data Store**.
This is the physically-simulated climate pathway Dr. Goroshi (IMD) recommended on 14 May 2026.
    """)

    import pandas as pd

    @st.cache_data
    def load_cordex():
        try:
            df = pd.read_csv("cordex_telangana_monthly.csv", index_col=0, parse_dates=True)
        except Exception:
            df = pd.read_csv("data/cordex_telangana_monthly.csv", index_col=0, parse_dates=True)
        return df

    try:
        cordex = load_cordex()
    except Exception as e:
        st.error("CORDEX data file not found. Upload cordex_telangana_monthly.csv to the repo root.")
        st.stop()

    # Annual aggregation
    annual = cordex.resample("YS").mean()
    annual.index = annual.index.year

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    t45_change = annual['tas_rcp45_C'].iloc[-1] - annual['tas_rcp45_C'].iloc[0]
    t85_change = annual['tas_rcp85_C'].iloc[-1] - annual['tas_rcp85_C'].iloc[0]
    c1.metric("RCP 4.5 Temp 2030", f"{annual['tas_rcp45_C'].iloc[-1]:.1f}°C", f"{t45_change:+.2f}°C vs 2026")
    c2.metric("RCP 8.5 Temp 2030", f"{annual['tas_rcp85_C'].iloc[-1]:.1f}°C", f"{t85_change:+.2f}°C vs 2026")
    c3.metric("RCP 4.5 Rainfall 2030", f"{annual['pr_rcp45_mmday'].iloc[-1]:.2f} mm/d", "monsoon-driven")
    c4.metric("RCP 8.5 Rainfall 2030", f"{annual['pr_rcp85_mmday'].iloc[-1]:.2f} mm/d", "high-emission path")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temperature Trajectory 2026–2030")
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=annual.index, y=annual['tas_rcp45_C'],
            mode="lines+markers", name="RCP 4.5 (Paris-aligned)",
            line=dict(color="#0A7B6E", width=3)))
        fig_t.add_trace(go.Scatter(x=annual.index, y=annual['tas_rcp85_C'],
            mode="lines+markers", name="RCP 8.5 (high emissions)",
            line=dict(color="#B91C1C", width=3)))
        fig_t.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Mean Temperature (°C)", xaxis_title="Year",
            legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.subheader("Rainfall Trajectory 2026–2030")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=annual.index, y=annual['pr_rcp45_mmday'],
            mode="lines+markers", name="RCP 4.5",
            line=dict(color="#0A7B6E", width=3), fill="tozeroy",
            fillcolor="rgba(10,123,110,0.08)"))
        fig_p.add_trace(go.Scatter(x=annual.index, y=annual['pr_rcp85_mmday'],
            mode="lines+markers", name="RCP 8.5",
            line=dict(color="#D97706", width=3)))
        fig_p.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0),
            yaxis_title="Mean Precipitation (mm/day)", xaxis_title="Year",
            legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("---")

    # Monthly seasonal cycle
    st.subheader("Monthly Climate Cycle — CORDEX-SA Downscaled (2026–2030 avg)")
    cordex_m = cordex.copy()
    cordex_m['month'] = cordex_m.index.month
    monthly_avg = cordex_m.groupby('month').mean()
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_season = go.Figure()
    fig_season.add_trace(go.Bar(x=MONTHS, y=monthly_avg['pr_rcp45_mmday'],
        name="Rainfall RCP4.5", marker_color="#0A7B6E", yaxis="y"))
    fig_season.add_trace(go.Scatter(x=MONTHS, y=monthly_avg['tas_rcp45_C'],
        name="Temp RCP4.5", line=dict(color="#B91C1C", width=3), yaxis="y2"))
    fig_season.update_layout(
        height=340, margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(title="Rainfall (mm/day)", side="left"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15),
        xaxis=dict(type="category"))
    st.plotly_chart(fig_season, use_container_width=True)
    st.caption("Clear monsoon signal — June-September rainfall peak, pre-monsoon April-May temperature maximum. "
               "This is the seasonal structure the drought model uses for forward projection.")

    st.markdown("---")
    st.info("""
**Why this matters for the jury:** Our 2026–2040 drought projections are no longer based on IPCC AR6
scaling factors applied to a baseline. They are driven by **actual regional climate model simulations**
from IITM-RegCM4-4 — the same Indian climate model used by IMD and IITM Pune for national assessments.
Sourced from Copernicus CDS, 0.44° resolution, RCP4.5 and RCP8.5 scenarios, daily data aggregated to
monthly and annual. **This is publication-grade climate methodology.**
    """)
