# -------------------------------------------------------
# INBOUND MONITORING STREAMLIT – PREMIUM EDITION
# -------------------------------------------------------
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List
# Make sure pydeck does NOT expect a Mapbox token
import pydeck as pdk
pdk.settings.mapbox_api_key = ""  # force fallback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

from src.databricks_db import (
    fetch_kpi_by_region,
    fetch_kpi_by_weather,
    fetch_kpi_daily,
    fetch_kpi_weekly,
    fetch_raw_sample
)
from src.services import (
    get_kpi_service,
    get_supplier_kpis_service,
)

# -------------------------------------------------------
# STREAMLIT CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="Inbound Monitoring – Real-Time Dashboard",
    layout="wide",
)

# -------------------------------------------------------
# GLOBAL CSS (tabs + headings)
# -------------------------------------------------------
st.markdown(
    """
<style>
/* --- Tabs --- */
div[data-baseweb="tab-list"] {
    display: flex;
    gap: 40px !important;
    justify-content: center;
    padding-top: 10px;
    padding-bottom: 10px;
}

div[data-baseweb="tab"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
    border-radius: 6px !important;
    color: #dddddd !important;
}

div[aria-selected="true"][data-baseweb="tab"] {
    color: #ff6b6b !important;
    font-size: 24px !important;
}

/* underline */
div[data-baseweb="tab-highlight"] {
    background-color: #ff6b6b !important;
    height: 4px !important;
    border-radius: 2px;
}

/* Bigger section titles */
.big-title {
    font-size: 32px !important;
    font-weight: 800 !important;
    padding: 6px 0px;
    color: #ffffff;
    text-shadow: 0px 0px 12px rgba(0,0,0,0.5);
}
</style>
""",
    unsafe_allow_html=True,
)
# -------------------------------------------------------
# GLOBAL SESSION STATE
# -------------------------------------------------------
if "event_log" not in st.session_state:
    st.session_state.event_log = []

if "live_trips" not in st.session_state:
    st.session_state.live_trips = []

if "live_metrics" not in st.session_state:
    st.session_state.live_metrics = {
        "total": 0,
        "late": 0,
        "severe": 0,
        "late_pct": 0.0,
    }

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# -------------------------------------------------------
# LOGGING
# -------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.event_log.append(f"[{ts}] {msg}")


# -------------------------------------------------------
# REAL-TIME TRIP SIMULATION
# -------------------------------------------------------
def simulate_realtime_changes(trips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add small random noise to give a 'live' feeling."""
    simulated: List[Dict[str, Any]] = []

    for t in trips:
        t = dict(t)  # ensure dict

        # Delay jitter
        delta = random.randint(-3, 3)
        t["delay_minutes"] = max(0, t.get("delay_minutes", 0) + delta)

        # Update status
        t["status"] = "Delivered Late" if t["delay_minutes"] > 0 else "Delivered"

        # Adjust actual arrival (if parseable)
        if "actual_arrival" in t and t["actual_arrival"] is not None:
            try:
                t["actual_arrival"] = pd.to_datetime(t["actual_arrival"]) + timedelta(
                    minutes=delta
                )
            except Exception:
                pass

        simulated.append(t)

    return simulated


def compute_live_metrics(trips: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trips)
    late = sum(
        1 for t in trips
        if str(t.get("status", "")).lower().startswith("delivered late")
    )
    severe = sum(1 for t in trips if t.get("delay_minutes", 0) > 30)

    return {
        "total": total,
        "late": late,
        "severe": severe,
        "late_pct": late / total if total else 0.0,
    }


def recompute_metrics_with_threshold(
    trips: List[Dict[str, Any]],
    threshold: int,
) -> Dict[str, Any]:
    """Metrics but with dynamic 'severe' threshold."""
    total = len(trips)
    late = sum(
        1 for t in trips
        if str(t.get("status", "")).lower().startswith("delivered late")
    )
    severe = sum(1 for t in trips if t.get("delay_minutes", 0) > threshold)

    return {
        "total": total,
        "late": late,
        "severe": severe,
        "late_pct": late / total if total else 0.0,
    }


def update_live_stream() -> None:
    from src.services import get_live_trips_service

    # 🔥 pull a big chunk from Gold, not just 200
    base_trips = get_live_trips_service(limit=None)  # or limit=None if you dare

    trips: List[Dict[str, Any]] = []
    for t in base_trips:
        if hasattr(t, "model_dump"):
            trips.append(t.model_dump())
        elif hasattr(t, "dict"):
            trips.append(t.dict())
        else:
            trips.append(dict(t))

    # keep your simulation if you want “live feeling”
    stream = simulate_realtime_changes(trips)

    st.session_state.live_trips = stream
    st.session_state.live_metrics = compute_live_metrics(stream)
    st.session_state["live_last_update"] = time.strftime("%H:%M:%S")
    log("Live stream updated from Gold view (limit=10 000).")

# -------------------------------------------------------
# PREMIUM HEADER
# -------------------------------------------------------
def header(title: str) -> None:
    st.markdown(
        f"""
        <div class="big-title">{title}</div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------
# CITY → COORDS (FOR ORIGIN / WAREHOUSE)
# -------------------------------------------------------
CITY_COORDS = {
    "Munich": (48.1351, 11.5820),
    "Frankfurt": (50.1109, 8.6821),
    "Cologne": (50.9375, 6.9603),
    "Essen": (51.4556, 7.0116),
    "Leipzig": (51.3397, 12.3731),
    "Berlin": (52.5200, 13.4050),
    "Hamburg": (53.5511, 9.9937),
    "Dortmund": (51.5136, 7.4653),
}


def city_coord(city: str) -> tuple[float, float]:
    """Map warehouse / city name to approximate lat/lon."""
    if not isinstance(city, str):
        return 50.0, 10.0
    return CITY_COORDS.get(city, (50.0, 10.0))


# -------------------------------------------------------
# LIVE MAP (PYDECK) – origin_city → warehouse
# -------------------------------------------------------
def live_map(trips: List[Dict[str, Any]], max_routes: int = 20) -> None:
    """
    PyDeck map:
    - Show a small random sample of trips as arcs origin_city → warehouse.
    - Uses columns: origin_city, warehouse, delay_minutes, trip_id, supplier, truck_id, status.
    """
    st.caption(f"live_map(): received {len(trips)} trips")

    if not trips:
        st.info("No trips to display on map.")
        return

    df = pd.DataFrame(trips).copy()

    # Require warehouse column for location
    if "warehouse" not in df.columns:
        st.error("Column 'warehouse' not found in live trips – cannot build map.")
        return

    # 1) Ensure delay_minutes exists and is numeric (but do NOT randomize)
    if "delay_minutes" not in df.columns:
        df["delay_minutes"] = 0
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0)

    # 2) Take a RANDOM SAMPLE of trips (10) instead of "worst delayed"
    SAMPLE_SIZE = 10
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    # if len(df) <= 10, we just show all of them

    if df.empty:
        st.info("No trips found to visualize.")
        return

    # -----------------------------------------
    # Build coordinates: origin_city → warehouse
    # -----------------------------------------
    origin_lats, origin_lons, dest_lats, dest_lons = [], [], [], []
    rng = np.random.default_rng(42)  # stable jitter

    for _, row in df.iterrows():
        origin_city = row.get("origin_city") or row.get("warehouse") or "Munich"
        warehouse = row.get("warehouse") or "Munich"

        o_lat, o_lon = city_coord(origin_city)
        d_lat, d_lon = city_coord(warehouse)

        # small jitter so routes don’t sit exactly on top of each other
        jitter = 0.15
        o_lat += (rng.random() - 0.5) * jitter
        o_lon += (rng.random() - 0.5) * jitter
        d_lat += (rng.random() - 0.5) * jitter
        d_lon += (rng.random() - 0.5) * jitter

        origin_lats.append(o_lat)
        origin_lons.append(o_lon)
        dest_lats.append(d_lat)
        dest_lons.append(d_lon)

    df["origin_lat"] = origin_lats
    df["origin_lon"] = origin_lons
    df["dest_lat"] = dest_lats
    df["dest_lon"] = dest_lons

    # Color by delay severity (using REAL delay_minutes from DB)
    def delay_color(d: float) -> list[int]:
        if d > 45:
            return [220, 50, 50]    # red
        if d > 15:
            return [230, 170, 60]   # orange
        return [70, 200, 120]       # green

    df["color"] = df["delay_minutes"].apply(delay_color)

    # Center the map on all routes
    center_lat = float(df[["origin_lat", "dest_lat"]].mean().mean())
    center_lon = float(df[["origin_lon", "dest_lon"]].mean().mean())

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=5,
        pitch=40,
        bearing=20,
    )

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=df,
        get_source_position=["origin_lon", "origin_lat"],
        get_target_position=["dest_lon", "dest_lat"],
        get_source_color="color",
        get_target_color="color",
        get_width=4,
        pickable=True,
        auto_highlight=True,
    )

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["dest_lon", "dest_lat"],
        get_radius=7000,
        get_color="[230, 230, 230]",
        pickable=True,
    )

    tooltip = {
        "html": "<b>Trip {trip_id}</b><br/>"
                "Origin: {origin_city}<br/>"
                "Warehouse: {warehouse}<br/>"
                "Supplier: {supplier}<br/>"
                "Vehicle: {truck_id}<br/>"
                "Status: {status}<br/>"
                "Delay: {delay_minutes} min",
        "style": {
            "backgroundColor": "rgba(0, 0, 0, 0.85)",
            "color": "white",
            "fontSize": "12px",
        },
    }

    deck = pdk.Deck(
        map_style=None,         # Carto / OSM background
        map_provider="carto",
        initial_view_state=view_state,
        layers=[arc_layer, point_layer],
        tooltip=tooltip,
    )

    st.pydeck_chart(deck)

# -------------------------------------------------------
# SIDEBAR LOG PANEL
# -------------------------------------------------------
with st.sidebar.expander("📝 System Log", expanded=True):
    if st.session_state.event_log:
        st.write("\n".join(st.session_state.event_log[-40:]))
    else:
        st.write("No logs yet.")

# =======================================================
#                  TABS LAYOUT
# =======================================================
tab_pipeline, tab_dashboard, tab_ml = st.tabs(
    ["🧱 Pipeline", "📊 Live Dashboard", "🤖 ML Playground"]
)

# -------------------------------------------------------
# 🧱 PIPELINE TAB
# -------------------------------------------------------
with tab_pipeline:
    header("Pipeline Overview")

    st.markdown(
        """
- **Bronze:** `inbound_monitoring.raw.logistics_deliveries` – raw CSV.  
- **Silver:** `inbound_monitoring.silver.logistics_trips` – cleaned & enriched.  
- **Gold:** `inbound_monitoring.gold.fact_trips` + KPI views – dashboards & ML.  


        """
    )

    col1, col2 = st.columns([3,1])

    with col1:
    # ----------------------------------------------------
    # PIPELINE DIAGRAM IMAGE UNDER BUTTONS
    # ----------------------------------------------------

        st.image(
            "static/pipeline_overview.png",  # <-- put your file here!
            caption="One-Click Logistics Intelligence Pipeline",
            width="stretch",
        )

    with col2:
        if st.button("🔄 Refresh KPIs Now"):
            log("Manual KPI refresh invoked (Gold rebuilt in Databricks jobs).")

        if st.button("⚡ Refresh Live Stream"):
            update_live_stream()

        toggle_label = (
            "🟢 Auto-Refresh ON" if st.session_state.auto_refresh else "🔴 Auto-Refresh OFF"
        )
        if st.button(toggle_label, key="auto_refresh_toggle"):
            st.session_state.auto_refresh = not st.session_state.auto_refresh
            log(f"Auto-refresh set to {st.session_state.auto_refresh}")
            st.rerun()

        if st.session_state.auto_refresh:
            st.caption("Auto-refreshing Live Dashboard every 3 seconds…")

    # --------------------------------------------------
    # Random RAW sample from Bronze layer
    # --------------------------------------------------
    st.markdown("### 🔍 Raw Data (Data before cleaning)")

    try:
        raw_sample = fetch_raw_sample(limit=20)
        raw_df = pd.DataFrame(raw_sample)
        st.dataframe(raw_df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load RAW sample from Databricks: {e}")
# -------------------------------------------------------
# 📊 LIVE DASHBOARD TAB – FULL KPI DASHBOARD + SLIDERS
# -------------------------------------------------------
with tab_dashboard:
    # Auto-refresh for live stream
    if st.session_state.auto_refresh:
        time.sleep(3)
        update_live_stream()
        st.rerun()

    header("Live Trips & KPI Dashboard")

    # 1) GLOBAL KPI METRICS FROM GOLD
    log("Fetching KPI summary from service layer…")
    kpi = get_kpi_service()

    total_deliveries = kpi.total_deliveries
    on_time_rate = kpi.on_time_rate
    late_rate = kpi.late_rate
    avg_delay_minutes = kpi.avg_delay_minutes

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Deliveries", f"{total_deliveries:,}")
    colB.metric("On-time Rate", f"{on_time_rate * 100:.1f}%")
    colC.metric("Late Rate", f"{late_rate * 100:.1f}%")
    colD.metric("Avg Delay (min)", f"{avg_delay_minutes:.1f}")

    st.markdown("---")

    # Ensure we have trips
    if not st.session_state.live_trips:
        update_live_stream()
    trips = st.session_state.live_trips

    st.markdown("### 🔎 Filter Trips by Delay Minutes")

    # Extract delay values from ALL trips
    all_delays = [
        float(t.get("delay_minutes", 0) or 0)
        for t in trips
    ]

    if all_delays:
        data_min = min(all_delays)
        data_max = max(all_delays)

        # Round a bit for nicer slider edges
        slider_min = float(max(0, int(data_min)))  # e.g. 0
        slider_max = float(int(data_max) + 1)  # e.g. 63 -> 64

        delay_min, delay_max = st.slider(
            "Show trips with delay between (minutes):",
            min_value=slider_min,
            max_value=slider_max,
            value=(slider_min, slider_max),
            step=1.0,
        )
    else:
        st.warning("No delay data available; using default delay range.")
        delay_min, delay_max = 0.0, 60.0

    # Filter the trips based on selected delay range
    filtered_trips = [t for t in trips if delay_min <= t.get("delay_minutes", 0) <= delay_max]

    # Recompute metrics on filtered trips
    metrics = compute_live_metrics(filtered_trips)

    st.markdown("### 🚚 Live Trips (filtered by delay range)")

    col_live1, col_live2 = st.columns([2, 1])

    with col_live1:
        df_filtered = pd.DataFrame(filtered_trips)
        st.write(f"Showing **{len(df_filtered):,}** trips in this range.")
        st.dataframe(df_filtered, width="stretch", height=400)  # 👈 shows all rows, scrollable

    with col_live2:
        st.metric("Trips in range", metrics["total"])
        st.metric("Late deliveries in range", metrics["late"])
        st.metric("Severe delays (>30 min)", metrics["severe"])
        st.metric("Late % (range)", f"{metrics['late_pct'] * 100:.1f}%")
        st.caption(f"Delay range: {delay_min:.1f}–{delay_max:.1f} minutes")

    # For the map, show the worst delayed routes within the selected range
    sorted_for_map = sorted(
        filtered_trips,
        key=lambda t: float(t.get("delay_minutes", 0) or 0),
        reverse=True,
    )

    top_for_map = sorted_for_map[:20]  # longest delays

    st.markdown("### 🗺 Live Map – Top delayed routes in current delay range")
    live_map(top_for_map, max_routes=20)

    st.markdown("---")

    # 3) SUPPLIER KPI DASHBOARD – BUBBLE CHART + SLIDER
    st.markdown("### 🏭 Supplier KPI Overview")

    supplier_filter = st.text_input(
        "Filter by Supplier (exact match, optional)", value="", key="supplier_filter"
    )

    on_time_min = st.slider(
        "Minimum on-time rate for suppliers shown",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.05,
    )

    log("Fetching supplier KPIs from service layer…")
    if supplier_filter.strip():
        suppliers = get_supplier_kpis_service(supplier_filter.strip())
    else:
        suppliers = get_supplier_kpis_service()

    supplier_records: List[Dict[str, Any]] = [
        s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in suppliers
    ]
    supplier_df = pd.DataFrame(supplier_records)

    if supplier_df.empty:
        st.warning("No supplier KPI data returned from Gold.")
    else:
        # Filter by on-time threshold
        supplier_df = supplier_df[supplier_df["on_time_rate"] >= on_time_min]

        if supplier_df.empty:
            st.info("No suppliers meet the selected on-time rate threshold.")
        else:
            st.write(
                f"Showing **{len(supplier_df)}** suppliers (≥ {on_time_min:.0%} on-time)."
            )

            # Take top N by deliveries for the visual
            top_n = min(50, len(supplier_df))
            sup_view = supplier_df.sort_values(
                "total_deliveries", ascending=False
            ).head(top_n)

            fig_sup = px.scatter(
                sup_view,
                x="on_time_rate",
                y="avg_delay_minutes",
                size="total_deliveries",
                color="late_rate",
                hover_name="supplier",
                hover_data={
                    "total_deliveries": True,
                    "on_time_rate": ":.2f",
                    "late_rate": ":.2f",
                    "avg_delay_minutes": ":.1f",
                },
                color_continuous_scale="RdYlGn_r",
                labels={
                    "on_time_rate": "On-time rate",
                    "avg_delay_minutes": "Avg delay (min)",
                    "late_rate": "Late rate",
                },
                title=f"Top {top_n} Suppliers – Volume vs On-time vs Delay",
            )

            fig_sup.update_layout(
                xaxis=dict(tickformat=".0%", title="On-time rate"),
                coloraxis_colorbar=dict(title="Late rate"),
                legend_title_text="",
                height=500,
            )

            st.plotly_chart(fig_sup, use_container_width=True)

            st.caption(
                "• Bubble size = total deliveries · Color = late rate · "
                "Y-axis = average delay in minutes."
            )

    st.markdown("---")

    # 4) REGION & WEATHER KPI – WEATHER AS RADIAL/POLAR
    st.markdown("### 🌍 Region & Weather Impact")

    region_df = pd.DataFrame(fetch_kpi_by_region())
    weather_df = pd.DataFrame(fetch_kpi_by_weather())

    col_reg, col_wea = st.columns(2)

    with col_reg:
        if region_df.empty:
            st.info("No region KPIs.")
        else:
            fig_region = px.bar(
                region_df.sort_values("on_time_rate", ascending=False),
                x="region",
                y="on_time_rate",
                hover_data=["total_deliveries", "late_rate", "avg_delay_minutes"],
                title="On-time Performance by Region",
            )
            fig_region.update_layout(
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_region, use_container_width=True)

    with col_wea:
        if weather_df.empty:
            st.info("No weather KPIs.")
        else:
            w = weather_df.sort_values("on_time_rate", ascending=False)
            fig_weather = px.bar_polar(
                w,
                r="late_rate",
                theta="weather_condition",
                color="avg_delay_minutes",
                color_continuous_scale="Turbo",
                title="Late Rate & Delay by Weather Condition",
                hover_data={
                    "total_deliveries": True,
                    "on_time_rate": ":.2f",
                    "late_rate": ":.2f",
                    "avg_delay_minutes": ":.1f",
                },
            )
            fig_weather.update_layout(
                polar=dict(
                    radialaxis=dict(
                        tickformat=".0%",
                        showline=False,
                    )
                ),
                height=500,
            )
            st.plotly_chart(fig_weather, use_container_width=True)

            st.caption(
                "• Distance from center = late rate · Color = avg delay · "
                "Hover for full KPIs per weather type."
            )

    st.markdown("---")

    # 5) DAILY & WEEKLY TRENDS – AREA + ROLLING + HEATMAP
    st.markdown("### 📈 Time-Series On-time Performance")

    daily_df = pd.DataFrame(fetch_kpi_daily())
    weekly_df = pd.DataFrame(fetch_kpi_weekly())

    col_day, col_week = st.columns(2)

    with col_day:
        if daily_df.empty:
            st.info("No daily KPI data.")
        else:
            daily = daily_df.sort_values("day").copy()
            daily["on_time_rate_rolling7"] = daily["on_time_rate"].rolling(7).mean()

            max_len = len(daily)
            window = st.slider(
                "Show last N days",
                min_value=7,
                max_value=min(365, max_len),
                value=min(90, max_len),
                step=7,
            )
            daily_window = daily.tail(window)

            fig_daily = go.Figure()

            # Area for daily OT rate
            fig_daily.add_trace(
                go.Scatter(
                    x=daily_window["day"],
                    y=daily_window["on_time_rate"],
                    mode="lines",
                    name="Daily on-time rate",
                    line=dict(width=0.5),
                    fill="tozeroy",
                    hovertemplate="Day=%{x}<br>On-time=%{y:.1%}<extra></extra>",
                )
            )

            # 7-day rolling
            fig_daily.add_trace(
                go.Scatter(
                    x=daily_window["day"],
                    y=daily_window["on_time_rate_rolling7"],
                    mode="lines",
                    name="7-day rolling avg",
                    line=dict(width=3),
                    hovertemplate="Day=%{x}<br>7d avg=%{y:.1%}<extra></extra>",
                )
            )

            fig_daily.update_layout(
                height=450,
                xaxis_title="Date",
                yaxis_title="On-time rate",
                yaxis_tickformat=".0%",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                title=f"Daily On-time Rate – last {window} days",
            )

            st.plotly_chart(fig_daily, use_container_width=True)

    with col_week:
        if weekly_df.empty:
            st.info("No weekly KPI data.")
        else:
            weekly = weekly_df.copy()
            weekly["week_start"] = pd.to_datetime(weekly["week_start"])
            weekly["year"] = weekly["week_start"].dt.year
            weekly["week"] = weekly["week_start"].dt.isocalendar().week

            # 🔧 FIX: aggregate per (year, week) so pivot has unique keys
            weekly_agg = (
                weekly.groupby(["year", "week"], as_index=False)
                .agg({"on_time_rate": "mean"})
            )

            pivot = weekly_agg.pivot(
                index="year",
                columns="week",
                values="on_time_rate",
            ).sort_index().sort_index(axis=1)

            fig_week = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                labels=dict(color="On-time rate"),
                title="On-time Rate by Year / Week",
            )
            fig_week.update_coloraxes(colorbar_tickformat=".0%")

            st.plotly_chart(fig_week, use_container_width=True)


# -------------------------------------------------------
# 🤖 ML PLAYGROUND TAB – REAL TRAINING + PREDICTION
# -------------------------------------------------------
with tab_ml:
    header("ML Playground – Train & Predict")

    # ========== 1) TRAIN MODELS ==========
    st.markdown("### 🧪 Train Models on Gold.fact_trips")

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    col_train_left, col_train_right = st.columns([1, 2])

    with col_train_left:
        st.markdown(
            "This will:\n"
            "- Load data from **Gold.fact_trips**\n"
            "- Train a **Late vs On-time classifier**\n"
            "- Train an **ETA (delay minutes) regressor**\n"
            "- Save them to `models/`\n"
        )
        if st.button("🏋️ Train / Re-train models"):
            with st.spinner("Training models on Gold data..."):
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/train-models",
                        timeout=300,
                    )
                    resp.raise_for_status()
                    train_metrics = resp.json()
                    st.success("Training completed.")
                except Exception as e:
                    st.error(f"Error calling /train-models: {e}")
                    train_metrics = None

    with col_train_right:
        if "train_metrics" in locals() and train_metrics:
            st.markdown("#### Training Metrics")
            st.json(train_metrics)
        else:
            st.info("Click the button to train models and see metrics here.")

    st.markdown("---")

    # ========== 2) PREDICT USING TRAINED MODELS ==========
    st.markdown("### 🔮 Predict Delay & ETA")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Input Features")

        distance_km = st.number_input("Distance (km)", min_value=0.0, value=50.0)
        estimated_time_min = st.number_input(
            "Estimated Travel Time (min)", min_value=0.0, value=60.0
        )
        delay_minutes = st.number_input(
            "Current Delay (min, can be 0)", min_value=-60.0, value=0.0
        )
        traffic = st.selectbox("Traffic", ["low", "medium", "high"], index=1)
        weather = st.selectbox(
            "Weather", ["clear", "rain", "snow", "storm"], index=1
        )
        warehouse = st.text_input("Warehouse", value="Munich")

        run_pred = st.button("⚡ Predict with trained models")

    if run_pred:
        payload = {
            "distance_km": distance_km,
            "estimated_time_min": estimated_time_min,
            "delay_minutes": delay_minutes,
            "traffic": traffic,
            "weather": weather,
            "warehouse": warehouse,
        }

        try:
            with st.spinner("Calling FastAPI /predict ..."):
                r = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=payload,
                    timeout=10,
                )
                r.raise_for_status()
                out = r.json()
                # Save prediction to history
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "distance_km": distance_km,
                    "estimated_time_min": estimated_time_min,
                    "delay_minutes": delay_minutes,
                    "traffic": traffic,
                    "weather": weather,
                    "warehouse": warehouse,
                    "prediction": out["prediction"],
                    "probability_late": out["probability_late"],
                    "eta_delay_minutes": out.get("eta_delay_minutes", None),
                }
                st.session_state.prediction_history.append(record)
        except Exception as e:
            st.error(f"Predict API error: {e}")
            out = None

        with col_right:
            if out:
                prob = float(out["probability_late"])
                pred = out["prediction"]
                eta_delay = float(out.get("eta_delay_minutes", 0.0))

                # Card styling
                if pred.lower() == "late":
                    color = "#ff4b4b"
                    icon = "🚨"
                    msg = "High risk of late delivery. Consider rerouting or rescheduling."
                else:
                    color = "#4caf50"
                    icon = "✅"
                    msg = "Delivery is likely to arrive on time."

                st.markdown(
                    f"""
                    <div style="
                        padding: 20px;
                        border-radius: 12px;
                        background-color: {color}20;
                        border: 1px solid {color};
                        margin-bottom: 16px;
                    ">
                        <h2 style="color:{color}; margin:0; padding:0;">
                            {icon} {pred}
                        </h2>
                        <p style="font-size:18px; margin-top:8px; margin-bottom:4px;">
                            Probability of being late:
                            <strong style="color:{color};">{prob:.0%}</strong>
                        </p>
                        <p style="font-size:16px; margin-top:4px; color:#ddd;">
                            Estimated delay:
                            <strong>{eta_delay:.1f} minutes</strong>
                        </p>
                        <p style="font-size:14px; margin-top:4px; color:#ccc;">
                            {msg}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 30], "color": "#1b5e20"},
                                {"range": [30, 60], "color": "#f9a825"},
                                {"range": [60, 100], "color": "#b71c1c"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 3},
                                "thickness": 0.75,
                                "value": prob * 100,
                            },
                        },
                    )
                )
                gauge_fig.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                )

                st.plotly_chart(gauge_fig, use_container_width=True)

                st.markdown(
                    """
                    <small>
                    The classifier estimates the probability that this shipment will arrive late.
                    The regressor estimates how many minutes earlier or later it will arrive.
                    </small>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Submit the form on the left to see prediction results.")

    st.markdown("---")
    st.markdown("### 📄 Prediction History")

    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Run a prediction above.")
    else:
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True)

        # CSV export
        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Prediction History (CSV)",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
