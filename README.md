
# Real-Time Inbound Tracking Platform 🚚⚡  
A fully interactive, real-time logistics intelligence platform integrating **Databricks Lakehouse**, **FastAPI**, **Streamlit**, and **Machine Learning** — all automated .

---

## 🎯 Project Purpose  
This platform demonstrates how a modern logistics company can track inbound shipments live, compute KPIs, predict delays using ML models, and visualize everything in real time with enterprise-grade dashboards.

---

# ✅ What We Built So Far (Full Summary)

---

## 🏗 1. Databricks Lakehouse Pipeline (Bronze → Silver → Gold)

### **Bronze Layer**
- Raw CSV ingestion  
- New uploads via GUI will **restart the entire ETL pipeline**  
- No business logic here

### **Silver Layer**
- Cleaning, standardizing, typing  
- Adding computed fields:  
  - `delay_minutes`  
  - `is_late`  
  - structured timestamps  
  - weather & traffic normalization

### **Gold Layer**
Business-ready tables + KPI views:

- `fact_trips`
- `v_kpi_by_supplier`
- `v_kpi_by_region`
- `v_kpi_by_weather`
- `v_kpi_daily`
- `v_kpi_weekly`

The **API & GUI read ONLY from Gold** — Databricks handles all heavy compute.

---

# 🚀 2. FastAPI Backend

### Core Features:
- `/live-trips` (Gold view)  
- `/kpi` (global metrics)  
- `/kpi/supplier`  
- `/kpi/region`  
- `/kpi/weather`  
- `/kpi/daily`  
- `/kpi/weekly`  
- `/predict` — ML delay prediction (classification + ETA regression)  

### ML Models Used
- RandomForestClassifier (late vs not late)
- GradientBoostingRegressor (ETA estimate)
- Exported with joblib  
- Loaded dynamically by the API  

---

# 🖥 3. Streamlit Application (Premium Edition)

We redesigned the GUI completely with a **professional dashboard layout**.

### Tabs:
---

### **🧱 PIPELINE TAB**
- Buttons:
  - **Refresh KPIs Now**
  - **Refresh Live Stream**
  - **Auto-Refresh ON/OFF** toggle  
- Large Pipeline Diagram Image  
- Raw sample preview table (randomized)
- Event log sidebar

---

### **📊 LIVE DASHBOARD**
- **Animated PyDeck Map**
  - ArcLayer showing routes (origin → warehouse)
  - Colored by delay severity
  - Hover info: supplier, truck, status, delay
- Live KPIs:
  - Total trips
  - Late trips
  - Severe delays
  - Late %
- Real-time simulation using jitter (+/- few minutes)
- Auto-refresh option every 3 seconds
- Interactive trip table

---

### **🤖 ML PLAYGROUND**
- Prediction widgets:
  - Distance slider
  - Estimated time
  - Current delay
  - Traffic
  - Weather
  - Warehouse
- Gauge speedometer result
- Fancy colored prediction card
- All prediction results **saved inside session_state**
- Export all predictions to CSV (`ml_predictions_export.csv`)

---

# 🗺 4. Advanced Map Implementation

We replaced circles with **true logistics paths**:

- Drawn using **ArcLayer**
- Route colored by delay:
  - Green = On time
  - Orange = Moderate delay
  - Red = Severe delay
- Hover shows:
  - Supplier  
  - Truck ID  
  - Status  
  - Delay  
- Automatic coordinate assignment for missing cities

---

# 📦 5. Export Features

- Export model predictions
- Export entire session’s ML tests
- Export Chart as HTML
- Export KPI results (if needed)

---

# 🔧 6. Improvements Prepared

- Better icons (replacing ugly stickman)
- Full-road-route drawing (OSRM API optional)
- Slider-based KPI filtering
- Plotly advanced visuals (Histogram, Treemap, Radar, Waterfall)
- Cloud dashboard export option (Plotly Cloud)
- Deployment-ready structure

---

# 📚 7. How to Run Locally

```bash
git clone https://github.com/JPdev6/real-time-inbound-tracking-platform
cd real-time-inbound-tracking-platform
pip install -r requirements.txt
```

### Start API:
```bash
uvicorn src.main:app --reload
```

### Start Streamlit GUI:
```bash
streamlit run streamlit_app.py
```

---

# 🌐 8. How Everything Connects

```
Databricks (Bronze → Silver → Gold)
          ↓
     FastAPI Backend
          ↓
   Streamlit Real-Time Dashboard
          ↓
   User Interactions + ML Predictions
```

A complete vertical pipeline from **raw data → ML → dashboard**.

---

# 💡 9. Next Steps (Recommended)

- Add **full path routing** with real GPS routes  
- Add slider-controlled KPI exploration  
- Replace PyDeck with Mapbox GL for full animations  
- Deploy:
  - API → Render / Azure  
  - Streamlit → Cloudflare Pages / Streamlit Cloud  
  - Databricks → Production workflows  

---

Made with ❤️ for your project presentation  
