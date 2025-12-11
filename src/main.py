from datetime import datetime
from typing import  List ,Dict ,Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .train_models import train_all_models
from .ml_model import predict_delay
from .schemas import (
    KPIResponse,
    SupplierKPI,
    SupplierScoreResponse,
    ETARequest,
    ETAResponse,
    PredictResponse,
    PredictRequest
)
from .services import (
    get_kpi_service,
    get_supplier_kpis_service,
    get_supplier_score_service
)

app = FastAPI(
    title="Real-Time Inbound Tracking Platform",
    version="0.1.0",
    description="FastAPI backend on top of Databricks Lakehouse (Silver/Gold).",
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
    <html>
    <head>
        <title>Inbound Monitoring Platform</title>
        <style>
            body {
                background-color: #0d0d0d;
                color: white;
                text-align: center;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            h1 {
                margin-top: 30px;
            }
            .container {
                margin-top: 25px;
            }
            .btn {
                display: inline-block;
                padding: 12px 25px;
                margin: 10px;
                font-size: 18px;
                border-radius: 8px;
                text-decoration: none;
                color: white;
                background: #1e90ff;
                transition: 0.2s;
            }
            .btn:hover {
                background: #63b3ff;
            }
            img {
                margin-top: 40px;
                max-width: 90%;
                border-radius: 12px;
                box-shadow: 0 0 20px rgba(255,255,255,0.15);
            }
        </style>
    </head>

    <body>
        <h1>Inbound Monitoring Platform</h1>

        <div class="container">
            <a href="/docs" class="btn">API Docs</a>
        </div>

        <img src="/static/pipeline_overview.png">
    </body>
    </html>
    """

@app.get("/kpi", response_model=KPIResponse)
def kpi():
    try:
        return get_kpi_service()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    result = predict_delay(payload.dict())
    return result


@app.post("/train-models")
def train_models_endpoint():
    """
    Trigger training from UI (Streamlit).
    Returns metrics for both models.
    """
    metrics = train_all_models()
    return metrics


@app.get("/kpi/by-supplier", response_model=List[SupplierKPI])
def kpi_by_supplier(supplier: str | None = None):
    """
    If ?supplier=... is provided → KPIs for that supplier only.
    If not → list of KPIs for all suppliers.
    """
    try:
        return get_supplier_kpis_service(supplier=supplier)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/supplier-score", response_model=SupplierScoreResponse)
def supplier_score(supplier: str):
    try:
        return get_supplier_score_service(supplier)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/eta", response_model=ETAResponse)
def eta(req: ETARequest):
    try:
        return estimate_eta(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))