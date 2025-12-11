#This connects Databricks SQL layer → schemas → API.

# src/services.py
from typing import List, Dict, Any, Optional
from .databricks_db import fetch_kpis, fetch_live_trips, fetch_supplier_kpis
from .schemas import KPIResponse, Trip, SupplierKPI, SupplierScoreResponse


def get_live_trips_service(limit: Optional[int] = 10000):
    """
    Read live trips from Gold. If limit=None → full table.
    Default 10k to avoid melting your laptop.
    """
    return fetch_live_trips(limit=limit)


def get_kpi_service() -> KPIResponse:
    data = fetch_kpis()
    return KPIResponse(
        total_deliveries=int(data.get("total_deliveries", 0) or 0),
        on_time_rate=float(data.get("on_time_rate", 0.0) or 0.0),
        late_rate=float(data.get("late_rate", 0.0) or 0.0),
        avg_delay_minutes=float(data.get("avg_delay_minutes", 0.0) or 0.0),
    )
from typing import List  # ensure imported


def get_supplier_kpis_service(supplier: str | None = None) -> List[SupplierKPI]:
    rows = fetch_supplier_kpis(supplier=supplier)
    return [
        SupplierKPI(
            supplier=row["supplier"],
            total_deliveries=int(row["total_deliveries"]),
            on_time_rate=float(row["on_time_rate"]),
            late_rate=float(row["late_rate"]),
            avg_delay_minutes=float(row["avg_delay_minutes"]),
        )
        for row in rows
    ]


def get_supplier_score_service(supplier: str) -> SupplierScoreResponse:
    rows = fetch_supplier_kpis(supplier=supplier)
    if not rows:
        # supplier not found → score 0
        return SupplierScoreResponse(
            supplier=supplier,
            score=0.0,
            late_rate=0.0,
            avg_delay_minutes=0.0,
        )

    r = rows[0]
    late_rate = float(r["late_rate"])
    avg_delay = float(r["avg_delay_minutes"])

    # simple scoring formula: lower late_rate & delay => higher score
    # 1.0 is perfect, 0.0 is worst
    score = (1.0 - late_rate) * 0.7 + (1.0 / (1.0 + max(avg_delay, 0.0))) * 0.3

    return SupplierScoreResponse(
        supplier=r["supplier"],
        score=round(score, 4),
        late_rate=late_rate,
        avg_delay_minutes=avg_delay,
    )