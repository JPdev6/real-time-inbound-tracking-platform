from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Trip(BaseModel):
    trip_id: int
    truck_id: Optional[str] = None
    supplier: Optional[str] = None
    warehouse: Optional[str] = None
    delay_minutes: Optional[float] = 0.0
    planned_arrival: datetime
    actual_arrival: Optional[datetime] = None
    origin_city: Optional[str] = None    # ✅ add this
    status: str


class KPIResponse(BaseModel):
    total_deliveries: int
    on_time_rate: float
    late_rate: float
    avg_delay_minutes: float


class PredictRequest(BaseModel):
    distance_km: float
    estimated_time_min: float
    delay_minutes: float
    traffic: str
    weather: str
    warehouse: str
    supplier: Optional[str] = None
    planned_arrival: Optional[datetime] = None

class PredictResponse(BaseModel):
    prediction: str
    probability_late: float
    eta_delay_minutes: float

class SupplierKPI(BaseModel):
    supplier: str
    total_deliveries: int
    on_time_rate: float
    late_rate: float
    avg_delay_minutes: float


class SupplierScoreResponse(BaseModel):
    supplier: str
    score: float
    late_rate: float
    avg_delay_minutes: float

class ETARequest(BaseModel):
    distance_km: float
    package_weight_kg: float
    region: Optional[str] = None
    weather: Optional[str] = None
    delivery_mode: Optional[str] = None


class ETAResponse(BaseModel):
    eta_hours: float
