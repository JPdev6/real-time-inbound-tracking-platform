from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .config import settings
from databricks import sql

@contextmanager
def get_connection():
    if not (settings.dbx_host and settings.dbx_http_path and settings.dbx_token):
        raise RuntimeError("Databricks connection env vars are missing")
    conn = sql.connect(
        server_hostname=settings.dbx_host,
        http_path=settings.dbx_http_path,
        access_token=settings.dbx_token,
    )
    try:
        yield conn
    finally:
        conn.close()

    #     •	yield conn – hands the open connection back to the with block that called get_connection().
    #     •	finally: conn.close() – guarantees we close the connection whether:
    #     •	everything works, or
    #     •	an exception gets thrown
    #
    # This is clean resource management: no zombie connections, no leaks.

def query_as_dicts(query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:

    #run any SQL and get back list of dicts
    params = params or {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
    return [dict(zip(cols, row)) for row in rows]
    # with get_connection() as conn: – opens DB connection and ensures it will be closed.
    # conn.cursor() – DB cursor for executing SQL.
    # cur.execute(query, params) – runs the SQL on the warehouse, safely passing parameters.
    # cur.description – metadata about columns that came back (names, types, etc.). We extract just the names: cols = [c[0] for c in cur.description].
    # cur.fetchall() – gets all rows as tuples, e.g. (1, "TRUCK_1", "DHL", ...).
    # It’s the generic helper – any SQL we need, we call this and get Python dictionaries back.

def fetch_live_trips(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch trips from Gold.fact_trips.
    If limit is None → return ALL rows (careful: can be big).
    Otherwise return most recent <limit> by planned_arrival.
    """
    sql = """
        SELECT
            trip_id,
            truck_id,
            supplier,
            warehouse,
            origin_city,
            planned_arrival,
            actual_arrival,
            delay_minutes,
            status
        FROM inbound_monitoring.gold.fact_trips
        ORDER BY planned_arrival DESC
    """

    params: Dict[str, Any] = {}
    if limit is not None:
        sql += "\n        LIMIT :limit"
        params["limit"] = limit

    return query_as_dicts(sql, params)

def fetch_kpis() -> Dict[str, Any]:
    fq_table = f"{settings.dbx_catalog}.{settings.dbx_kpi_table}"  # e.g. inbound_monitoring.gold.fact_trips

    sql_query = f"""
        SELECT
            COUNT(*) AS total_deliveries,
            AVG(CASE WHEN is_late THEN 1.0 ELSE 0.0 END) AS late_rate,
            1.0 - AVG(CASE WHEN is_late THEN 1.0 ELSE 0.0 END) AS on_time_rate,
            AVG(delay_minutes) AS avg_delay_minutes
        FROM {fq_table}
    """
    #This query computes KPIs directly in the warehouse:
    # COUNT(*) → how many deliveries total.
    # AVG(CASE WHEN is_late THEN 1.0 ELSE 0.0 END) → fraction of late deliveries.
    #So the heavy KPI math happens inside Databricks, not in Python. That’s exactly what you want.
    rows = query_as_dicts(sql_query)
    if not rows:
        return {
            "total_deliveries": 0,
            "on_time_rate": 0.0,
            "late_rate": 0.0,
            "avg_delay_minutes": 0.0,
        }
    #The service layer then wraps this into a KPIResponse Pydantic model and sends it to the client.
    #If the table is empty for some reason, we return a default KPI payload with zeros.
    #That way /kpi never explodes with “index out of range” or NoneType issue
    # Databricks can return Decimal types → cast to float in service layer
    return rows[0]

def fetch_supplier_kpis(supplier: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read supplier-level KPIs from the v_kpi_by_supplier view."""
    fq_view = f"{settings.dbx_catalog}.gold.v_kpi_by_supplier"

    base_query = f"""
        SELECT supplier,
               total_deliveries,
               on_time_rate,
               late_rate,
               avg_delay_minutes
        FROM {fq_view}
    """

    params: Dict[str, Any] = {}
    if supplier:
        base_query += " WHERE supplier = %(supplier)s"
        params["supplier"] = supplier

    return query_as_dicts(base_query, params)

def fetch_kpi_by_region() -> List[Dict[str, Any]]:
    fq_table = f"{settings.dbx_catalog}.gold.v_kpi_by_region"
    sql_query = f"SELECT * FROM {fq_table}"
    return query_as_dicts(sql_query)

def fetch_kpi_by_weather() -> List[Dict[str, Any]]:
    fq_table = f"{settings.dbx_catalog}.gold.v_kpi_by_weather"
    sql_query = f"SELECT * FROM {fq_table}"
    return query_as_dicts(sql_query)

def fetch_kpi_daily() -> List[Dict[str, Any]]:
    fq_table = f"{settings.dbx_catalog}.gold.v_kpi_daily"
    sql_query = f"SELECT * FROM {fq_table} ORDER BY day"
    return query_as_dicts(sql_query)

def fetch_kpi_weekly() -> List[Dict[str, Any]]:
    fq_table = f"{settings.dbx_catalog}.gold.v_kpi_weekly"
    sql_query = f"SELECT * FROM {fq_table} ORDER BY week_start"
    return query_as_dicts(sql_query)

def fetch_raw_sample(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Return ALL columns from the RAW table (no shuffle).
    We sort by order_datetime because planned_arrival does NOT exist in RAW.
    """
    sql_query = f"""
        SELECT *
        FROM inbound_monitoring.raw.logistics_deliveries
        ORDER BY order_datetime DESC
        LIMIT %(limit)s
    """
    return query_as_dicts(sql_query, {"limit": limit})