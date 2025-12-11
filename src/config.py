import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    db_backend: str = os.getenv("DB_BACKEND", "databricks")

    dbx_host: str = os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
    dbx_http_path: str = os.getenv("DATABRICKS_HTTP_PATH", "")
    dbx_token: str = os.getenv("DATABRICKS_TOKEN", "")

    dbx_catalog: str = os.getenv("DATABRICKS_CATALOG", "inbound_monitoring")
    dbx_trip_view: str = os.getenv("DATABRICKS_TRIP_VIEW", "gold.v_live_trips")
    dbx_kpi_table: str = os.getenv("DATABRICKS_KPI_TABLE", "gold.fact_trips")


settings = Settings()