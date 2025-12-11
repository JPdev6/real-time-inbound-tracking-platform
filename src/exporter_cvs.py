from pathlib import Path
from typing import Dict

import pandas as pd
from databricks import sql
from config import settings
from databricks_db import query_as_dicts

VIEWS: Dict[str, str] = {
    "kpi_by_supplier": f"{settings.dbx_catalog}.gold.v_kpi_by_supplier",
    "kpi_by_region": f"{settings.dbx_catalog}.gold.v_kpi_by_region",
    "kpi_by_weather": f"{settings.dbx_catalog}.gold.v_kpi_by_weather",
    "kpi_daily": f"{settings.dbx_catalog}.gold.v_kpi_daily",
    "kpi_weekly": f"{settings.dbx_catalog}.gold.v_kpi_weekly",
    "live_trips": f"{settings.dbx_catalog}.gold.v_live_trips"
    # add more if you want:
}


def export_view_to_csv(logical_name: str, fully_qualified_view: str, limit: int | None = None) -> Path:
    """
    Export a Databricks view to a local CSV file in ./exports.
    If limit is set, only that many rows are exported (useful for very large tables).
    """
    print(f"➡ Exporting {fully_qualified_view} ...")

    sql = f"SELECT * FROM {fully_qualified_view}"
    params = {}

    if limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = limit

    rows = query_as_dicts(sql, params)
    df = pd.DataFrame(rows)

    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)

    out_path = exports_dir / f"{logical_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Wrote {len(df)} rows to {out_path}")
    return out_path


def main():
    # Set a LIMIT if you want to restrict size for Tableau Public demos
    row_limit = None  # e.g. 10000

    for logical_name, fq_view in VIEWS.items():
        try:
            export_view_to_csv(logical_name, fq_view, limit=row_limit)
        except Exception as exc:
            print(f"❌ Failed to export {fq_view}: {exc}")


if __name__ == "__main__":
    main()