import pytest

from src.config import settings
from src.databricks_db import query_as_dicts


def fq(object_name: str) -> str:
    """
    Build fully qualified table/view name inside the configured catalog.
    Example: fq("silver.trips") -> "inbound_monitoring.silver.trips"
    """
    return f"{settings.dbx_catalog}.{object_name}"


# ---------- SILVER LAYER TESTS ----------
#Silver trips table exists and has data
@pytest.mark.databricks
def test_silver_trips_not_empty():
    rows = query_as_dicts(f"SELECT COUNT(*) AS cnt FROM {fq('silver.logistics_trips')}")
    assert rows[0]["cnt"] > 0, "Silver trips table is empty – ETL to silver did not run or produced no data."

#trip_id is fully populated and unique
@pytest.mark.databricks
def test_silver_trip_id_not_null_and_unique():
    sql = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(trip_id) AS non_null_trip_id,
            COUNT(DISTINCT trip_id) AS distinct_trip_id
        FROM {fq('silver.logistics_trips')}
    """
    rows = query_as_dicts(sql)
    r = rows[0]

    assert r["total_rows"] == r["non_null_trip_id"], "Found NULL trip_id values in silver.logistics_trips."
    assert r["total_rows"] == r["distinct_trip_id"], "trip_id is not unique in silver.logistics_trips."


#planned_arrival is never NULL in Silver
@pytest.mark.databricks
def test_silver_planned_arrival_not_null():
    sql = f"""
        SELECT COUNT(*) AS missing
        FROM {fq('silver.logistics_trips')}
        WHERE planned_arrival IS NULL
    """
    rows = query_as_dicts(sql)
    assert rows[0]["missing"] == 0, "Some rows in silver.logistics_trips have NULL planned_arrival."

#status uses only expected values
@pytest.mark.databricks
def test_silver_status_in_allowed_values():
    sql = f"""
        SELECT DISTINCT status
        FROM {fq('silver.logistics_trips')}
    """
    rows = query_as_dicts(sql)
    values = {row["status"] for row in rows}

    allowed = {
        "Planned",
        "Late",
        "Delivered",
        "Delivered Late",
        "Cancelled",
        "Failed",
    }
    invalid = values - allowed
    assert not invalid, f"Found invalid status values in silver.logistics_trips: {invalid}"


# ---------- GOLD LAYER TESTS ----------
#Gold fact table exists and has data
@pytest.mark.databricks
def test_gold_fact_trips_not_empty():
    rows = query_as_dicts(f"SELECT COUNT(*) AS cnt FROM {fq('gold.fact_trips')}")
    assert rows[0]["cnt"] > 0, "Gold fact_trips table is empty – ETL from silver to gold failed."

#delay_minutes isn’t all NULL
@pytest.mark.databricks
def test_gold_delay_minutes_not_all_null():
    sql = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(delay_minutes) AS non_null_delay
        FROM {fq('gold.fact_trips')}
    """
    rows = query_as_dicts(sql)
    r = rows[0]

    assert r["non_null_delay"] > 0, "All delay_minutes are NULL – delay calculation not applied."
    assert r["non_null_delay"] <= r["total_rows"], "More non-null delays than rows – something is wrong."

#is_late only holds boolean-style values
@pytest.mark.databricks
def test_gold_is_late_is_boolean_like():
    sql = f"""
        SELECT DISTINCT CAST(is_late AS STRING) AS val
        FROM {fq('gold.fact_trips')}
    """
    rows = query_as_dicts(sql)
    values = {row["val"] for row in rows}

    allowed = {"true", "false", "0", "1"}
    invalid = values - allowed
    assert not invalid, f"Unexpected values in is_late: {invalid}"


# ---------- LIVE VIEW TEST (OPTIONAL BUT NICE) ----------
#Live view status values are also valid
@pytest.mark.databricks
def test_live_view_status_in_allowed_values():
    sql = f"""
        SELECT DISTINCT status
        FROM {fq('gold.v_live_trips')}
    """
    rows = query_as_dicts(sql)
    values = {row["status"] for row in rows}

    allowed = {"Planned", "Late", "Delivered", "Delivered Late"}
    invalid = values - allowed
    assert not invalid, f"Found invalid status values in v_live_trips: {invalid}"