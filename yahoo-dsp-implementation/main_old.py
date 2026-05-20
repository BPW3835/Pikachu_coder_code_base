"""
main.py
───────
Google Cloud Run Function entry point for Yahoo DSP Reporting pipeline.
Triggered via HTTP (Cloud Scheduler, Pub/Sub push, or manual).
"""

import os
import json
import pendulum
import functions_framework
from loguru import logger

from reporting import YahooReporting


# ═══════════════════════════════════════════════════════
# CONFIGURATION (from environment variables)
# ═══════════════════════════════════════════════════════

GCS_BUCKET = os.environ.get("GCS_BUCKET", "your-staging-bucket")
GCS_PREFIX = os.environ.get("GCS_PREFIX", "yahoo_dsp/reporting")
BQ_TABLE_ID = os.environ.get("BQ_TABLE_ID", "project.dataset.yahoo_dsp_report")
WRITE_DISPOSITION = os.environ.get("WRITE_DISPOSITION", "WRITE_APPEND")
DEFAULT_TIMEZONE = os.environ.get("REPORT_TIMEZONE", "UTC")
DEFAULT_ACCOUNT_IDS: list[int] = json.loads(
    os.environ.get("ACCOUNT_IDS", "[]")
)


# ═══════════════════════════════════════════════════════
# SCHEMA — Manually defined, strict BQ schema
# Modify columns to match your actual report dimensions & metrics.
# ═══════════════════════════════════════════════════════

REPORT_SCHEMA: list[dict] = [
    {"name": "Day",            "type": "DATE",    "mode": "NULLABLE"},
    {"name": "Advertiser_ID",  "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Advertiser",     "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Campaign_ID",    "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Campaign",       "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Line_ID",        "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Line",           "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Impressions",    "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Clicks",         "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Total_Spend",    "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Conversions",    "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video_Views",    "type": "INTEGER", "mode": "NULLABLE"},
]


# ═══════════════════════════════════════════════════════
# REPORT CONFIG BUILDER
# ═══════════════════════════════════════════════════════

def build_report_config(
    lookback_days: int = 1,
    timezone: str = DEFAULT_TIMEZONE,
    account_ids: list[int] | None = None,
) -> dict:
    """
    Build a Custom-Range report config with Day granularity.
    Dates are auto-calculated with the correct timezone offset.

    Args:
        lookback_days: How many days back from today.
        timezone:      IANA timezone string for the report.
        account_ids:   Advertiser IDs to scope the report.
    """
    tz = pendulum.timezone(timezone)
    now = pendulum.now(tz)

    # Lookback window: [T-lookback_days 00:00:00, T-1 23:59:59]
    start = now.subtract(days=lookback_days).start_of("day")
    end = now.subtract(days=1).end_of("day")

    # Format with TZ offset (e.g. 2026-05-04T00:00:00-04:00)
    fmt = "YYYY-MM-DDTHH:mm:ssZ"

    config = {
        "reportOption": {
            "timezone": timezone,
            "currency": 4,                              # USD
            "dimensionTypeIds": [1, 4, 5, 6],           # Day, Advertiser, Campaign, Line
            "metricTypeIds": [1, 2, 23, 44, 46],        # Impressions, Clicks, Spend…
            "filterOptions": [],
            "accountIds": account_ids or DEFAULT_ACCOUNT_IDS,
        },
        "dateTypeId": 11,           # Custom Range
        "intervalTypeId": 2,        # Day granularity
        "startDate": start.format(fmt),
        "endDate": end.format(fmt),
    }

    logger.info(
        f"Report config built | {start.format(fmt)} → {end.format(fmt)} "
        f"| lookback={lookback_days}d | tz={timezone}"
    )
    return config


# ═══════════════════════════════════════════════════════
# CLOUD RUN FUNCTION ENTRY POINT
# ═══════════════════════════════════════════════════════

@functions_framework.http
def main(request):
    """
    HTTP entry point for Cloud Run Functions.

    Optional JSON body overrides:
    {
        "lookback_days": 7,
        "timezone": "UTC",
        "account_ids": [12345, 67890],
        "gcs_bucket": "my-bucket",
        "bq_table_id": "proj.ds.table"
    }
    """
    try:
        logger.info("☁️  Cloud Run Function triggered.")

        # Parse optional overrides from request body
        body = request.get_json(silent=True) or {}
        lookback_days = body.get("lookback_days", 1)
        timezone = body.get("timezone", DEFAULT_TIMEZONE)
        account_ids = body.get("account_ids", DEFAULT_ACCOUNT_IDS)
        gcs_bucket = body.get("gcs_bucket", GCS_BUCKET)
        bq_table_id = body.get("bq_table_id", BQ_TABLE_ID)

        # Timestamped GCS blob path (avoids overwrites)
        ts = pendulum.now("UTC").format("YYYYMMDD_HHmmss")
        gcs_blob_path = f"{GCS_PREFIX}/report_{ts}.csv"

        # Build report config
        report_config = build_report_config(
            lookback_days=lookback_days,
            timezone=timezone,
            account_ids=account_ids,
        )

        # Execute pipeline
        reporter = YahooReporting()
        reporter.run_pipeline(
            report_config=report_config,
            schema_fields=REPORT_SCHEMA,
            gcs_bucket=gcs_bucket,
            gcs_blob_path=gcs_blob_path,
            bq_table_id=bq_table_id,
            write_disposition=WRITE_DISPOSITION,
        )

        return (
            json.dumps({"status": "success", "message": "Pipeline completed."}),
            200,
            {"Content-Type": "application/json"},
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return (
            json.dumps({"status": "error", "message": str(e)}),
            500,
            {"Content-Type": "application/json"},
        )