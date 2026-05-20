"""
main.py
───────
Local entry point for Yahoo DSP Reporting pipeline.
Run directly: python main.py

Environment variables required:
    YAHOO_DSP_CLIENT_ID      — Yahoo DSP API client ID
    YAHOO_DSP_CLIENT_SECRET  — Yahoo DSP API client secret
"""

import os
import sys
import time
import pendulum
from loguru import logger

from reporting import YahooReporting


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

# Report timezone — must match the offset used in startDate/endDate
REPORT_TIMEZONE = "America/New_York"

# Lookback window in days (7 = last 7 days)
LOOKBACK_DAYS = 7

# Advertiser account IDs — replace with your actual IDs
ACCOUNT_IDS: list[int] = [
    # 12345,
    # 67890,
]

# Local output CSV path
LOCAL_OUTPUT_PATH = f"yahoo_dsp_report_output_{pendulum.now("America/New_York").date()}_{pendulum.now("America/New_York")}.csv"


# ═══════════════════════════════════════════════════════
# DIMENSION & METRIC IDS
# ═══════════════════════════════════════════════════════
# Reference: https://help.yahooinc.com/dsp-api/docs/dimensions
# Reference: https://help.yahooinc.com/dsp-api/docs/metrics
#
# Below are all the dimension/metric IDs documented in the
# Yahoo DSP API docs. Adjust based on your seat's access.
# NOTE: Bidding metrics (173, 583, 584) require dataSource=15
#       and only work with dimensions 4, 5, 6, 149.
#       They are excluded here for the standard report.
# ═══════════════════════════════════════════════════════

# All standard dimension type IDs from Yahoo DSP API docs
DIMENSION_TYPE_IDS: list[int] = [
    1,      # Day / Date
    3,      # Creative Size
    4,      # Advertiser
    5,      # Campaign
    6,      # Line
    19,     # Country
    56,     # Seat
    64,     # Advertiser Group
    149,    # Package
]

# All standard metric type IDs from Yahoo DSP API docs
# (excluding bidding-only metrics 173, 583, 584)
METRIC_TYPE_IDS: list[int] = [
    1,      # Impressions
    2,      # Clicks
    11,     # Post-Click Conversions
    23,     # CTR (Click-Through Rate)
    41,     # Video Starts
    43,     # Video Completions
    44,     # Total Spend
    46,     # eCPM
    110,    # Post-View Conversions
]


# ═══════════════════════════════════════════════════════
# BQ SCHEMA (manually defined, strict)
# Update column names to match Yahoo CSV headers exactly.
# These are placeholder names — after first run, check the
# actual CSV headers and update accordingly.
# ═══════════════════════════════════════════════════════

REPORT_SCHEMA: list[dict] = [
    {"name": "Day",                   "type": "DATE",    "mode": "NULLABLE"},
    {"name": "Creative Size",         "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Advertiser ID",         "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Advertiser",            "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Campaign ID",           "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Campaign",              "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Line ID",               "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Line",                  "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Country",               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Seat ID",               "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Seat",                  "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Advertiser Group ID",   "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Advertiser Group",      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Package ID",            "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Package",               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Impressions",           "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Clicks",                "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Post-Click Conversions","type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "CTR",                   "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video Starts",          "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Completions",     "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Total Spend",           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "eCPM",                  "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post-View Conversions", "type": "FLOAT64", "mode": "NULLABLE"},
]


# ═══════════════════════════════════════════════════════
# REPORT CONFIG BUILDER
# ═══════════════════════════════════════════════════════

def build_report_config() -> dict:
    """
    Build a Custom-Range (dateTypeId=11) report config
    with Day granularity (intervalTypeId=2).

    Dates are auto-calculated based on LOOKBACK_DAYS
    with correct timezone offset appended.
    """
    tz = pendulum.timezone(REPORT_TIMEZONE)
    now = pendulum.now(tz)

    # Lookback: [T-LOOKBACK_DAYS 00:00:00 , T-1 23:59:59]
    start = now.subtract(days=LOOKBACK_DAYS).start_of("day")
    end = now.subtract(days=1).end_of("day")

    # Format with timezone offset (e.g. 2026-04-29T00:00:00-04:00)
    fmt = "YYYY-MM-DDTHH:mm:ssZ"

    config = {
        "reportOption": {
            "timezone": REPORT_TIMEZONE,
            "currency": 4,                          # USD
            "dimensionTypeIds": DIMENSION_TYPE_IDS,
            "metricTypeIds": METRIC_TYPE_IDS,
            "filterOptions": [],
            "accountIds": ACCOUNT_IDS,
        },
        "dateTypeId": 11,           # Custom Range
        "intervalTypeId": 2,        # Day granularity
        "startDate": start.format(fmt),
        "endDate": end.format(fmt),
    }

    logger.info(
        f"Report config built:\n"
        f"  Date range : {start.format(fmt)} → {end.format(fmt)}\n"
        f"  Lookback   : {LOOKBACK_DAYS} days\n"
        f"  Timezone   : {REPORT_TIMEZONE}\n"
        f"  Dimensions : {DIMENSION_TYPE_IDS}\n"
        f"  Metrics    : {METRIC_TYPE_IDS}\n"
        f"  Accounts   : {ACCOUNT_IDS or 'all (default seat)'}"
    )
    return config


# ═══════════════════════════════════════════════════════
# MAIN — LOCAL ENTRY POINT
# ═══════════════════════════════════════════════════════

def main():
    """
    Local pipeline execution.
    Validates env vars → builds config → runs full pipeline.
    """
    logger.info("=" * 60)
    logger.info("📍 Yahoo DSP Reporting Pipeline — LOCAL MODE")
    logger.info("=" * 60)

    # ── Validate credentials ──
    client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        logger.error(
            "❌ Missing credentials. Set environment variables:\n"
            "   export YAHOO_DSP_CLIENT_ID='your_client_id'\n"
            "   export YAHOO_DSP_CLIENT_SECRET='your_client_secret'"
        )
        sys.exit(1)

    logger.success(f"Credentials loaded | client_id: {client_id[:8]}...")

    # ── Build report config ──
    report_config = build_report_config()

    # ── Instantiate reporter and run pipeline ──
    reporter = YahooReporting()
    df = reporter.run_pipeline(
        report_config=report_config,
        schema_fields=REPORT_SCHEMA,
        local_output_path=LOCAL_OUTPUT_PATH,
    )

    # ── Print sample output for inspection ──
    if df is not None and not df.is_empty():
        logger.info(f"\n📊 Sample output (first 5 rows):\n{df.head(5)}")
        logger.info(f"📊 Schema:\n{df.schema}")
    else:
        logger.warning("No data returned from Yahoo DSP.")


if __name__ == "__main__":
    main()