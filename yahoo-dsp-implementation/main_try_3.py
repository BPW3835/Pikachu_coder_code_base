"""
main.py
───────
Local entry point for Yahoo DSP Reporting pipeline.

Two run modes:
  backfill : dateTypeId=10 (All Time) — pull maximum historical data
  daily    : dateTypeId=2  (Yesterday) — pull yesterday + COB check

Usage:
  python main.py --mode backfill
  python main.py --mode daily
  python main.py --mode daily --skip-cob

Environment variables required:
  YAHOO_DSP_CLIENT_ID
  YAHOO_DSP_CLIENT_SECRET
"""

import os
import sys
import argparse
import pendulum
from loguru import logger

from reporting import YahooReporting


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

REPORT_TIMEZONE = "America/New_York"

# Advertiser account IDs — replace with your actual IDs
# If empty, Yahoo defaults to your seat's advertisers
ACCOUNT_IDS: list[int] = [
    # 12345,
    # 67890,
]

# Output directory for local CSVs
OUTPUT_DIR = "output"


# ═══════════════════════════════════════════════════════
# DIMENSION & METRIC IDS
# ═══════════════════════════════════════════════════════
# Full reference:
#   Dimensions → https://help.yahooinc.com/dsp-api/docs/dimensions
#   Metrics    → https://help.yahooinc.com/dsp-api/docs/metrics
#
# NOTE: Dimension 1 (Day) → the Day column header comes from this.
#       But the Day VALUE is populated ONLY when intervalTypeId=2 (Day).
#       If intervalTypeId=1 (Cumulative), Day will be EMPTY.
#
# NOTE: Bidding metrics (173, 583, 584) require dataSource=15 and
#       ONLY work with dimensions 4, 5, 6, 149.
#       They are excluded here for the standard report.
# ═══════════════════════════════════════════════════════

DIMENSION_TYPE_IDS: list[int] = [
    1,      # Day (populated by intervalTypeId=2)
    3,      # Creative Size
    4,      # Advertiser
    5,      # Campaign
    6,      # Line
    19,     # Country
    56,     # Seat
    64,     # Advertiser Group
    149,    # Package
]

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
# BQ SCHEMA — Manually defined, strict
# Column names MUST match Yahoo CSV headers exactly.
# Run once → check actual headers → update if needed.
# ═══════════════════════════════════════════════════════

REPORT_SCHEMA: list[dict] = [
    # ── Dimension columns ──
    {"name": "Day",                    "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Creative Size",          "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Advertiser",             "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Campaign",               "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Line",                   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Country",                "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Seat",                   "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Advertiser Group",       "type": "STRING",    "mode": "NULLABLE"},
    {"name": "Package",                "type": "STRING",    "mode": "NULLABLE"},
    # ── Metric columns ──
    {"name": "Impressions",            "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "Clicks",                 "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "Post-Click Conversions", "type": "FLOAT64",   "mode": "NULLABLE"},
    {"name": "CTR",                    "type": "FLOAT64",   "mode": "NULLABLE"},
    {"name": "Video Starts",           "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "Video Completions",      "type": "INTEGER",   "mode": "NULLABLE"},
    {"name": "Total Spend",            "type": "FLOAT64",   "mode": "NULLABLE"},
    {"name": "eCPM",                   "type": "FLOAT64",   "mode": "NULLABLE"},
    {"name": "Post-View Conversions",  "type": "FLOAT64",   "mode": "NULLABLE"},
    # ── Metadata columns (added by pipeline) ──
    {"name": "ingestion_timestamp",    "type": "STRING",    "mode": "NULLABLE"},
    {"name": "lookback_date_range",    "type": "STRING",    "mode": "NULLABLE"},
]


# ═══════════════════════════════════════════════════════
# CONFIG BUILDERS
# ═══════════════════════════════════════════════════════

# def build_backfill_config() -> dict:
#     """
#     BACKFILL mode: dateTypeId=10 (All Time) + intervalTypeId=2 (Day).
#     Extracts the MAXIMUM historical data available in the account.
#     No startDate/endDate needed for All Time.

#     Yahoo DSP does not document a hard lookback limit —
#     All Time goes back to account inception.
#     For very large accounts this can return millions of rows.
#     """
#     config = {
#         "reportOption": {
#             "timezone": REPORT_TIMEZONE,
#             "currency": 4,                          # USD
#             "dimensionTypeIds": DIMENSION_TYPE_IDS,
#             "metricTypeIds": METRIC_TYPE_IDS,
#             "filterOptions": [],
#             "accountIds": ACCOUNT_IDS,
#         },
#         "dateTypeId": 10,           # ← All Time (maximum lookback)
#         "intervalTypeId": 2,        # ← Day granularity (populates Day column)
#     }
#     # NOTE: startDate/endDate are NOT required for dateTypeId=10

#     logger.info(
#         f"BACKFILL config built:\n"
#         f"  Date range   : ALL TIME (max lookback)\n"
#         f"  Interval     : Day (intervalTypeId=2)\n"
#         f"  Timezone     : {REPORT_TIMEZONE}\n"
#         f"  Dimensions   : {DIMENSION_TYPE_IDS}\n"
#         f"  Metrics      : {METRIC_TYPE_IDS}\n"
#         f"  Accounts     : {ACCOUNT_IDS or 'all (default seat)'}"
#     )
#     return config


def build_backfill_config() -> dict:
    """
    Backfill for last 13 months using custom range.
    MUCH faster than All Time.
    """

    tz = pendulum.timezone(REPORT_TIMEZONE)
    now = pendulum.now(tz)

    # ✅ 13 months back
    start = now.subtract(months=13).start_of("day")
    end = now.end_of("day")

    fmt = "YYYY-MM-DDTHH:mm:ssZ"

    config = {
        "reportOption": {
            "timezone": REPORT_TIMEZONE,
            "currency": 4,
            "dimensionTypeIds": DIMENSION_TYPE_IDS,
            "metricTypeIds": METRIC_TYPE_IDS,
            "filterOptions": [],
            "accountIds": ACCOUNT_IDS,
        },
        "dateTypeId": 11,  # ✅ Custom Range
        "intervalTypeId": 2,  # ✅ Day (IMPORTANT for Day column)
        "startDate": start.format(fmt),
        "endDate": end.format(fmt),
    }

    logger.info(
        f"BACKFILL (13 months) config:\n"
        f"  Start: {start.format(fmt)}\n"
        f"  End  : {end.format(fmt)}"
    )

    return config



def build_daily_config() -> dict:
    """
    DAILY mode: dateTypeId=2 (Yesterday) + intervalTypeId=2 (Day).
    Extracts yesterday's finalized data only.
    Designed to run once per day via scheduler.
    No startDate/endDate needed for preset dateTypeId=2.
    """
    config = {
        "reportOption": {
            "timezone": REPORT_TIMEZONE,
            "currency": 4,
            "dimensionTypeIds": DIMENSION_TYPE_IDS,
            "metricTypeIds": METRIC_TYPE_IDS,
            "filterOptions": [],
            "accountIds": ACCOUNT_IDS,
        },
        "dateTypeId": 2,            # ← Yesterday
        "intervalTypeId": 2,        # ← Day granularity
    }

    tz = pendulum.timezone(REPORT_TIMEZONE)
    yesterday = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")

    logger.info(
        f"DAILY config built:\n"
        f"  Date range   : Yesterday ({yesterday})\n"
        f"  Interval     : Day (intervalTypeId=2)\n"
        f"  Timezone     : {REPORT_TIMEZONE}\n"
        f"  Dimensions   : {DIMENSION_TYPE_IDS}\n"
        f"  Metrics      : {METRIC_TYPE_IDS}\n"
        f"  Accounts     : {ACCOUNT_IDS or 'all (default seat)'}"
    )
    return config


# ═══════════════════════════════════════════════════════
# MAIN — LOCAL ENTRY POINT
# ═══════════════════════════════════════════════════════

def main():
    """
    CLI entry point. Parses --mode and --skip-cob flags,
    builds config, optionally checks COB, runs pipeline.
    """
    # ── CLI Arguments ──
    parser = argparse.ArgumentParser(
        description="Yahoo DSP Reporting Pipeline — Local Mode"
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="daily",
        help=(
            "backfill: All Time historical data (max lookback) | "
            "daily: Yesterday's data with COB check"
        ),
    )
    parser.add_argument(
        "--skip-cob",
        action="store_true",
        help="Skip Close of Books check in daily mode",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"📍 Yahoo DSP Reporting Pipeline — LOCAL MODE ({args.mode.upper()})")
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

    # ── Create output directory ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Build config per mode ──
    if args.mode == "backfill":
        report_config = build_backfill_config()
    else:
        report_config = build_daily_config()

    # ── Instantiate reporter ──
    reporter = YahooReporting()

    # ── COB check (daily mode only, when account IDs are known) ──
    if args.mode == "daily" and ACCOUNT_IDS and not args.skip_cob:
        tz = pendulum.timezone(REPORT_TIMEZONE)
        yesterday_yyyymmdd = pendulum.now(tz).subtract(days=1).format("YYYYMMDD")

        logger.info(
            f"📚 Close of Books check for {yesterday_yyyymmdd} "
            f"({len(ACCOUNT_IDS)} advertiser(s))..."
        )

        cob_ok = reporter.wait_for_cob(
            advertiser_ids=ACCOUNT_IDS,
            target_day=yesterday_yyyymmdd,
            max_retries=12,         # 12 × 5 min = 60 min max wait
            retry_delay_secs=300,   # 5 minutes between retries
        )

        if not cob_ok:
            logger.error(
                "❌ Books not closed. Data may be incomplete. "
                "Re-run later or use --skip-cob to force extraction."
            )
            sys.exit(1)

    elif args.mode == "daily" and not ACCOUNT_IDS and not args.skip_cob:
        logger.warning(
            "⚠️  ACCOUNT_IDS is empty — skipping COB check. "
            "Set ACCOUNT_IDS to enable Close of Books verification."
        )

    elif args.mode == "daily" and args.skip_cob:
        logger.warning("⚠️  --skip-cob flag set. Skipping COB check.")

    # ── Run pipeline ──
    df = reporter.run_pipeline(
        report_config=report_config,
        schema_fields=REPORT_SCHEMA,
        output_dir=OUTPUT_DIR,
    )

    # ── Print sample for inspection ──
    if df is not None and not df.is_empty():
        logger.info(f"\n📊 Sample output (first 5 rows):\n{df.head(5)}")
        logger.info(f"\n📊 DataFrame schema:\n{df.schema}")
        logger.info(f"\n📊 Total rows: {df.shape[0]} | Total cols: {df.shape[1]}")
    else:
        logger.warning("No data returned from Yahoo DSP.")


if __name__ == "__main__":
    main()