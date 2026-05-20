"""
main.py
-------
Yahoo DSP Reporting Pipeline - CLI entrypoint.

Usage:
    python main.py --mode daily
    python main.py --mode daily --skip-cob
    python main.py --mode backfill
    python main.py --mode backfill --months 1 --workers 4
    python main.py --mode backfill --start 2025-06-01 --end 2026-05-06
    python main.py --mode backfill --groups campaign_delivery bidding
"""

import os
import argparse
import pendulum
from loguru import logger

from reporting import (
    YahooReporting,
    ALL_REPORT_GROUPS,
    CampaignDeliveryReport,
    CreativePerformanceReport,
    VideoPerformanceReport,
    ViewabilityReport,
    ConversionDetailReport,
    CostFeeReport,
    GeoDeviceReport,
    SupplyInventoryReport,
    AudienceReachReport,
    BiddingReport,
)

# ===========================================================
# CONFIGURATION
# ===========================================================

REPORT_TIMEZONE = "America/New_York"
ACCOUNT_IDS: list[int] = []        # empty = all advertisers with data
OUTPUT_DIR  = "output"
TEMP_DIR    = "output/temp"

ENABLED_GROUPS = [
    CampaignDeliveryReport,
    CreativePerformanceReport,
    VideoPerformanceReport,
    ViewabilityReport,
    ConversionDetailReport,
    CostFeeReport,
    GeoDeviceReport,
    SupplyInventoryReport,
    AudienceReachReport,
    BiddingReport,
]

# ===========================================================
# OVERRIDE BUILDERS
# ===========================================================

def build_daily_overrides() -> dict:
    """DAILY mode: dateTypeId=2 (Yesterday)."""
    tz = pendulum.timezone(REPORT_TIMEZONE)
    yesterday = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")
    logger.info(
        f"DAILY overrides:\n"
        f"  Date range : Yesterday ({yesterday})\n"
        f"  TZ         : {REPORT_TIMEZONE}"
    )
    return {
        "timezone":       REPORT_TIMEZONE,
        "currency":       4,
        "accountIds":     ACCOUNT_IDS if ACCOUNT_IDS else [],
        "dateTypeId":     2,
        "intervalTypeId": 2,
    }


def build_backfill_base_overrides() -> dict:
    """
    BACKFILL mode: Base overrides WITHOUT date fields.
    Dates are injected per-day by fetch_single_day().
    """
    logger.info(
        f"BACKFILL base overrides:\n"
        f"  TZ       : {REPORT_TIMEZONE}\n"
        f"  Currency : USD (4)\n"
        f"  Accounts : {ACCOUNT_IDS or 'all with data'}"
    )
    return {
        "timezone":   REPORT_TIMEZONE,
        "currency":   4,
        "accountIds": ACCOUNT_IDS if ACCOUNT_IDS else [],
    }


# ===========================================================
# MAIN
# ===========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Yahoo DSP Reporting Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["daily", "backfill"], default="daily",
        help="daily = yesterday; backfill = multi-month historical"
    )
    parser.add_argument(
        "--skip-cob", action="store_true",
        help="Skip Close of Books check (daily mode)"
    )
    parser.add_argument(
        "--months", type=int, default=13,
        help="Backfill: months to look back from yesterday (default: 13)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Backfill: parallel threads per month (default: 4)"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Backfill: explicit start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="Backfill: explicit end date YYYY-MM-DD"
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Only run specific report group names (space-separated)"
    )
    args = parser.parse_args()

    # -- Banner --
    logger.info("=" * 60)
    logger.info(f"Yahoo DSP Multi-Group Pipeline - {args.mode.upper()}")
    logger.info("=" * 60)

    # -- Credentials --
    client_id = os.environ.get("YAHOO_DSP_CLIENT_ID", "")
    if client_id:
        logger.success(f"Credentials loaded | client_id: {client_id[:8]}...")
    else:
        logger.warning("YAHOO_DSP_CLIENT_ID not set - helpers will handle auth")

    # -- Select report groups --
    groups = ENABLED_GROUPS
    if args.groups:
        groups = [g for g in ALL_REPORT_GROUPS if g.name in args.groups]
        if not groups:
            logger.error(f"No matching groups for: {args.groups}")
            return

    group_names = [g.name for g in groups]
    logger.info(f"Groups to run ({len(groups)}): {group_names}")

    # -- Init reporter --
    reporter = YahooReporting()

    # ===========================================================
    # DAILY MODE
    # ===========================================================
    if args.mode == "daily":
        overrides = build_daily_overrides()
        results = reporter.run_all_groups(
            report_groups=groups,
            config_overrides=overrides,
            output_dir=OUTPUT_DIR,
            skip_cob=args.skip_cob,
        )

        logger.info("\nFINAL SUMMARY:")
        for name, df in results.items():
            if df is not None:
                logger.info(f"  [OK] {name} -> {df.shape[0]:,} rows x {df.shape[1]} cols")
            else:
                logger.info(f"  [FAIL] {name}")

    # ===========================================================
    # BACKFILL MODE
    # ===========================================================
    elif args.mode == "backfill":
        tz = pendulum.timezone(REPORT_TIMEZONE)

        # Determine date range
        if args.start and args.end:
            start_date = pendulum.parse(args.start, tz=tz)
            end_date   = pendulum.parse(args.end, tz=tz)
        else:
            end_date   = pendulum.yesterday(tz)
            start_date = end_date.subtract(months=args.months)

        logger.info(
            f"BACKFILL CONFIG:\n"
            f"  Start    : {start_date.to_date_string()}\n"
            f"  End      : {end_date.to_date_string()}\n"
            f"  Months   : ~{args.months}\n"
            f"  Workers  : {args.workers}\n"
            f"  Groups   : {len(groups)}"
        )

        base_overrides = build_backfill_base_overrides()

        results = reporter.backfill_all_groups(
            report_groups=groups,
            start_date=start_date,
            end_date=end_date,
            base_overrides=base_overrides,
            output_dir=OUTPUT_DIR,
            temp_dir=TEMP_DIR,
            max_workers=args.workers,
        )

        logger.info("\nBACKFILL FINAL SUMMARY:")
        for name, path in results.items():
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"  [OK] {name} -> {path} ({size_mb:.1f} MB)")
            else:
                logger.info(f"  [FAIL] {name} -> NO DATA")


if __name__ == "__main__":
    main()
