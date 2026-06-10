"""
main.py — Yahoo DSP Reporting Pipeline CLI

Usage:
    python main.py --mode daily --skip-cob
    python main.py --mode backfill --months 13 --window-days 2 --workers 4
    python main.py --mode backfill --months 1 --window-days 5 --workers 4
    python main.py --mode backfill --groups campaign_delivery bidding
"""

import os
import sys
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
    AudienceReachReport,
    BiddingReport,
    SupplyInventoryReport,
    GeoDeviceReport,
)

REPORT_TIMEZONE = "America/New_York"
ACCOUNT_IDS: list[int] = []

ENABLED_GROUPS = [
    CampaignDeliveryReport,
    CreativePerformanceReport,
    VideoPerformanceReport,
    ViewabilityReport,
    ConversionDetailReport,
    CostFeeReport,
    AudienceReachReport,
    BiddingReport,
    GeoDeviceReport,
    SupplyInventoryReport,
]


def build_daily_overrides(target_date=None) -> dict:
    """
    If target_date is given (YYYY-MM-DD), fetch that specific day.
    Otherwise, fetch yesterday (default daily behavior).
    """
    tz = pendulum.timezone(REPORT_TIMEZONE)

    if target_date:
        # Custom date → dateTypeId=11 (Custom Range)
        target = pendulum.parse(target_date, tz=tz)
        fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
        start_str = target.start_of("day").format(fmt)
        end_str = target.end_of("day").format(fmt)
        logger.info(f"DAILY (custom): {target_date}, TZ={REPORT_TIMEZONE}")
        return {
            "timezone":       REPORT_TIMEZONE,
            "currency":       4,
            "accountIds":     ACCOUNT_IDS if ACCOUNT_IDS else [],
            "dateTypeId":     11,
            "intervalTypeId": 2,
            "startDate":      start_str,
            "endDate":        end_str,
        }
    else:
        # Default: yesterday
        yesterday_str = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")
        logger.info(f"DAILY: Yesterday ({yesterday_str}), TZ={REPORT_TIMEZONE}")
        return {
            "timezone":       REPORT_TIMEZONE,
            "currency":       4,
            "accountIds":     ACCOUNT_IDS if ACCOUNT_IDS else [],
            "dateTypeId":     2,
            "intervalTypeId": 2,
        }


def main():
    parser = argparse.ArgumentParser(description="Yahoo DSP Reporting Pipeline")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--skip-cob", action="store_true")
    parser.add_argument("--months", type=int, default=13,
                        help="Backfill: months to look back (default 13)")
    parser.add_argument("--window-days", type=int, default=2,
                        help="Backfill: days per API window (default 2, try 5 for speed)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Backfill: parallel threads per window (default 4)")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Run specific groups by REPORT_NAME")
    parser.add_argument("--date", type=str, default=None,
                        help="Daily: override date (YYYY-MM-DD). Default=yesterday")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Yahoo DSP Pipeline - {args.mode.upper()}")
    logger.info("=" * 60)

    client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        logger.error(
            "Missing credentials. Set:\n"
            "  export YAHOO_DSP_CLIENT_ID='...'\n"
            "  export YAHOO_DSP_CLIENT_SECRET='...'"
        )
        sys.exit(1)
    logger.success(f"Credentials loaded | client_id: {client_id[:8]}...")

    if args.groups:
        all_by_name = {g.REPORT_NAME: g for g in ALL_REPORT_GROUPS}
        groups = [all_by_name[s] for s in args.groups if s in all_by_name]
        if not groups:
            logger.error(f"No matching groups: {args.groups}")
            sys.exit(1)
    else:
        groups = ENABLED_GROUPS

    logger.info(f"Groups ({len(groups)}): {[g.REPORT_NAME for g in groups]}")
    reporter = YahooReporting()

    # ══════════════════════════════════════
    # DAILY
    # ══════════════════════════════════════
    if args.mode == "daily":
        output_dir = "output_daily"
        os.makedirs(output_dir, exist_ok=True)
        overrides = build_daily_overrides(target_date=args.date)

        results = reporter.run_all_groups(
            report_groups=groups,
            config_overrides=overrides,
            output_dir=output_dir,
            skip_cob=args.skip_cob,
            advertiser_ids_for_cob=ACCOUNT_IDS or None,
        )

        logger.info("\nFINAL SUMMARY:")
        for name, df in results.items():
            if df is not None and not df.is_empty():
                logger.info(f"  OK   {name:30s} -> {df.shape>8,} rows x {df.shape[1]} cols")
            elif df is not None:
                logger.info(f"  EMPTY {name:30s} -> 0 rows")
            else:
                logger.info(f"  FAIL {name:30s}")

    # ══════════════════════════════════════
    # BACKFILL
    # ══════════════════════════════════════
    elif args.mode == "backfill":
        output_dir = "output_backfill"
        os.makedirs(output_dir, exist_ok=True)

        results = reporter.backfill_all_groups(
            report_groups=groups,
            tz=REPORT_TIMEZONE,
            currency=4,
            account_ids=ACCOUNT_IDS or None,
            months_back=args.months,
            window_days=args.window_days,
            max_workers=args.workers,
            worker_sleep=3,
            output_dir=output_dir,
        )

        logger.info("\nBACKFILL SUMMARY:")
        for name, path in results.items():
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"  OK   {name} -> {path} ({size_mb:.1f} MB)")
            else:
                logger.info(f"  FAIL {name} -> NO DATA")


if __name__ == "__main__":
    main()