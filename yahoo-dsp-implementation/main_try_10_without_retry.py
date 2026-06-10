"""
main.py — Yahoo DSP Reporting Pipeline CLI

Usage:
    # Daily (yesterday)
    python main.py --mode daily --skip-cob

    # Daily (specific date)
    python main.py --mode daily --skip-cob --date 2025-06-15

    # Backfill 13 months (sequential, day-by-day, random sleep)
    python main.py --mode backfill --skip-cob --months 13

    # Backfill specific groups only
    python main.py --mode backfill --skip-cob --months 3 --groups campaign_delivery bidding

Environment variables:
    YAHOO_DSP_CLIENT_ID
    YAHOO_DSP_CLIENT_SECRET
"""

import os
import sys
import time
import random
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

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

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

# Random sleep choices (seconds) between backfill days
BACKFILL_SLEEP_CHOICES = [120, 110, 130, 100, 150]


# ═══════════════════════════════════════════════════════════
# OVERRIDE BUILDERS
# ═══════════════════════════════════════════════════════════

def build_daily_overrides(target_date=None) -> dict:
    """
    If target_date (YYYY-MM-DD) given → dateTypeId=11 (Custom Range) for that day.
    Otherwise → dateTypeId=2 (Yesterday).
    """
    tz = pendulum.timezone(REPORT_TIMEZONE)

    if target_date:
        target = pendulum.parse(target_date, tz=tz)
        fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
        start_str = target.start_of("day").format(fmt)
        end_str = target.end_of("day").format(fmt)
        logger.info(f"DAILY (custom date): {target_date}, TZ={REPORT_TIMEZONE}")
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
        yesterday_str = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")
        logger.info(f"DAILY: Yesterday ({yesterday_str}), TZ={REPORT_TIMEZONE}")
        return {
            "timezone":       REPORT_TIMEZONE,
            "currency":       4,
            "accountIds":     ACCOUNT_IDS if ACCOUNT_IDS else [],
            "dateTypeId":     2,
            "intervalTypeId": 2,
        }


def build_single_day_overrides(day_str) -> dict:
    """Build overrides for one specific day (used in backfill loop)."""
    tz = pendulum.timezone(REPORT_TIMEZONE)
    target = pendulum.parse(day_str, tz=tz)
    fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
    return {
        "timezone":       REPORT_TIMEZONE,
        "currency":       4,
        "accountIds":     ACCOUNT_IDS if ACCOUNT_IDS else [],
        "dateTypeId":     11,
        "intervalTypeId": 2,
        "startDate":      target.start_of("day").format(fmt),
        "endDate":        target.end_of("day").format(fmt),
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Yahoo DSP Reporting — Multi-Group Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["daily", "backfill"], default="daily",
        help="daily = single day | backfill = sequential day-by-day",
    )
    parser.add_argument("--skip-cob", action="store_true")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Daily: override date (YYYY-MM-DD). Default=yesterday",
    )
    parser.add_argument(
        "--months", type=int, default=13,
        help="Backfill: months to look back (default 13)",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Run specific groups by REPORT_NAME",
    )
    args = parser.parse_args()

    # ── Banner ──
    logger.info("=" * 60)
    logger.info(f"📍 Yahoo DSP Pipeline — {args.mode.upper()}")
    logger.info("=" * 60)

    # ── Credentials ──
    client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        logger.error(
            "❌ Missing credentials. Set:\n"
            "   export YAHOO_DSP_CLIENT_ID='...'\n"
            "   export YAHOO_DSP_CLIENT_SECRET='...'"
        )
        sys.exit(1)
    logger.success(f"Credentials loaded | client_id: {client_id[:8]}...")

    # ── Resolve groups ──
    if args.groups:
        all_by_name = {g.REPORT_NAME: g for g in ALL_REPORT_GROUPS}
        groups_to_run = []
        for slug in args.groups:
            if slug in all_by_name:
                groups_to_run.append(all_by_name[slug])
            else:
                logger.warning(f"⚠️  Unknown group '{slug}'. Available: {list(all_by_name.keys())}")
        if not groups_to_run:
            logger.error("❌ No valid groups. Exiting.")
            sys.exit(1)
    else:
        groups_to_run = ENABLED_GROUPS

    logger.info(f"Groups ({len(groups_to_run)}): {[g.REPORT_NAME for g in groups_to_run]}")
    reporter = YahooReporting()

    # ══════════════════════════════════════
    # DAILY MODE
    # ══════════════════════════════════════
    if args.mode == "daily":
        output_dir = "output_daily"
        os.makedirs(output_dir, exist_ok=True)
        overrides = build_daily_overrides(target_date=args.date)

        results = reporter.run_all_groups(
            report_groups=groups_to_run,
            config_overrides=overrides,
            output_dir=output_dir,
            skip_cob=args.skip_cob,
            advertiser_ids_for_cob=ACCOUNT_IDS if ACCOUNT_IDS else None,
        )

        # Summary
        logger.info("\n📊 FINAL SUMMARY:")
        for name, df in results.items():
            if df is not None and not df.is_empty():
                logger.info(f"  ✅ {name:30s} → {df.shape[0]:>8,} rows × {df.shape[1]} cols")
            elif df is not None:
                logger.info(f"  ⚠️  {name:30s} → 0 rows (empty)")
            else:
                logger.info(f"  ❌ {name:30s} → FAILED")

    # ══════════════════════════════════════
    # BACKFILL MODE — Sequential day-by-day
    # ══════════════════════════════════════
    elif args.mode == "backfill":
        output_dir = "output_backfill"
        os.makedirs(output_dir, exist_ok=True)

        tz = pendulum.timezone(REPORT_TIMEZONE)
        end_date = pendulum.yesterday(tz)
        start_date = end_date.subtract(months=args.months)

        # Generate all days (oldest → newest)
        all_days = []
        cursor = start_date
        while cursor <= end_date:
            all_days.append(cursor.format("YYYY-MM-DD"))
            cursor = cursor.add(days=1)

        total_days = len(all_days)
        total_api_calls = total_days * len(groups_to_run)

        logger.info("=" * 60)
        logger.info("🔄 BACKFILL PLAN")
        logger.info(f"  Range      : {all_days[0]} → {all_days[-1]}")
        logger.info(f"  Total days : {total_days}")
        logger.info(f"  Groups     : {len(groups_to_run)}")
        logger.info(f"  Total POST : {total_api_calls}")
        logger.info(f"  Sleep pool : {BACKFILL_SLEEP_CHOICES}s")
        logger.info("=" * 60)

        t0 = time.time()
        day_results = {}

        for day_idx, day_str in enumerate(all_days, 1):
            t_day = time.time()
            logger.info(f"\n{'─' * 50}")
            logger.info(f"📅 Day {day_idx}/{total_days}: {day_str}")
            logger.info(f"{'─' * 50}")

            overrides = build_single_day_overrides(day_str)

            results = reporter.run_all_groups(
                report_groups=groups_to_run,
                config_overrides=overrides,
                output_dir=output_dir,
                skip_cob=True,
                advertiser_ids_for_cob=None,
            )

            # Track results
            ok = sum(1 for v in results.values() if v is not None and not v.is_empty())
            fail = len(results) - ok
            elapsed_day = time.time() - t_day
            day_results[day_str] = {"ok": ok, "fail": fail, "time": elapsed_day}

            logger.info(f"  Day {day_str}: {ok} OK, {fail} fail | {elapsed_day:.1f}s")

            # Random sleep between days (skip after last day)
            if day_idx < total_days:
                sleep_secs = random.choice(BACKFILL_SLEEP_CHOICES)
                logger.info(f"  💤 Sleeping {sleep_secs}s before next day ...")
                time.sleep(sleep_secs)

        # ── Backfill Summary ──
        elapsed_total = time.time() - t0
        total_ok = sum(d["ok"] for d in day_results.values())
        total_fail = sum(d["fail"] for d in day_results.values())

        logger.info("\n" + "=" * 60)
        logger.info("🏁 BACKFILL COMPLETE")
        logger.info(f"  Days processed : {total_days}")
        logger.info(f"  Groups OK      : {total_ok}")
        logger.info(f"  Groups FAIL    : {total_fail}")
        logger.info(f"  Total time     : {elapsed_total / 3600:.1f} hours")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()