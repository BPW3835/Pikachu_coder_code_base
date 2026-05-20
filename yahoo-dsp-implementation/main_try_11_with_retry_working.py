"""
main.py — Yahoo DSP Reporting Pipeline CLI

Usage:
    # Daily (yesterday)
    python main.py --mode daily --skip-cob

    # Daily (specific date)
    python main.py --mode daily --skip-cob --date 2025-06-15

    # Backfill 13 months (sequential, day-by-day, random sleep, auto-retry)
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

BUCKET_NAME = "mdlz-yahoo-dsp-reports_sr4w"  
REPORT_TIMEZONE = "America/New_York"
ACCOUNT_IDS: list[int] = []
DATASET_ID = "dev-mdp-paidmedia-svc-sr4w.010_mdlz_dl_yahooads"

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
MAX_RETRY_ROUNDS = 2                          # how many retry passes after primary run
RETRY_SLEEP_CHOICES = [60, 75, 90, 100, 120]  # sleep between each retry item

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
# RETRY HELPER
# ═══════════════════════════════════════════════════════════

def run_retry_queue(reporter, retry_queue, output_dir, max_rounds=MAX_RETRY_ROUNDS):
    """
    Retry failed (day, group) pairs.

    Args:
        reporter    : YahooReporting instance
        retry_queue : list of {"day": str, "group": ReportGroupConfig class}
        output_dir  : base output directory
        max_rounds  : max retry passes

    Returns:
        still_failed : list of {"day": str, "group_name": str} that never recovered
    """
    if not retry_queue:
        logger.info("✅ No failures to retry.")
        return []

    still_pending = list(retry_queue)  # copy

    for round_num in range(1, max_rounds + 1):
        if not still_pending:
            break

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🔁 RETRY ROUND {round_num}/{max_rounds} — {len(still_pending)} item(s)")
        logger.info("=" * 60)

        # Log what we're retrying
        for item in still_pending:
            logger.info(f"   📌 {item['day']} | {item['group'].REPORT_NAME}")

        round_failures = []

        for idx, item in enumerate(still_pending, 1):
            day_str = item["day"]
            grp = item["group"]
            name = grp.REPORT_NAME

            logger.info(f"\n  [{round_num}.{idx}] Retrying {name} for {day_str} ...")

            overrides = build_single_day_overrides(day_str)

            try:
                df = reporter.run_report_group(grp, overrides, output_dir)

                if df is not None and not df.is_empty():
                    logger.success(f"  ✅ RETRY OK: {name} | {day_str} | {df.shape[0]} rows")
                else:
                    logger.warning(f"  ⚠️  RETRY EMPTY: {name} | {day_str} | 0 rows")
                    round_failures.append(item)

            except Exception as e:
                logger.error(f"  ❌ RETRY FAILED: {name} | {day_str} | {e}")
                round_failures.append(item)

            # Sleep between retries (skip after last)
            if idx < len(still_pending):
                sleep_secs = random.choice(RETRY_SLEEP_CHOICES)
                logger.debug(f"  💤 Retry sleep: {sleep_secs}s")
                time.sleep(sleep_secs)

        # Update pending list for next round
        still_pending = round_failures

        if still_pending:
            logger.warning(
                f"  Round {round_num} done — {len(still_pending)} still failed"
            )
        else:
            logger.success(f"  ✅ Round {round_num} — all retries succeeded!")

    # Build final failure list
    still_failed = [
        {"day": item["day"], "group_name": item["group"].REPORT_NAME}
        for item in still_pending
    ]

    if still_failed:
        logger.error("")
        logger.error("=" * 60)
        logger.error(f"❌ PERMANENTLY FAILED after {max_rounds} retry rounds: {len(still_failed)} item(s)")
        for f in still_failed:
            logger.error(f"   ❌ {f['day']} | {f['group_name']}")
        logger.error("=" * 60)

    return still_failed


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
    parser.add_argument(
        "--max-retries", type=int, default=MAX_RETRY_ROUNDS,
        help=f"Max retry rounds for failed reports (default {MAX_RETRY_ROUNDS})",
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

        # Determine the day string for retry overrides
        if args.date:
            day_str = args.date
        else:
            tz = pendulum.timezone(REPORT_TIMEZONE)
            day_str = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")

        results = reporter.run_all_groups(
            report_groups=groups_to_run,
            config_overrides=overrides,
            output_dir=output_dir,
            skip_cob=args.skip_cob,
            advertiser_ids_for_cob=ACCOUNT_IDS if ACCOUNT_IDS else None,
        )

        # ── Check for failures → build retry queue ──
        retry_queue = []
        for grp in groups_to_run:
            name = grp.REPORT_NAME
            df = results.get(name)
            if df is None or df.is_empty():
                retry_queue.append({"day": day_str, "group": grp})

        # ── Summary ──
        logger.info("\n📊 DAILY SUMMARY (primary run):")
        for name, df in results.items():
            if df is not None and not df.is_empty():
                logger.info(f"  ✅ {name:30s} → {df.shape[0]:>8,} rows × {df.shape[1]} cols")
            elif df is not None:
                logger.info(f"  ⚠️  {name:30s} → 0 rows (empty)")
            else:
                logger.info(f"  ❌ {name:30s} → FAILED")

        # ── Retry ──
        if retry_queue:
            logger.warning(f"\n⚠️  {len(retry_queue)} group(s) failed — starting retry ...")
            still_failed = run_retry_queue(
                reporter, retry_queue, output_dir, max_rounds=args.max_retries
            )
            if not still_failed:
                logger.success("🎉 All retries succeeded — daily run fully complete.")
        else:
            logger.success("🎉 All groups succeeded — no retries needed.")

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
        logger.info(f"  Range        : {all_days[0]} → {all_days[-1]}")
        logger.info(f"  Total days   : {total_days}")
        logger.info(f"  Groups       : {len(groups_to_run)}")
        logger.info(f"  Total POST   : {total_api_calls}")
        logger.info(f"  Sleep pool   : {BACKFILL_SLEEP_CHOICES}s")
        logger.info(f"  Max retries  : {args.max_retries}")
        logger.info("=" * 60)

        t0 = time.time()
        retry_queue = []   # collect all failures across all days
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

            # ── Check each group for this day ──
            ok = 0
            fail = 0
            for grp in groups_to_run:
                name = grp.REPORT_NAME
                df = results.get(name)
                if df is not None and not df.is_empty():
                    ok += 1
                else:
                    fail += 1
                    retry_queue.append({"day": day_str, "group": grp})
                    logger.warning(f"  ⚠️  QUEUED FOR RETRY: {name} | {day_str}")

            elapsed_day = time.time() - t_day
            day_results[day_str] = {"ok": ok, "fail": fail, "time": elapsed_day}
            logger.info(f"  Day {day_str}: {ok} OK, {fail} fail | {elapsed_day:.1f}s")

            # Random sleep between days (skip after last day)
            if day_idx < total_days:
                sleep_secs = random.choice(BACKFILL_SLEEP_CHOICES)
                logger.info(f"  💤 Sleeping {sleep_secs}s before next day ...")
                time.sleep(sleep_secs)

        # ── Primary pass summary ──
        elapsed_primary = time.time() - t0
        total_ok = sum(d["ok"] for d in day_results.values())
        total_fail = sum(d["fail"] for d in day_results.values())

        logger.info("\n" + "=" * 60)
        logger.info("📋 PRIMARY PASS COMPLETE")
        logger.info(f"  Days processed : {total_days}")
        logger.info(f"  Groups OK      : {total_ok}")
        logger.info(f"  Groups FAIL    : {total_fail}")
        logger.info(f"  Time           : {elapsed_primary / 3600:.1f} hours")
        logger.info(f"  Retry queue    : {len(retry_queue)} item(s)")
        logger.info("=" * 60)

        # ── Retry phase ──
        still_failed = run_retry_queue(
            reporter, retry_queue, output_dir, max_rounds=args.max_retries
        )

        # ── Final summary ──
        elapsed_total = time.time() - t0
        logger.info("\n" + "=" * 60)
        logger.info("🏁 BACKFILL FULLY COMPLETE")
        logger.info(f"  Total time         : {elapsed_total / 3600:.1f} hours")
        logger.info(f"  Primary OK         : {total_ok}")
        logger.info(f"  Retried            : {len(retry_queue)}")
        logger.info(f"  Recovered on retry : {len(retry_queue) - len(still_failed)}")
        logger.info(f"  Permanently failed : {len(still_failed)}")

        if still_failed:
            logger.error("\n  ❌ PERMANENTLY FAILED:")
            for f in still_failed:
                logger.error(f"     {f['day']} | {f['group_name']}")
        else:
            logger.success("  🎉 Zero failures — backfill 100% complete!")

        logger.info("=" * 60)


if __name__ == "__main__":
    main()