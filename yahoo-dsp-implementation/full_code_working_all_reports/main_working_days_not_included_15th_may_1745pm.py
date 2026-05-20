"""
main.py — Yahoo DSP Reporting Pipeline
═══════════════════════════════════════════════════════════════

Dual entry point:
  1. Cloud Function (triggered by Cloud Scheduler via HTTP POST)
  2. CLI (for local development / testing)

Cloud Scheduler JSON body examples:
    {"mode": "daily", "lookback_days": 7}
    {"mode": "daily", "date": "2025-06-15", "lookback_days": 7}
    {"mode": "backfill", "months": 13}
    {"mode": "backfill", "months": 13, "date": "2026-05-15"}
    {"mode": "backfill", "months": 3, "groups": ["campaign_delivery", "bidding"]}

CLI examples:
    python main.py --mode daily --skip-cob --lookback-days 7
    python main.py --mode daily --skip-cob --date 2025-06-15 --lookback-days 7
    python main.py --mode backfill --skip-cob --months 13
    python main.py --mode backfill --skip-cob --months 13 --date 2026-05-15
"""

import os
import sys
import time
import random
import argparse
import traceback
from collections import OrderedDict

import pendulum
import functions_framework
from loguru import logger

from helpers import YahooHelpers
from reporting import (
    API_TYPE,
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


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

REPORT_TIMEZONE = "America/New_York"
ACCOUNT_IDS: list[int] = []

IS_CLOUD = bool(os.environ.get("K_SERVICE") or os.environ.get("FUNCTION_TARGET"))
OUTPUT_BASE = "/tmp" if IS_CLOUD else "."

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

BACKFILL_SLEEP_CHOICES = [220, 210, 230, 200, 190]
MAX_RETRY_ROUNDS = 2
RETRY_SLEEP_CHOICES = [60, 75, 90, 100, 120]
DEFAULT_LOOKBACK_DAYS = 7


# ═══════════════════════════════════════════════════════════════
# OVERRIDE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_daily_overrides(target_date=None) -> dict:
    tz = pendulum.timezone(REPORT_TIMEZONE)
    if target_date:
        target = pendulum.parse(target_date, tz=tz)
        fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
        logger.info(f"DAILY (custom date): {target_date}, TZ={REPORT_TIMEZONE}")
        return {
            "timezone": REPORT_TIMEZONE, "currency": 4,
            "accountIds": ACCOUNT_IDS if ACCOUNT_IDS else [],
            "dateTypeId": 11, "intervalTypeId": 2,
            "startDate": target.start_of("day").format(fmt),
            "endDate": target.end_of("day").format(fmt),
        }
    else:
        yesterday_str = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")
        logger.info(f"DAILY: Yesterday ({yesterday_str}), TZ={REPORT_TIMEZONE}")
        return {
            "timezone": REPORT_TIMEZONE, "currency": 4,
            "accountIds": ACCOUNT_IDS if ACCOUNT_IDS else [],
            "dateTypeId": 2, "intervalTypeId": 2,
        }


def build_single_day_overrides(day_str) -> dict:
    tz = pendulum.timezone(REPORT_TIMEZONE)
    target = pendulum.parse(day_str, tz=tz)
    fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
    return {
        "timezone": REPORT_TIMEZONE, "currency": 4,
        "accountIds": ACCOUNT_IDS if ACCOUNT_IDS else [],
        "dateTypeId": 11, "intervalTypeId": 2,
        "startDate": target.start_of("day").format(fmt),
        "endDate": target.end_of("day").format(fmt),
    }


# ═══════════════════════════════════════════════════════════════
# CLOUD LOADING — GCS + BQ (Safe Idempotent)
# ═══════════════════════════════════════════════════════════════

def load_results_to_cloud(helper, results, groups_to_run, api_type=API_TYPE):
    gcs_bucket = helper.gcs_bucket
    bq_dataset = helper.bq_dataset
    project_id = helper.project_id

    if not gcs_bucket:
        logger.info("GCS_BUCKET not set — skipping cloud upload")
        return

    grp_map = {g.REPORT_NAME: g for g in groups_to_run}
    cloud_ok = 0
    cloud_fail = 0

    for name, result in results.items():
        if result is None or result.get("filepath") is None:
            continue

        filepath = result["filepath"]
        day_str = result["day_str"]
        df = result.get("df")
        grp = grp_map.get(name)
        if not grp:
            continue

        try:
            known_schema = grp.build_schema()
            dynamic_schema = helper.build_dynamic_schema_from_csv(filepath, known_schema)

            # 1. Upload to GCS
            filename = os.path.basename(filepath)
            blob_path = f"{api_type}/{name}/{day_str}/{filename}"
            gcs_uri = helper.upload_file_to_gcs(gcs_bucket, blob_path, filepath)

            # 2. BQ load (safe idempotent)
            if bq_dataset and project_id:
                table_name = f"010_mdlz_dl_yahoo_dsp_reporting_{name}"
                table_id = f"{project_id}.{bq_dataset}.{table_name}"
                new_row_count = df.shape[0] if df is not None else 0

                helper.ensure_bq_table(table_id, dynamic_schema)
                helper.safe_idempotent_load(
                    gcs_uri=gcs_uri,
                    table_id=table_id,
                    schema_fields=dynamic_schema,
                    day_str=day_str,
                    new_row_count=new_row_count,
                )
                cloud_ok += 1
            else:
                logger.info(f"BQ_DATASET not set — GCS only for {name}")
                cloud_ok += 1

        except Exception as e:
            cloud_fail += 1
            logger.error(f"Cloud load failed for {name}/{day_str}: {e}")

    logger.info(f"Cloud load: {cloud_ok} OK, {cloud_fail} FAIL")


# ═══════════════════════════════════════════════════════════════
# RUN SINGLE DAY — shared by daily + lookback + backfill
# ═══════════════════════════════════════════════════════════════

def run_single_day(reporter, helper, day_str, groups_to_run, output_dir,
                   pipeline_mode="daily", api_type=API_TYPE):
    """
    Run all report groups for a single date.
    Returns: (results_dict, retry_queue_list)
    Also writes audit trail per group.
    """
    logger.info(f"\n{'─' * 50}")
    logger.info(f"📅 Processing date: {day_str} (mode={pipeline_mode})")
    logger.info(f"{'─' * 50}")

    overrides = build_single_day_overrides(day_str)

    results = reporter.run_all_groups(
        report_groups=groups_to_run,
        config_overrides=overrides,
        output_dir=output_dir,
        skip_cob=True,
        advertiser_ids_for_cob=ACCOUNT_IDS if ACCOUNT_IDS else None,
    )

    # Cloud load
    load_results_to_cloud(helper, results, groups_to_run, api_type)

    # Audit trail + collect failures
    retry_queue = []
    for grp in groups_to_run:
        name = grp.REPORT_NAME
        res = results.get(name)

        if res and res.get("df") is not None and not res["df"].is_empty():
            row_count = res["df"].shape[0]
            logger.info(f"  ✅ {name:30s} → {row_count:>8,} rows")
            helper.log_audit_trail(name, day_str, row_count, "COMPLETED", pipeline_mode)
        elif res and res.get("filepath") is not None:
            logger.info(f"  ⚠️  {name:30s} → 0 rows (empty)")
            helper.log_audit_trail(name, day_str, 0, "EMPTY", pipeline_mode)
        else:
            logger.info(f"  ❌ {name:30s} → FAILED")
            helper.log_audit_trail(name, day_str, 0, "FAILED", pipeline_mode)
            retry_queue.append({"day": day_str, "group": grp})

    return results, retry_queue


# ═══════════════════════════════════════════════════════════════
# RETRY HELPER
# ═══════════════════════════════════════════════════════════════

def run_retry_queue(reporter, helper, retry_queue, output_dir,
                    groups_to_run, mode="daily",
                    max_rounds=MAX_RETRY_ROUNDS, api_type=API_TYPE):
    if not retry_queue:
        logger.info("No failures to retry.")
        return []

    still_pending = list(retry_queue)

    for round_num in range(1, max_rounds + 1):
        if not still_pending:
            break

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"RETRY ROUND {round_num}/{max_rounds} — {len(still_pending)} item(s)")
        logger.info("=" * 60)

        for item in still_pending:
            logger.info(f"   {item['day']} | {item['group'].REPORT_NAME}")

        round_failures = []

        for idx, item in enumerate(still_pending, 1):
            day_str = item["day"]
            grp = item["group"]
            name = grp.REPORT_NAME

            logger.info(f"  [{round_num}.{idx}] Retrying {name} for {day_str} ...")
            overrides = build_single_day_overrides(day_str)

            try:
                result = reporter.run_report_group(grp, overrides, output_dir, api_type)

                if (result is not None and result["df"] is not None
                        and not result["df"].is_empty()):
                    row_count = result["df"].shape[0]
                    logger.success(f"  RETRY OK: {name} | {day_str} | {row_count} rows")
                    if helper.gcs_bucket:
                        load_results_to_cloud(helper, {name: result}, [grp], api_type)
                    helper.log_audit_trail(name, day_str, row_count, "COMPLETED", mode)
                else:
                    logger.warning(f"  RETRY EMPTY: {name} | {day_str}")
                    round_failures.append(item)
                    helper.log_audit_trail(name, day_str, 0, "EMPTY", mode)

            except Exception as e:
                logger.error(f"  RETRY FAILED: {name} | {day_str} | {e}")
                round_failures.append(item)
                helper.log_audit_trail(name, day_str, 0, "FAILED", mode, error_message=str(e))

            if idx < len(still_pending):
                sleep_secs = random.choice(RETRY_SLEEP_CHOICES)
                logger.debug(f"  Retry sleep: {sleep_secs}s")
                time.sleep(sleep_secs)

        still_pending = round_failures
        if still_pending:
            logger.warning(f"  Round {round_num} done — {len(still_pending)} still failed")
        else:
            logger.success(f"  Round {round_num} — all retries succeeded!")

    still_failed = [
        {"day": item["day"], "group_name": item["group"].REPORT_NAME}
        for item in still_pending
    ]

    if still_failed:
        logger.error("")
        logger.error("=" * 60)
        logger.error(f"PERMANENTLY FAILED after {max_rounds} retry rounds: {len(still_failed)} item(s)")
        for f in still_failed:
            logger.error(f"   {f['day']} | {f['group_name']}")
        logger.error("=" * 60)

    return still_failed


# ═══════════════════════════════════════════════════════════════
# REPORTING PIPELINE (daily + backfill)
# ═══════════════════════════════════════════════════════════════

def run_pipeline(mode="daily", target_date=None, months=13,
                 groups=None, skip_cob=True, max_retries=MAX_RETRY_ROUNDS,
                 lookback_days=DEFAULT_LOOKBACK_DAYS):
    """
    Core reporting pipeline — supports daily and backfill modes.
    Returns a dict with status for Cloud Function HTTP response.
    """

    # —— Resolve report groups ——
    if groups:
        all_by_name = {g.REPORT_NAME: g for g in ALL_REPORT_GROUPS}
        groups_to_run = [all_by_name[s] for s in groups if s in all_by_name]
        if not groups_to_run:
            logger.error(f"No matching groups: {groups}")
            return {"status": "error", "message": "no matching groups"}
    else:
        groups_to_run = ENABLED_GROUPS

    logger.info(f"Groups ({len(groups_to_run)}): {[g.REPORT_NAME for g in groups_to_run]}")

    reporter = YahooReporting()
    helper = reporter.helper

    # Credentials from Secret Manager
    if not helper.client_id or not helper.client_secret:
        logger.error("Missing credentials from Secret Manager")
        return {"status": "error", "message": "missing credentials"}
    logger.success(f"Credentials loaded | client_id: {helper.client_id[:8]}...")

    # ══════════════════════════════════════════
    # DAILY MODE (with lookback recovery)
    # ══════════════════════════════════════════
    if mode == "daily":
        output_dir = os.path.join(OUTPUT_BASE, "output_daily")
        os.makedirs(output_dir, exist_ok=True)

        tz = pendulum.timezone(REPORT_TIMEZONE)

        # Primary date: yesterday or custom
        if target_date:
            yesterday_str = target_date
        else:
            yesterday_str = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")

        # ── Step 1: Build list of (day, group) items to process ──
        # Always include yesterday for all groups
        items_to_process = []
        for grp in groups_to_run:
            items_to_process.append({"day": yesterday_str, "group": grp})

        logger.info("=" * 60)
        logger.info(f"📅 DAILY MODE | Primary date: {yesterday_str} | Lookback: {lookback_days} days")
        logger.info("=" * 60)

        # ── Step 2: Lookback recovery from audit trail ──
        if lookback_days and lookback_days > 0:
            logger.info(f"🔍 Checking audit trail for incomplete dates (last {lookback_days} days) ...")

            expected_names = [g.REPORT_NAME for g in groups_to_run]
            incomplete_items = helper.get_incomplete_dates_from_audit(
                lookback_days, expected_names
            )

            # Map group_name back to group class
            grp_map = {g.REPORT_NAME: g for g in groups_to_run}

            # Merge: avoid duplicates with yesterday
            existing_keys = {(item["day"], item["group"].REPORT_NAME) for item in items_to_process}

            for item in incomplete_items:
                grp = grp_map.get(item["group_name"])
                if not grp:
                    continue
                key = (item["day"], item["group_name"])
                if key not in existing_keys:
                    items_to_process.append({"day": item["day"], "group": grp})
                    existing_keys.add(key)
        else:
            logger.info("🔍 Lookback disabled (lookback_days=0) — processing primary date only")

        # ── Step 3: Group items by date for sequential processing ──
        date_groups = OrderedDict()
        for item in items_to_process:
            day = item["day"]
            if day not in date_groups:
                date_groups[day] = []
            # Avoid duplicate groups per date
            if item["group"].REPORT_NAME not in [g.REPORT_NAME for g in date_groups[day]]:
                date_groups[day].append(item["group"])

        # Sort dates
        date_groups = OrderedDict(sorted(date_groups.items()))
        total_dates = len(date_groups)

        logger.info(f"📋 Dates to process: {total_dates}")
        for day, grps in date_groups.items():
            tag = "⭐ primary" if day == yesterday_str else "🔄 recovery"
            logger.info(f"  {tag} | {day} → {len(grps)} group(s): {[g.REPORT_NAME for g in grps]}")

        # ── Step 4: Process each date ──
        t0 = time.time()
        all_retry_queue = []

        for day_idx, (day_str, grps) in enumerate(date_groups.items(), 1):
            logger.info(f"\n{'═' * 60}")
            logger.info(f"📅 Date {day_idx}/{total_dates}: {day_str} ({len(grps)} groups)")
            logger.info(f"{'═' * 60}")

            results, retry_queue = run_single_day(
                reporter, helper, day_str, grps, output_dir,
                pipeline_mode="daily",
            )
            all_retry_queue.extend(retry_queue)

            # Sleep between dates
            if day_idx < total_dates:
                sleep_secs = random.choice([30, 45, 60])
                logger.info(f"  💤 Inter-date sleep: {sleep_secs}s")
                time.sleep(sleep_secs)

        # ── Step 5: Retry all failures ──
        still_failed = []
        if all_retry_queue:
            logger.warning(f"\n{len(all_retry_queue)} group(s) failed across all dates — retrying ...")
            still_failed = run_retry_queue(
                reporter, helper, all_retry_queue, output_dir,
                groups_to_run, mode="daily", max_rounds=max_retries,
            )
            if not still_failed:
                logger.success("All retries succeeded — daily run complete.")
        else:
            logger.success("All groups succeeded on all dates — no retries needed.")

        # ── Step 6: Summary ──
        elapsed = time.time() - t0
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📊 DAILY SUMMARY")
        logger.info(f"  Dates processed  : {total_dates}")
        logger.info(f"  Primary date     : {yesterday_str}")
        logger.info(f"  Lookback days    : {lookback_days}")
        logger.info(f"  Failed (final)   : {len(still_failed)}")
        logger.info(f"  Elapsed          : {elapsed:.1f}s")
        logger.info(f"{'=' * 60}")

        return {
            "status": "ok", "mode": "daily",
            "primary_date": yesterday_str,
            "dates_processed": total_dates,
            "lookback_days": lookback_days,
            "failed": len(still_failed),
        }

    # ══════════════════════════════════════════
    # BACKFILL MODE (with optional end date)
    # ══════════════════════════════════════════
    elif mode == "backfill":
        output_dir = os.path.join(OUTPUT_BASE, "output_backfill")
        os.makedirs(output_dir, exist_ok=True)

        tz = pendulum.timezone(REPORT_TIMEZONE)

        # End date: custom date or yesterday
        if target_date:
            end_date = pendulum.parse(target_date, tz=tz)
            logger.info(f"BACKFILL: Custom end date = {target_date}")
        else:
            end_date = pendulum.yesterday(tz)
            logger.info(f"BACKFILL: End date = yesterday ({end_date.format('YYYY-MM-DD')})")

        start_date = end_date.subtract(months=months)

        all_days = []
        cursor = start_date
        while cursor <= end_date:
            all_days.append(cursor.format("YYYY-MM-DD"))
            cursor = cursor.add(days=1)

        total_days = len(all_days)

        logger.info("=" * 60)
        logger.info("BACKFILL PLAN")
        logger.info(f"  Range        : {all_days[0]} -> {all_days[-1]}")
        logger.info(f"  Total days   : {total_days}")
        logger.info(f"  Groups       : {len(groups_to_run)}")
        logger.info(f"  Total POST   : {total_days * len(groups_to_run)}")
        logger.info(f"  Max retries  : {max_retries}")
        logger.info("=" * 60)

        t0 = time.time()
        all_retry_queue = []
        day_results = {}

        for day_idx, day_str in enumerate(all_days, 1):
            t_day = time.time()

            results, retry_queue = run_single_day(
                reporter, helper, day_str, groups_to_run, output_dir,
                pipeline_mode="backfill",
            )
            all_retry_queue.extend(retry_queue)

            elapsed_day = time.time() - t_day
            ok = sum(1 for grp in groups_to_run
                     if results.get(grp.REPORT_NAME) is not None
                     and results[grp.REPORT_NAME].get("filepath") is not None)
            fail = len(groups_to_run) - ok

            day_results[day_str] = {"ok": ok, "fail": fail, "time": elapsed_day}
            logger.info(f"  Day {day_str}: {ok} OK, {fail} fail | {elapsed_day:.1f}s")

            if day_idx < total_days:
                sleep_secs = random.choice(BACKFILL_SLEEP_CHOICES)
                logger.info(f"  Sleeping {sleep_secs}s ...")
                time.sleep(sleep_secs)

        # Primary summary
        elapsed_primary = time.time() - t0
        total_ok = sum(d["ok"] for d in day_results.values())
        total_fail = sum(d["fail"] for d in day_results.values())

        logger.info("\n" + "=" * 60)
        logger.info("PRIMARY PASS COMPLETE")
        logger.info(f"  Days: {total_days} | OK: {total_ok} | FAIL: {total_fail}")
        logger.info(f"  Time: {elapsed_primary / 3600:.1f}h | Retry queue: {len(all_retry_queue)}")
        logger.info("=" * 60)

        # Retry
        still_failed = run_retry_queue(
            reporter, helper, all_retry_queue, output_dir,
            groups_to_run, mode="backfill", max_rounds=max_retries,
        )

        elapsed_total = time.time() - t0
        logger.info("\n" + "=" * 60)
        logger.info("BACKFILL FULLY COMPLETE")
        logger.info(f"  Total time     : {elapsed_total / 3600:.1f} hours")
        logger.info(f"  Primary OK     : {total_ok}")
        logger.info(f"  Retried        : {len(all_retry_queue)}")
        logger.info(f"  Recovered      : {len(all_retry_queue) - len(still_failed)}")
        logger.info(f"  Perm. failed   : {len(still_failed)}")

        if still_failed:
            logger.error("\n  PERMANENTLY FAILED:")
            for f in still_failed:
                logger.error(f"     {f['day']} | {f['group_name']}")
        else:
            logger.success("  Zero failures — backfill 100% complete!")
        logger.info("=" * 60)

        return {
            "status": "ok", "mode": "backfill", "days": total_days,
            "start_date": all_days[0], "end_date": all_days[-1],
            "failed": len(still_failed),
        }

    else:
        return {"status": "error", "message": f"unknown mode: {mode}"}


# ═══════════════════════════════════════════════════════════════
# CLOUD FUNCTION ENTRY POINT
# ═══════════════════════════════════════════════════════════════

@functions_framework.http
def yahoo_dsp_handler(request):
    """
    HTTP Cloud Function entry point — triggered by Cloud Scheduler.

    Accepts both GET and POST:
        GET  (no body)  → defaults to daily mode with lookback
        POST (JSON body) → configurable

    JSON body examples:
        {"mode": "daily", "lookback_days": 7}
        {"mode": "daily", "date": "2025-06-15", "lookback_days": 7}
        {"mode": "backfill", "months": 13}
        {"mode": "backfill", "months": 13, "date": "2026-05-15"}
    """
    logger.info("=" * 60)
    logger.info("Cloud Function triggered via HTTP")
    logger.info(f"Method: {request.method}")
    logger.info("=" * 60)

    request_json = {}
    if request.method == "POST":
        request_json = request.get_json(silent=True) or {}

    mode          = request_json.get("mode", "daily")
    target_date   = request_json.get("date")
    months        = request_json.get("months", 13)
    groups        = request_json.get("groups")
    max_retries   = request_json.get("max_retries", MAX_RETRY_ROUNDS)
    lookback_days = request_json.get("lookback_days", DEFAULT_LOOKBACK_DAYS)

    logger.info(
        f"Config: mode={mode}, date={target_date}, months={months}, "
        f"lookback_days={lookback_days}"
    )

    try:
        result = run_pipeline(
            mode=mode,
            target_date=target_date,
            months=months,
            groups=groups,
            skip_cob=True,
            max_retries=max_retries,
            lookback_days=lookback_days,
        )
        status_code = 200 if result.get("status") == "ok" else 500
        return (result, status_code)

    except Exception as e:
        logger.error(f"Cloud Function error: {e}\n{traceback.format_exc()}")
        return ({"status": "error", "message": str(e)}, 500)


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT (local development only)
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Yahoo DSP Reporting Pipeline")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--skip-cob", action="store_true")
    parser.add_argument("--date", type=str, default=None,
                        help="Daily: primary date. Backfill: end date. (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=13)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRY_ROUNDS)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Daily mode: days to look back in audit trail for recovery")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Yahoo DSP Reporting — {args.mode.upper()} (CLI)")
    logger.info("=" * 60)

    run_pipeline(
        mode=args.mode,
        target_date=args.date,
        months=args.months,
        groups=args.groups,
        skip_cob=args.skip_cob,
        max_retries=args.max_retries,
        lookback_days=args.lookback_days,
    )


if __name__ == "__main__":
    main()