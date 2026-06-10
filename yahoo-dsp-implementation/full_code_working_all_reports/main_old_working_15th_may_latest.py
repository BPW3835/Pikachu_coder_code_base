"""
main.py — Yahoo DSP Reporting Pipeline
═══════════════════════════════════════════════════════════════

Dual entry point:
  1. Cloud Function (triggered by Cloud Scheduler via HTTP POST)
  2. CLI (for local development / testing)

Cloud Scheduler JSON body examples:
    {"mode": "daily"}
    {"mode": "daily", "date": "2025-06-15"}
    {"mode": "backfill", "months": 13}
    {"mode": "backfill", "months": 3, "groups": ["campaign_delivery", "bidding"]}

CLI examples:
    python main.py --mode daily --skip-cob
    python main.py --mode daily --skip-cob --date 2025-06-15
    python main.py --mode backfill --skip-cob --months 13

Environment variables:
    YAHOO_DSP_CLIENT_ID      — required
    YAHOO_DSP_CLIENT_SECRET  — required
    PROJECT_ID               — GCP project (for GCS/BQ)
    GCS_BUCKET               — enables GCS upload
    BQ_DATASET               — enables BQ load
"""

import os
import sys
import time
import random
import argparse
import traceback

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
                # table_id = f"{project_id}.{bq_dataset}.{api_type}_{name}"
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
                 groups=None, skip_cob=True, max_retries=MAX_RETRY_ROUNDS):
    """
    Core reporting pipeline — supports daily and backfill modes.
    Returns a dict with status for Cloud Function HTTP response.
    """

    # —— Credentials ——
    # client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    # client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()
    # if not client_id or not client_secret:
    #     logger.error(
    #         "Missing credentials. Set:\n"
    #         "   YAHOO_DSP_CLIENT_ID\n"
    #         "   YAHOO_DSP_CLIENT_SECRET"
    #     )
    #     return {"status": "error", "message": "missing credentials"}
    # logger.success(f"Credentials loaded | client_id: {client_id[:8]}...")

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
    
    
    if not helper.client_id or not helper.client_secret:
            logger.error("Missing credentials from Secret Manager")
            return {"status": "error", "message": "missing credentials"}
    logger.success(f"Credentials loaded | client_id: {helper.client_id[:8]}...")

    # ══════════════════════════════════════════
    # DAILY MODE
    # ══════════════════════════════════════════
    if mode == "daily":
        output_dir = os.path.join(OUTPUT_BASE, "output_daily")
        os.makedirs(output_dir, exist_ok=True)
        overrides = build_daily_overrides(target_date=target_date)

        if target_date:
            day_str = target_date
        else:
            tz = pendulum.timezone(REPORT_TIMEZONE)
            day_str = pendulum.now(tz).subtract(days=1).format("YYYY-MM-DD")

        results = reporter.run_all_groups(
            report_groups=groups_to_run,
            config_overrides=overrides,
            output_dir=output_dir,
            skip_cob=skip_cob,
            advertiser_ids_for_cob=ACCOUNT_IDS if ACCOUNT_IDS else None,
        )

        # Cloud load
        load_results_to_cloud(helper, results, groups_to_run)

        # Identify failures for retry
        retry_queue = []
        for grp in groups_to_run:
            name = grp.REPORT_NAME
            res = results.get(name)
            if res is None or res.get("filepath") is None:
                retry_queue.append({"day": day_str, "group": grp})

        # Summary + Audit Trail
        logger.info("\nDAILY SUMMARY (primary run):")
        for grp in groups_to_run:
            name = grp.REPORT_NAME
            res = results.get(name)
            if res and res.get("df") is not None and not res["df"].is_empty():
                row_count = res["df"].shape[0]
                logger.info(
                    f"  {name:30s} -> {row_count:>8,} rows "
                    f"x {res['df'].shape[1]} cols"
                )
                helper.log_audit_trail(name, day_str, row_count, "COMPLETED", "daily")
            elif res:
                logger.info(f"  {name:30s} -> 0 rows (empty)")
                helper.log_audit_trail(name, day_str, 0, "EMPTY", "daily")
            else:
                logger.info(f"  {name:30s} -> FAILED")
                # Audit for FAILED logged during retry

        # Retry failed groups
        still_failed = []
        if retry_queue:
            logger.warning(f"\n{len(retry_queue)} group(s) failed — retrying ...")
            still_failed = run_retry_queue(
                reporter, helper, retry_queue, output_dir,
                groups_to_run, mode="daily", max_rounds=max_retries,
            )
            if not still_failed:
                logger.success("All retries succeeded — daily run complete.")
        else:
            logger.success("All groups succeeded — no retries needed.")

        return {
            "status": "ok", "mode": "daily", "day": day_str,
            "failed": len(still_failed),
        }

    # ══════════════════════════════════════════
    # BACKFILL MODE
    # ══════════════════════════════════════════
    elif mode == "backfill":
        output_dir = os.path.join(OUTPUT_BASE, "output_backfill")
        os.makedirs(output_dir, exist_ok=True)

        tz = pendulum.timezone(REPORT_TIMEZONE)
        end_date = pendulum.yesterday(tz)
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
        retry_queue = []
        day_results = {}

        for day_idx, day_str in enumerate(all_days, 1):
            t_day = time.time()
            logger.info(f"\nDay {day_idx}/{total_days}: {day_str}")

            overrides = build_single_day_overrides(day_str)
            results = reporter.run_all_groups(
                report_groups=groups_to_run,
                config_overrides=overrides,
                output_dir=output_dir,
                skip_cob=True,
                advertiser_ids_for_cob=None,
            )

            # Cloud load for this day
            load_results_to_cloud(helper, results, groups_to_run)

            ok = 0
            fail = 0
            for grp in groups_to_run:
                name = grp.REPORT_NAME
                res = results.get(name)
                if res is not None and res.get("filepath") is not None:
                    ok += 1
                    row_ct = res["df"].shape[0] if res.get("df") is not None else 0
                    helper.log_audit_trail(name, day_str, row_ct, "COMPLETED", "backfill")
                else:
                    fail += 1
                    retry_queue.append({"day": day_str, "group": grp})
                    logger.warning(f"  QUEUED FOR RETRY: {name} | {day_str}")

            elapsed_day = time.time() - t_day
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
        logger.info(f"  Time: {elapsed_primary / 3600:.1f}h | Retry queue: {len(retry_queue)}")
        logger.info("=" * 60)

        # Retry
        still_failed = run_retry_queue(
            reporter, helper, retry_queue, output_dir,
            groups_to_run, mode="backfill", max_rounds=max_retries,
        )

        elapsed_total = time.time() - t0
        logger.info("\n" + "=" * 60)
        logger.info("BACKFILL FULLY COMPLETE")
        logger.info(f"  Total time     : {elapsed_total / 3600:.1f} hours")
        logger.info(f"  Primary OK     : {total_ok}")
        logger.info(f"  Retried        : {len(retry_queue)}")
        logger.info(f"  Recovered      : {len(retry_queue) - len(still_failed)}")
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
        GET  (no body)  → defaults to daily mode (yesterday)
        POST (JSON body) → configurable

    JSON body examples:
        {"mode": "daily"}
        {"mode": "daily", "date": "2025-06-15"}
        {"mode": "backfill", "months": 13}
        {"mode": "backfill", "months": 3, "groups": ["campaign_delivery"]}

    Returns: tuple(dict, int) — JSON response + HTTP status code.
    """
    logger.info("=" * 60)
    logger.info("Cloud Function triggered via HTTP")
    logger.info(f"Method: {request.method}")
    logger.info("=" * 60)

    # ── Parse request body (GET = empty defaults, POST = JSON) ──
    request_json = {}
    if request.method == "POST":
        request_json = request.get_json(silent=True) or {}

    mode        = request_json.get("mode", "daily")
    target_date = request_json.get("date")
    months      = request_json.get("months", 13)
    groups      = request_json.get("groups")
    max_retries = request_json.get("max_retries", MAX_RETRY_ROUNDS)

    logger.info(f"Config: mode={mode}, date={target_date}, months={months}")

    try:
        result = run_pipeline(
            mode=mode,
            target_date=target_date,
            months=months,
            groups=groups,
            skip_cob=True,
            max_retries=max_retries,
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
    parser = argparse.ArgumentParser(description="Yahoo DSP Reporting Pipeline (Local)")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--skip-cob", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--months", type=int, default=13)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRY_ROUNDS)
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
    )


if __name__ == "__main__":
    main()