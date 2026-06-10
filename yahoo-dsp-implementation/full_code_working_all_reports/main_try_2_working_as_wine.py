"""
main.py — Yahoo DSP Reporting Pipeline

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
import pendulum
import functions_framework
from loguru import logger
import traceback
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


from dsp_traffic import (
    API_TYPE as TRAFFIC_API_TYPE,
    YahooTraffic,
    ALL_TRAFFIC_ENDPOINTS,
    ENDPOINT_REGISTRY,
)



# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

REPORT_TIMEZONE = "America/New_York"
ACCOUNT_IDS: list[int] = []

# Cloud Function writes to /tmp (only writable dir); local uses cwd
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


# ═══════════════════════════════════════════════════════════
# OVERRIDE BUILDERS
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# CLOUD LOADING — GCS + BQ
# ═══════════════════════════════════════════════════════════

# def load_results_to_cloud(helper, results, groups_to_run, api_type=API_TYPE):
#     gcs_bucket = helper.gcs_bucket
#     bq_dataset = helper.bq_dataset
#     project_id = helper.project_id

#     if not gcs_bucket:
#         logger.info("GCS_BUCKET not set — skipping cloud upload")
#         return

#     grp_map = {g.REPORT_NAME: g for g in groups_to_run}
#     cloud_ok = 0
#     cloud_fail = 0

#     for name, result in results.items():
#         if result is None or result.get("filepath") is None:
#             continue

#         filepath = result["filepath"]
#         day_str = result["day_str"]
#         grp = grp_map.get(name)
#         if not grp:
#             continue

#         try:
#             known_schema = grp.build_schema()
#             dynamic_schema = helper.build_dynamic_schema_from_csv(filepath, known_schema)

#             filename = os.path.basename(filepath)
#             blob_path = f"{api_type}/{name}/{day_str}/{filename}"
#             gcs_uri = helper.upload_file_to_gcs(gcs_bucket, blob_path, filepath)

#             if bq_dataset and project_id:
#                 # Table name: {api_type}_{report_name}
#                 table_id = f"{project_id}.{bq_dataset}.{api_type}_{name}"
#                 helper.ensure_bq_table(table_id, dynamic_schema)
#                 helper.load_gcs_to_bq(gcs_uri, table_id, dynamic_schema)
#                 cloud_ok += 1
#             else:
#                 logger.info(f"BQ_DATASET not set — GCS only for {name}")
#                 cloud_ok += 1

#         except Exception as e:
#             cloud_fail += 1
#             logger.error(f"Cloud load failed for {name}/{day_str}: {e}")

#     logger.info(f"☁️  Cloud load: {cloud_ok} OK, {cloud_fail} FAIL")

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

            # 1. Upload to GCS (always — even if 0 rows, for audit trail)
            filename = os.path.basename(filepath)
            blob_path = f"{api_type}/{name}/{day_str}/{filename}"
            gcs_uri = helper.upload_file_to_gcs(gcs_bucket, blob_path, filepath)

            # 2. BQ load (safe idempotent)
            if bq_dataset and project_id:
                table_id = f"{project_id}.{bq_dataset}.{api_type}_{name}"
                new_row_count = df.shape[0] if df is not None else 0

                # Ensure table exists / evolve schema
                helper.ensure_bq_table(table_id, dynamic_schema)

                # Safe idempotent: INSERT first → DELETE old
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

    logger.info(f"☁️  Cloud load: {cloud_ok} OK, {cloud_fail} FAIL")
# ═══════════════════════════════════════════════════════════
# RETRY HELPER
# ═══════════════════════════════════════════════════════════

def run_retry_queue(reporter, helper, retry_queue, output_dir,
                    groups_to_run, max_rounds=MAX_RETRY_ROUNDS,
                    api_type=API_TYPE):
    if not retry_queue:
        logger.info("✅ No failures to retry.")
        return []

    still_pending = list(retry_queue)

    for round_num in range(1, max_rounds + 1):
        if not still_pending:
            break

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"🔁 RETRY ROUND {round_num}/{max_rounds} — {len(still_pending)} item(s)")
        logger.info("=" * 60)

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
                result = reporter.run_report_group(grp, overrides, output_dir, api_type)

                if (result is not None and result["df"] is not None
                        and not result["df"].is_empty()):
                    logger.success(
                        f"  ✅ RETRY OK: {name} | {day_str} | "
                        f"{result['df'].shape[0]} rows"
                    )
                    # Cloud load the recovered result
                    if helper.gcs_bucket:
                        load_results_to_cloud(
                            helper, {name: result}, [grp], api_type
                        )
                else:
                    logger.warning(f"  ⚠️  RETRY EMPTY: {name} | {day_str}")
                    round_failures.append(item)

            except Exception as e:
                logger.error(f"  ❌ RETRY FAILED: {name} | {day_str} | {e}")
                round_failures.append(item)

            if idx < len(still_pending):
                sleep_secs = random.choice(RETRY_SLEEP_CHOICES)
                logger.debug(f"  💤 Retry sleep: {sleep_secs}s")
                time.sleep(sleep_secs)

        still_pending = round_failures
        if still_pending:
            logger.warning(f"  Round {round_num} done — {len(still_pending)} still failed")
        else:
            logger.success(f"  ✅ Round {round_num} — all retries succeeded!")

    still_failed = [
        {"day": item["day"], "group_name": item["group"].REPORT_NAME}
        for item in still_pending
    ]

    if still_failed:
        logger.error("")
        logger.error("=" * 60)
        logger.error(
            f"❌ PERMANENTLY FAILED after {max_rounds} retry rounds: "
            f"{len(still_failed)} item(s)"
        )
        for f in still_failed:
            logger.error(f"   ❌ {f['day']} | {f['group_name']}")
        logger.error("=" * 60)

    return still_failed


# ═══════════════════════════════════════════════════════════
# CORE PIPELINE (called by both CF and CLI)
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# TRAFFIC PIPELINE
# ═══════════════════════════════════════════════════════════

ENABLED_TRAFFIC_ENDPOINTS = ALL_TRAFFIC_ENDPOINTS

def run_traffic_pipeline(endpoints=None, max_retries=MAX_RETRY_ROUNDS,
                         start_time=None, end_time=None):
    client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        logger.error("❌ Missing credentials.")
        return {"status": "error", "message": "missing credentials"}

    if endpoints:
        eps_to_run = [ENDPOINT_REGISTRY[e] for e in endpoints if e in ENDPOINT_REGISTRY]
        if not eps_to_run:
            logger.error(f"❌ No matching endpoints: {endpoints}")
            return {"status": "error", "message": "no matching endpoints"}
    else:
        eps_to_run = ENABLED_TRAFFIC_ENDPOINTS

    logger.info(f"Traffic endpoints ({len(eps_to_run)}): {[e.ENDPOINT_NAME for e in eps_to_run]}")

    traffic = YahooTraffic()
    helper = traffic.helper

    output_dir = os.path.join(OUTPUT_BASE, "output_traffic")
    os.makedirs(output_dir, exist_ok=True)

    results = traffic.run_all_endpoints(
        eps_to_run, output_dir, TRAFFIC_API_TYPE,
        start_time=start_time,
        end_time=end_time,
    )

    # ── Cloud load ──
    # Build a minimal "group" wrapper so load_results_to_cloud works
    _traffic_grp_map = {}
    for ep in eps_to_run:
        # Create a simple object with build_schema that returns dynamic schema
        _traffic_grp_map[ep.ENDPOINT_NAME] = ep

    if helper.gcs_bucket:
        gcs_bucket = helper.gcs_bucket
        bq_dataset = helper.bq_dataset
        project_id = helper.project_id
        cloud_ok = 0
        cloud_fail = 0

        for name, result in results.items():
            if result is None or result.get("filepath") is None:
                continue

            filepath = result["filepath"]
            day_str = result["day_str"]
            df = result.get("df")

            try:
                # Dynamic schema from CSV headers (all STRING for traffic)
                dynamic_schema = []
                if df is not None:
                    for col in df.columns:
                        dynamic_schema.append({
                            "name": col, "type": "STRING", "mode": "NULLABLE"
                        })

                new_row_count = df.shape[0] if df is not None else 0
                if new_row_count == 0:
                    continue

                # GCS upload
                filename = os.path.basename(filepath)
                blob_path = f"{TRAFFIC_API_TYPE}/{name}/{day_str}/{filename}"
                gcs_uri = helper.upload_file_to_gcs(gcs_bucket, blob_path, filepath)

                # BQ load
                if bq_dataset and project_id:
                    table_id = f"{project_id}.{bq_dataset}.{TRAFFIC_API_TYPE}_{name}"
                    helper.ensure_bq_table(table_id, dynamic_schema)
                    helper.safe_idempotent_load(
                        gcs_uri=gcs_uri,
                        table_id=table_id,
                        schema_fields=dynamic_schema,
                        day_str=day_str,
                        new_row_count=new_row_count,
                    )
                    cloud_ok += 1

            except Exception as e:
                cloud_fail += 1
                logger.error(f"Cloud load failed for {name}/{day_str}: {e}")

        logger.info(f"☁️  Traffic cloud load: {cloud_ok} OK, {cloud_fail} FAIL")

    return {
        "status": "ok", "api": "traffic",
        "endpoints": len(eps_to_run),
    }



def run_pipeline(mode="daily", target_date=None, months=13,
                 groups=None, skip_cob=True, max_retries=MAX_RETRY_ROUNDS):
    """
    Core pipeline logic — mode-agnostic.
    Returns a dict with status for Cloud Function HTTP response.
    """

    # ── Credentials ──
    client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        logger.error(
            "❌ Missing credentials. Set:\n"
            "   YAHOO_DSP_CLIENT_ID\n"
            "   YAHOO_DSP_CLIENT_SECRET"
        )
        return {"status": "error", "message": "missing credentials"}
    logger.success(f"Credentials loaded | client_id: {client_id[:8]}...")

    # ── Resolve groups ──
    if groups:
        all_by_name = {g.REPORT_NAME: g for g in ALL_REPORT_GROUPS}
        groups_to_run = [all_by_name[s] for s in groups if s in all_by_name]
        if not groups_to_run:
            logger.error(f"❌ No matching groups: {groups}")
            return {"status": "error", "message": "no matching groups"}
    else:
        groups_to_run = ENABLED_GROUPS

    logger.info(f"Groups ({len(groups_to_run)}): {[g.REPORT_NAME for g in groups_to_run]}")

    reporter = YahooReporting()
    helper = reporter.helper

    # ══════════════════════════════════════
    # DAILY MODE
    # ══════════════════════════════════════
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

        # Check failures
        retry_queue = []
        for grp in groups_to_run:
            name = grp.REPORT_NAME
            res = results.get(name)
            if res is None or res.get("filepath") is None:
                retry_queue.append({"day": day_str, "group": grp})

        # Summary
        logger.info("\n📊 DAILY SUMMARY (primary run):")
        for name, res in results.items():
            if res and res.get("df") is not None and not res["df"].is_empty():
                logger.info(
                    f"  ✅ {name:30s} → {res['df'].shape[0]:>8,} rows "
                    f"× {res['df'].shape[1]} cols"
                )
            elif res:
                logger.info(f"  ⚠️  {name:30s} → 0 rows (empty)")
            else:
                logger.info(f"  ❌ {name:30s} → FAILED")

        # Retry
        still_failed = []
        if retry_queue:
            logger.warning(f"\n⚠️  {len(retry_queue)} group(s) failed — retrying ...")
            still_failed = run_retry_queue(
                reporter, helper, retry_queue, output_dir,
                groups_to_run, max_retries
            )
            if not still_failed:
                logger.success("🎉 All retries succeeded — daily run complete.")
        else:
            logger.success("🎉 All groups succeeded — no retries needed.")

        return {
            "status": "ok", "mode": "daily", "day": day_str,
            "failed": len(still_failed),
        }

    # ══════════════════════════════════════
    # BACKFILL MODE
    # ══════════════════════════════════════
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
        logger.info("🔄 BACKFILL PLAN")
        logger.info(f"  Range        : {all_days[0]} → {all_days[-1]}")
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

            # Cloud load for this day
            load_results_to_cloud(helper, results, groups_to_run)

            ok = 0
            fail = 0
            for grp in groups_to_run:
                name = grp.REPORT_NAME
                res = results.get(name)
                if res is not None and res.get("filepath") is not None:
                    ok += 1
                else:
                    fail += 1
                    retry_queue.append({"day": day_str, "group": grp})
                    logger.warning(f"  ⚠️  QUEUED FOR RETRY: {name} | {day_str}")

            elapsed_day = time.time() - t_day
            day_results[day_str] = {"ok": ok, "fail": fail, "time": elapsed_day}
            logger.info(f"  Day {day_str}: {ok} OK, {fail} fail | {elapsed_day:.1f}s")

            if day_idx < total_days:
                sleep_secs = random.choice(BACKFILL_SLEEP_CHOICES)
                logger.info(f"  💤 Sleeping {sleep_secs}s ...")
                time.sleep(sleep_secs)

        # Primary summary
        elapsed_primary = time.time() - t0
        total_ok = sum(d["ok"] for d in day_results.values())
        total_fail = sum(d["fail"] for d in day_results.values())

        logger.info("\n" + "=" * 60)
        logger.info("📋 PRIMARY PASS COMPLETE")
        logger.info(f"  Days: {total_days} | OK: {total_ok} | FAIL: {total_fail}")
        logger.info(f"  Time: {elapsed_primary / 3600:.1f}h | Retry queue: {len(retry_queue)}")
        logger.info("=" * 60)

        # Retry
        still_failed = run_retry_queue(
            reporter, helper, retry_queue, output_dir,
            groups_to_run, max_retries
        )

        elapsed_total = time.time() - t0
        logger.info("\n" + "=" * 60)
        logger.info("🏁 BACKFILL FULLY COMPLETE")
        logger.info(f"  Total time     : {elapsed_total / 3600:.1f} hours")
        logger.info(f"  Primary OK     : {total_ok}")
        logger.info(f"  Retried        : {len(retry_queue)}")
        logger.info(f"  Recovered      : {len(retry_queue) - len(still_failed)}")
        logger.info(f"  Perm. failed   : {len(still_failed)}")

        if still_failed:
            logger.error("\n  ❌ PERMANENTLY FAILED:")
            for f in still_failed:
                logger.error(f"     {f['day']} | {f['group_name']}")
        else:
            logger.success("  🎉 Zero failures — backfill 100% complete!")
        logger.info("=" * 60)

        return {
            "status": "ok", "mode": "backfill", "days": total_days,
            "failed": len(still_failed),
        }

    else:
        return {"status": "error", "message": f"unknown mode: {mode}"}


# ═══════════════════════════════════════════════════════════
# CLOUD FUNCTION ENTRY POINT
# ═══════════════════════════════════════════════════════════

# @functions_framework.http
# def yahoo_dsp_handler(request):
#     """
#     HTTP Cloud Function entry point — triggered by Cloud Scheduler.

#     Cloud Scheduler sends HTTP POST with JSON body:
#         Daily (auto, yesterday):  {"mode": "daily"}
#         Daily (specific date):    {"mode": "daily", "date": "2025-06-15"}
#         Backfill (13 months):     {"mode": "backfill", "months": 13}
#         Backfill (subset):        {"mode": "backfill", "months": 3,
#                                    "groups": ["campaign_delivery"]}

#     Returns:
#         200 + JSON on success
#         500 + JSON on failure
#     """
#     logger.info("=" * 60)
#     logger.info("☁️  Cloud Function triggered via HTTP")
#     logger.info("=" * 60)

#     request_json = request.get_json(silent=True) or {}

#     mode = request_json.get("mode", "daily")
#     target_date = request_json.get("date")
#     months = request_json.get("months", 13)
#     groups = request_json.get("groups")
#     max_retries = request_json.get("max_retries", MAX_RETRY_ROUNDS)

#     logger.info(
#         f"Request params: mode={mode}, date={target_date}, "
#         f"months={months}, groups={groups}, max_retries={max_retries}"
#     )

#     try:
#         result = run_pipeline(
#             mode=mode,
#             target_date=target_date,
#             months=months,
#             groups=groups,
#             skip_cob=True,
#             max_retries=max_retries,
#         )

#         status_code = 200 if result.get("status") == "ok" else 500
#         return (result, status_code)

#     except Exception as e:
#         logger.error(f"☁️  Cloud Function error: {e}\n{traceback.format_exc()}")
#         return ({"status": "error", "message": str(e)}, 500)


@functions_framework.http
def yahoo_dsp_handler(request):
    logger.info("=" * 60)
    logger.info("☁️  Cloud Function triggered via HTTP")
    logger.info("=" * 60)

    request_json = request.get_json(silent=True) or {}

    api = request_json.get("api", "reporting")  # "reporting" | "traffic"
    mode = request_json.get("mode", "daily")
    target_date = request_json.get("date")
    months = request_json.get("months", 13)
    groups = request_json.get("groups")
    endpoints = request_json.get("endpoints")
    max_retries = request_json.get("max_retries", MAX_RETRY_ROUNDS)
    start_time = request_json.get("start_time")
    end_time = request_json.get("end_time")


    logger.info(f"Request: api={api}, mode={mode}, date={target_date}")

    try:
        
        if api == "traffic":
                result = run_traffic_pipeline(
                    endpoints=endpoints,
                    max_retries=max_retries,
                    start_time=start_time,
                    end_time=end_time,
                )
        else:
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
        logger.error(f"☁️  Cloud Function error: {e}\n{traceback.format_exc()}")
        return ({"status": "error", "message": str(e)}, 500)

# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT (local development)
# ═══════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(description="Yahoo DSP Reporting Pipeline")
#     parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
#     parser.add_argument("--skip-cob", action="store_true")
#     parser.add_argument("--date", type=str, default=None)
#     parser.add_argument("--months", type=int, default=13)
#     parser.add_argument("--groups", nargs="*", default=None)
#     parser.add_argument("--max-retries", type=int, default=MAX_RETRY_ROUNDS)
#     args = parser.parse_args()

#     logger.info("=" * 60)
#     logger.info(f"📍 Yahoo DSP Pipeline — {args.mode.upper()} (CLI)")
#     logger.info("=" * 60)

#     run_pipeline(
#         mode=args.mode,
#         target_date=args.date,
#         months=args.months,
#         groups=args.groups,
#         skip_cob=args.skip_cob,
#         max_retries=args.max_retries,
#     )

def main():
    parser = argparse.ArgumentParser(description="Yahoo DSP Pipeline")
    parser.add_argument("--api", choices=["reporting", "traffic"], default="reporting")
    parser.add_argument("--mode", choices=["daily", "backfill"], default="daily")
    parser.add_argument("--skip-cob", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--months", type=int, default=13)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--endpoints", nargs="*", default=None)
    
    parser.add_argument("--start-time", type=str, default=None,
                        help="Traffic API startTime filter (ISO format)")
    parser.add_argument("--end-time", type=str, default=None,
                        help="Traffic API endTime filter (ISO format)")

    parser.add_argument("--max-retries", type=int, default=MAX_RETRY_ROUNDS)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"📍 Yahoo DSP Pipeline — {args.api.upper()} / {args.mode.upper()} (CLI)")
    logger.info("=" * 60)

    
    if args.api == "traffic":
            run_traffic_pipeline(
                endpoints=args.endpoints,
                max_retries=args.max_retries,
                start_time=args.start_time,
                end_time=args.end_time,
            )

    else:
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