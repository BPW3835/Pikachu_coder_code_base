"""
beam_pipeline.py
────────────────
Apache Beam entry point for Yahoo DSP Reporting — Multi-Group Pipeline.

Usage:
  python beam_pipeline.py --mode daily --output-dir output
  python beam_pipeline.py --mode backfill --output-dir output
  python beam_pipeline.py --mode backfill --groups campaign_delivery bidding

Env vars: YAHOO_DSP_CLIENT_ID, YAHOO_DSP_CLIENT_SECRET
"""

import os
import sys
import argparse

import pendulum
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from loguru import logger

from beam_reporting import (
    ALL_REPORT_GROUPS,
    build_report_configs,
    generate_backfill_dates,
    generate_daily_dates,
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
    SequentialExtractFn,
    ParseCSVFn,
    CastTypesFn,
    WritePartitionedCSVFn,
    LogSummaryFn,
)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

REPORT_TIMEZONE = "America/New_York"

ACCOUNT_IDS: list[int] = [
    # 12345,
    # 67890,
]

DEFAULT_OUTPUT_DIR = "output"
BACKFILL_MONTHS   = 13
RATE_LIMIT_SECONDS = 2.0

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


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════


def filter_groups_by_name(groups, names):
    """Return only those groups whose REPORT_NAME is in *names*."""
    if not names:
        return list(groups)
    name_set = set(names)
    matched = [g for g in groups if g.REPORT_NAME in name_set]
    missing = name_set - {g.REPORT_NAME for g in matched}
    if missing:
        logger.warning(f"Requested groups not found (skipping): {sorted(missing)}")
    return matched


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════


def run(args, beam_args):
    """Build and execute the Apache Beam pipeline."""

    # ── Determine dates ─────────────────────────────────────────
    if args.mode == "backfill":
        dates = generate_backfill_dates(REPORT_TIMEZONE, BACKFILL_MONTHS)
    else:
        dates = generate_daily_dates(REPORT_TIMEZONE)

    # ── Filter & serialise report groups ────────────────────────
    selected_groups = filter_groups_by_name(ENABLED_GROUPS, args.groups)
    if not selected_groups:
        logger.error("No report groups selected. Exiting.")
        sys.exit(1)

    report_configs = build_report_configs(selected_groups)

    # ── Summary log ─────────────────────────────────────────────
    total_api_calls = len(dates) * len(report_configs)
    est_seconds = total_api_calls * (args.rate_limit + 30.0)
    est_hours   = est_seconds / 3600

    logger.info("=" * 70)
    logger.info("  Yahoo DSP Reporting — Apache Beam Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Mode            : {args.mode}")
    logger.info(f"  Date range      : {dates[-1]}  →  {dates[0]}  ({len(dates)} days)")
    logger.info(f"  Report groups   : {[c['report_name'] for c in report_configs]}")
    logger.info(f"  Total API calls : {total_api_calls:,}")
    logger.info(f"  Rate-limit pause: {args.rate_limit}s")
    logger.info(f"  Est. runtime    : {est_seconds:,.0f}s  ≈  {est_hours:.1f}h")
    logger.info(f"  Output dir      : {args.output_dir}")
    logger.info(f"  Skip on error   : {args.skip_errors}")
    logger.info("=" * 70)

    # FIX 3: Only pass beam_args to PipelineOptions
    # (not our custom --mode, --groups, etc.)
    pipeline_options = PipelineOptions(
        beam_args,
        runner="DirectRunner",
        direct_num_workers=1,
        direct_running_mode="multi_threading",
    )

    # ── Build & run pipeline ────────────────────────────────────
    with beam.Pipeline(options=pipeline_options) as p:
        _ = (
            p
            | "Seed" >> beam.Create([None])

            | "SequentialExtract" >> beam.ParDo(
                SequentialExtractFn(
                    dates=dates,
                    report_configs=report_configs,
                    timezone=REPORT_TIMEZONE,
                    currency=4,
                    account_ids=ACCOUNT_IDS if ACCOUNT_IDS else None,
                    rate_limit_sec=args.rate_limit,
                    skip_on_error=args.skip_errors,
                )
            )

            | "ParseCSV"   >> beam.ParDo(ParseCSVFn())
            | "CastTypes"  >> beam.ParDo(CastTypesFn())

            | "KeyByPartition" >> beam.Map(
                lambda row: ((row["_date"], row["_report_name"]), row)
            )

            | "GroupByPartition"    >> beam.GroupByKey()

            | "WritePartitionedCSV" >> beam.ParDo(
                WritePartitionedCSVFn(output_dir=args.output_dir)
            )

            | "LogSummary" >> beam.ParDo(LogSummaryFn())
        )

    logger.info("🎉 Pipeline completed successfully.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Yahoo DSP Reporting — Apache Beam Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python beam_pipeline.py --mode daily\n"
            "  python beam_pipeline.py --mode backfill --output-dir output\n"
            "  python beam_pipeline.py --mode backfill --groups campaign_delivery bidding\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="daily",
        help="backfill = 13-month range; daily = yesterday only.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help=(
            "Run specific groups by REPORT_NAME slug, e.g.: "
            "campaign_delivery video_performance bidding. "
            "Omit to run all enabled groups."
        ),
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=RATE_LIMIT_SECONDS,
        help=f"Seconds to sleep between API calls (default: {RATE_LIMIT_SECONDS}).",
    )
    parser.add_argument(
        "--skip-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip on API errors. Use --no-skip-errors to fail fast.",
    )

    # FIX 3: parse_known_args separates our flags from Beam's flags
    args, beam_args = parser.parse_known_args()
    run(args, beam_args)


if __name__ == "__main__":
    main()
