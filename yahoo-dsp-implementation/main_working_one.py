"""
main_v2.py
──────────
Local entry point for Yahoo DSP Reporting — Multi-Group Pipeline.

Each logical report group (campaign delivery, creative, video, etc.)
is executed independently, producing its own CSV file.

Usage examples:
    # Run ALL enabled groups for yesterday
    python main_v2.py --mode daily --skip-cob

    # Run ALL enabled groups with 13-month backfill
    python main_v2.py --mode backfill --skip-cob

    # Run only specific groups
    python main_v2.py --mode daily --skip-cob --groups campaign_delivery video_performance

    # Run ONLY the bidding report (requires dataSource=15, USD)
    python main_v2.py --mode daily --skip-cob --groups bidding

Environment variables required:
    YAHOO_DSP_CLIENT_ID
    YAHOO_DSP_CLIENT_SECRET
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


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

REPORT_TIMEZONE = "America/New_York"

# Advertiser account IDs — fill in your actual IDs.
# If empty, Yahoo defaults to your seat's advertisers.
ACCOUNT_IDS: list[int] = [
    # 12345,
    # 67890,
]

OUTPUT_DIR = "output"


# ═══════════════════════════════════════════════════════════════
# ENABLED REPORT GROUPS
# Comment / uncomment to control which groups run.
# ═══════════════════════════════════════════════════════════════

ENABLED_GROUPS = [
    CampaignDeliveryReport,
    CreativePerformanceReport,
    VideoPerformanceReport,
    ViewabilityReport,
    ConversionDetailReport,
    CostFeeReport,
    AudienceReachReport,
    BiddingReport,  # ← uncomment when ready (requires dataSource=15, currency=4 USD)
    GeoDeviceReport,
    SupplyInventoryReport,
]


# ═══════════════════════════════════════════════════════════════
# CONFIG BUILDERS
# ═══════════════════════════════════════════════════════════════


def build_backfill_overrides() -> dict:
    """
    BACKFILL: Custom range, 1 month back → yesterday.
    intervalTypeId=2 (Day) to populate Day column.
    """
    tz = pendulum.timezone(REPORT_TIMEZONE)
    end_dt = pendulum.yesterday(tz)
    start_dt = end_dt.subtract(days=2)

    fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
    start_str = start_dt.start_of("day").format(fmt)
    end_str = end_dt.end_of("day").format(fmt)

    logger.info(
        f"BACKFILL overrides:\n"
        f"  Start : {start_str}\n"
        f"  End   : {end_str}\n"
        f"  TZ    : {REPORT_TIMEZONE}"
    )
    return {
        "timezone": REPORT_TIMEZONE,
        "currency": 4,
        "accountIds": ACCOUNT_IDS if ACCOUNT_IDS else [],
        "dateTypeId": 11,              # Custom Range
        "intervalTypeId": 2,           # Day granularity
        "startDate": start_str,
        "endDate": end_str,
    }


def build_daily_overrides() -> dict:
    yesterday_str = (
        pendulum.now(REPORT_TIMEZONE).subtract(days=1).format("YYYY-MM-DD")
    )
    logger.info(
        f"DAILY overrides:\n"
        f"  Date range : Yesterday ({yesterday_str})\n"
        f"  TZ         : {REPORT_TIMEZONE}"
    )
    return {
        "timezone": REPORT_TIMEZONE,
        "currency": 4,
        "accountIds": ACCOUNT_IDS if ACCOUNT_IDS else [],
        "dateTypeId": 2,               # Yesterday
        "intervalTypeId": 2,           # Day
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Yahoo DSP Reporting — Multi-Group Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="daily",
        help="backfill = 13-month custom range | daily = yesterday",
    )
    parser.add_argument(
        "--skip-cob",
        action="store_true",
        help="Skip Close of Books check",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help=(
            "Run specific groups by REPORT_NAME slug, e.g.: "
            "campaign_delivery video_performance bidding"
        ),
    )
    args = parser.parse_args()

    # ── Banner ──
    logger.info("=" * 60)
    logger.info(f"📍 Yahoo DSP Multi-Group Pipeline — {args.mode.upper()}")
    logger.info("=" * 60)

    # ── Validate credentials ──
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

    # ── Build overrides ──
    if args.mode == "backfill":
        OUTPUT_DIR = "output_backfill"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        overrides = build_backfill_overrides()
    else:
        OUTPUT_DIR = "output_daily"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        overrides = build_daily_overrides()

    # ── Resolve groups ──
    if args.groups:
        # Map all known groups by name
        all_by_name = {g.REPORT_NAME: g for g in ALL_REPORT_GROUPS}
        groups_to_run = []
        for slug in args.groups:
            if slug in all_by_name:
                groups_to_run.append(all_by_name[slug])
            else:
                logger.warning(
                    f"⚠️  Unknown group '{slug}'. "
                    f"Available: {list(all_by_name.keys())}"
                )
        if not groups_to_run:
            logger.error("❌ No valid groups matched. Exiting.")
            sys.exit(1)
    else:
        groups_to_run = ENABLED_GROUPS

    logger.info(
        f"Groups to run ({len(groups_to_run)}): "
        f"{[g.REPORT_NAME for g in groups_to_run]}"
    )

    # ── Create output dir ──
    # os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Run pipeline ──
    reporter = YahooReporting()
    results = reporter.run_all_groups(
        report_groups=groups_to_run,
        config_overrides=overrides,
        output_dir=OUTPUT_DIR,
        skip_cob=args.skip_cob,
        advertiser_ids_for_cob=ACCOUNT_IDS if ACCOUNT_IDS else None,
    )

    # ── Final summary ──
    logger.info("\n📊 FINAL SUMMARY:")
    for name, df in results.items():
        if df is not None and not df.is_empty():
            logger.info(f"  ✅ {name:30s} → {df.shape[0]:>8,} rows × {df.shape[1]} cols")
        elif df is not None:
            logger.info(f"  ⚠️  {name:30s} → 0 rows (empty)")
        else:
            logger.info(f"  ❌ {name:30s} → FAILED")


if __name__ == "__main__":
    main()
