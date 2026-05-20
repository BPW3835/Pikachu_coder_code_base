"""
main.py — Yahoo DSP Reporting Pipeline CLI

Usage:
    python main.py --mode daily
    python main.py --mode daily --skip-cob
    python main.py --mode backfill --months 13 --workers 4
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


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

REPORT_TIMEZONE     = "America/New_York"
ACCOUNT_IDS: list[int] = []          # empty → all advertisers
DAILY_OUTPUT_DIR    = "output/daily"
BACKFILL_OUTPUT_DIR = "output/backfill"

ENABLED_GROUPS = [
    CampaignDeliveryReport(),
    CreativePerformanceReport(),
    VideoPerformanceReport(),
    ViewabilityReport(),
    ConversionDetailReport(),
    CostFeeReport(),
    GeoDeviceReport(),
    SupplyInventoryReport(),
    AudienceReachReport(),
    BiddingReport(),
]


# ═══════════════════════════════════════════════════════
# OVERRIDE BUILDERS
# ═══════════════════════════════════════════════════════

def build_daily_overrides() -> dict:
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


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Yahoo DSP Reporting Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["daily", "backfill"], default="daily",
    )
    parser.add_argument("--skip-cob", action="store_true")
    parser.add_argument(
        "--months", type=int, default=13,
        help="Backfill: how many months back (default 13)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Backfill: parallel threads per month (default 4)",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Filter report groups by name (e.g. campaign_delivery bidding)",
    )
    args = parser.parse_args()

    # ── Banner ──
    logger.info("=" * 60)
    logger.info(f"📍 Yahoo DSP Multi-Group Pipeline — {args.mode.upper()}")
    logger.info("=" * 60)

    # ── Credentials check ──
    client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        logger.error(
            "❌ Missing credentials. Set:\n"
            "   export YAHOO_DSP_CLIENT_ID='...'\n"
            "   export YAHOO_DSP_CLIENT_SECRET='...'"
        )
        return
    logger.success(f"Credentials loaded | client_id: {client_id[:8]}...")

    # ── Select groups ──
    groups = ENABLED_GROUPS
    if args.groups:
        groups = [g for g in ALL_REPORT_GROUPS if g.name in args.groups]
        if not groups:
            logger.error(f"No matching groups for: {args.groups}")
            return

    logger.info(f"Groups to run ({len(groups)}): {[g.name for g in groups]}")

    # ── Init reporter ──
    reporter = YahooReporting()

    # ═══════════════════════════════════════════════════
    # DAILY MODE
    # ═══════════════════════════════════════════════════
    if args.mode == "daily":
        overrides = build_daily_overrides()

        results = reporter.run_all_groups(
            groups=groups,
            output_dir=DAILY_OUTPUT_DIR,
            overrides=overrides,
            account_ids=ACCOUNT_IDS if ACCOUNT_IDS else None,
            skip_cob=args.skip_cob,
        )

        # Summary
        logger.info("\n📊 DAILY FINAL SUMMARY:")
        for name, df in results.items():
            if df is not None:
                logger.info(f"  ✅ {name} → {df.shape[0]:,} rows × {df.shape[1]} cols")
            else:
                logger.info(f"  ❌ {name} → FAILED or EMPTY")

    # ═══════════════════════════════════════════════════
    # BACKFILL MODE
    # ═══════════════════════════════════════════════════
    elif args.mode == "backfill":
        logger.info(
            f"BACKFILL CONFIG:\n"
            f"  Months  : {args.months}\n"
            f"  Workers : {args.workers}\n"
            f"  TZ      : {REPORT_TIMEZONE}\n"
            f"  Groups  : {len(groups)}"
        )

        results = reporter.backfill_all_groups(
            groups=groups,
            months_back=args.months,
            output_dir=BACKFILL_OUTPUT_DIR,
            tz=REPORT_TIMEZONE,
            currency=4,
            account_ids=ACCOUNT_IDS if ACCOUNT_IDS else None,
            max_workers=args.workers,
        )

        # Summary
        logger.info("\n📊 BACKFILL FINAL SUMMARY:")
        for name, path in results.items():
            if path and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                logger.info(f"  ✅ {name} → {path} ({size_mb:.1f} MB)")
            else:
                logger.info(f"  ❌ {name} → NO DATA")


if __name__ == "__main__":
    main()
