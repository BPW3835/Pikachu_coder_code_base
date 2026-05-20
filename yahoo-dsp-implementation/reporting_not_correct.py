"""
reporting.py
────────────
Yahoo DSP Reporting API module – v3 with multi-threaded backfill.

Features:
  • Pydantic payload validation
  • Report creation / status polling / CSV download
  • Polars transformation & BQ-type casting
  • Close-of-Books (COB) check
  • Single-group pipeline (run_report_group / run_all_groups)
  • **NEW** Multi-threaded backfill:
        backfill_all_groups
          └─ backfill_report_group  (sequential over months)
               └─ backfill_month   (parallel over days, max_workers=4)
                    └─ fetch_single_day
"""

import io
import os
import gc
import time
import json
import shutil
import traceback
import concurrent.futures
from pathlib import Path
from typing import Optional

import requests
import pendulum
import polars as pl
from pydantic import BaseModel, Field, model_validator
from loguru import logger
from tqdm import tqdm

from helpers import YahooHelpers


# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS — Payload Validation
# ═══════════════════════════════════════════════════════════

class FilterOptionValue(BaseModel):
    id: int | str
    name: Optional[str] = None

class FilterOption(BaseModel):
    dimensionTypeId: int
    isExcluded: bool = False
    includeValues: list[FilterOptionValue] = []

class SpecObject(BaseModel):
    metricTypeId: int
    type: str
    value: int | float

class HavingObject(BaseModel):
    type: str
    havingSpecs: list[SpecObject]

class LimitSpecColumn(BaseModel):
    direction: str = "DESCENDING"
    metricTypeId: int

class LimitSpec(BaseModel):
    columns: list[LimitSpecColumn]
    limit: int = 10000
    type: str = "default"

class ReportOption(BaseModel):
    timezone: str = "America/New_York"
    currency: int = 4
    dimensionTypeIds: list[int] = []
    metricTypeIds: list[int] = []
    filterOptions: list[FilterOption] = []
    having: Optional[HavingObject] = None
    limitSpec: Optional[LimitSpec] = None
    accountIds: list[int] = []

class ReportPayload(BaseModel):
    reportOption: ReportOption
    dateTypeId: int = Field(..., ge=1, le=11)
    intervalTypeId: int = Field(..., ge=1, le=5)
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    dataSource: Optional[int] = None

    @model_validator(mode="after")
    def _custom_range_requires_dates(self):
        if self.dateTypeId == 11 and (not self.startDate or not self.endDate):
            raise ValueError(
                "startDate and endDate are required when dateTypeId=11"
            )
        return self


# ═══════════════════════════════════════════════════════════
# REPORT GROUP CONFIG — Base + 10 Groups
# ═══════════════════════════════════════════════════════════

class ReportGroupConfig:
    """Base class for a report group configuration."""
    name: str = "base"
    dimension_ids: list[int] = []
    metric_ids: list[int] = []
    data_source: Optional[int] = None
    schema_fields: list[dict] = []

    @classmethod
    def build_payload(cls, overrides: dict) -> dict:
        ro = {
            "timezone": overrides.get("timezone", "America/New_York"),
            "currency": overrides.get("currency", 4),
            "dimensionTypeIds": cls.dimension_ids,
            "metricTypeIds": cls.metric_ids,
            "filterOptions": overrides.get("filterOptions", []),
            "accountIds": overrides.get("accountIds", []),
        }
        payload = {
            "reportOption": ro,
            "dateTypeId": overrides.get("dateTypeId", 2),
            "intervalTypeId": overrides.get("intervalTypeId", 2),
        }
        if cls.data_source is not None:
            payload["dataSource"] = cls.data_source
        if payload["dateTypeId"] == 11:
            payload["startDate"] = overrides.get("startDate")
            payload["endDate"] = overrides.get("endDate")
        return payload


# ── 1. Campaign Delivery ──
class CampaignDeliveryReport(ReportGroupConfig):
    name = "campaign_delivery"
    dimension_ids = [4, 5, 6, 149, 56, 64, 19, 63, 1225, 1267]
    metric_ids = [1, 2, 11, 23, 41, 42, 43, 44, 45, 137, 138, 171, 305, 317, 328]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Package Id", "type": "STRING"},
        {"name": "Package", "type": "STRING"},
        {"name": "Seat Id", "type": "STRING"},
        {"name": "Seat", "type": "STRING"},
        {"name": "Advertiser Group Id", "type": "STRING"},
        {"name": "Advertiser Group", "type": "STRING"},
        {"name": "Country Id", "type": "STRING"},
        {"name": "Country", "type": "STRING"},
        {"name": "Media Type Id", "type": "STRING"},
        {"name": "Media Type", "type": "STRING"},
        {"name": "Line Channel Type", "type": "STRING"},
        {"name": "Campaign Client status", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Clicks", "type": "INTEGER"},
        {"name": "CTR", "type": "FLOAT64"},
        {"name": "Conversion", "type": "FLOAT64"},
        {"name": "CPC", "type": "FLOAT64"},
        {"name": "Conversion Rate", "type": "FLOAT64"},
        {"name": "CPA", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "Advertiser eCPM", "type": "FLOAT64"},
        {"name": "ROAS Action Value", "type": "FLOAT64"},
        {"name": "ROAS", "type": "FLOAT64"},
        {"name": "Cross device conversions", "type": "FLOAT64"},
        {"name": "Dynamic Conversion Value", "type": "FLOAT64"},
        {"name": "Very first conversion", "type": "FLOAT64"},
        {"name": "3rd Party Conversion", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class CreativePerformanceReport(ReportGroupConfig):
    name = "creative_performance"
    dimension_ids = [4, 5, 6, 7, 8, 3, 174]
    metric_ids = [1, 2, 11, 23, 44, 45, 41, 43]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Ad Id", "type": "STRING"},
        {"name": "Ad", "type": "STRING"},
        {"name": "Creative Id", "type": "STRING"},
        {"name": "Creative Name", "type": "STRING"},
        {"name": "Creative Size Id", "type": "STRING"},
        {"name": "Creative Size", "type": "STRING"},
        {"name": "Ad Format Id", "type": "STRING"},
        {"name": "Ad Format", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Clicks", "type": "INTEGER"},
        {"name": "CTR", "type": "FLOAT64"},
        {"name": "Conversion", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "Advertiser eCPM", "type": "FLOAT64"},
        {"name": "CPC", "type": "FLOAT64"},
        {"name": "CPA", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class VideoPerformanceReport(ReportGroupConfig):
    name = "video_performance"
    dimension_ids = [4, 5, 6, 7]
    metric_ids = [1, 25, 26, 27, 28, 29, 62, 63, 64, 65, 169, 44]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Ad Id", "type": "STRING"},
        {"name": "Ad", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Start Views", "type": "INTEGER"},
        {"name": "25% Completions", "type": "INTEGER"},
        {"name": "50% Completions", "type": "INTEGER"},
        {"name": "75% Completions", "type": "INTEGER"},
        {"name": "100% Completions", "type": "INTEGER"},
        {"name": "25% Completion Rate", "type": "FLOAT64"},
        {"name": "50% Completion Rate", "type": "FLOAT64"},
        {"name": "75% Completion Rate", "type": "FLOAT64"},
        {"name": "100% Completion Rate", "type": "FLOAT64"},
        {"name": "CPCV", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class ViewabilityReport(ReportGroupConfig):
    name = "viewability"
    dimension_ids = [4, 5, 6]
    metric_ids = [1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 44, 168, 365]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Yahoo+ Measurable Impressions", "type": "INTEGER"},
        {"name": "Yahoo+ Viewable Impressions", "type": "INTEGER"},
        {"name": "Yahoo+ Viewable Rate", "type": "FLOAT64"},
        {"name": "Yahoo+ Nonviewable impressions", "type": "INTEGER"},
        {"name": "Yahoo+ Non-Measurable Impressions", "type": "INTEGER"},
        {"name": "Yahoo+ Measurable Rate", "type": "FLOAT64"},
        {"name": "Impression Distribution Viewable", "type": "FLOAT64"},
        {"name": "Impression Distribution Nonviewable", "type": "FLOAT64"},
        {"name": "Impression Distribution Nonmeasurable", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "VCPM", "type": "FLOAT64"},
        {"name": "Cost Per Valid Video View (CPV)", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class ConversionDetailReport(ReportGroupConfig):
    name = "conversion_detail"
    dimension_ids = [4, 5, 6, 46, 54, 147]
    metric_ids = [1, 23, 109, 110, 111, 112, 113, 114, 44, 317, 461, 462, 463, 464]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Beacon Id", "type": "STRING"},
        {"name": "Beacon", "type": "STRING"},
        {"name": "Beacon Type ID", "type": "STRING"},
        {"name": "Days to Conversion", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Conversion", "type": "FLOAT64"},
        {"name": "View Through Conversion", "type": "FLOAT64"},
        {"name": "Click Through Conversion", "type": "FLOAT64"},
        {"name": "View Through Conversion Rate", "type": "FLOAT64"},
        {"name": "Click Through Conversion Rate", "type": "FLOAT64"},
        {"name": "View Through CPA", "type": "FLOAT64"},
        {"name": "Click Through CPA", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "Very first conversion", "type": "FLOAT64"},
        {"name": "Very First View Through Conversion", "type": "FLOAT64"},
        {"name": "Very First Click Through Conversion", "type": "FLOAT64"},
        {"name": "Very First View Through CPA", "type": "FLOAT64"},
        {"name": "Very First Click Through CPA", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class CostFeeReport(ReportGroupConfig):
    name = "cost_fee"
    dimension_ids = [4, 5, 6]
    metric_ids = [1, 44, 129, 188, 192, 196, 200, 204, 208, 212, 216, 260, 322, 714, 715, 716, 717, 724]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "Seat Management Fee", "type": "FLOAT64"},
        {"name": "Order Management Fee", "type": "FLOAT64"},
        {"name": "Ad Serving Cost", "type": "FLOAT64"},
        {"name": "Ad Verification Cost", "type": "FLOAT64"},
        {"name": "Audience Measurement Cost", "type": "FLOAT64"},
        {"name": "Brand Safety Cost", "type": "FLOAT64"},
        {"name": "Viewability Cost", "type": "FLOAT64"},
        {"name": "Other Cost", "type": "FLOAT64"},
        {"name": "Total Variable Cost", "type": "FLOAT64"},
        {"name": "Total Vendor Fee", "type": "FLOAT64"},
        {"name": "Programmatic Guaranteed Deal Fee", "type": "FLOAT64"},
        {"name": "Total Data Fee", "type": "FLOAT64"},
        {"name": "Platform Fee", "type": "FLOAT64"},
        {"name": "Inventory ECPM", "type": "FLOAT64"},
        {"name": "Inventory Cost %", "type": "FLOAT64"},
        {"name": "Inventory Cost", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class GeoDeviceReport(ReportGroupConfig):
    name = "geo_device"
    dimension_ids = [4, 5, 6, 19, 20, 34, 122, 126]
    metric_ids = [1, 2, 11, 23, 44, 45]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Country Id", "type": "STRING"},
        {"name": "Country", "type": "STRING"},
        {"name": "State", "type": "STRING"},
        {"name": "Device Type Id", "type": "STRING"},
        {"name": "Device Type Name", "type": "STRING"},
        {"name": "Mobile OS", "type": "STRING"},
        {"name": "Connection Type", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Clicks", "type": "INTEGER"},
        {"name": "CTR", "type": "FLOAT64"},
        {"name": "Conversion", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "Advertiser eCPM", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class SupplyInventoryReport(ReportGroupConfig):
    name = "supply_inventory"
    dimension_ids = [4, 5, 6, 9, 106, 154, 1228]
    metric_ids = [1, 2, 11, 44, 45]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Domain", "type": "STRING"},
        {"name": "Inventory Type", "type": "STRING"},
        {"name": "Inventory Auction Type", "type": "STRING"},
        {"name": "Exchange Id", "type": "STRING"},
        {"name": "Exchange", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Clicks", "type": "INTEGER"},
        {"name": "CTR", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "Advertiser eCPM", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class AudienceReachReport(ReportGroupConfig):
    name = "audience_reach"
    dimension_ids = [4, 5, 6, 27, 28, 104]
    metric_ids = [1, 2, 11, 44, 45, 53, 115, 430, 523]
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Age Id", "type": "STRING"},
        {"name": "Age", "type": "STRING"},
        {"name": "Gender Id", "type": "STRING"},
        {"name": "Gender", "type": "STRING"},
        {"name": "Targeted Audience ID: Group A", "type": "STRING"},
        {"name": "Targeted Audience: Group A", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Clicks", "type": "INTEGER"},
        {"name": "CTR", "type": "FLOAT64"},
        {"name": "Advertiser Spending", "type": "FLOAT64"},
        {"name": "Advertiser eCPM", "type": "FLOAT64"},
        {"name": "Average Frequency", "type": "FLOAT64"},
        {"name": "Unique Viewers", "type": "INTEGER"},
        {"name": "Unique People", "type": "INTEGER"},
        {"name": "Average People Frequency", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


class BiddingReport(ReportGroupConfig):
    name = "bidding"
    dimension_ids = [4, 5, 6, 149]
    metric_ids = [1, 173, 254, 583, 584]
    data_source = 15
    schema_fields = [
        {"name": "Day", "type": "STRING"},
        {"name": "Advertiser Id", "type": "STRING"},
        {"name": "Advertiser", "type": "STRING"},
        {"name": "Campaign Id", "type": "STRING"},
        {"name": "Campaign Name", "type": "STRING"},
        {"name": "Line Id", "type": "STRING"},
        {"name": "Line", "type": "STRING"},
        {"name": "Package Id", "type": "STRING"},
        {"name": "Package", "type": "STRING"},
        {"name": "Impressions", "type": "INTEGER"},
        {"name": "Qualified Bid Requests", "type": "INTEGER"},
        {"name": "Bid Responses", "type": "INTEGER"},
        {"name": "Available Bid Requests", "type": "INTEGER"},
        {"name": "Average Win Rate", "type": "FLOAT64"},
        {"name": "ingestion_timestamp", "type": "STRING"},
        {"name": "lookback_date_range", "type": "STRING"},
    ]


ALL_REPORT_GROUPS = [
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


# ═══════════════════════════════════════════════════════════
# YAHOO REPORTING CLASS
# ═══════════════════════════════════════════════════════════


class YahooReporting:
    """
    End-to-end Yahoo DSP Reporting API handler.

    Modes:
      • Daily / single run  → run_report_group / run_all_groups
      • Backfill            → backfill_all_groups
            └─ backfill_report_group  (sequential months)
                 └─ backfill_month    (parallel days, ThreadPoolExecutor)
                      └─ fetch_single_day
    """

    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"

    DATE_TYPE_LABELS = {
        1: "Today", 2: "Yesterday", 3: "This_Week",
        4: "Last_7_Days", 5: "Last_Week", 6: "Last_14_Days",
        7: "MTD", 8: "Last_30_Days", 9: "Last_Month", 10: "All_Time",
    }

    def __init__(self):
        try:
            self.report_endpoint = (
                "http://api-sched-v3.admanagerplus.yahoo.com"
                "/yamplus_api/extreport/"
            )
            self.cob_endpoint = (
                "http://api-sched-v3.admanagerplus.yahoo.com"
                "/yamplus_api/extreport/cob"
            )
            self.helper = YahooHelpers()

            # Polling tunables
            self.poll_interval     = 15
            self.poll_max_attempts = 200
            self.poll_backoff      = 1.2
            self.poll_max_wait     = 120

            logger.success("YahooReporting initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────

    def _extract_date_range_label(self, config: dict) -> str:
        dt = config.get("dateTypeId")
        if dt == 10:
            return "ALL_TIME"
        if dt == 11:
            s = config.get("startDate", "?")[:10]
            e = config.get("endDate", "?")[:10]
            return f"{s} to {e}"
        return self.DATE_TYPE_LABELS.get(dt, f"dateTypeId_{dt}")

    def _add_metadata(self, df: pl.DataFrame, config: dict) -> pl.DataFrame:
        ts = pendulum.now("UTC").to_iso8601_string()
        dr = self._extract_date_range_label(config)
        return df.with_columns(
            pl.lit(ts).alias("ingestion_timestamp"),
            pl.lit(dr).alias("lookback_date_range"),
        )

    @staticmethod
    def _generate_filename(prefix: str = "yahoo_dsp") -> str:
        ts = pendulum.now("UTC").format("YYYY-MM-DD_HH-mm-ss")
        return f"{prefix}_{ts}.csv"

    @staticmethod
    def _cast_expression(col_name: str, bq_type: str) -> pl.Expr:
        bq_type = bq_type.upper()
        if bq_type == "STRING":
            return pl.col(col_name).cast(pl.Utf8).alias(col_name)
        if bq_type in ("INTEGER", "INT64"):
            return pl.col(col_name).cast(pl.Int64, strict=False).alias(col_name)
        if bq_type in ("FLOAT", "FLOAT64", "NUMERIC"):
            return pl.col(col_name).cast(pl.Float64, strict=False).alias(col_name)
        if bq_type in ("BOOLEAN", "BOOL"):
            return (
                pl.when(pl.col(col_name).str.to_lowercase().is_in(["true", "1"]))
                .then(True)
                .when(pl.col(col_name).str.to_lowercase().is_in(["false", "0"]))
                .then(False)
                .otherwise(None)
                .alias(col_name)
            )
        if bq_type == "DATE":
            return pl.col(col_name).str.to_date(strict=False).alias(col_name)
        if bq_type in ("DATETIME", "TIMESTAMP"):
            return pl.col(col_name).str.to_datetime(strict=False).alias(col_name)
        return pl.col(col_name).cast(pl.Utf8).alias(col_name)

    # ─────────────────────────────────────────────────────────
    # CORE API METHODS
    # ─────────────────────────────────────────────────────────

    def create_report(self, payload: dict) -> str:
        try:
            token = self.helper.get_access_token()
            headers = self.helper.build_auth_headers(token)

            dims = len(payload.get("reportOption", {}).get("dimensionTypeIds", []))
            mets = len(payload.get("reportOption", {}).get("metricTypeIds", []))
            logger.info(f"Creating report — dims={dims}, metrics={mets}")
            logger.debug(f"Payload:\n{json.dumps(payload, indent=2)}")

            resp = requests.post(
                self.report_endpoint, headers=headers,
                json=payload, timeout=60,
            )
            if resp.status_code not in (200, 201, 202):
                raise Exception(
                    f"Create report failed [{resp.status_code}]: {resp.text}"
                )
            body = resp.json()
            report_id = body.get("customerReportId")
            if not report_id:
                raise Exception(f"No customerReportId: {body}")

            logger.success(
                f"Report created | ID: {report_id} | Status: {body.get('status')}"
            )
            return report_id
        except Exception as e:
            logger.error(f"Create report error: {e}\n{traceback.format_exc()}")
            raise

    def poll_report_status(self, report_id: str) -> dict:
        try:
            logger.info(f"Polling report {report_id} ...")
            wait = self.poll_interval

            with tqdm(
                total=self.poll_max_attempts, desc="⏳ Polling",
                unit="req", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for attempt in range(1, self.poll_max_attempts + 1):
                    token = self.helper.get_access_token()
                    headers = self.helper.build_auth_headers(token)
                    resp = requests.get(
                        f"{self.report_endpoint}{report_id}",
                        headers=headers, timeout=60,
                    )
                    if resp.status_code != 200:
                        pbar.update(1)
                        time.sleep(wait)
                        wait = min(wait * self.poll_backoff, self.poll_max_wait)
                        continue

                    body = resp.json()
                    status = body.get("status", "").upper()
                    pbar.set_postfix_str(f"Status={status}")
                    pbar.update(1)

                    if status == self.SUCCESS:
                        logger.success(f"Report ready | Rows: {body.get('numRows', '?')}")
                        return body
                    if status == self.FAILED:
                        raise Exception(f"Report FAILED: {body}")

                    time.sleep(wait)
                    wait = min(wait * self.poll_backoff, self.poll_max_wait)

            raise TimeoutError(f"Report {report_id} not ready after {self.poll_max_attempts} attempts.")
        except Exception as e:
            logger.error(f"Polling error: {e}\n{traceback.format_exc()}")
            raise

    def download_report_csv(self, url: str) -> str:
        try:
            logger.info("Downloading report CSV ...")
            resp = requests.get(url, timeout=300)
            if resp.status_code != 200:
                raise Exception(f"Download failed [{resp.status_code}]")
            logger.success(f"Downloaded | {len(resp.text):,} chars")
            return resp.text
        except Exception as e:
            logger.error(f"Download error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────────────────────────────────────────────────
    # TRANSFORM
    # ─────────────────────────────────────────────────────────

    def transform_to_polars(
        self, csv_text: str, schema_fields: list[dict], config: dict
    ) -> pl.DataFrame:
        try:
            df = pl.read_csv(
                io.StringIO(csv_text),
                infer_schema_length=0,
                null_values=["", "null", "NULL", "None", "N/A"],
            )
            logger.info(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")

            # Add metadata
            df = self._add_metadata(df, config)

            # Cast
            type_map = {f["name"]: f["type"] for f in schema_fields}
            schema_cols = set(type_map)
            df_cols = set(df.columns)

            missing = schema_cols - df_cols
            extra = df_cols - schema_cols
            if missing:
                logger.warning(f"Schema cols missing in data: {missing}")
            if extra:
                logger.info(f"Dropping cols not in schema: {extra}")

            keep = [c for c in df.columns if c in schema_cols]
            exprs = [self._cast_expression(c, type_map[c]) for c in keep]
            df = df.select(exprs)
            logger.success(f"Transform done | {df.shape[0]} rows × {df.shape[1]} cols")
            return df
        except Exception as e:
            logger.error(f"Transform error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────────────────────────────────────────────────
    # COB (Close of Books)
    # ─────────────────────────────────────────────────────────

    def check_close_of_books(self, advertiser_id: int, day: str) -> dict:
        try:
            token = self.helper.get_access_token()
            headers = self.helper.build_auth_headers(token)
            resp = requests.get(
                self.cob_endpoint,
                params={"advertiserId": advertiser_id, "day": day},
                headers=headers, timeout=60,
            )
            if resp.status_code != 200:
                raise Exception(f"COB failed [{resp.status_code}]: {resp.text}")
            return resp.json()
        except Exception as e:
            logger.error(f"COB error: {e}\n{traceback.format_exc()}")
            raise

    def wait_for_cob(
        self, advertiser_ids, target_day, max_retries=12, retry_delay_secs=300
    ) -> bool:
        try:
            for attempt in range(1, max_retries + 1):
                all_ok = True
                for adv_id in advertiser_ids:
                    r = self.check_close_of_books(adv_id, target_day)
                    if str(r.get("dayClosed", "false")).lower() != "true":
                        all_ok = False
                        break
                if all_ok:
                    logger.success(f"✅ COB closed for all on {target_day}")
                    return True
                if attempt < max_retries:
                    logger.info(f"COB retry {attempt}/{max_retries} — waiting {retry_delay_secs}s ...")
                    time.sleep(retry_delay_secs)
            return False
        except Exception as e:
            logger.error(f"COB wait error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────────────────────────────────────────────────
    # SINGLE GROUP PIPELINE (daily / ad-hoc)
    # ─────────────────────────────────────────────────────────

    def run_report_group(
        self, report_group, overrides: dict, output_dir: str = "output"
    ) -> Optional[pl.DataFrame]:
        t0 = time.time()
        name = report_group.name.upper()
        try:
            logger.info("=" * 60)
            logger.info(f"▶ [{name}] Starting report group")
            logger.info("=" * 60)

            payload = report_group.build_payload(overrides)
            validated = ReportPayload(**payload).model_dump(exclude_none=True)

            report_id = self.create_report(validated)
            status = self.poll_report_status(report_id)
            url = status.get("url")
            if not url:
                raise Exception("No download URL")

            csv_text = self.download_report_csv(url)
            df = self.transform_to_polars(csv_text, report_group.schema_fields, payload)

            os.makedirs(output_dir, exist_ok=True)
            fname = self._generate_filename(report_group.name)
            fpath = os.path.join(output_dir, fname)
            df.write_csv(fpath)

            elapsed = time.time() - t0
            logger.success(
                f"✅ [{name}] DONE | {df.shape[0]} rows × {df.shape[1]} cols "
                f"| {elapsed:.1f}s → {os.path.abspath(fpath)}"
            )
            return df
        except Exception as e:
            logger.error(f"❌ [{name}] FAILED: {e}\n{traceback.format_exc()}")
            return None

    def run_all_groups(
        self, report_groups, config_overrides, output_dir="output", skip_cob=False
    ) -> dict:
        results = {}
        logger.info(f"🚀 Running {len(report_groups)} report group(s) ...")
        with tqdm(total=len(report_groups), desc="📊 Report Groups", unit="grp") as pbar:
            for rg in report_groups:
                df = self.run_report_group(rg, config_overrides, output_dir)
                results[rg.name] = df
                pbar.update(1)
        ok = sum(1 for v in results.values() if v is not None)
        logger.info(f"Summary: {ok}/{len(results)} groups succeeded")
        return results

    # ═══════════════════════════════════════════════════════════
    # BACKFILL ENGINE — Multi-threaded
    # ═══════════════════════════════════════════════════════════

    def fetch_single_day(
        self,
        report_group,
        day_date: pendulum.Date,
        base_overrides: dict,
    ) -> Optional[pl.DataFrame]:
        """
        Fetch ONE day's data for one report group.
        Returns a small DataFrame or None on error.
        Thread-safe: creates its own payload, no shared state mutation.
        """
        day_str = day_date.format("YYYY-MM-DD")
        name = report_group.name
        tz = base_overrides.get("timezone", "America/New_York")
        fmt = "YYYY-MM-DD[T]HH:mm:ssZ"

        try:
            tz_obj = pendulum.timezone(tz)
            start_dt = pendulum.instance(day_date, tz=tz_obj).start_of("day")
            end_dt   = pendulum.instance(day_date, tz=tz_obj).end_of("day")

            day_overrides = {
                **base_overrides,
                "dateTypeId": 11,
                "intervalTypeId": 2,
                "startDate": start_dt.format(fmt),
                "endDate":   end_dt.format(fmt),
            }

            payload = report_group.build_payload(day_overrides)
            validated = ReportPayload(**payload).model_dump(exclude_none=True)

            report_id = self.create_report(validated)
            status = self.poll_report_status(report_id)
            url = status.get("url")
            if not url:
                logger.warning(f"[{name}] {day_str}: No URL returned")
                return None

            csv_text = self.download_report_csv(url)
            df = self.transform_to_polars(csv_text, report_group.schema_fields, payload)

            logger.info(f"[{name}] {day_str}: {df.shape[0]} rows fetched")
            return df

        except Exception as e:
            logger.warning(f"[{name}] {day_str}: FAILED — {e}")
            return None

    def backfill_month(
        self,
        report_group,
        year: int,
        month: int,
        base_overrides: dict,
        output_dir: str = "output/temp",
        max_workers: int = 4,
    ) -> Optional[str]:
        """
        Backfill one calendar month using parallel daily fetches.

        Flow:
          1. Generate list of days in the month
          2. ThreadPoolExecutor → fetch_single_day per day
          3. Collect results, concatenate into one DataFrame
          4. Write monthly CSV → return filepath
          5. Free memory (del + gc.collect)

        Returns filepath of the monthly CSV, or None if no data.
        """
        name = report_group.name
        month_label = f"{year}-{month:02d}"
        logger.info(f"[{name}] 📅 Backfilling month: {month_label}")

        # Generate all days in the month
        tz = base_overrides.get("timezone", "America/New_York")
        first_day = pendulum.date(year, month, 1)
        last_day = first_day.end_of("month")

        # Clip to yesterday (don't fetch future / today)
        yesterday = pendulum.yesterday(tz)
        if last_day > yesterday:
            last_day = yesterday
        if first_day > yesterday:
            logger.warning(f"[{name}] {month_label}: all days in the future — skipping")
            return None

        days = []
        d = first_day
        while d <= last_day:
            days.append(d)
            d = d.add(days=1)

        logger.info(f"[{name}] {month_label}: {len(days)} days to fetch (workers={max_workers})")

        # Parallel fetch
        daily_dfs: list[pl.DataFrame] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_day = {
                executor.submit(self.fetch_single_day, report_group, day, base_overrides): day
                for day in days
            }
            with tqdm(
                total=len(days),
                desc=f"  📆 {name} {month_label}",
                unit="day",
                leave=False,
            ) as pbar:
                for future in concurrent.futures.as_completed(future_to_day):
                    day = future_to_day[future]
                    try:
                        result = future.result()
                        if result is not None and not result.is_empty():
                            daily_dfs.append(result)
                    except Exception as e:
                        logger.warning(f"[{name}] {day}: thread error — {e}")
                    pbar.update(1)

        if not daily_dfs:
            logger.warning(f"[{name}] {month_label}: no data collected")
            return None

        # Concatenate daily DataFrames
        month_df = pl.concat(daily_dfs, how="diagonal_relaxed")
        logger.info(f"[{name}] {month_label}: {month_df.shape[0]} total rows")

        # Write monthly CSV
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{name}_{month_label}.csv")
        month_df.write_csv(filepath)
        logger.success(f"[{name}] {month_label}: saved → {filepath}")

        # Free memory
        del daily_dfs, month_df
        gc.collect()

        return filepath

    def backfill_report_group(
        self,
        report_group,
        start_date: pendulum.Date,
        end_date: pendulum.Date,
        base_overrides: dict,
        output_dir: str = "output",
        temp_dir: str = "output/temp",
        max_workers: int = 4,
    ) -> Optional[str]:
        """
        Backfill a single report group across multiple months.

        Flow:
          1. Generate (year, month) list from start_date to end_date
          2. Sequential loop: backfill_month for each (parallel inside)
          3. Combine monthly CSVs → final CSV using Polars lazy scan
          4. Delete temp files
          5. Return final filepath

        Memory-efficient: uses scan_csv (lazy) to combine without
        loading all months into RAM simultaneously.
        """
        name = report_group.name
        t0 = time.time()
        logger.info("=" * 60)
        logger.info(f"🔄 [{name.upper()}] BACKFILL: {start_date} → {end_date}")
        logger.info("=" * 60)

        # Generate (year, month) tuples
        months = []
        cursor = start_date.start_of("month")
        end_month = end_date.start_of("month")
        while cursor <= end_month:
            months.append((cursor.year, cursor.month))
            cursor = cursor.add(months=1)

        logger.info(f"[{name}] {len(months)} month(s) to backfill")

        # Sequential month processing (parallel inside each month)
        monthly_csvs: list[str] = []
        for year, month in months:
            csv_path = self.backfill_month(
                report_group, year, month,
                base_overrides, temp_dir, max_workers,
            )
            if csv_path:
                monthly_csvs.append(csv_path)

        if not monthly_csvs:
            logger.warning(f"[{name}] No data from any month")
            return None

        # Combine monthly CSVs using lazy scan (memory-efficient)
        os.makedirs(output_dir, exist_ok=True)
        start_str = start_date.format("YYYY-MM-DD")
        end_str = end_date.format("YYYY-MM-DD")
        final_path = os.path.join(
            output_dir, f"{name}_backfill_{start_str}_{end_str}.csv"
        )

        logger.info(f"[{name}] Combining {len(monthly_csvs)} monthly CSVs → {final_path}")

        if len(monthly_csvs) == 1:
            # Single month — just move/rename
            shutil.move(monthly_csvs[0], final_path)
        else:
            # Multiple months — lazy concat + sink
            lazy_frames = [
                pl.scan_csv(f, infer_schema_length=0)
                for f in monthly_csvs
            ]
            combined = pl.concat(lazy_frames, how="diagonal_relaxed")
            combined.sink_csv(final_path)

            # Delete temp monthly CSVs
            for f in monthly_csvs:
                try:
                    os.remove(f)
                except OSError:
                    pass

        # Cleanup empty temp dir
        try:
            temp_path = Path(temp_dir)
            if temp_path.exists() and not any(temp_path.iterdir()):
                temp_path.rmdir()
        except OSError:
            pass

        elapsed = time.time() - t0
        logger.success(
            f"✅ [{name.upper()}] BACKFILL COMPLETE | {elapsed:.1f}s → {final_path}"
        )

        gc.collect()
        return final_path

    def backfill_all_groups(
        self,
        report_groups: list,
        start_date: pendulum.Date,
        end_date: pendulum.Date,
        base_overrides: dict,
        output_dir: str = "output",
        temp_dir: str = "output/temp",
        max_workers: int = 4,
    ) -> dict:
        """
        Backfill ALL report groups.

        Args:
            report_groups:  List of ReportGroupConfig subclasses
            start_date:     Backfill start (pendulum.Date)
            end_date:       Backfill end (pendulum.Date)
            base_overrides: {timezone, currency, accountIds}
            output_dir:     Final CSV output directory
            temp_dir:       Temp directory for monthly CSVs
            max_workers:    Thread pool size for daily fetches

        Returns:
            {report_name: final_csv_path | None}
        """
        t0 = time.time()
        results = {}

        logger.info("=" * 60)
        logger.info(f"🚀 BACKFILL ALL GROUPS: {start_date} → {end_date}")
        logger.info(f"   Groups: {len(report_groups)} | Workers: {max_workers}")
        logger.info("=" * 60)

        with tqdm(
            total=len(report_groups),
            desc="📊 Backfill Groups",
            unit="grp",
        ) as pbar:
            for rg in report_groups:
                try:
                    path = self.backfill_report_group(
                        rg, start_date, end_date,
                        base_overrides, output_dir, temp_dir, max_workers,
                    )
                    results[rg.name] = path
                except Exception as e:
                    logger.error(f"❌ [{rg.name.upper()}] BACKFILL FAILED: {e}")
                    results[rg.name] = None
                pbar.update(1)

        # Summary
        elapsed = time.time() - t0
        ok = sum(1 for v in results.values() if v is not None)
        logger.info("=" * 60)
        logger.info(f"🏁 BACKFILL COMPLETE | {ok}/{len(results)} groups | {elapsed:.1f}s")
        for name, path in results.items():
            status = "✅" if path else "❌"
            logger.info(f"  {status} {name}: {path or 'NO DATA'}")
        logger.info("=" * 60)

        return results
