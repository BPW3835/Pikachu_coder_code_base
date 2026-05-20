"""
reporting.py
────────────
Yahoo DSP Reporting API module – production-ready rewrite.

Handles:
  • Pydantic payload validation
  • Report creation / status polling / CSV download
  • Polars transformation (cast to BQ-compatible types)
  • Close-of-Books (COB) check with retry
  • Pipeline orchestration for multiple report groups
  • Dimension-to-CSV-column mapping & metric-name registry

GCS / BQ load steps are commented out for local development.
"""

import io
import os
import json
import time
import traceback
from typing import Optional
import gc
import shutil
import concurrent.futures
from pathlib import Path
import pendulum
import polars as pl
import requests
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from helpers import YahooHelpers


# ═══════════════════════════════════════════════════════════════
# DIMENSION → CSV COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════
# Each key is a dimensionTypeId; the value is a list of
# (column_name, bq_type) tuples that Yahoo returns in the CSV.

DIMENSION_CSV_COLUMNS: dict[int, list[tuple[str, str]]] = {
    3:    [("Creative Size Id", "STRING"), ("Creative Size", "STRING")],
    4:    [("Advertiser Id", "STRING"), ("Advertiser", "STRING")],
    5:    [("Campaign Id", "STRING"), ("Campaign Name", "STRING")],
    6:    [("Line Id", "STRING"), ("Line", "STRING")],
    7:    [("Ad Id", "STRING"), ("Ad", "STRING")],
    8:    [("Creative Id", "STRING"), ("Creative", "STRING")],
    9:    [("Domain Name", "STRING")],
    19:   [("Country Id", "STRING"), ("Country", "STRING")],
    20:   [("Region Id", "STRING"), ("Region", "STRING")],
    23:   [("Device Category Id", "STRING"), ("Device Category", "STRING")],
    25:   [("Browser Type ID", "STRING"), ("Browser Type Name", "STRING")],
    27:   [("Age ID", "STRING"), ("Age", "STRING")],
    28:   [("Gender ID", "STRING"), ("Gender", "STRING")],
    34:   [("Device Type Id", "STRING"), ("Device Type", "STRING")],
    46:   [("Beacon ID", "STRING"), ("Beacon", "STRING")],
    50:   [("Deal ID", "STRING"), ("Deal", "STRING")],
    54:   [("Beacon Type Id", "STRING")],
    55:   [("App Id", "STRING"), ("App Name", "STRING")],
    56:   [("Seat Id", "STRING"), ("Seat", "STRING")],
    63:   [("Media Type Id", "STRING"), ("Media Type", "STRING")],
    64:   [("Advertiser Group Id", "STRING"), ("Advertiser Group", "STRING")],
    106:  [("Inventory Type", "STRING")],
    122:  [("Mobile OS ID", "STRING"), ("Mobile OS", "STRING")],
    126:  [("Connection Type", "STRING")],
    147:  [("Days To Conversion", "STRING")],
    149:  [("Package Id", "STRING"), ("Package", "STRING")],
    153:  [("Connected TV Device", "STRING")],
    154:  [("Programmatic Guaranteed Impression", "STRING"), ("Auction Type", "STRING")],
    166:  [("Channel Type Id", "STRING")],
    167:  [("Video Content Length ID", "STRING"), ("Video Content Length", "STRING")],
    169:  [("Video Ad Placement ID", "STRING"), ("Video Ad Placement", "STRING")],
    171:  [("Ads.txt & App-ads.txt Declaration", "STRING")],
    174:  [("Ad Format Id", "STRING"), ("Ad Format", "STRING")],
    182:  [("Video Player Size", "STRING")],
    1028: [("Connected TV Make", "STRING")],
    1029: [("Connected TV Model", "STRING")],
    1033: [("Deal Type Description", "STRING")],
    1037: [("Video Creative Duration", "STRING")],
    1225: [("Line Channel Type", "STRING")],
    1226: [("Line Channel Subtype", "STRING")],
    1228: [("Exchange Id", "STRING"), ("Exchange", "STRING")],
    1240: [("Video Ad Placement (IAB) ID", "STRING"), ("Video Ad Placement IAB", "STRING")],
    1267: [("Campaign Client status", "STRING")],
}


# ═══════════════════════════════════════════════════════════════
# METRIC ID → CSV COLUMN NAME
# ═══════════════════════════════════════════════════════════════

METRIC_NAMES: dict[int, str] = {
    1:   "Impressions",
    2:   "Clicks",
    11:  "CTR",
    23:  "Conversion",
    25:  "Start Views",
    26:  "25% Complete",
    27:  "50% Complete",
    28:  "75% Complete",
    29:  "100% Complete",
    30:  "Drop Rate at 0%",
    31:  "Drop Rate at 25%",
    32:  "Drop Rate at 50%",
    33:  "Drop Rate at 75%",
    41:  "CPC",
    42:  "Conversion Rate",
    43:  "CPA",
    44:  "Advertiser Spending",
    45:  "Advertiser eCPM",
    49:  "Network Profit",
    50:  "Network Profit %",
    51:  "Network eCPM",
    62:  "25% Completion Rate",
    63:  "50% Completion Rate",
    64:  "75% Completion Rate",
    65:  "100% Completion Rate",
    100: "Yahoo+ Measurable Impressions",
    101: "Yahoo+ Viewable Impressions",
    102: "Yahoo+ Viewability Rate",
    103: "Yahoo+ Nonviewable Impressions",
    104: "Yahoo+ Non-Measurable Impressions",
    105: "Yahoo+ Measurable Rate",
    106: "Impression Distribution Viewable",
    107: "Impression Distribution Nonviewable",
    108: "Impression Distribution Nonmeasurable",
    109: "View Through Conversion",
    110: "Click Through Conversion",
    111: "View Through ConversionRate",
    112: "Click Through ConversionRate",
    113: "View Through CPA",
    114: "Click Through CPA",
    129: "Seat Management Fee",
    137: "ROAS Action Value",
    138: "ROAS",
    168: "vCPM",
    169: "CPCV",
    171: "Cross device conversions",
    173: "Qualified Count",
    180: "Yahoo Data Fees",
    184: "Third Party Data Fees",
    188: "Order Management Fee",
    192: "Ad Serving Cost",
    196: "Ad Verification Cost",
    200: "Audience Measurement Cost",
    204: "Brand Safety Cost",
    208: "Viewability Cost",
    212: "Other Cost",
    216: "Total Variable Cost",
    254: "Bid Responses",
    260: "Total Vendor Fee",
    296: "Bid Price Without Multiplier",
    302: "Bid CPM",
    305: "Dynamic Conversion Value",
    317: "Very first conversion",
    322: "Programmatic Guaranteed Deal Fee",
    328: "3rd Party Conversion",
    329: "Audience Impressions",
    365: "CPV",
    382: "Bid Shading Cost Savings",
    386: "Companion Impressions",
    387: "Companion Clicks",
    426: "Estimated Billable Amount",
    430: "Unique People",
    446: "Unique Households",
    447: "Average Frequency Per HouseHold",
    448: "Average Impression Multiplier",
    461: "Very First View Through Conversion",
    462: "Very First Click Through Conversion",
    463: "Very First View Through Conversion CPA",
    464: "Very First Click Through Conversion CPA",
    465: "View Through Dynamic Conversion Value",
    466: "View Through Dynamic Conversion ROAS",
    467: "Click Through Dynamic Conversion Value",
    468: "Click Through Dynamic Conversion ROAS",
    469: "View Through Conversion Value",
    470: "View Through Conversion ROAS",
    471: "Click Through Conversion Value",
    472: "Click Through Conversion ROAS",
    473: "Very First View Through Conversion Value",
    474: "Very First View Through Conversion ROAS",
    475: "Very First Click Through Conversion Value",
    476: "Very First Click Through Conversion ROAS",
    523: "Average People Frequency",
    560: "Yahoo+ Demo Fee",
    564: "Yahoo Segment Fee",
    570: "Third Party Segment Fee",
    583: "Available Bid Requests",
    584: "Average Win Rate",
    714: "Total Data Fee",
    715: "Platform Fee",
    716: "Inventory eCPM",
    717: "Inventory Cost %",
    721: "Yahoo+ Viewable Impressions GroupM",
    723: "Yahoo+ Viewability Rate GroupM",
    724: "Inventory Cost",
}

# Metrics whose BQ type should be INTEGER; everything else defaults to FLOAT64.
INTEGER_METRIC_IDS: set[int] = {
    1, 2, 23, 25, 26, 27, 28, 29,
    100, 101, 103, 104, 109, 110,
    171, 173, 254, 317, 328, 329,
    386, 387, 430, 446, 461, 462, 583, 721,
}


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS — Payload Validation
# ═══════════════════════════════════════════════════════════════


class FilterOptionValue(BaseModel):
    """Single value inside filterOptions.includeValues."""
    id: int | str
    name: Optional[str] = None


class FilterOption(BaseModel):
    """Dimension-level filter for report payload."""
    dimensionTypeId: int
    isExcluded: bool = False
    includeValues: list[FilterOptionValue] = []


class SpecObject(BaseModel):
    """Metric-level spec used inside having.havingSpecs."""
    metricTypeId: int
    type: str       # equalTo | greaterThan | lessThan | not
    value: int | float


class HavingObject(BaseModel):
    """Metric-level filter (having clause)."""
    type: str       # and | or | not
    havingSpecs: list[SpecObject]


class LimitSpecColumn(BaseModel):
    """Single sort column for limitSpec."""
    direction: str = "DESCENDING"
    metricTypeId: int


class LimitSpec(BaseModel):
    """Controls row limit and sort order."""
    columns: list[LimitSpecColumn]
    limit: int = 10_000
    type: str = "default"


class ReportOption(BaseModel):
    """
    reportOption object — defines dimensions, metrics,
    filters, timezone, and currency.
    """
    timezone: str = "America/New_York"
    currency: int = 4                           # 1=Seat 2=Adv 3=Campaign 4=USD
    dimensionTypeIds: list[int] = []
    metricTypeIds: list[int] = []
    filterOptions: list[FilterOption] = []
    having: Optional[HavingObject] = None
    limitSpec: Optional[LimitSpec] = None
    accountIds: list[int] = []


class ReportPayload(BaseModel):
    """
    Full reporting payload for POST /extreport/.

    dateTypeId:
        1=Today  2=Yesterday  3=This Week  4=Last 7 Days
        5=Last Week  6=Last 14 Days  7=MTD  8=Last 30 Days
        9=Last Month  10=All Time  11=Custom Range

    intervalTypeId:
        1=Cumulative  2=Day  3=Month  4=Week  5=Hour

    IMPORTANT: intervalTypeId=2 (Day) populates the Day column.
    """
    reportOption: ReportOption
    dateTypeId: int = Field(..., ge=1, le=11)
    intervalTypeId: int = Field(..., ge=1, le=5)
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    dataSource: Optional[int] = None            # 15 for bidding reports

    @model_validator(mode="after")
    def _custom_range_requires_dates(self):
        """startDate & endDate are mandatory when dateTypeId=11."""
        if self.dateTypeId == 11 and (not self.startDate or not self.endDate):
            raise ValueError(
                "startDate and endDate are required when dateTypeId=11 (Custom Range)"
            )
        return self


# ═══════════════════════════════════════════════════════════════
# REPORT GROUP CONFIGURATION (base + subclasses)
# ═══════════════════════════════════════════════════════════════


class ReportGroupConfig:
    """
    Base class for a report group.
    Subclasses set REPORT_NAME, DIMENSION_IDS, METRIC_IDS,
    and optionally DATA_SOURCE.
    """

    REPORT_NAME: str = "base"
    DIMENSION_IDS: list[int] = []
    METRIC_IDS: list[int] = []
    DATA_SOURCE: int | None = None

    _DATE_TYPE_LABELS: dict[int, str] = {
        1: "Today", 2: "Yesterday", 3: "This_Week",
        4: "Last_7_Days", 5: "Last_Week", 6: "Last_14_Days",
        7: "MTD", 8: "Last_30_Days", 9: "Last_Month", 10: "All_Time",
    }

    # ── schema builder ──────────────────────────────────────

    @classmethod
    def build_schema(cls) -> list[dict]:
        """
        Auto-build a BQ schema from DIMENSION_IDS + METRIC_IDS.
        Order: Day → dimension cols → metric cols → metadata cols.
        """
        schema: list[dict] = [
            {"name": "Day", "type": "STRING", "mode": "NULLABLE"},
        ]

        # Dimension columns
        for dim_id in cls.DIMENSION_IDS:
            cols = DIMENSION_CSV_COLUMNS.get(dim_id, [])
            for col_name, col_type in cols:
                schema.append({"name": col_name, "type": col_type, "mode": "NULLABLE"})

        # Metric columns
        for met_id in cls.METRIC_IDS:
            met_name = METRIC_NAMES.get(met_id)
            if met_name is None:
                logger.warning(f"Metric ID {met_id} not in METRIC_NAMES — skipped.")
                continue
            bq_type = "INTEGER" if met_id in INTEGER_METRIC_IDS else "FLOAT64"
            schema.append({"name": met_name, "type": bq_type, "mode": "NULLABLE"})

        # Metadata columns
        schema.append({"name": "ingestion_timestamp", "type": "STRING", "mode": "NULLABLE"})
        schema.append({"name": "lookback_date_range", "type": "STRING", "mode": "NULLABLE"})

        return schema

    # ── payload builder ─────────────────────────────────────

    @classmethod
    def build_payload(cls, overrides: dict | None = None) -> dict:
        """
        Build a validated report payload dict.
        *overrides* can supply dateTypeId, intervalTypeId, timezone,
        currency, startDate, endDate, filterOptions, accountIds, etc.
        """
        overrides = overrides or {}

        report_option: dict = {
            "timezone": overrides.get("timezone", "America/New_York"),
            "currency": overrides.get("currency", 4),
            "dimensionTypeIds": cls.DIMENSION_IDS,
            "metricTypeIds": cls.METRIC_IDS,
            "filterOptions": overrides.get("filterOptions", []),
            "accountIds": overrides.get("accountIds", []),
        }

        payload: dict = {
            "reportOption": report_option,
            "dateTypeId": overrides.get("dateTypeId", 2),
            "intervalTypeId": overrides.get("intervalTypeId", 2),
        }

        if cls.DATA_SOURCE is not None:
            payload["dataSource"] = cls.DATA_SOURCE

        # Custom range dates
        if payload["dateTypeId"] == 11:
            payload["startDate"] = overrides.get("startDate")
            payload["endDate"] = overrides.get("endDate")

        # Validate via Pydantic
        validated = ReportPayload(**payload)
        return validated.model_dump(exclude_none=True)

    # ── date range label ────────────────────────────────────

    @classmethod
    def date_range_label(cls, config: dict) -> str:
        """Human-readable date range string for metadata column."""
        dt = config.get("dateTypeId")
        if dt == 10:
            return "ALL_TIME"
        if dt == 11:
            s = config.get("startDate", "?")[:10]
            e = config.get("endDate", "?")[:10]
            return f"{s} to {e}"
        return cls._DATE_TYPE_LABELS.get(dt, f"dateTypeId_{dt}")


# ──────────────────────────────────────────────────────────────
# Concrete report groups
# ──────────────────────────────────────────────────────────────


class CampaignDeliveryReport(ReportGroupConfig):
    REPORT_NAME = "campaign_delivery"
    DIMENSION_IDS = [4, 5, 6, 149, 56, 64, 19, 63, 1225, 1267]
    METRIC_IDS = [1, 2, 11, 23, 41, 42, 43, 44, 45, 137, 138, 171, 305, 317, 328]


class CreativePerformanceReport(ReportGroupConfig):
    REPORT_NAME = "creative_performance"
    DIMENSION_IDS = [4, 5, 6, 7, 8, 3, 174]
    METRIC_IDS = [1, 2, 11, 23, 44, 45, 41, 43]


class VideoPerformanceReport(ReportGroupConfig):
    REPORT_NAME = "video_performance"
    DIMENSION_IDS = [4, 5, 6, 3, 169, 167, 182, 1037, 1240]
    METRIC_IDS = [
        1, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        62, 63, 64, 65, 168, 169, 365, 386, 387, 44,
    ]


class ViewabilityReport(ReportGroupConfig):
    REPORT_NAME = "viewability"
    DIMENSION_IDS = [4, 5, 6, 34, 106, 19]
    METRIC_IDS = [1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 721, 723, 44]


class ConversionDetailReport(ReportGroupConfig):
    REPORT_NAME = "conversion_detail"
    DIMENSION_IDS = [4, 5, 6, 46, 54, 147]
    METRIC_IDS = [
        23, 109, 110, 111, 112, 113, 114, 137, 138, 305,
        317, 461, 462, 463, 464, 465, 466, 467, 468,
        469, 470, 471, 472, 473, 474, 475, 476, 44,
    ]


class CostFeeReport(ReportGroupConfig):
    REPORT_NAME = "cost_fee"
    DIMENSION_IDS = [4, 5, 6, 149]
    METRIC_IDS = [
        44, 49, 50, 51, 129, 180, 184, 188, 192, 196,
        200, 204, 208, 212, 216, 260, 322, 426, 560,
        564, 570, 714, 715, 716, 717, 724,
    ]


class GeoDeviceReport(ReportGroupConfig):
    REPORT_NAME = "geo_device"
    DIMENSION_IDS = [4, 5, 19, 20, 34, 23, 122, 126]
    METRIC_IDS = [1, 2, 11, 23, 44, 45, 41, 43]


class SupplyInventoryReport(ReportGroupConfig):
    REPORT_NAME = "supply_inventory"
    DIMENSION_IDS = [4, 5, 6, 1228, 1033, 154, 106, 55, 166, 171]
    METRIC_IDS = [1, 2, 11, 44, 45, 23, 41, 43]


class AudienceReachReport(ReportGroupConfig):
    REPORT_NAME = "audience_reach"
    DIMENSION_IDS = [4, 5, 6, 166, 1225]
    METRIC_IDS = [1, 2, 44, 329, 430, 446, 447, 448, 523]


class BiddingReport(ReportGroupConfig):
    REPORT_NAME = "bidding"
    DIMENSION_IDS = [4, 5, 6, 149]
    METRIC_IDS = [1, 173, 583, 584]
    DATA_SOURCE = 15

    @classmethod
    def build_schema(cls) -> list[dict]:
        """Bidding reports have a fixed, minimal schema."""
        return [
            {"name": "Day",                     "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Advertiser Id",            "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Advertiser",               "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Campaign Id",              "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Campaign Name",            "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Line Id",                  "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Line",                     "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Package Id",               "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Package",                  "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Impressions",              "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Qualified Count",           "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Available Bid Requests",   "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Average Win Rate",         "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "ingestion_timestamp",      "type": "STRING",  "mode": "NULLABLE"},
            {"name": "lookback_date_range",      "type": "STRING",  "mode": "NULLABLE"},
        ]


# All available report groups
ALL_REPORT_GROUPS: list[type[ReportGroupConfig]] = [
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
# YAHOO REPORTING CLASS
# ═══════════════════════════════════════════════════════════════


class YahooReporting:
    """
    End-to-end Yahoo DSP Reporting API handler.

    Pipeline steps (per report group):
      1. Build schema from group config
      2. Build & validate payload
      3. POST  /extreport/           → create report
      4. GET   /extreport/{id}       → poll until SUCCESS / FAILED
      5. Download CSV from temp URL
      6. Parse CSV → Polars DataFrame (all Utf8)
      7. Add metadata columns
      8. Cast columns per BQ schema; drop extras
      9. Save CSV locally with timestamped filename
    """

    REPORT_ENDPOINT = (
        "https://api-sched-v3.admanagerplus.yahoo.com"
        "/yamplus_api/extreport/"
    )
    COB_ENDPOINT = (
        "https://api-sched-v3.admanagerplus.yahoo.com"
        "/yamplus_api/extreport/cob"
    )

    def __init__(self):
        try:
            self.helper = YahooHelpers()
            self.poll_interval = 15          # seconds — initial wait
            self.poll_max_attempts = 200     # ~20 min ceiling
            self.poll_backoff = 1.2          # multiplicative backoff
            self.poll_max_wait = 120         # cap per-cycle wait (s)
            logger.success("YahooReporting initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── auth helper ──────────────────────────────

    def _auth_headers(self) -> dict:
        """Obtain (or reuse cached) auth headers."""
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)
    
    
    def list_all_advertisers(self) -> list[dict]:
        """
        GET /dsp/api/v1/traffic/advertisers
        Returns all advertiser accounts under your seat.
        """
        url = "https://api-sched-v3.admanagerplus.yahoo.com/traffic/api/v1/advertisers"
        headers = self._auth_headers()

        resp = requests.get(url, headers=headers, timeout=60)

        if resp.status_code != 200:
            logger.error(f"List advertisers failed [{resp.status_code}]: {resp.text}")
            resp.raise_for_status()

        body = resp.json()
        advertisers = body.get("response", [])

        logger.info(f"Found {len(advertisers)} advertiser(s) under seat")
        for adv in advertisers:
            logger.info(
                f"  ID: {adv.get('id'):<12} | "
                f"Name: {adv.get('name', 'N/A'):<40} | "
                f"Status: {adv.get('status', 'N/A')}"
            )
        print(advertisers)
        return advertisers

    # ─────────────── 1. CREATE REPORT ─────────────────────────

    def create_report(self, payload: dict) -> str:
        """
        POST /extreport/ — submit report creation request.
        Returns the customerReportId (UUID) for polling.
        """
        try:
            headers = self._auth_headers()
            dims = payload.get("reportOption", {}).get("dimensionTypeIds", [])
            mets = payload.get("reportOption", {}).get("metricTypeIds", [])
            logger.info(f"Creating report — dims={len(dims)}, metrics={len(mets)}")
            logger.debug(f"Payload:\n{json.dumps(payload, indent=2)}")

            resp = requests.post(
                self.REPORT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=60,
            )

            if resp.status_code not in (200, 201, 202):
                logger.error(f"Create report HTTP {resp.status_code}: {resp.text}")
                resp.raise_for_status()

            body = resp.json()
            report_id = body.get("customerReportId")
            if not report_id:
                raise Exception(f"No customerReportId in response: {body}")

            logger.success(
                f"Report created | ID: {report_id} | Status: {body.get('status')}"
            )
            return report_id

        except Exception as e:
            logger.error(f"Error creating report: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 2. POLL STATUS ───────────────────────────

    def poll_report_status(self, report_id: str) -> dict:
        """
        GET /extreport/{id} — poll with exponential backoff.
        Returns full status dict on SUCCESS (contains download URL).
        """
        try:
            logger.info(f"Polling report {report_id} ...")
            wait = self.poll_interval

            with tqdm(
                total=self.poll_max_attempts,
                desc="⏳ Polling",
                unit="req",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for attempt in range(1, self.poll_max_attempts + 1):
                    headers = self._auth_headers()

                    resp = requests.get(
                        f"{self.REPORT_ENDPOINT}{report_id}",
                        headers=headers,
                        timeout=60,
                    )

                    if resp.status_code != 200:
                        logger.warning(
                            f"Attempt {attempt}: HTTP {resp.status_code} — retrying"
                        )
                        pbar.update(1)
                        time.sleep(wait)
                        wait = min(wait * self.poll_backoff, self.poll_max_wait)
                        continue

                    body = resp.json()
                    status = body.get("status", "").upper()
                    pbar.set_postfix_str(f"Status={status}")
                    pbar.update(1)

                    if status == "SUCCESS":
                        logger.success(
                            f"Report ready | Rows: {body.get('numRows', '?')}"
                        )
                        return body

                    if status == "FAILED":
                        raise Exception(f"Report generation FAILED: {body}")

                    logger.info(
                        f"Attempt {attempt}/{self.poll_max_attempts} "
                        f"| Status: {status} | Retrying in {wait:.0f}s"
                    )
                    time.sleep(wait)
                    wait = min(wait * self.poll_backoff, self.poll_max_wait)

            raise TimeoutError(
                f"Report {report_id} not ready after {self.poll_max_attempts} attempts."
            )
        except Exception as e:
            logger.error(f"Polling error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 3. DOWNLOAD CSV ──────────────────────────

    def download_report_csv(self, download_url: str) -> str:
        """Download CSV from Yahoo's temporary URL (valid 24 h)."""
        try:
            logger.info("Downloading report CSV ...")
            resp = requests.get(download_url, timeout=300)
            if resp.status_code != 200:
                raise Exception(
                    f"Download failed [{resp.status_code}]: {resp.text[:500]}"
                )
            logger.success(f"Downloaded | {len(resp.text):,} chars")
            return resp.text
        except Exception as e:
            logger.error(f"Download error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 4. CLOSE OF BOOKS ────────────────────────

    def check_cob(self, advertiser_id: int, day: str) -> bool:
        """
        GET /extreport/cob — check if data is finalised for
        a given advertiser + day (YYYYMMDD).
        Returns True if dayClosed == 'true'.
        """
        try:
            logger.info(f"COB check → advertiser={advertiser_id}, day={day}")
            headers = self._auth_headers()

            resp = requests.get(
                self.COB_ENDPOINT,
                params={"advertiserId": advertiser_id, "day": day},
                headers=headers,
                timeout=60,
            )
            if resp.status_code != 200:
                raise Exception(f"COB failed [{resp.status_code}]: {resp.text}")

            body = resp.json()
            is_closed = str(body.get("dayClosed", "false")).lower() == "true"
            logger.success(f"COB result: {body} → closed={is_closed}")
            return is_closed
        except Exception as e:
            logger.error(f"COB error: {e}\n{traceback.format_exc()}")
            raise

    def wait_for_cob(
        self,
        advertiser_ids: list[int],
        target_day: str,
        max_retries: int = 12,
        retry_delay_secs: int = 300,
    ) -> bool:
        """
        Wait for COB to be confirmed for ALL advertisers on target_day.
        Default: 12 retries × 5 min = 60 min max wait.
        Returns True if all closed, False on timeout.
        """
        try:
            logger.info(
                f"Waiting for COB on {target_day} for "
                f"{len(advertiser_ids)} advertiser(s)..."
            )
            with tqdm(
                total=max_retries,
                desc="📚 COB check",
                unit="attempt",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for attempt in range(1, max_retries + 1):
                    all_closed = True
                    for adv_id in advertiser_ids:
                        if not self.check_cob(adv_id, target_day):
                            logger.warning(
                                f"COB NOT closed → advertiser={adv_id}, day={target_day}"
                            )
                            all_closed = False
                            break

                    pbar.set_postfix_str(
                        "ALL CLOSED ✅" if all_closed else "PENDING ⏳"
                    )
                    pbar.update(1)

                    if all_closed:
                        logger.success(
                            f"✅ COB closed for all advertisers on {target_day}"
                        )
                        return True

                    if attempt < max_retries:
                        logger.info(
                            f"COB retry {attempt}/{max_retries} — "
                            f"waiting {retry_delay_secs}s..."
                        )
                        time.sleep(retry_delay_secs)

            logger.error(
                f"❌ COB not closed after {max_retries} retries for {target_day}"
            )
            return False
        except Exception as e:
            logger.error(f"COB wait error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 5. CAST EXPRESSION ───────────────────────

    @staticmethod
    def _cast_expression(col_name: str, bq_type: str) -> pl.Expr:
        """Return a Polars expression to cast Utf8 → target BQ type."""
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

        logger.warning(f"Unknown BQ type '{bq_type}' for '{col_name}'; keeping Utf8.")
        return pl.col(col_name).cast(pl.Utf8).alias(col_name)

    # ─────────────── 6. TRANSFORM TO POLARS ───────────────────

    def transform_to_polars(
        self,
        csv_text: str,
        schema: list[dict],
        lookback_range: str,
    ) -> pl.DataFrame:
        """
        Parse CSV text → Polars DataFrame, add metadata, cast types.
        """
        try:
            df = pl.read_csv(
                io.StringIO(csv_text),
                infer_schema_length=0,
                null_values=["", "null", "NULL", "None", "N/A"],
            )
            logger.info(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
            logger.info(f"Columns from Yahoo: {df.columns}")

            # Add metadata
            ingestion_ts = pendulum.now("UTC").to_iso8601_string()
            df = df.with_columns(
                pl.lit(ingestion_ts).alias("ingestion_timestamp"),
                pl.lit(lookback_range).alias("lookback_date_range"),
            )

            # Build type map & reconcile columns
            type_map = {f["name"]: f["type"] for f in schema}
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

    # ─────────────── 7. RUN SINGLE GROUP ──────────────────────

    def run_report_group(
        self,
        report_group: type[ReportGroupConfig],
        config_overrides: dict,
        output_dir: str = "output",
    ) -> Optional[pl.DataFrame]:
        """
        Execute the full pipeline for a single report group.
        Returns the final DataFrame, or None on error.
        """
        t0 = time.time()
        name = report_group.REPORT_NAME.upper()
        try:
            logger.info("=" * 60)
            logger.info(f"▶ [{name}] Starting report group")
            logger.info("=" * 60)

            # 1. Schema
            schema = report_group.build_schema()
            logger.info(f"Schema: {len(schema)} columns")

            # 2. Payload
            payload = report_group.build_payload(config_overrides)
            ds = payload.get("dataSource")
            dims = payload["reportOption"]["dimensionTypeIds"]
            mets = payload["reportOption"]["metricTypeIds"]
            logger.info(
                f"Payload built — dims={len(dims)}, metrics={len(mets)}, "
                f"dataSource={ds}"
            )

            # 3. Create report
            report_id = self.create_report(payload)

            # 4. Poll
            status_resp = self.poll_report_status(report_id)
            url = status_resp.get("url")
            if not url:
                raise Exception("Report succeeded but no download URL returned.")

            # 5. Download
            csv_text = self.download_report_csv(url)

            # 6. Transform
            lookback = report_group.date_range_label(payload)
            df = self.transform_to_polars(csv_text, schema, lookback)

            if df.is_empty():
                logger.warning(f"[{name}] 0 rows returned — nothing to save.")
                return df

            # 7. Save CSV
            os.makedirs(output_dir, exist_ok=True)
            ts = pendulum.now("UTC").format("YYYY-MM-DD_HH-mm-ss")
            filename = f"{report_group.REPORT_NAME}_{ts}.csv"
            filepath = os.path.join(output_dir, filename)
            df.write_csv(filepath)
            abs_path = os.path.abspath(filepath)

            elapsed = time.time() - t0
            logger.success(
                f"✅ [{name}] DONE | {df.shape[0]} rows × {df.shape[1]} cols "
                f"| {elapsed:.1f}s → {abs_path}"
            )
            return df

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"❌ [{name}] FAILED after {elapsed:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            return None

    # ─────────────── 8. RUN ALL GROUPS ────────────────────────

    def run_all_groups(
        self,
        report_groups: list[type[ReportGroupConfig]],
        config_overrides: dict,
        output_dir: str = "output",
        skip_cob: bool = False,
        advertiser_ids_for_cob: list[int] | None = None,
    ) -> dict[str, Optional[pl.DataFrame]]:
        """
        Run every report group in sequence.
        Optionally checks COB once before starting.
        Returns {report_name: DataFrame | None}.
        """
        results: dict[str, Optional[pl.DataFrame]] = {}

        # ── COB check ──
        adv_ids = advertiser_ids_for_cob or []
        if not skip_cob and adv_ids:
            tz = config_overrides.get("timezone", "America/New_York")
            yesterday = pendulum.now(tz).subtract(days=1).format("YYYYMMDD")
            logger.info(
                f"📚 COB check for {yesterday} "
                f"({len(adv_ids)} advertiser(s))..."
            )
            cob_ok = self.wait_for_cob(adv_ids, yesterday)
            if not cob_ok:
                logger.error(
                    "❌ Books not closed. Use skip_cob=True to force."
                )
                return results
        elif not adv_ids and not skip_cob:
            logger.warning("⚠️  No advertiser IDs — skipping COB check.")

        # ── Run groups ──
        group_names = [g.REPORT_NAME for g in report_groups]
        logger.info(f"🚀 Running {len(report_groups)} report group(s) ...")

        with tqdm(
            total=len(report_groups),
            desc="📊 Report Groups",
            unit="group",
        ) as pbar:
            for grp in report_groups:
                df = self.run_report_group(grp, config_overrides, output_dir)
                results[grp.REPORT_NAME] = df
                pbar.update(1)

        # ── Summary ──
        ok = sum(1 for v in results.values() if v is not None and not v.is_empty())
        fail = len(results) - ok
        logger.info("=" * 60)
        logger.info(f"📋 SUMMARY: {ok} succeeded, {fail} failed/empty out of {len(results)}")
        for name, df in results.items():
            rows = df.shape[0] if df is not None else "ERROR"
            logger.info(f"   {name:30s} → {rows} rows")
        logger.info("=" * 60)

        return results

    # ═══════════════════════════════════════════════════════
    # BACKFILL ENGINE — Rate-Limited Sliding-Window
    # ═══════════════════════════════════════════════════════

    def _fetch_window_for_group(self, group, overrides):
        """Fetch one date-window for one report group. Thread-safe."""
        name = group.REPORT_NAME
        try:
            schema = group.build_schema()
            payload = group.build_payload(overrides)
            report_id = self.create_report(payload)
            status_resp = self.poll_report_status(report_id)
            url = status_resp.get("url")
            if not url:
                logger.warning(f"[{name}] No download URL")
                return None
            csv_text = self.download_report_csv(url)
            lookback = group.date_range_label(payload)
            df = self.transform_to_polars(csv_text, schema, lookback)
            return df if not df.is_empty() else None
        except Exception as e:
            logger.error(f"[{name}] Window fetch failed: {e}")
            return None

    def backfill_all_groups(self, report_groups, tz="America/New_York", currency=4,
                            account_ids=None, months_back=13, window_days=2,
                            max_workers=4, worker_sleep=3, output_dir="output_backfill"):
        """Backfill all groups via sliding windows. Rate-limit aware (100 POST/hr)."""
        t0 = time.time()
        tz_obj = pendulum.timezone(tz)
        end_date = pendulum.yesterday(tz_obj)
        start_date = end_date.subtract(months=months_back)

        # Generate sliding windows (newest first)
        windows = []
        cursor = end_date
        while cursor >= start_date:
            w_start = cursor.subtract(days=window_days - 1)
            if w_start < start_date:
                w_start = start_date
            windows.append((w_start, cursor))
            cursor = w_start.subtract(days=1)

        total_windows = len(windows)
        num_groups = len(report_groups)
        total_posts = total_windows * num_groups
        max_win_per_hour = max(1, 100 // num_groups)
        window_interval = 3600.0 / max_win_per_hour
        est_hours = total_windows * window_interval / 3600

        logger.info("=" * 60)
        logger.info("BACKFILL PLAN")
        logger.info(f"  Range      : {start_date.to_date_string()} -> {end_date.to_date_string()}")
        logger.info(f"  Window     : {window_days} day(s)")
        logger.info(f"  Windows    : {total_windows}")
        logger.info(f"  Groups     : {num_groups}")
        logger.info(f"  Total POST : {total_posts}")
        logger.info(f"  Workers    : {max_workers}")
        logger.info(f"  Rate       : {max_win_per_hour} win/hr ({window_interval:.0f}s gap)")
        logger.info(f"  Est. time  : {est_hours:.1f} hours")
        logger.info("=" * 60)

        temp_base = os.path.join(output_dir, "_temp")
        for grp in report_groups:
            os.makedirs(os.path.join(temp_base, grp.REPORT_NAME), exist_ok=True)

        fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
        post_count = 0

        with tqdm(total=total_windows, desc="Backfill", unit="win") as pbar:
            for idx, (w_start, w_end) in enumerate(windows):
                t_win = time.time()
                w_label = f"{w_start.to_date_string()} -> {w_end.to_date_string()}"
                pbar.set_postfix_str(w_label)

                overrides = {
                    "timezone": tz,
                    "currency": currency,
                    "accountIds": account_ids or [],
                    "dateTypeId": 11,
                    "intervalTypeId": 2,
                    "startDate": pendulum.instance(w_start, tz=tz_obj).start_of("day").format(fmt),
                    "endDate": pendulum.instance(w_end, tz=tz_obj).end_of("day").format(fmt),
                }

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_map = {
                        pool.submit(self._fetch_window_for_group, grp, overrides): grp
                        for grp in report_groups
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        grp = future_map[future]
                        try:
                            df = future.result()
                            if df is not None:
                                temp_path = os.path.join(temp_base, grp.REPORT_NAME, f"w_{idx:04d}.csv")
                                df.write_csv(temp_path)
                                del df
                        except Exception as e:
                            logger.error(f"[{grp.REPORT_NAME}] {w_label}: {e}")
                        time.sleep(worker_sleep)

                post_count += num_groups
                pbar.update(1)

                elapsed_win = time.time() - t_win
                sleep_needed = max(0, window_interval - elapsed_win)
                if sleep_needed > 0 and idx < total_windows - 1:
                    logger.debug(f"Rate limit sleep: {sleep_needed:.0f}s (POST total: {post_count})")
                    time.sleep(sleep_needed)

        # Combine temp CSVs per group (lazy scan = memory-safe)
        logger.info("Combining temp files ...")
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        ts_str = pendulum.now("UTC").format("YYYY-MM-DD_HH-mm-ss")

        for grp in report_groups:
            name = grp.REPORT_NAME
            csv_files = sorted(Path(os.path.join(temp_base, name)).glob("*.csv"))
            if not csv_files:
                logger.warning(f"[{name}] No data")
                results[name] = None
                continue

            final_path = os.path.join(output_dir, f"{name}_backfill_{ts_str}.csv")
            if len(csv_files) == 1:
                shutil.move(str(csv_files[0]), final_path)
            else:
                lazy = [pl.scan_csv(str(f), infer_schema_length=0) for f in csv_files]
                pl.concat(lazy, how="diagonal_relaxed").sink_csv(final_path)

            size_mb = os.path.getsize(final_path) / (1024 * 1024)
            logger.success(f"[{name}] -> {final_path} ({size_mb:.1f} MB)")
            results[name] = final_path

        # Cleanup temp
        try:
            shutil.rmtree(temp_base)
        except Exception:
            pass
        gc.collect()

        elapsed_total = time.time() - t0
        ok = sum(1 for v in results.values() if v)
        logger.info("=" * 60)
        logger.info(f"BACKFILL DONE | {ok}/{num_groups} groups | {elapsed_total/3600:.1f}h | {post_count} POSTs")
        for name, path in results.items():
            st = "OK" if path else "FAIL"
            logger.info(f"  {st} {name}: {path or 'NO DATA'}")
        logger.info("=" * 60)
        return results



if __name__ == "__main__":
    cls_obj = YahooReporting()
    cls_obj.list_all_advertisers()
    