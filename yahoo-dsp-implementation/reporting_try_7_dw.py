"""
reporting.py
────────────
Yahoo DSP Reporting API module — CORRECTED FINAL VERSION.

Fixes vs previous iteration
────────────────────────────
1.  Schema is built DYNAMICALLY from DIMENSION_CSV_COLUMNS + METRIC_NAMES
    dicts so column names always match what Yahoo actually returns.
2.  BiddingReport uses ONLY metric IDs [1, 173, 583, 584] (no 254/302/382).
3.  Daily output → output/daily ; backfill → output/backfill.
4.  Endpoint uses HTTPS.
"""

import io
import os
import gc
import json
import time
import shutil
import traceback
import concurrent.futures
from pathlib import Path
from typing import Optional

import pendulum
import polars as pl
import requests
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from helpers import YahooHelpers


# ═══════════════════════════════════════════════════════════════
# DIMENSION → CSV COLUMN MAPPING  (Yahoo actual header names)
# ═══════════════════════════════════════════════════════════════

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
# METRIC ID → CSV HEADER NAME
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
    173: "Qualified Bid Requests",
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


# Metrics that are INTEGER (rest default to FLOAT64)
INTEGER_METRIC_IDS: set[int] = {
    1, 2, 23, 25, 26, 27, 28, 29,
    100, 101, 103, 104, 109, 110,
    171, 173, 254, 305, 317, 328,
    329, 386, 387, 430, 446,
    461, 462, 583,
}


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS — Payload Validation
# ═══════════════════════════════════════════════════════════════

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
    limit: int = 10_000
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
            raise ValueError("startDate and endDate required for dateTypeId=11")
        return self


# ═══════════════════════════════════════════════════════════════
# REPORT GROUP CONFIG — Base + Subclasses
# ═══════════════════════════════════════════════════════════════

DATE_TYPE_LABELS: dict[int, str] = {
    1: "Today", 2: "Yesterday", 3: "This_Week",
    4: "Last_7_Days", 5: "Last_Week", 6: "Last_14_Days",
    7: "MTD", 8: "Last_30_Days", 9: "Last_Month", 10: "All_Time",
}


class ReportGroupConfig:
    """Base report group — subclasses only set name, dimension_ids, metric_ids."""

    name: str = "base"
    dimension_ids: list[int] = []
    metric_ids: list[int] = []
    data_source: Optional[int] = None

    # ── Dynamic schema builder ──
    def build_schema(self) -> list[dict]:
        """Auto-generate BQ schema from DIMENSION_CSV_COLUMNS + METRIC_NAMES."""
        schema: list[dict] = []
        # Day column (populated when intervalTypeId=2)
        schema.append({"name": "Day", "type": "STRING", "mode": "NULLABLE"})
        # Dimension columns
        for dim_id in self.dimension_ids:
            cols = DIMENSION_CSV_COLUMNS.get(dim_id, [])
            for col_name, col_type in cols:
                schema.append({"name": col_name, "type": col_type, "mode": "NULLABLE"})
        # Metric columns
        for met_id in self.metric_ids:
            met_name = METRIC_NAMES.get(met_id, f"Metric_{met_id}")
            met_type = "INTEGER" if met_id in INTEGER_METRIC_IDS else "FLOAT64"
            schema.append({"name": met_name, "type": met_type, "mode": "NULLABLE"})
        # Metadata
        schema.append({"name": "ingestion_timestamp", "type": "STRING", "mode": "NULLABLE"})
        schema.append({"name": "lookback_date_range", "type": "STRING", "mode": "NULLABLE"})
        return schema

    # ── Payload builder ──
    def build_payload(self, overrides: dict | None = None) -> dict:
        """Build validated API payload dict."""
        ov = overrides or {}
        config = {
            "reportOption": {
                "timezone":         ov.get("timezone", "America/New_York"),
                "currency":         ov.get("currency", 4),
                "dimensionTypeIds": self.dimension_ids,
                "metricTypeIds":    self.metric_ids,
                "filterOptions":    ov.get("filterOptions", []),
                "accountIds":       ov.get("accountIds", []),
            },
            "dateTypeId":     ov.get("dateTypeId", 2),
            "intervalTypeId": ov.get("intervalTypeId", 2),
        }
        if self.data_source is not None:
            config["dataSource"] = self.data_source
        if ov.get("startDate"):
            config["startDate"] = ov["startDate"]
        if ov.get("endDate"):
            config["endDate"] = ov["endDate"]
        # Validate through Pydantic
        validated = ReportPayload(**config)
        return validated.model_dump(exclude_none=True)

    # ── Date range label ──
    def date_range_label(self, overrides: dict | None = None) -> str:
        ov = overrides or {}
        dt = ov.get("dateTypeId", 2)
        if dt == 10:
            return "ALL_TIME"
        if dt == 11:
            s = ov.get("startDate", "?")[:10]
            e = ov.get("endDate", "?")[:10]
            return f"{s} to {e}"
        return DATE_TYPE_LABELS.get(dt, f"dateTypeId_{dt}")


# ── Concrete Groups ──────────────────────────────────────────

class CampaignDeliveryReport(ReportGroupConfig):
    name = "campaign_delivery"
    dimension_ids = [4, 5, 6, 149, 56, 64, 19, 63, 1225, 1267]
    metric_ids = [1, 2, 11, 23, 41, 42, 43, 44, 45, 137, 138, 171, 305, 317, 328]

class CreativePerformanceReport(ReportGroupConfig):
    name = "creative_performance"
    dimension_ids = [4, 5, 6, 7, 8, 3, 174]
    metric_ids = [1, 2, 11, 23, 44, 45, 41, 43]

class VideoPerformanceReport(ReportGroupConfig):
    name = "video_performance"
    dimension_ids = [4, 5, 6, 3, 169, 167, 182, 1037, 1240]
    metric_ids = [1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 62, 63, 64, 65, 168, 169, 365, 386, 387, 44]

class ViewabilityReport(ReportGroupConfig):
    name = "viewability"
    dimension_ids = [4, 5, 6, 34, 106, 19]
    metric_ids = [1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 721, 723, 44]

class ConversionDetailReport(ReportGroupConfig):
    name = "conversion_detail"
    dimension_ids = [4, 5, 6, 46, 54, 147]
    metric_ids = [23, 109, 110, 111, 112, 113, 114, 137, 138, 305, 317,
                  461, 462, 463, 464, 465, 466, 467, 468,
                  469, 470, 471, 472, 473, 474, 475, 476, 44]

class CostFeeReport(ReportGroupConfig):
    name = "cost_fee"
    dimension_ids = [4, 5, 6, 149]
    metric_ids = [44, 49, 50, 51, 129, 180, 184, 188, 192, 196, 200, 204, 208,
                  212, 216, 260, 322, 426, 560, 564, 570, 714, 715, 716, 717, 724]

class GeoDeviceReport(ReportGroupConfig):
    name = "geo_device"
    dimension_ids = [4, 5, 19, 20, 34, 23, 122, 126]
    metric_ids = [1, 2, 11, 23, 44, 45, 41, 43]

class SupplyInventoryReport(ReportGroupConfig):
    name = "supply_inventory"
    dimension_ids = [4, 5, 6, 1228, 1033, 154, 106, 55, 166, 171]
    metric_ids = [1, 2, 11, 44, 45, 23, 41, 43]

class AudienceReachReport(ReportGroupConfig):
    name = "audience_reach"
    dimension_ids = [4, 5, 6, 166, 1225]
    metric_ids = [1, 2, 44, 329, 430, 446, 447, 448, 523]

class BiddingReport(ReportGroupConfig):
    """
    Bidding report — requires dataSource=15 and currency=4 (USD).
    Only dimensions 4,5,6,149 and metrics 1,173,583,584 are allowed.
    """
    name = "bidding"
    dimension_ids = [4, 5, 6, 149]
    metric_ids = [1, 173, 583, 584]       # ← NO 254, NO 302, NO 382
    data_source = 15

    def build_schema(self) -> list[dict]:
        """Fixed schema for bidding report."""
        return [
            {"name": "Day",                    "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Advertiser Id",          "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Advertiser",             "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Campaign Id",            "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Campaign Name",          "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Line Id",                "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Line",                   "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Package Id",             "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Package",                "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Impressions",            "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Qualified Bid Requests", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Available Bid Requests", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Average Win Rate",       "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "ingestion_timestamp",    "type": "STRING",  "mode": "NULLABLE"},
            {"name": "lookback_date_range",    "type": "STRING",  "mode": "NULLABLE"},
        ]

    def build_payload(self, overrides: dict | None = None) -> dict:
        """Force currency=4 for bidding reports."""
        ov = dict(overrides or {})
        ov["currency"] = 4
        return super().build_payload(ov)


ALL_REPORT_GROUPS: list[ReportGroupConfig] = [
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


# ═══════════════════════════════════════════════════════════════
# YAHOO REPORTING CLASS
# ═══════════════════════════════════════════════════════════════

class YahooReporting:
    """End-to-end Yahoo DSP Reporting handler."""

    REPORT_ENDPOINT = (
        "https://api-sched-v3.admanagerplus.yahoo.com"
        "/yamplus_api/extreport/"
    )
    COB_ENDPOINT = (
        "https://api-sched-v3.admanagerplus.yahoo.com"
        "/yamplus_api/extreport/cob"
    )

    def __init__(self):
        self.helper = YahooHelpers()
        self.poll_interval    = 15
        self.poll_max_attempts = 200
        self.poll_backoff     = 1.2
        self.poll_max_wait    = 120
        logger.success("YahooReporting initialised.")

    # ─── Auth headers ─────────────────────────────────────
    def _auth_headers(self) -> dict:
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)

    # ─── 1. CREATE REPORT ─────────────────────────────────
    def create_report(self, payload: dict) -> str:
        headers = self._auth_headers()
        dims = payload.get("reportOption", {}).get("dimensionTypeIds", [])
        mets = payload.get("reportOption", {}).get("metricTypeIds", [])
        logger.info(f"Creating report — dims={len(dims)}, metrics={len(mets)}")
        logger.debug(f"Payload:\n{json.dumps(payload, indent=2)}")

        resp = requests.post(
            self.REPORT_ENDPOINT, headers=headers,
            json=payload, timeout=60,
        )
        if resp.status_code not in (200, 201, 202):
            logger.error(f"Create report failed [{resp.status_code}]: {resp.text}")
            resp.raise_for_status()

        body = resp.json()
        report_id = body.get("customerReportId")
        if not report_id:
            raise Exception(f"No customerReportId: {body}")
        logger.success(f"Report created | ID: {report_id} | Status: {body.get('status')}")
        return report_id

    # ─── 2. POLL STATUS ───────────────────────────────────
    def poll_report_status(self, report_id: str) -> dict:
        wait = self.poll_interval
        with tqdm(total=self.poll_max_attempts, desc="⏳ Polling",
                  unit="req", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            for attempt in range(1, self.poll_max_attempts + 1):
                headers = self._auth_headers()
                resp = requests.get(
                    f"{self.REPORT_ENDPOINT}{report_id}",
                    headers=headers, timeout=60,
                )
                if resp.status_code != 200:
                    logger.warning(f"Attempt {attempt}: HTTP {resp.status_code}")
                    pbar.update(1)
                    time.sleep(wait)
                    wait = min(wait * self.poll_backoff, self.poll_max_wait)
                    continue

                body = resp.json()
                status = body.get("status", "").upper()
                pbar.set_postfix_str(f"Status={status}")
                pbar.update(1)

                if status == "SUCCESS":
                    logger.success(f"Report ready | Rows: {body.get('numRows', '?')}")
                    return body
                if status == "FAILED":
                    raise Exception(f"Report FAILED: {body}")

                time.sleep(wait)
                wait = min(wait * self.poll_backoff, self.poll_max_wait)

        raise TimeoutError(f"Report {report_id} not ready after {self.poll_max_attempts} attempts.")

    # ─── 3. DOWNLOAD CSV ──────────────────────────────────
    def download_report_csv(self, url: str) -> str:
        logger.info("Downloading report CSV ...")
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        logger.success(f"Downloaded | {len(resp.text):,} chars")
        return resp.text

    # ─── 4. COB CHECK ─────────────────────────────────────
    def check_cob(self, advertiser_id: int, day: str) -> dict:
        headers = self._auth_headers()
        resp = requests.get(
            self.COB_ENDPOINT,
            params={"advertiserId": advertiser_id, "day": day},
            headers=headers, timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        logger.info(f"COB({advertiser_id}, {day}): {body}")
        return body

    def wait_for_cob(self, advertiser_ids: list[int], target_day: str,
                     max_retries: int = 12, retry_delay: int = 300) -> bool:
        with tqdm(total=max_retries, desc="📚 COB", unit="attempt") as pbar:
            for attempt in range(1, max_retries + 1):
                all_closed = True
                for aid in advertiser_ids:
                    result = self.check_cob(aid, target_day)
                    if str(result.get("dayClosed", "false")).lower() != "true":
                        all_closed = False
                        break
                pbar.set_postfix_str("ALL CLOSED ✅" if all_closed else "PENDING ⏳")
                pbar.update(1)
                if all_closed:
                    logger.success(f"COB closed for all on {target_day}")
                    return True
                if attempt < max_retries:
                    time.sleep(retry_delay)
        logger.error(f"COB not closed after {max_retries} retries")
        return False

    # ─── 5. CAST EXPRESSION ───────────────────────────────
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
        return pl.col(col_name).cast(pl.Utf8).alias(col_name)

    # ─── 6. TRANSFORM CSV → POLARS ───────────────────────
    def transform_to_polars(
        self,
        csv_text: str,
        schema_fields: list[dict],
        overrides: dict | None = None,
    ) -> pl.DataFrame:
        """Parse CSV, add metadata columns, cast types, align to schema."""
        df = pl.read_csv(
            io.StringIO(csv_text),
            infer_schema_length=0,
            null_values=["", "null", "NULL", "None", "N/A"],
        )
        logger.info(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
        logger.info(f"Columns from Yahoo: {df.columns}")

        # Add metadata
        ts = pendulum.now("UTC").to_iso8601_string()
        ov = overrides or {}
        dt = ov.get("dateTypeId", 2)
        if dt == 10:
            dr = "ALL_TIME"
        elif dt == 11:
            dr = f"{ov.get('startDate', '?')[:10]} to {ov.get('endDate', '?')[:10]}"
        else:
            dr = DATE_TYPE_LABELS.get(dt, f"dateTypeId_{dt}")

        df = df.with_columns(
            pl.lit(ts).alias("ingestion_timestamp"),
            pl.lit(dr).alias("lookback_date_range"),
        )

        # Align to schema
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

    # ─── 7. RUN SINGLE REPORT GROUP (DAILY) ──────────────
    def run_report_group(
        self,
        group: ReportGroupConfig,
        output_dir: str,
        overrides: dict | None = None,
    ) -> Optional[pl.DataFrame]:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info(f"▶ [{group.name.upper()}] Starting report group")
        logger.info("=" * 60)
        try:
            schema = group.build_schema()
            logger.info(f"Schema: {len(schema)} columns")
            payload = group.build_payload(overrides)
            dims = payload.get("reportOption", {}).get("dimensionTypeIds", [])
            mets = payload.get("reportOption", {}).get("metricTypeIds", [])
            ds = payload.get("dataSource")
            logger.info(f"Payload built — dims={len(dims)}, metrics={len(mets)}, dataSource={ds}")

            report_id = self.create_report(payload)
            status = self.poll_report_status(report_id)
            url = status.get("url")
            if not url:
                raise Exception("No download URL")
            csv_text = self.download_report_csv(url)
            df = self.transform_to_polars(csv_text, schema, overrides)

            if df.is_empty():
                logger.warning(f"[{group.name}] 0 rows returned")
                return df

            os.makedirs(output_dir, exist_ok=True)
            ts = pendulum.now("UTC").format("YYYY-MM-DD_HH-mm-ss")
            fname = f"{group.name}_{ts}.csv"
            fpath = os.path.join(output_dir, fname)
            df.write_csv(fpath)
            elapsed = time.time() - t0
            logger.success(
                f"✅ [{group.name.upper()}] DONE | "
                f"{df.shape[0]} rows × {df.shape[1]} cols | "
                f"{elapsed:.1f}s → {os.path.abspath(fpath)}"
            )
            return df
        except Exception as e:
            logger.error(f"❌ [{group.name}] FAILED: {e}\n{traceback.format_exc()}")
            return None

    # ─── 8. RUN ALL GROUPS (DAILY) ────────────────────────
    def run_all_groups(
        self,
        groups: list[ReportGroupConfig],
        output_dir: str,
        overrides: dict | None = None,
        account_ids: list[int] | None = None,
        skip_cob: bool = False,
    ) -> dict[str, Optional[pl.DataFrame]]:
        # COB check
        if account_ids and not skip_cob:
            tz = (overrides or {}).get("timezone", "America/New_York")
            yesterday = pendulum.now(tz).subtract(days=1).format("YYYYMMDD")
            if not self.wait_for_cob(account_ids, yesterday):
                logger.error("COB not closed — aborting")
                return {g.name: None for g in groups}
        elif not account_ids and not skip_cob:
            logger.warning("⚠️  No advertiser IDs — skipping COB check.")

        results: dict[str, Optional[pl.DataFrame]] = {}
        logger.info(f"🚀 Running {len(groups)} report group(s) ...")

        with tqdm(total=len(groups), desc="📊 Report Groups", unit="group") as pbar:
            for group in groups:
                df = self.run_report_group(group, output_dir, overrides)
                results[group.name] = df
                pbar.update(1)
        return results

    # ─── 9. BACKFILL: FETCH SINGLE DAY ───────────────────
    def fetch_single_day(
        self,
        group: ReportGroupConfig,
        day_str: str,
        tz: str,
        currency: int,
        account_ids: list[int] | None,
    ) -> Optional[pl.DataFrame]:
        """Fetch one day's data for a group. Returns DataFrame or None."""
        try:
            tz_obj = pendulum.timezone(tz)
            day = pendulum.parse(day_str, tz=tz_obj)
            fmt = "YYYY-MM-DDTHH:mm:ssZ"
            overrides = {
                "timezone":       tz,
                "currency":       currency,
                "accountIds":     account_ids or [],
                "dateTypeId":     11,
                "intervalTypeId": 2,
                "startDate":      day.start_of("day").format(fmt),
                "endDate":        day.end_of("day").format(fmt),
            }
            payload = group.build_payload(overrides)
            report_id = self.create_report(payload)
            status = self.poll_report_status(report_id)
            url = status.get("url")
            if not url:
                return None
            csv_text = self.download_report_csv(url)
            schema = group.build_schema()
            df = self.transform_to_polars(csv_text, schema, overrides)
            return df if not df.is_empty() else None
        except Exception as e:
            logger.error(f"fetch_single_day({group.name}, {day_str}) failed: {e}")
            return None

    # ─── 10. BACKFILL: ONE MONTH ──────────────────────────
    def backfill_month(
        self,
        group: ReportGroupConfig,
        month_start: "pendulum.DateTime",
        month_end: "pendulum.DateTime",
        tz: str,
        currency: int,
        account_ids: list[int] | None,
        max_workers: int = 4,
    ) -> list[pl.DataFrame]:
        days = []
        cursor = month_start
        while cursor <= month_end:
            days.append(cursor.format("YYYY-MM-DD"))
            cursor = cursor.add(days=1)

        frames: list[pl.DataFrame] = []
        logger.info(f"[{group.name}] Backfilling {len(days)} days ({month_start.format('YYYY-MM')})")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.fetch_single_day, group, d, tz, currency, account_ids): d
                for d in days
            }
            with tqdm(total=len(days), desc=f"  {month_start.format('YYYY-MM')}", unit="day") as pbar:
                for fut in concurrent.futures.as_completed(futures):
                    df = fut.result()
                    if df is not None:
                        frames.append(df)
                    pbar.update(1)
        return frames

    # ─── 11. BACKFILL: ONE GROUP ACROSS MONTHS ────────────
    def backfill_report_group(
        self,
        group: ReportGroupConfig,
        months_back: int,
        output_dir: str,
        tz: str = "America/New_York",
        currency: int = 4,
        account_ids: list[int] | None = None,
        max_workers: int = 4,
    ) -> Optional[str]:
        t0 = time.time()
        logger.info(f"{'='*60}\n🔄 BACKFILL [{group.name.upper()}] — {months_back} months\n{'='*60}")

        tz_obj = pendulum.timezone(tz)
        now = pendulum.now(tz_obj)
        all_lazy: list[pl.LazyFrame] = []

        for m in range(months_back):
            ref = now.subtract(months=m)
            m_start = ref.start_of("month")
            m_end = min(ref.end_of("month"), now.subtract(days=1))
            if m_start > m_end:
                continue
            frames = self.backfill_month(group, m_start, m_end, tz, currency, account_ids, max_workers)
            for f in frames:
                all_lazy.append(f.lazy())
            gc.collect()

        if not all_lazy:
            logger.warning(f"[{group.name}] No data collected")
            return None

        combined = pl.concat(all_lazy).collect()
        os.makedirs(output_dir, exist_ok=True)
        ts = pendulum.now("UTC").format("YYYY-MM-DD_HH-mm-ss")
        fpath = os.path.join(output_dir, f"{group.name}_backfill_{ts}.csv")
        combined.write_csv(fpath)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        logger.success(
            f"✅ BACKFILL [{group.name.upper()}] DONE | "
            f"{combined.shape[0]:,} rows | {size_mb:.1f} MB | {elapsed:.0f}s → {fpath}"
        )
        return fpath

    # ─── 12. BACKFILL: ALL GROUPS ─────────────────────────
    def backfill_all_groups(
        self,
        groups: list[ReportGroupConfig],
        months_back: int,
        output_dir: str,
        tz: str = "America/New_York",
        currency: int = 4,
        account_ids: list[int] | None = None,
        max_workers: int = 4,
    ) -> dict[str, Optional[str]]:
        results: dict[str, Optional[str]] = {}
        with tqdm(total=len(groups), desc="📊 Backfill Groups", unit="group") as pbar:
            for group in groups:
                path = self.backfill_report_group(
                    group, months_back, output_dir, tz, currency, account_ids, max_workers
                )
                results[group.name] = path
                pbar.update(1)
        return results
