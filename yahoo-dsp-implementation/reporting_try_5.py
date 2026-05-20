"""
reporting_v2.py
───────────────
Yahoo DSP Reporting API — Multi-Group Architecture.

Each logical report group is a separate class that defines:
  • REPORT_NAME   – slug used in file-names and logs
  • DIMENSION_IDS – Yahoo dimension-type IDs
  • METRIC_IDS    – Yahoo metric-type IDs
  • DATA_SOURCE   – set to 15 only for the Bidding report

The YahooReporting orchestrator iterates over groups, builds payloads,
creates / polls / downloads reports, transforms CSV → Polars DataFrame,
and saves one CSV per group.

GCS / BigQuery upload hooks are stubbed (commented) for cloud deploy.
"""

import io
import os
import json
import time
import traceback
from typing import Optional

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
# Each key is a dimensionTypeId; value is a list of
# (csv_column_name, bq_type) that Yahoo returns for that dim.

DIMENSION_CSV_COLUMNS: dict[int, list[tuple[str, str]]] = {
    3:    [("Creative Size ID", "STRING"), ("Creative Size", "STRING")],
    4:    [("Advertiser ID", "STRING"), ("Advertiser", "STRING")],
    5:    [("Campaign ID", "STRING"), ("Campaign Name", "STRING")],
    6:    [("Line ID", "STRING"), ("Line", "STRING")],
    7:    [("Ad ID", "STRING"), ("Ad", "STRING")],
    8:    [("Creative ID", "STRING"), ("Creative Name", "STRING")],
    9:    [("Domain Name", "STRING")],
    19:   [("Country ID", "STRING"), ("Country", "STRING")],
    20:   [("State", "STRING")],
    23:   [("Device Category ID", "STRING"), ("Device Category Name", "STRING")],
    25:   [("Browser Type ID", "STRING"), ("Browser Type Name", "STRING")],
    27:   [("Age ID", "STRING"), ("Age", "STRING")],
    28:   [("Gender ID", "STRING"), ("Gender", "STRING")],
    34:   [("Device Type Id", "STRING"), ("Device Type Name", "STRING")],
    46:   [("Beacon ID", "STRING"), ("Beacon", "STRING")],
    50:   [("Deal ID", "STRING"), ("Deal", "STRING")],
    54:   [("Beacon Type ID", "STRING")],
    55:   [("App Name", "STRING")],
    56:   [("Seat ID", "STRING"), ("Seat Name", "STRING")],
    63:   [("Media Type", "STRING")],
    64:   [("Advertiser Group", "STRING")],
    106:  [("Inventory Type", "STRING")],
    122:  [("Mobile OS", "STRING")],
    126:  [("Connection Type", "STRING")],
    147:  [("Days to Conversion", "STRING")],
    149:  [("Package ID", "STRING"), ("Package", "STRING")],
    153:  [("Connected TV Device", "STRING")],
    154:  [("Inventory Auction Type", "STRING")],
    166:  [("Channel Type", "STRING")],
    167:  [("Video Content Length", "STRING")],
    169:  [("Video Ad Placement", "STRING")],
    171:  [("Ads.txt Declaration", "STRING")],
    174:  [("Ad Format ID", "STRING"), ("Ad Format", "STRING")],
    182:  [("Video Player Size", "STRING")],
    1028: [("Connected TV Make", "STRING")],
    1029: [("Connected TV Model", "STRING")],
    1033: [("Deal Type Description", "STRING")],
    1037: [("Video Creative Duration", "STRING")],
    1225: [("Line Channel Type", "STRING")],
    1226: [("Line Channel Subtype", "STRING")],
    1228: [("Exchange ID", "STRING"), ("Exchange", "STRING")],
    1240: [("Video Ad Placement (IAB)", "STRING")],
    1267: [("Campaign Client Status", "STRING")],
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
    26:  "25% Completions",
    27:  "50% Completions",
    28:  "75% Completions",
    29:  "100% Completions",
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
    102: "Yahoo+ Viewable Rate",
    103: "Yahoo+ Nonviewable impressions",
    104: "Yahoo+ Non-Measurable Impressions",
    105: "Yahoo+ Measurable Rate",
    106: "Impression Distribution Viewable",
    107: "Impression Distribution Nonviewable",
    108: "Impression Distribution Nonmeasurable",
    109: "View Through Conversion",
    110: "Click Through Conversion",
    111: "View Through Conversion Rate",
    112: "Click Through Conversion Rate",
    113: "View Through CPA",
    114: "Click Through CPA",
    129: "Seat Management Fee",
    137: "Conversion Value",
    138: "ROAS",
    168: "VCPM",
    169: "CPCV",
    171: "Cross-device Conversions",
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
    317: "Very First Conversion",
    322: "Programmatic Guaranteed Deal Fee",
    328: "Third Party Conversions",
    329: "Audience Impressions",
    365: "Cost Per Valid Video View (CPV)",
    382: "Bid Shading Cost Savings",
    386: "Companion Impressions",
    387: "Companion Clicks",
    426: "Estimated Billable Amount",
    430: "Unique People",
    446: "Unique Households",
    447: "Average Frequency per Household",
    448: "Average Impression Multiplier",
    461: "Very First View Through Conversion",
    462: "Very First Click Through Conversion",
    463: "Very First View Through CPA",
    464: "Very First Click Through CPA",
    465: "View Through Dynamic Conversion Value",
    466: "View Through Dynamic Conversion ROAS",
    467: "Click Through Dynamic Conversion Value",
    468: "Click Through Dynamic Conversion ROAS",
    469: "View Through Conversion Value",
    470: "View Through ROAS",
    471: "Click Through Conversion Value",
    472: "Click Through ROAS",
    473: "Very First View Through Conversion Value",
    474: "Very First View Through ROAS",
    475: "Very First Click Through Conversion Value",
    476: "Very First Click Through ROAS",
    523: "Average People Frequency",
    560: "Yahoo+ Demo Fee",
    564: "Yahoo Segment Fee",
    570: "Third Party Segment Fee",
    583: "Available Bid Requests",
    584: "Average Win Rate",
    714: "Total Data Fee",
    715: "Platform Fee",
    716: "Inventory ECPM",
    717: "Inventory Cost %",
    721: "Yahoo+ Viewable Impressions GroupM",
    723: "Yahoo+ Viewability Rate GroupM",
    724: "Inventory Cost",
}

# Metric IDs whose values are integer counts (rest → FLOAT64)
INTEGER_METRIC_IDS: set[int] = {
    1, 2, 23, 25, 26, 27, 28, 29,
    100, 101, 103, 104, 109, 110,
    171, 173, 254, 317, 328, 329,
    386, 387, 430, 446, 461, 462,
    583, 721,
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
    type: str
    value: int | float


class HavingObject(BaseModel):
    """Metric-level filter (having clause)."""
    type: str
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
    """reportOption object — dims, metrics, filters, tz, currency."""
    timezone: str = "America/New_York"
    currency: int = 4
    dimensionTypeIds: list[int] = []
    metricTypeIds: list[int] = []
    filterOptions: list[FilterOption] = []
    having: Optional[HavingObject] = None
    limitSpec: Optional[LimitSpec] = None
    accountIds: list[int] = []


class ReportPayload(BaseModel):
    """
    Full reporting payload for POST /extreport/.

    dateTypeId:  1=Today 2=Yesterday 3=This Week 4=Last 7d
                 5=Last Week 6=Last 14d 7=MTD 8=Last 30d
                 9=Last Month 10=All Time 11=Custom Range

    intervalTypeId: 1=Cumulative 2=Day 3=Month 4=Week 5=Hour
    """
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
                "startDate and endDate are required when dateTypeId=11 (Custom Range)"
            )
        return self


# ═══════════════════════════════════════════════════════════════
# REPORT GROUP BASE CLASS
# ═══════════════════════════════════════════════════════════════


class ReportGroupConfig:
    """
    Base class for a logical report group.

    Subclasses define REPORT_NAME, DIMENSION_IDS, METRIC_IDS,
    and optionally DATA_SOURCE.

    The class auto-generates:
      • BQ-compatible schema  (build_schema)
      • API request payload   (build_payload)
    """

    REPORT_NAME: str = "base"
    DIMENSION_IDS: list[int] = []
    METRIC_IDS: list[int] = []
    DATA_SOURCE: int | None = None

    # ── Human-readable date-type labels ──────────────────────
    _DATE_TYPE_LABELS: dict[int, str] = {
        1: "Today", 2: "Yesterday", 3: "This_Week",
        4: "Last_7_Days", 5: "Last_Week", 6: "Last_14_Days",
        7: "MTD", 8: "Last_30_Days", 9: "Last_Month", 10: "All_Time",
    }

    @classmethod
    def build_schema(cls) -> list[dict]:
        """
        Auto-generate a BQ schema from DIMENSION_IDS and METRIC_IDS.

        Column order:  Day → dim cols → metric cols → metadata cols
        """
        schema: list[dict] = []

        # 1. Day column (always first — populated by intervalTypeId=2)
        schema.append({"name": "Day", "type": "STRING", "mode": "NULLABLE"})

        # 2. Dimension columns
        for dim_id in cls.DIMENSION_IDS:
            cols = DIMENSION_CSV_COLUMNS.get(dim_id, [])
            for col_name, bq_type in cols:
                schema.append({"name": col_name, "type": bq_type, "mode": "NULLABLE"})

        # 3. Metric columns
        for met_id in cls.METRIC_IDS:
            col_name = METRIC_NAMES.get(met_id, f"metric_{met_id}")
            bq_type = "INTEGER" if met_id in INTEGER_METRIC_IDS else "FLOAT64"
            schema.append({"name": col_name, "type": bq_type, "mode": "NULLABLE"})

        # 4. Metadata columns (added by pipeline)
        schema.append({"name": "ingestion_timestamp", "type": "STRING", "mode": "NULLABLE"})
        schema.append({"name": "lookback_date_range", "type": "STRING", "mode": "NULLABLE"})

        return schema

    @classmethod
    def build_payload(
        cls,
        timezone: str = "America/New_York",
        currency: int = 4,
        account_ids: list[int] | None = None,
        date_type_id: int = 2,
        interval_type_id: int = 2,
        start_date: str | None = None,
        end_date: str | None = None,
        filter_options: list[dict] | None = None,
    ) -> dict:
        """Build the API request payload dict for this report group."""
        raw = {
            "reportOption": {
                "timezone": timezone,
                "currency": currency,
                "dimensionTypeIds": cls.DIMENSION_IDS,
                "metricTypeIds": cls.METRIC_IDS,
                "filterOptions": filter_options or [],
                "accountIds": account_ids or [],
            },
            "dateTypeId": date_type_id,
            "intervalTypeId": interval_type_id,
        }
        if start_date:
            raw["startDate"] = start_date
        if end_date:
            raw["endDate"] = end_date
        if cls.DATA_SOURCE is not None:
            raw["dataSource"] = cls.DATA_SOURCE

        # Validate through Pydantic
        validated = ReportPayload(**raw)
        return validated.model_dump(exclude_none=True)

    @classmethod
    def date_range_label(cls, date_type_id: int,
                         start_date: str | None = None,
                         end_date: str | None = None) -> str:
        if date_type_id == 10:
            return "ALL_TIME"
        if date_type_id == 11 and start_date and end_date:
            return f"{start_date[:10]} to {end_date[:10]}"
        return cls._DATE_TYPE_LABELS.get(date_type_id, f"dateTypeId_{date_type_id}")


# ═══════════════════════════════════════════════════════════════
# REPORT GROUP SUBCLASSES  (10 groups)
# ═══════════════════════════════════════════════════════════════


class CampaignDeliveryReport(ReportGroupConfig):
    """Core delivery KPIs at campaign / line / package grain."""
    REPORT_NAME = "campaign_delivery"
    DIMENSION_IDS = [4, 5, 6, 149, 56, 64, 19, 63, 1225, 1267]
    METRIC_IDS = [1, 2, 11, 23, 41, 42, 43, 44, 45, 137, 138, 171, 305, 317, 328]


class CreativePerformanceReport(ReportGroupConfig):
    """Creative- and ad-level performance."""
    REPORT_NAME = "creative_performance"
    DIMENSION_IDS = [4, 5, 6, 7, 8, 3, 174]
    METRIC_IDS = [1, 2, 11, 23, 44, 45, 41, 43]


class VideoPerformanceReport(ReportGroupConfig):
    """Video quartile completions, drop rates, CPCV."""
    REPORT_NAME = "video_performance"
    DIMENSION_IDS = [4, 5, 6, 3, 169, 167, 182, 1037, 1240]
    METRIC_IDS = [1, 25, 26, 27, 28, 29, 30, 31, 32, 33, 62, 63, 64, 65, 168, 169, 365, 386, 387, 44]


class ViewabilityReport(ReportGroupConfig):
    """Yahoo+ viewability metrics by device and inventory."""
    REPORT_NAME = "viewability"
    DIMENSION_IDS = [4, 5, 6, 34, 106, 19]
    METRIC_IDS = [1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 721, 723, 44]


class ConversionDetailReport(ReportGroupConfig):
    """View-through / click-through conversions, ROAS, CPA detail."""
    REPORT_NAME = "conversion_detail"
    DIMENSION_IDS = [4, 5, 6, 46, 54, 147]
    METRIC_IDS = [
        23, 109, 110, 111, 112, 113, 114, 137, 138, 305, 317,
        461, 462, 463, 464, 465, 466, 467, 468,
        469, 470, 471, 472, 473, 474, 475, 476, 44,
    ]


class CostFeeReport(ReportGroupConfig):
    """Platform fees, data fees, vendor costs, inventory cost."""
    REPORT_NAME = "cost_fee"
    DIMENSION_IDS = [4, 5, 6, 149]
    METRIC_IDS = [
        44, 49, 50, 51, 129, 180, 184, 188, 192, 196,
        200, 204, 208, 212, 216, 260, 322, 426,
        560, 564, 570, 714, 715, 716, 717, 724,
    ]


class GeoDeviceReport(ReportGroupConfig):
    """Geographic + device breakdown for delivery metrics."""
    REPORT_NAME = "geo_device"
    DIMENSION_IDS = [4, 5, 19, 20, 34, 23, 122, 126]
    METRIC_IDS = [1, 2, 11, 23, 44, 45, 41, 43]


class SupplyInventoryReport(ReportGroupConfig):
    """Exchange, deal type, inventory, app, channel breakdown."""
    REPORT_NAME = "supply_inventory"
    DIMENSION_IDS = [4, 5, 6, 1228, 1033, 154, 106, 55, 166, 171]
    METRIC_IDS = [1, 2, 11, 44, 45, 23, 41, 43]


class AudienceReachReport(ReportGroupConfig):
    """Unique people, households, audience impressions."""
    REPORT_NAME = "audience_reach"
    DIMENSION_IDS = [4, 5, 6, 166, 1225]
    METRIC_IDS = [1, 2, 44, 329, 430, 446, 447, 448, 523]


# class BiddingReport(ReportGroupConfig):
#     """
#     Bidding insights — requires dataSource=15 and restricted dims.
#     Only Advertiser (4), Campaign (5), Line (6), Package (149) allowed.
#     Currency MUST be USD (4).
#     """
#     REPORT_NAME = "bidding"
#     DIMENSION_IDS = [4, 5, 6, 149]
#     METRIC_IDS = [1, 173, 583, 584]   # ← ONLY these 4 are supported
#     DATA_SOURCE = 15

class BiddingReport(ReportGroupConfig):

    REPORT_NAME = "bidding"
    DIMENSION_IDS = [4, 5, 6, 149]
    METRIC_IDS = [1, 173, 583, 584]   # already fixed
    DATA_SOURCE = 15

    @classmethod
    def build_schema(cls):
        """
        Override schema because Yahoo Bidding report
        does NOT return dimension columns.
        """
        return [
            {"name": "Day", "type": "STRING", "mode": "NULLABLE"},
            {"name": "Impressions", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Qualified Bid Requests", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Available Bid Requests", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Average Win Rate", "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "ingestion_timestamp", "type": "STRING", "mode": "NULLABLE"},
            {"name": "lookback_date_range", "type": "STRING", "mode": "NULLABLE"},
        ]


# Convenience list of all groups
ALL_REPORT_GROUPS: list[type[ReportGroupConfig]] = [
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


# ═══════════════════════════════════════════════════════════════
# YAHOO REPORTING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════


class YahooReporting:
    """
    End-to-end Yahoo DSP Reporting API handler (multi-group).

    Pipeline per group:
        1. Build schema + payload from ReportGroupConfig subclass
        2. (Optional) COB check
        3. POST  /extreport/           → create report
        4. GET   /extreport/{id}       → poll until SUCCESS
        5. Download CSV from temp URL
        6. Parse CSV → Polars DataFrame
        7. Add metadata columns
        8. Cast columns per auto-generated BQ schema
        9. Save CSV locally
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
            self.poll_interval = 15        # seconds — initial wait
            self.poll_max_attempts = 200   # ~50 min ceiling with backoff
            self.poll_backoff = 1.2
            self.poll_max_wait = 120       # cap per-cycle
            logger.success("YahooReporting initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── auth helper ──────────────────────

    def _auth_headers(self) -> dict:
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)

    # ───────────────────── 1. CREATE REPORT ─────────────────

    def create_report(self, payload: dict) -> str:
        """POST /extreport/ — returns customerReportId."""
        try:
            headers = self._auth_headers()
            logger.info(
                f"Creating report — dims={len(payload['reportOption']['dimensionTypeIds'])}, "
                f"metrics={len(payload['reportOption']['metricTypeIds'])}"
            )
            logger.debug(f"Payload:\n{json.dumps(payload, indent=2)}")

            resp = requests.post(
                self.REPORT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # ✅ KEY FIX: log Yahoo error body BEFORE raising
            if resp.status_code not in (200, 201, 202):
                logger.error(
                    f"Yahoo API rejected request [{resp.status_code}]: {resp.text}"
                )
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
            logger.error(f"create_report error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 2. POLL STATUS ───────────────────

    def poll_report_status(self, report_id: str) -> dict:
        """GET /extreport/{id} — poll with exponential backoff."""
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

            raise TimeoutError(
                f"Report {report_id} not ready after {self.poll_max_attempts} attempts."
            )
        except Exception as e:
            logger.error(f"poll error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 3. DOWNLOAD CSV ──────────────────

    def download_report_csv(self, download_url: str) -> str:
        """Download CSV text from Yahoo temporary URL (valid 24h)."""
        try:
            logger.info("Downloading report CSV ...")
            resp = requests.get(download_url, timeout=300)
            if resp.status_code != 200:
                raise Exception(f"Download failed [{resp.status_code}]: {resp.text[:500]}")
            logger.success(f"Downloaded | {len(resp.text):,} chars")
            return resp.text
        except Exception as e:
            logger.error(f"Download error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 4. CLOSE OF BOOKS ────────────────

    def check_cob(self, advertiser_id: int, day: str) -> bool:
        """GET /extreport/cob — returns True if books are closed."""
        try:
            headers = self._auth_headers()
            resp = requests.get(
                self.COB_ENDPOINT,
                params={"advertiserId": advertiser_id, "day": day},
                headers=headers,
                timeout=60,
            )
            if resp.status_code != 200:
                logger.warning(f"COB check HTTP {resp.status_code}: {resp.text}")
                return False
            body = resp.json()
            closed = str(body.get("dayClosed", "false")).lower() == "true"
            logger.info(f"COB adv={advertiser_id} day={day} → {'CLOSED' if closed else 'OPEN'}")
            return closed
        except Exception as e:
            logger.error(f"COB error: {e}\n{traceback.format_exc()}")
            return False

    def wait_for_cob(
        self,
        advertiser_ids: list[int],
        target_day: str,
        max_retries: int = 12,
        retry_delay_secs: int = 300,
    ) -> bool:
        """Wait for COB for ALL advertisers. Returns True when all closed."""
        logger.info(
            f"Waiting for COB on {target_day} for "
            f"{len(advertiser_ids)} advertiser(s) ..."
        )
        for attempt in range(1, max_retries + 1):
            all_closed = all(
                self.check_cob(adv, target_day) for adv in advertiser_ids
            )
            if all_closed:
                logger.success(f"✅ COB closed for all on {target_day}")
                return True
            if attempt < max_retries:
                logger.info(f"COB retry {attempt}/{max_retries} — waiting {retry_delay_secs}s")
                time.sleep(retry_delay_secs)
        logger.error(f"❌ COB not closed after {max_retries} retries")
        return False

    # ───────────────────── 5. TRANSFORM CSV → POLARS ────────

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

    def transform_to_polars(
        self,
        csv_text: str,
        schema: list[dict],
        lookback_range: str,
    ) -> pl.DataFrame:
        """Parse CSV → Polars, add metadata, cast types, drop extras."""
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

            # Cast per schema
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

    # ───────────────────── 6. SINGLE GROUP RUNNER ───────────

    def run_report_group(
        self,
        report_group: type[ReportGroupConfig],
        config_overrides: dict,
        output_dir: str = "output",
    ) -> pl.DataFrame | None:
        """
        Execute the full pipeline for ONE report group.

        Args:
            report_group:     ReportGroupConfig subclass
            config_overrides: dict with timezone, currency, account_ids,
                              date_type_id, interval_type_id,
                              start_date (opt), end_date (opt)
            output_dir:       local directory for CSVs

        Returns:
            Polars DataFrame on success; None on failure.
        """
        name = report_group.REPORT_NAME
        t0 = time.time()
        try:
            logger.info("=" * 60)
            logger.info(f"▶ [{name.upper()}] Starting report group")
            logger.info("=" * 60)

            # 1. Schema
            schema = report_group.build_schema()
            logger.info(f"Schema: {len(schema)} columns")

            # 2. Payload
            payload = report_group.build_payload(**config_overrides)
            logger.info(
                f"Payload built — dims={len(report_group.DIMENSION_IDS)}, "
                f"metrics={len(report_group.METRIC_IDS)}, "
                f"dataSource={report_group.DATA_SOURCE}"
            )

            # 3. Create report
            report_id = self.create_report(payload)

            # 4. Poll
            status_resp = self.poll_report_status(report_id)
            url = status_resp.get("url")
            if not url:
                raise Exception("Report succeeded but no URL returned.")

            # 5. Download
            csv_text = self.download_report_csv(url)

            # 6. Transform
            lookback = report_group.date_range_label(
                config_overrides.get("date_type_id", 2),
                config_overrides.get("start_date"),
                config_overrides.get("end_date"),
            )
            df = self.transform_to_polars(csv_text, schema, lookback)

            if df.is_empty():
                logger.warning(f"[{name}] 0 rows returned — skipping save.")
                return df

            # 7. Save CSV
            os.makedirs(output_dir, exist_ok=True)
            ts = pendulum.now("UTC").format("YYYY-MM-DD_HH-mm-ss")
            filename = f"{name}_{ts}.csv"
            filepath = os.path.join(output_dir, filename)
            df.write_csv(filepath)
            abs_path = os.path.abspath(filepath)

            elapsed = time.time() - t0
            logger.success(
                f"✅ [{name.upper()}] DONE | {df.shape[0]} rows × "
                f"{df.shape[1]} cols | {elapsed:.1f}s → {abs_path}"
            )
            return df

        except Exception as e:
            logger.error(
                f"❌ [{name.upper()}] FAILED after {time.time() - t0:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            return None

    # ───────────────────── 7. ALL-GROUPS RUNNER ─────────────

    def run_all_groups(
        self,
        report_groups: list[type[ReportGroupConfig]],
        config_overrides: dict,
        output_dir: str = "output",
        skip_cob: bool = False,
        advertiser_ids_for_cob: list[int] | None = None,
    ) -> dict[str, pl.DataFrame | None]:
        """
        Iterate through report groups and run each.

        Returns dict of {report_name: DataFrame | None}.
        Errors per group are logged but do NOT stop the pipeline.
        """
        results: dict[str, pl.DataFrame | None] = {}
        total = len(report_groups)

        # ── Optional COB check (once for all groups) ──
        if not skip_cob and advertiser_ids_for_cob:
            tz = config_overrides.get("timezone", "America/New_York")
            yesterday_yyyymmdd = (
                pendulum.now(tz).subtract(days=1).format("YYYYMMDD")
            )
            logger.info(f"📚 COB check for {yesterday_yyyymmdd} ...")
            cob_ok = self.wait_for_cob(
                advertiser_ids=advertiser_ids_for_cob,
                target_day=yesterday_yyyymmdd,
            )
            if not cob_ok:
                logger.error("❌ Books not closed — aborting all groups.")
                return results
        elif skip_cob:
            logger.warning("⚠️  COB check skipped (--skip-cob).")
        else:
            logger.warning("⚠️  No advertiser IDs — skipping COB check.")

        # ── Run each group ──
        logger.info(f"🚀 Running {total} report group(s) ...")
        with tqdm(
            total=total,
            desc="📊 Report Groups",
            unit="group",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for grp in report_groups:
                pbar.set_postfix_str(grp.REPORT_NAME)
                df = self.run_report_group(
                    report_group=grp,
                    config_overrides=config_overrides,
                    output_dir=output_dir,
                )
                results[grp.REPORT_NAME] = df
                pbar.update(1)

        # ── Summary ──
        ok = sum(1 for v in results.values() if v is not None and not v.is_empty())
        fail = total - ok
        logger.info("=" * 60)
        logger.info(
            f"🏁 PIPELINE COMPLETE | {ok}/{total} succeeded, {fail} failed/empty"
        )
        for name, df in results.items():
            rows = df.shape[0] if df is not None else "FAILED"
            logger.info(f"   {name:30s} → {rows} rows")
        logger.info("=" * 60)

        return results
