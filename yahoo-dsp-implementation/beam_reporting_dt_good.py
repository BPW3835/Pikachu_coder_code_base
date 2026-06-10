"""
beam_reporting.py
─────────────────
Yahoo DSP Reporting API module — Apache Beam edition.

Handles:
  • Pydantic payload validation
  • Report creation / status polling / CSV download
  • Apache Beam DoFns for parsing, type-casting, and partitioned CSV writing
  • Sequential API extraction inside a single DoFn to respect rate limits
  • Dimension-to-CSV-column mapping & metric-name registry
  • Backfill (up to 13 months) and daily modes
  • Close-of-Books (COB) check with retry (optional, via YahooAPIClient)

GCS / BQ load steps are commented out for local development.
"""

import io
import os
import csv
import json
import time
import traceback
from typing import Optional, List, Dict, Tuple, Any

import pendulum
import requests
import apache_beam as beam
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from helpers import YahooHelpers

# ═══════════════════════════════════════════════════════════════
# DIMENSION → CSV COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════
# Each key is a dimensionTypeId; the value is a list of
# (column_name, bq_type) tuples that Yahoo returns in the CSV.

DIMENSION_CSV_COLUMNS: Dict[int, List[Tuple[str, str]]] = {
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

METRIC_NAMES: Dict[int, str] = {
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
INTEGER_METRIC_IDS: set = {
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
    includeValues: List[FilterOptionValue] = []


class SpecObject(BaseModel):
    """Metric-level spec used inside having.havingSpecs."""
    metricTypeId: int
    type: str       # equalTo | greaterThan | lessThan | not
    value: int | float


class HavingObject(BaseModel):
    """Metric-level filter (having clause)."""
    type: str       # and | or | not
    havingSpecs: List[SpecObject]


class LimitSpecColumn(BaseModel):
    """Single sort column for limitSpec."""
    direction: str = "DESCENDING"
    metricTypeId: int


class LimitSpec(BaseModel):
    """Controls row limit and sort order."""
    columns: List[LimitSpecColumn]
    limit: int = 10_000
    type: str = "default"


class ReportOption(BaseModel):
    """
    reportOption object — defines dimensions, metrics,
    filters, timezone, and currency.
    """
    timezone: str = "America/New_York"
    currency: int = 4               # 1=Seat 2=Adv 3=Campaign 4=USD
    dimensionTypeIds: List[int] = []
    metricTypeIds: List[int] = []
    filterOptions: List[FilterOption] = []
    having: Optional[HavingObject] = None
    limitSpec: Optional[LimitSpec] = None
    accountIds: List[int] = []


class ReportPayload(BaseModel):
    """
    Full reporting payload for POST /extreport/.
    dateTypeId = 11 → custom range (startDate/endDate required)
    dateTypeId = 2  → yesterday
    intervalTypeId = 2  → Day granularity
    """
    reportOption: ReportOption
    dateTypeId: int = Field(..., ge=1, le=11)
    intervalTypeId: int = Field(..., ge=1, le=5)
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    dataSource: Optional[int] = None        # 15 for bidding reports

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
    REPORT_NAME: str = ""
    DIMENSION_IDS: List[int] = []
    METRIC_IDS: List[int] = []
    DATA_SOURCE: Optional[int] = None


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


ALL_REPORT_GROUPS: List[type] = [
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
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def build_report_configs(groups: List[type]) -> List[Dict[str, Any]]:
    """
    Convert ReportGroupConfig subclasses into plain dicts
    that are serialisable and safe to pass between DoFns.
    """
    configs = []
    for grp in groups:
        configs.append({
            "report_name":   grp.REPORT_NAME,
            "dimension_ids": list(grp.DIMENSION_IDS),
            "metric_ids":    list(grp.METRIC_IDS),
            "data_source":   grp.DATA_SOURCE,
        })
    return configs


def generate_backfill_dates(timezone: str, months_back: int = 13) -> List[str]:
    """
    Generate a list of date strings (YYYY-MM-DD) in DESCENDING order
    from yesterday going back *months_back* months.
    """
    tz = pendulum.timezone(timezone)
    end_dt = pendulum.yesterday(tz)
    start_dt = end_dt.subtract(months=months_back)
    dates = []
    cursor = end_dt
    while cursor >= start_dt:
        dates.append(cursor.format("YYYY-MM-DD"))
        cursor = cursor.subtract(days=1)
    logger.info(
        f"Backfill dates generated: {len(dates)} days "
        f"({dates[-1]} -> {dates[0]})"
    )
    return dates


def generate_daily_dates(timezone: str) -> List[str]:
    """Return a single-element list containing yesterday's date string."""
    tz = pendulum.timezone(timezone)
    yesterday = pendulum.yesterday(tz).format("YYYY-MM-DD")
    logger.info(f"Daily mode — single date: {yesterday}")
    return [yesterday]


# ═══════════════════════════════════════════════════════════════
# YAHOO API CLIENT
# ═══════════════════════════════════════════════════════════════

class YahooAPIClient:
    """
    Thin wrapper around Yahoo DSP Reporting API.
    Responsible for auth, payload building, report creation,
    status polling, and CSV download.

    Uses the same endpoints as the original reporting.py module:
      REPORT_ENDPOINT = https://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/
      COB_ENDPOINT    = https://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/cob
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
        self._helpers = YahooHelpers()
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        logger.info("YahooAPIClient initialised.")

    # ───── Auth ─────────────────────────────────────────────────

    def _get_token(self) -> str:
        """
        Return a valid access token, refreshing if expired.

        FIX: YahooHelpers.get_access_token() returns the raw token
        string directly (not a dict). Handle both cases gracefully.
        """
        now = time.time()
        if self._token and now < self._token_expiry:
            return self._token

        logger.info("Refreshing Yahoo DSP access token …")
        token_data = self._helpers.get_access_token()

        if isinstance(token_data, dict):
            # If helper returns a dict like {"access_token": "...", "expires_in": N}
            self._token = token_data.get("access_token", str(token_data))
            expires_in = token_data.get("expires_in", 3500)
        else:
            # Helper returns the raw token string directly
            self._token = str(token_data)
            expires_in = 3500   # default ~1 hour

        self._token_expiry = now + expires_in - 60
        logger.debug(
            f"Token refreshed | length={len(self._token)} | "
            f"expires_in ~{expires_in}s"
        )
        return self._token

    def _headers(self) -> Dict[str, str]:
        """Build auth headers using the helper's build_auth_headers if available."""
        token = self._get_token()
        # Try to use the helper's own header builder (matches original code)
        if hasattr(self._helpers, "build_auth_headers"):
            return self._helpers.build_auth_headers(token)
        # Fallback: construct manually
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ───── Payload construction ─────────────────────────────────

    @staticmethod
    def build_payload(
        report_config: Dict[str, Any],
        date_str: str,
        timezone: str = "America/New_York",
        currency: int = 4,
        account_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Build a validated ReportPayload dict for a single-day extraction.

        Uses dateTypeId=11 (Custom Range) with startDate == endDate == date_str
        and intervalTypeId=2 (Day) to populate the Day column — matching the
        original reporting.py approach.

        Parameters
        ----------
        report_config : dict produced by build_report_configs()
        date_str      : YYYY-MM-DD date for which data is requested
        timezone      : IANA timezone string
        currency      : 1=Seat 2=Adv 3=Campaign 4=USD
        account_ids   : advertiser account IDs (optional)
        """
        # Format dates as ISO-8601 matching the original code
        fmt = "YYYY-MM-DD[T]HH:mm:ssZ"
        tz = pendulum.timezone(timezone)
        day_dt = pendulum.parse(date_str, tz=tz)
        start_str = day_dt.start_of("day").format(fmt)
        end_str = day_dt.end_of("day").format(fmt)

        report_option: dict = {
            "timezone": timezone,
            "currency": currency,
            "dimensionTypeIds": report_config["dimension_ids"],
            "metricTypeIds": report_config["metric_ids"],
            "filterOptions": [],
            "accountIds": account_ids or [],
        }

        payload: dict = {
            "reportOption": report_option,
            "dateTypeId": 11,           # Custom Range
            "intervalTypeId": 2,        # Day granularity
            "startDate": start_str,
            "endDate": end_str,
        }

        # Add dataSource for bidding reports (dataSource=15)
        data_source = report_config.get("data_source")
        if data_source is not None:
            payload["dataSource"] = data_source

        # Validate via Pydantic
        validated = ReportPayload(**payload)
        return validated.model_dump(exclude_none=True)

    # ───── API calls ────────────────────────────────────────────

    def create_report(self, payload: Dict) -> str:
        """POST /extreport/ — submit report creation request. Returns report ID."""
        logger.debug(f"POST {self.REPORT_ENDPOINT}")
        logger.debug(f"Payload:\n{json.dumps(payload, indent=2)}")

        resp = requests.post(
            self.REPORT_ENDPOINT,
            headers=self._headers(),
            json=payload,
            timeout=120,
        )

        if resp.status_code not in (200, 201, 202):
            logger.error(f"Create report HTTP {resp.status_code}: {resp.text}")
            resp.raise_for_status()

        body = resp.json()
        report_id = body.get("customerReportId")
        if not report_id:
            raise RuntimeError(f"No customerReportId in response: {body}")

        logger.info(
            f"Report created | ID: {report_id} | Status: {body.get('status')}"
        )
        return str(report_id)

    def poll_report(
        self,
        report_id: str,
        initial_wait: float = 15.0,
        max_attempts: int = 200,
        backoff_factor: float = 1.2,
        max_per_cycle_wait: float = 120.0,
    ) -> Dict:
        """
        GET /extreport/{id} with exponential backoff (matching original).
        Returns the response JSON once status is 'SUCCESS'.
        Raises RuntimeError on 'FAILED' or timeout.
        """
        url = f"{self.REPORT_ENDPOINT}{report_id}"
        wait = initial_wait

        for attempt in range(1, max_attempts + 1):
            resp = requests.get(url, headers=self._headers(), timeout=120)

            if resp.status_code != 200:
                logger.warning(
                    f"Attempt {attempt}: HTTP {resp.status_code} — retrying"
                )
                time.sleep(wait)
                wait = min(wait * backoff_factor, max_per_cycle_wait)
                continue

            body = resp.json()
            status = (body.get("status") or "").upper()

            if status == "SUCCESS":
                logger.info(
                    f"Report {report_id} ready | Rows: {body.get('numRows', '?')}"
                )
                return body

            if status == "FAILED":
                raise RuntimeError(f"Report {report_id} FAILED: {body}")

            logger.debug(
                f"Attempt {attempt}/{max_attempts} | Status: {status} | "
                f"Retrying in {wait:.0f}s"
            )
            time.sleep(wait)
            wait = min(wait * backoff_factor, max_per_cycle_wait)

        raise TimeoutError(
            f"Report {report_id} not ready after {max_attempts} attempts."
        )

    def download_csv(self, report_data: Dict) -> str:
        """Extract download URL from poll response and GET the CSV text."""
        download_url = report_data.get("url")
        if not download_url:
            raise ValueError(f"No download URL in: {report_data}")

        logger.info(f"Downloading report CSV …")
        resp = requests.get(download_url, timeout=300)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Download failed [{resp.status_code}]: {resp.text[:500]}"
            )

        csv_text = resp.text
        logger.info(f"Downloaded | {len(csv_text):,} chars")
        return csv_text

    def extract_single_day_report(
        self,
        report_config: Dict[str, Any],
        date_str: str,
        timezone: str = "America/New_York",
        currency: int = 4,
        account_ids: Optional[List[int]] = None,
    ) -> str:
        """
        End-to-end: build payload -> create -> poll -> download -> return CSV text.
        """
        payload = self.build_payload(
            report_config, date_str, timezone, currency, account_ids
        )
        report_id = self.create_report(payload)
        report_data = self.poll_report(report_id)
        csv_text = self.download_csv(report_data)
        return csv_text

    # ───── Utility ──────────────────────────────────────────────

    def list_all_advertisers(self) -> List[Dict]:
        """
        GET /traffic/api/v1/advertisers
        Returns all advertiser accounts under your seat.
        """
        url = (
            "https://api-sched-v3.admanagerplus.yahoo.com"
            "/traffic/api/v1/advertisers"
        )
        resp = requests.get(url, headers=self._headers(), timeout=60)

        if resp.status_code != 200:
            logger.error(
                f"List advertisers failed [{resp.status_code}]: {resp.text}"
            )
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
        return advertisers

    def check_cob(self, advertiser_id: int, day: str) -> bool:
        """
        GET /extreport/cob — check if data is finalised for
        a given advertiser + day (YYYYMMDD).
        Returns True if dayClosed == 'true'.
        """
        logger.info(f"COB check -> advertiser={advertiser_id}, day={day}")
        resp = requests.get(
            self.COB_ENDPOINT,
            params={"advertiserId": advertiser_id, "day": day},
            headers=self._headers(),
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"COB failed [{resp.status_code}]: {resp.text}")

        body = resp.json()
        is_closed = str(body.get("dayClosed", "false")).lower() == "true"
        logger.info(f"COB result: {body} -> closed={is_closed}")
        return is_closed


# ═══════════════════════════════════════════════════════════════
# APACHE BEAM DoFns
# ═══════════════════════════════════════════════════════════════

class SequentialExtractFn(beam.DoFn):
    """
    Receives a single seed element (ignored) and sequentially
    iterates through every (date, report_config) combination.

    **Why sequential?**
    Yahoo DSP's reporting API has strict rate limits. Running
    multiple workers in parallel would quickly trigger 429s.
    By placing ALL extraction inside one DoFn.process() call
    we guarantee a single thread of API calls with a configurable
    sleep between each.

    Yields: (date_str, report_name, config_dict, csv_text)
    """

    def __init__(
        self,
        dates: List[str],
        report_configs: List[Dict[str, Any]],
        timezone: str = "America/New_York",
        currency: int = 4,
        account_ids: Optional[List[int]] = None,
        rate_limit_sec: float = 2.0,
        skip_on_error: bool = True,
    ):
        super().__init__()
        self._dates = dates
        self._configs = report_configs
        self._timezone = timezone
        self._currency = currency
        self._account_ids = account_ids or []
        self._rate_limit_sec = rate_limit_sec
        self._skip_on_error = skip_on_error
        self._client: Optional[YahooAPIClient] = None

    def setup(self):
        """Called once per worker — initialise the API client."""
        self._client = YahooAPIClient()
        logger.info("SequentialExtractFn.setup() — API client ready.")

    def process(self, seed_element):
        total = len(self._dates) * len(self._configs)
        counter = 0
        for date_str in self._dates:                    # descending (most recent first)
            for cfg in self._configs:
                counter += 1
                report_name = cfg["report_name"]
                logger.info(
                    f"[{counter}/{total}] Extracting {report_name} "
                    f"for {date_str} …"
                )
                try:
                    csv_text = self._client.extract_single_day_report(
                        report_config=cfg,
                        date_str=date_str,
                        timezone=self._timezone,
                        currency=self._currency,
                        account_ids=self._account_ids,
                    )
                    yield (date_str, report_name, cfg, csv_text)
                except Exception:
                    logger.error(
                        f"FAILED: {report_name} / {date_str}\n"
                        f"{traceback.format_exc()}"
                    )
                    if not self._skip_on_error:
                        raise

                # Rate-limit sleep between every API interaction
                if counter < total:
                    time.sleep(self._rate_limit_sec)


class ParseCSVFn(beam.DoFn):
    """
    Input : (date_str, report_name, config_dict, csv_text)
    Output: individual row dicts with _date and _report_name injected.
    """

    def process(self, element):
        date_str, report_name, _cfg, csv_text = element
        reader = csv.DictReader(io.StringIO(csv_text))
        row_count = 0
        for row in reader:
            row["_date"] = date_str
            row["_report_name"] = report_name
            row_count += 1
            yield row
        logger.info(
            f"ParseCSV: {report_name}/{date_str} -> {row_count:,} rows"
        )


class CastTypesFn(beam.DoFn):
    """
    Cast every column value to its BigQuery-compatible Python type
    based on the DIMENSION_CSV_COLUMNS and METRIC_NAMES registries.

    • Dimension columns -> str
    • Integer metrics   -> int
    • Float metrics     -> float
    • Day column        -> str
    • Internal keys (_*) -> pass-through
    """

    def setup(self):
        # Build lookup sets once per worker
        self._int_metric_names = {
            METRIC_NAMES[mid] for mid in INTEGER_METRIC_IDS if mid in METRIC_NAMES
        }
        self._float_metric_names = {
            name
            for mid, name in METRIC_NAMES.items()
            if mid not in INTEGER_METRIC_IDS
        }
        self._dimension_col_names = set()
        for cols in DIMENSION_CSV_COLUMNS.values():
            for col_name, _bq_type in cols:
                self._dimension_col_names.add(col_name)

    def process(self, row: Dict[str, Any]):
        cast_row: Dict[str, Any] = {}
        for key, value in row.items():
            if key.startswith("_"):
                # internal metadata — pass through
                cast_row[key] = value
            elif key == "Day":
                cast_row[key] = str(value) if value else ""
            elif key in self._int_metric_names:
                try:
                    cast_row[key] = int(float(value)) if value else 0
                except (ValueError, TypeError):
                    cast_row[key] = 0
            elif key in self._float_metric_names:
                try:
                    cast_row[key] = float(value) if value else 0.0
                except (ValueError, TypeError):
                    cast_row[key] = 0.0
            elif key in self._dimension_col_names:
                cast_row[key] = str(value) if value else ""
            else:
                # Unknown column — keep as string
                cast_row[key] = str(value) if value else ""
        yield cast_row


class WritePartitionedCSVFn(beam.DoFn):
    """
    Input : ((date_str, report_name), iterable_of_rows)
    Output: filepath string of the written CSV.

    Directory structure:
        {output_dir}/{report_name}/{date_str}/{report_name}_{date_str}.csv

    Column ordering:
        'Day' first -> dimension columns (sorted) -> metric columns (sorted)

    FIX: Uses only yield (no return) to avoid Beam yield/return conflict.
    """

    def __init__(self, output_dir: str = "output"):
        super().__init__()
        self._output_dir = output_dir

    def process(self, element):
        (date_str, report_name), rows = element
        rows = list(rows)

        if not rows:
            logger.warning(
                f"No rows for {report_name}/{date_str} — skipping write."
            )
            # FIX: Do NOT use 'return' here — just don't yield anything.
            # The generator will naturally end without producing output.
        else:
            # ── Build ordered column list ───────────────────────────
            all_keys = set()
            for r in rows:
                all_keys.update(r.keys())

            # Remove internal keys from output
            internal_keys = {k for k in all_keys if k.startswith("_")}
            writable_keys = all_keys - internal_keys

            # Classify columns
            dim_names = set()
            for cols in DIMENSION_CSV_COLUMNS.values():
                for col_name, _ in cols:
                    dim_names.add(col_name)

            metric_names_set = set(METRIC_NAMES.values())

            day_cols = sorted(k for k in writable_keys if k == "Day")
            dim_cols = sorted(k for k in writable_keys if k in dim_names and k != "Day")
            met_cols = sorted(k for k in writable_keys if k in metric_names_set)
            other_cols = sorted(
                writable_keys - set(day_cols) - set(dim_cols) - set(met_cols)
            )

            fieldnames = day_cols + dim_cols + met_cols + other_cols

            # ── Write CSV ───────────────────────────────────────────
            dir_path = os.path.join(self._output_dir, report_name, date_str)
            os.makedirs(dir_path, exist_ok=True)
            filename = f"{report_name}_{date_str}.csv"
            filepath = os.path.join(dir_path, filename)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                for r in rows:
                    # strip internal keys before writing
                    clean = {k: v for k, v in r.items() if not k.startswith("_")}
                    writer.writerow(clean)

            logger.info(f"✓ Wrote {len(rows):,} rows -> {filepath}")
            yield filepath


class LogSummaryFn(beam.DoFn):
    """Log each written file path and maintain a running count."""

    def __init__(self):
        super().__init__()
        self._counter = 0

    def process(self, filepath: str):
        self._counter += 1
        logger.info(f"[SUMMARY #{self._counter}] {filepath}")
        yield filepath

    def teardown(self):
        logger.info(f"Pipeline complete — {self._counter} CSV files written.")


# ═══════════════════════════════════════════════════════════════
# CONVENIENCE: stand-alone test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    client = YahooAPIClient()
    client.list_all_advertisers()
