"""
reporting.py
────────────
Yahoo DSP Reporting API module – production-ready.

Handles:
  • Pydantic payload validation
  • Report creation / status polling / CSV download
  • Polars transformation (cast to BQ-compatible types)
  • Close-of-Books (COB) check with retry
  • Pipeline orchestration for multiple report groups
  • Dynamic report group generation from metrics.py
  • report_type column injection per API type

GCS / BQ load steps are handled by main.py via helpers.py.
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

from helpers import YahooHelpers, sanitize_bq_column_name

from metrics import (
    REPORTING_API_DIMENSION_CSV_COLUMNS  as DIMENSION_CSV_COLUMNS,
    REPORTING_API_METRIC_NAMES           as METRIC_NAMES,
    REPORTING_API_INTEGER_METRIC_IDS     as INTEGER_METRIC_IDS,
    REPORTING_API_REPORT_GROUP_CONFIGS   as REPORT_GROUP_CONFIGS,
)

# API type identifier — used in report_type column and GCS paths
API_TYPE = "reporting_api"


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
    """
    dateTypeId:  1=Today 2=Yesterday ... 11=Custom Range
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
                "startDate and endDate are required when dateTypeId=11"
            )
        return self


# ═══════════════════════════════════════════════════════════════
# REPORT GROUP CONFIGURATION (base class)
# ═══════════════════════════════════════════════════════════════

class ReportGroupConfig:
    REPORT_NAME: str = "base"
    DIMENSION_IDS: list[int] = []
    METRIC_IDS: list[int] = []
    DATA_SOURCE: int | None = None

    _DATE_TYPE_LABELS: dict[int, str] = {
        1: "Today", 2: "Yesterday", 3: "This_Week",
        4: "Last_7_Days", 5: "Last_Week", 6: "Last_14_Days",
        7: "MTD", 8: "Last_30_Days", 9: "Last_Month", 10: "All_Time",
    }

    @classmethod
    def build_schema(cls) -> list[dict]:
        schema: list[dict] = [
            {"name": "Day", "type": "STRING", "mode": "NULLABLE"},
        ]
        for dim_id in cls.DIMENSION_IDS:
            cols = DIMENSION_CSV_COLUMNS.get(dim_id, [])
            for col_name, col_type in cols:
                schema.append({"name": col_name, "type": col_type, "mode": "NULLABLE"})
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
        schema.append({"name": "report_type", "type": "STRING", "mode": "NULLABLE"})
        return schema

    @classmethod
    def build_payload(cls, overrides: dict | None = None) -> dict:
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
        if payload["dateTypeId"] == 11:
            payload["startDate"] = overrides.get("startDate")
            payload["endDate"] = overrides.get("endDate")
        validated = ReportPayload(**payload)
        return validated.model_dump(exclude_none=True)

    # @classmethod
    # def date_range_label(cls, config: dict) -> str:
    #     dt = config.get("dateTypeId")
    #     if dt == 10:
    #         return "ALL_TIME"
    #     if dt == 11:
    #         s = config.get("startDate", "?")[:10]
    #         e = config.get("endDate", "?")[:10]
    #         return f"{s} to {e}"
    #     return cls._DATE_TYPE_LABELS.get(dt, f"dateTypeId_{dt}")

    @classmethod
    def date_range_label(cls, config: dict) -> str:
        dt = config.get("dateTypeId")
        if dt == 10:
            return "ALL_TIME"
        if dt == 11:
            s = config.get("startDate", "?")[:10]
            e = config.get("endDate", "?")[:10]
            if s == e:
                return s  # single day → just the date
            return f"{s} to {e}"
        # dateTypeId=2 (Yesterday) → resolve to actual date
        if dt == 2:
            tz_str = config.get("reportOption", {}).get("timezone", "America/New_York")
            return pendulum.now(tz_str).subtract(days=1).format("YYYY-MM-DD")
        return cls._DATE_TYPE_LABELS.get(dt, f"dateTypeId_{dt}")

# ═══════════════════════════════════════════════════════════════
# DYNAMIC REPORT GROUP GENERATION from metrics.py
# ═══════════════════════════════════════════════════════════════

def _build_report_group_class(name: str, config: dict) -> type:
    attrs = {
        "REPORT_NAME":   name,
        "DIMENSION_IDS": config["dimension_ids"],
        "METRIC_IDS":    config["metric_ids"],
        "DATA_SOURCE":   config.get("data_source"),
    }

    schema_override = config.get("schema_override")
    if schema_override:
        override_data = list(schema_override)
        # Auto-add report_type if missing from override
        if not any(f["name"] == "report_type" for f in override_data):
            override_data.append({"name": "report_type", "type": "STRING", "mode": "NULLABLE"})

        @classmethod
        def build_schema(cls) -> list[dict]:
            return list(override_data)
        attrs["build_schema"] = build_schema

    class_name = "".join(word.capitalize() for word in name.split("_")) + "Report"
    return type(class_name, (ReportGroupConfig,), attrs)


REPORT_GROUP_REGISTRY: dict[str, type[ReportGroupConfig]] = {}
for _name, _config in REPORT_GROUP_CONFIGS.items():
    _cls = _build_report_group_class(_name, _config)
    REPORT_GROUP_REGISTRY[_name] = _cls

# Named exports
CampaignDeliveryReport    = REPORT_GROUP_REGISTRY["campaign_delivery"]
CreativePerformanceReport = REPORT_GROUP_REGISTRY["creative_performance"]
VideoPerformanceReport    = REPORT_GROUP_REGISTRY["video_performance"]
ViewabilityReport         = REPORT_GROUP_REGISTRY["viewability"]
ConversionDetailReport    = REPORT_GROUP_REGISTRY["conversion_detail"]
CostFeeReport             = REPORT_GROUP_REGISTRY["cost_fee"]
GeoDeviceReport           = REPORT_GROUP_REGISTRY["geo_device"]
SupplyInventoryReport     = REPORT_GROUP_REGISTRY["supply_inventory"]
AudienceReachReport       = REPORT_GROUP_REGISTRY["audience_reach"]
BiddingReport             = REPORT_GROUP_REGISTRY["bidding"]

ALL_REPORT_GROUPS: list[type[ReportGroupConfig]] = list(REPORT_GROUP_REGISTRY.values())


# ═══════════════════════════════════════════════════════════════
# YAHOO REPORTING CLASS
# ═══════════════════════════════════════════════════════════════

class YahooReporting:
    """
    End-to-end Yahoo DSP Reporting API handler.

    run_report_group returns:
        {"df": pl.DataFrame, "filepath": str, "day_str": str}
        or None on error
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
            self.poll_interval = 15
            self.poll_max_attempts = 200
            self.poll_backoff = 1.2
            self.poll_max_wait = 120
            logger.success("YahooReporting initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── auth helper ──────────────────────────────
    def auth_headers(self) -> dict:
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)

    def list_all_advertisers(self) -> list[dict]:
        url = "https://api-sched-v3.admanagerplus.yahoo.com/traffic/api/v1/advertisers"
        headers = self.auth_headers()
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            logger.error(f"List advertisers failed [{resp.status_code}]: {resp.text}")
            resp.raise_for_status()
        body = resp.json()
        advertisers = body.get("response", [])
        logger.info(f"Found {len(advertisers)} advertiser(s) under seat")
        for adv in advertisers:
            logger.info(
                f"  ID: {adv.get('id'):<12} "
                f"Name: {adv.get('name', 'N/A'):<40} "
                f"Status: {adv.get('status', 'N/A')}"
            )
        print(advertisers)
        return advertisers

    # ─────────────── 1. CREATE REPORT ─────────────────────────
    def create_report(self, payload: dict) -> str:
        try:
            headers = self.auth_headers()
            dims = payload.get("reportOption", {}).get("dimensionTypeIds", [])
            mets = payload.get("reportOption", {}).get("metricTypeIds", [])
            logger.info(f"Creating report — dims={len(dims)}, metrics={len(mets)}")
            logger.debug(f"Payload:\n{json.dumps(payload, indent=2)}")

            resp = requests.post(
                self.REPORT_ENDPOINT, headers=headers,
                json=payload, timeout=60,
            )
            if resp.status_code not in (200, 201, 202):
                logger.error(f"Create report HTTP {resp.status_code}: {resp.text}")
                resp.raise_for_status()

            body = resp.json()
            report_id = body.get("customerReportId")
            if not report_id:
                raise Exception(f"No customerReportId in response: {body}")
            logger.success(f"Report created | ID: {report_id} | Status: {body.get('status')}")
            return report_id
        except Exception as e:
            logger.error(f"Error creating report: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 2. POLL STATUS ───────────────────────────
    def poll_report_status(self, report_id: str) -> dict:
        try:
            logger.info(f"Polling report {report_id} ...")
            wait = self.poll_interval
            with tqdm(
                total=self.poll_max_attempts, desc="⏳ Polling", unit="req",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for attempt in range(1, self.poll_max_attempts + 1):
                    headers = self.auth_headers()
                    resp = requests.get(
                        f"{self.REPORT_ENDPOINT}{report_id}",
                        headers=headers, timeout=60,
                    )
                    if resp.status_code != 200:
                        logger.warning(f"Attempt {attempt}: HTTP {resp.status_code} — retrying")
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

    # ─────────────── 4. CLOSE OF BOOKS ────────────────────────
    def check_cob(self, advertiser_id: int, day: str) -> bool:
        try:
            logger.info(f"COB check → advertiser={advertiser_id}, day={day}")
            headers = self.auth_headers()
            resp = requests.get(
                self.COB_ENDPOINT,
                params={"advertiserId": advertiser_id, "day": day},
                headers=headers, timeout=60,
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

    def wait_for_cob(self, advertiser_ids, target_day,
                     max_retries=12, retry_delay_secs=300):
        try:
            logger.info(
                f"Waiting for COB on {target_day} for "
                f"{len(advertiser_ids)} advertiser(s)..."
            )
            with tqdm(
                total=max_retries, desc="📚 COB check", unit="attempt",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for attempt in range(1, max_retries + 1):
                    all_closed = True
                    for adv_id in advertiser_ids:
                        if not self.check_cob(adv_id, target_day):
                            logger.warning(f"COB NOT closed → advertiser={adv_id}, day={target_day}")
                            all_closed = False
                            break
                    pbar.set_postfix_str("ALL CLOSED ✅" if all_closed else "PENDING ⏳")
                    pbar.update(1)
                    if all_closed:
                        logger.success(f"✅ COB closed for all advertisers on {target_day}")
                        return True
                    if attempt < max_retries:
                        logger.info(f"COB retry {attempt}/{max_retries} — waiting {retry_delay_secs}s...")
                        time.sleep(retry_delay_secs)

            logger.error(f"❌ COB not closed after {max_retries} retries for {target_day}")
            return False
        except Exception as e:
            logger.error(f"COB wait error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 5. CAST EXPRESSION ───────────────────────
    @staticmethod
    def cast_expression(col_name: str, bq_type: str) -> pl.Expr:
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
    # def transform_to_polars(self, csv_text, schema, lookback_range,
    #                         report_type_value=None):
    #     """
    #     Parse CSV → Polars DataFrame, add metadata, cast types.
    #     report_type_value: e.g. "reporting_api_audience_reach"
    #     """
    #     try:
    #         df = pl.read_csv(
    #             io.StringIO(csv_text),
    #             infer_schema_length=0,
    #             null_values=["", "null", "NULL", "None", "N/A"],
    #         )
    #         logger.info(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
    #         logger.info(f"Columns from Yahoo: {df.columns}")

    #         # Add metadata columns
    #         ingestion_ts = pendulum.now("UTC").to_iso8601_string()
    #         meta_exprs = [
    #             pl.lit(ingestion_ts).alias("ingestion_timestamp"),
    #             pl.lit(lookback_range).alias("lookback_date_range"),
    #         ]
    #         if report_type_value:
    #             meta_exprs.append(pl.lit(report_type_value).alias("report_type"))
    #         df = df.with_columns(meta_exprs)

    #         # Reconcile & cast
    #         type_map = {f["name"]: f["type"] for f in schema}
    #         schema_cols = set(type_map)
    #         df_cols = set(df.columns)

    #         missing = schema_cols - df_cols
    #         extra = df_cols - schema_cols
    #         if missing:
    #             logger.warning(f"Schema cols missing in data: {missing}")
    #         if extra:
    #             logger.info(f"Dropping cols not in schema: {extra}")

    #         keep = [c for c in df.columns if c in schema_cols]
    #         exprs = [self.cast_expression(c, type_map[c]) for c in keep]
    #         df = df.select(exprs)

    #         logger.success(f"Transform done | {df.shape[0]} rows × {df.shape[1]} cols")
    #         return df
    #     except Exception as e:
    #         logger.error(f"Transform error: {e}\n{traceback.format_exc()}")
    #         raise
    
    # def transform_to_polars(self, csv_text, schema, lookback_range,
    #                         report_type_value=None):
    #     try:
    #         df = pl.read_csv(
    #             io.StringIO(csv_text),
    #             infer_schema_length=0,
    #             null_values=["", "null", "NULL", "None", "N/A"],
    #         )
    #         logger.info(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
    #         logger.info(f"Columns from Yahoo: {df.columns}")

    #         # ── Sanitize column names for BQ compatibility ──
    #         rename_map = {}
    #         for col in df.columns:
    #             safe = sanitize_bq_column_name(col)
    #             if safe != col:
    #                 rename_map[col] = safe
    #         if rename_map:
    #             logger.info(f"Renaming {len(rename_map)} columns for BQ compatibility:")
    #             for old, new in rename_map.items():
    #                 logger.info(f"  '{old}' → '{new}'")
    #             df = df.rename(rename_map)

    #         # ── Also sanitize schema field names to match ──
    #         sanitized_schema = []
    #         for f in schema:
    #             sanitized_schema.append({
    #                 **f,
    #                 "name": sanitize_bq_column_name(f["name"]),
    #             })

    #         # Add metadata columns
    #         ingestion_ts = pendulum.now("UTC").to_iso8601_string()
    #         meta_exprs = [
    #             pl.lit(ingestion_ts).alias("ingestion_timestamp"),
    #             pl.lit(lookback_range).alias("lookback_date_range"),
    #         ]
    #         if report_type_value:
    #             meta_exprs.append(pl.lit(report_type_value).alias("report_type"))
    #         df = df.with_columns(meta_exprs)

    #         # Reconcile & cast using sanitized schema
    #         type_map = {f["name"]: f["type"] for f in sanitized_schema}
    #         schema_cols = set(type_map)
    #         df_cols = set(df.columns)

    #         missing = schema_cols - df_cols
    #         extra = df_cols - schema_cols
    #         if missing:
    #             logger.warning(f"Schema cols missing in data: {missing}")
    #         if extra:
    #             logger.info(f"Dropping cols not in schema: {extra}")

    #         keep = [c for c in df.columns if c in schema_cols]
    #         exprs = [self.cast_expression(c, type_map[c]) for c in keep]
    #         df = df.select(exprs)

    #         logger.success(f"Transform done | {df.shape[0]} rows × {df.shape[1]} cols")
    #         return df

    #     except Exception as e:
    #         logger.error(f"Transform error: {e}\n{traceback.format_exc()}")
    #         raise
    
    def transform_to_polars(self, csv_text, schema, lookback_range,
                            report_type_value=None):
        try:
            df = pl.read_csv(
                io.StringIO(csv_text),
                infer_schema_length=0,
                null_values=["", "null", "NULL", "None", "N/A"],
            )
            logger.info(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
            logger.info(f"Columns from Yahoo: {df.columns}")

            # ── Sanitize column names for BQ compatibility ──
            rename_map = {}
            for col in df.columns:
                safe = sanitize_bq_column_name(col)
                if safe != col:
                    rename_map[col] = safe
            if rename_map:
                logger.info(f"Renaming {len(rename_map)} columns for BQ compatibility:")
                for old, new in rename_map.items():
                    logger.info(f"  '{old}' → '{new}'")
                df = df.rename(rename_map)

            # ── Sanitize schema field names to match ──
            sanitized_schema = []
            for f in schema:
                sanitized_schema.append({
                    **f,
                    "name": sanitize_bq_column_name(f["name"]),
                })

            # ── Add metadata columns ──
            ingestion_ts = pendulum.now("UTC").to_iso8601_string()
            meta_exprs = [
                pl.lit(ingestion_ts).alias("ingestion_timestamp"),
                pl.lit(lookback_range).alias("lookback_date_range"),
            ]
            if report_type_value:
                meta_exprs.append(pl.lit(report_type_value).alias("report_type"))
            df = df.with_columns(meta_exprs)

            # ── Build type map from known schema ──
            type_map = {f["name"]: f["type"] for f in sanitized_schema}
            schema_cols = set(type_map)
            df_cols = set(df.columns)

            missing = schema_cols - df_cols
            extra = df_cols - schema_cols

            if missing:
                logger.warning(f"Schema cols missing in data: {missing}")

            # ── Cast known columns by their BQ type ──
            cast_exprs = []
            for c in df.columns:
                if c in type_map:
                    cast_exprs.append(self.cast_expression(c, type_map[c]))
                else:
                    # NEW / unknown column → keep as STRING (dynamic schema)
                    cast_exprs.append(pl.col(c).cast(pl.Utf8).alias(c))

            if extra:
                logger.info(
                    f"New columns not in base schema (will be added dynamically): "
                    f"{extra}"
                )

            df = df.select(cast_exprs)

            logger.success(f"Transform done | {df.shape[0]} rows × {df.shape[1]} cols")
            return df

        except Exception as e:
            logger.error(f"Transform error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 7. RUN SINGLE GROUP ──────────────────────
    def run_report_group(self, report_group, config_overrides,
                         output_dir="output", api_type=API_TYPE):
        """
        Execute the full pipeline for a single report group.

        Returns:
            {"df": pl.DataFrame, "filepath": str, "day_str": str}
            or None on error
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
            logger.info(f"Payload built — dims={len(dims)}, metrics={len(mets)}, dataSource={ds}")

            # 3. Create report
            report_id = self.create_report(payload)

            # 4. Poll
            status_resp = self.poll_report_status(report_id)
            url = status_resp.get("url")
            if not url:
                raise Exception("Report succeeded but no download URL returned.")

            # 5. Download
            csv_text = self.download_report_csv(url)

            # 6. Transform (with report_type injection)
            lookback = report_group.date_range_label(payload)
            report_type_value = f"{api_type}_{report_group.REPORT_NAME}"
            df = self.transform_to_polars(csv_text, schema, lookback, report_type_value)

            if df.is_empty():
                logger.warning(f"[{name}] 0 rows returned — nothing to save.")
                return {"df": df, "filepath": None, "day_str": None}

            # 7. Save CSV — {output_dir}/{report_name}/{day}/{report_name}_{day}_{epoch}.csv
            date_type = payload.get("dateTypeId")
            if date_type == 11:
                day_str = payload.get("startDate", "")[:10]
            elif date_type == 2:
                tz_str = payload.get("reportOption", {}).get("timezone", "America/New_York")
                day_str = pendulum.now(tz_str).subtract(days=1).format("YYYY-MM-DD")
            else:
                day_str = pendulum.now("UTC").format("YYYY-MM-DD")

            epoch = int(time.time())
            group_dir = os.path.join(output_dir, report_group.REPORT_NAME, day_str)
            os.makedirs(group_dir, exist_ok=True)
            filename = f"{report_group.REPORT_NAME}_{day_str}_{epoch}.csv"
            filepath = os.path.join(group_dir, filename)
            df.write_csv(filepath)
            abs_path = os.path.abspath(filepath)

            elapsed = time.time() - t0
            logger.success(
                f"✅ [{name}] DONE | {df.shape[0]} rows × {df.shape[1]} cols "
                f"| {elapsed:.1f}s → {abs_path}"
            )
            return {"df": df, "filepath": filepath, "day_str": day_str}

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"❌ [{name}] FAILED after {elapsed:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            return None

    # ─────────────── 8. RUN ALL GROUPS ────────────────────────
    def run_all_groups(self, report_groups, config_overrides,
                       output_dir="output", skip_cob=False,
                       advertiser_ids_for_cob=None, api_type=API_TYPE):
        """
        Run every report group in sequence.
        Returns {report_name: {"df":..,"filepath":..,"day_str":..} | None}
        """
        results: dict[str, Optional[dict]] = {}

        # ── COB check ──
        adv_ids = advertiser_ids_for_cob or []
        if not skip_cob and adv_ids:
            tz = config_overrides.get("timezone", "America/New_York")
            yesterday = pendulum.now(tz).subtract(days=1).format("YYYYMMDD")
            logger.info(f"📚 COB check for {yesterday} ({len(adv_ids)} advertiser(s))...")
            cob_ok = self.wait_for_cob(adv_ids, yesterday)
            if not cob_ok:
                logger.error("❌ Books not closed. Use skip_cob=True to force.")
                return results
        elif not adv_ids and not skip_cob:
            logger.warning("⚠️ No advertiser IDs — skipping COB check.")

        # ── Run groups ──
        group_names = [g.REPORT_NAME for g in report_groups]
        logger.info(f"🚀 Running {len(report_groups)} report group(s) with names as {group_names} ")

        with tqdm(total=len(report_groups), desc="📊 Report Groups", unit="group") as pbar:
            for grp in report_groups:
                result = self.run_report_group(grp, config_overrides, output_dir, api_type)
                results[grp.REPORT_NAME] = result
                pbar.update(1)

        # ── Summary ──
        ok = sum(
            1 for v in results.values()
            if v is not None and v["df"] is not None and not v["df"].is_empty()
        )
        fail = len(results) - ok
        logger.info("=" * 60)
        logger.info(f"📋 SUMMARY: {ok} succeeded, {fail} failed/empty out of {len(results)}")
        for name, res in results.items():
            if res is not None and res["df"] is not None:
                rows = res["df"].shape[0]
            else:
                rows = "ERROR"
            logger.info(f"   {name:30s} → {rows} rows")
        logger.info("=" * 60)

        return results


if __name__ == "__main__":
    cls_obj = YahooReporting()
    cls_obj.list_all_advertisers()