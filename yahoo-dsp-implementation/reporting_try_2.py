"""
reporting.py
────────────
Yahoo DSP Reporting API module.
Handles: payload validation (Pydantic), report creation,
status polling, CSV download, Polars transformation,
and end-to-end pipeline orchestration.

GCS/BQ load steps are commented out for local development.
"""

import io
import os
import time
import traceback
import requests
import polars as pl
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from loguru import logger
from tqdm import tqdm

from helpers import YahooHelpers


# ═══════════════════════════════════════════════════════
# PYDANTIC MODELS — Payload Validation
# ═══════════════════════════════════════════════════════


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
    limit: int = 10000
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
    Validates all fields per Yahoo DSP API spec.

    dateTypeId reference:
        1=Today  2=Yesterday  3=This Week  4=Last 7 Days
        5=Last Week  6=Last 14 Days  7=MTD  8=Last 30 Days
        9=Last Month  10=All Time  11=Custom Range

    intervalTypeId reference:
        1=Cumulative  2=Day  3=Month  4=Week  5=Hour
    """
    reportOption: ReportOption
    dateTypeId: int = Field(..., ge=1, le=11)
    intervalTypeId: int = Field(..., ge=1, le=5)
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    dataSource: Optional[int] = None            # 15 for bidding reports

    @model_validator(mode="after")
    def _custom_range_requires_dates(self):
        """startDate & endDate are mandatory when dateTypeId = 11 (Custom Range)."""
        if self.dateTypeId == 11 and (not self.startDate or not self.endDate):
            raise ValueError(
                "startDate and endDate are required when dateTypeId=11 (Custom Range)"
            )
        return self


# ═══════════════════════════════════════════════════════
# YAHOO REPORTING CLASS
# ═══════════════════════════════════════════════════════


class YahooReporting:
    """
    End-to-end Yahoo DSP Reporting API handler.

    Pipeline steps (local mode):
      1. Validate payload (Pydantic)
      2. POST  /extreport/           → create report
      3. GET   /extreport/{id}       → poll until SUCCESS / FAILED
      4. Download CSV from temp URL
      5. Parse CSV → Polars DataFrame (all Utf8 initially)
      6. Cast columns per BQ schema; drop columns not in schema
      7. Export DF → CSV file locally
      # 8. (COMMENTED) Upload CSV → GCS
      # 9. (COMMENTED) Load GCS → BigQuery
    """

    # Report status constants returned by Yahoo API
    _SUCCESS = "SUCCESS"
    _FAILED = "FAILED"

    def __init__(self):
        try:
            # Yahoo DSP Reporting API endpoints
            self.report_endpoint = (
                "http://api-sched-v3.admanagerplus.yahoo.com"
                "/yamplus_api/extreport/"
            )
            self.cob_endpoint = (
                "http://api-sched-v3.admanagerplus.yahoo.com"
                "/yamplus_api/extreport/cob"
            )

            # Shared helper for auth, GCS, BQ
            self.helper = YahooHelpers()

            # Polling configuration for report status check
            self.poll_interval = 15       # seconds between first poll
            self.poll_max_attempts = 80   # max polls (~20 min ceiling)
            self.poll_backoff = 1.2       # multiplicative backoff factor
            self.poll_max_wait = 120      # max seconds between polls

            logger.success("YahooReporting initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 1. VALIDATE PAYLOAD ───────────────

    def build_payload(self, config: dict) -> dict:
        """
        Validate raw config dict through Pydantic models.
        Returns a clean dict ready for the Yahoo API POST body.
        Raises ValidationError if config is invalid.
        """
        try:
            logger.info("Validating report payload via Pydantic...")
            payload = ReportPayload(**config)
            result = payload.model_dump(exclude_none=True)
            logger.success("Payload validated successfully.")
            return result
        except Exception as e:
            logger.error(f"Payload validation failed: {e}")
            raise

    # ─────────────── 2. CREATE REPORT ──────────────────

    def create_report(self, payload: dict) -> str:
        """
        POST /extreport/ — submit the report creation request to Yahoo.
        Returns the customerReportId (UUID) for subsequent polling.
        """
        try:
            token = self.helper.get_access_token()
            headers = self.helper.build_auth_headers(token)

            logger.info("Creating report via POST /extreport/ ...")
            resp = requests.post(
                self.report_endpoint,
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code not in (200, 201, 202):
                raise Exception(
                    f"Create report failed [{resp.status_code}]: {resp.text}"
                )

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

    # ─────────────── 3. POLL STATUS ────────────────────

    def poll_status(self, report_id: str) -> dict:
        """
        GET /extreport/{customerReportId} — poll with exponential backoff.
        Uses tqdm progress bar for visual feedback.

        Returns full status response dict on SUCCESS (contains download URL).
        Raises Exception on FAILED or timeout.
        """
        try:
            logger.info(f"Polling report status for ID: {report_id} ...")
            wait = self.poll_interval

            with tqdm(
                total=self.poll_max_attempts,
                desc="⏳ Polling report status",
                unit="req",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ) as pbar:
                for attempt in range(1, self.poll_max_attempts + 1):
                    # Refresh token if needed (helper handles caching)
                    token = self.helper.get_access_token()
                    headers = self.helper.build_auth_headers(token)

                    resp = requests.get(
                        f"{self.report_endpoint}{report_id}",
                        headers=headers,
                        timeout=60,
                    )

                    # Handle non-200 (transient network issue etc.)
                    if resp.status_code != 200:
                        logger.warning(
                            f"Attempt {attempt}: HTTP {resp.status_code} — retrying..."
                        )
                        pbar.update(1)
                        time.sleep(wait)
                        wait = min(wait * self.poll_backoff, self.poll_max_wait)
                        continue

                    body = resp.json()
                    status = body.get("status", "").upper()
                    pbar.set_postfix_str(f"Status={status}")
                    pbar.update(1)

                    if status == self._SUCCESS:
                        logger.success(
                            f"Report ready | Rows: {body.get('numRows', '?')}"
                        )
                        return body

                    if status == self._FAILED:
                        raise Exception(f"Report generation FAILED: {body}")

                    # SUBMITTED / PROCESSING — wait and retry with backoff
                    logger.info(
                        f"Attempt {attempt}/{self.poll_max_attempts} "
                        f"| Status: {status} | Retrying in {wait:.0f}s"
                    )
                    time.sleep(wait)
                    wait = min(wait * self.poll_backoff, self.poll_max_wait)

            raise TimeoutError(
                f"Report {report_id} did not complete after "
                f"{self.poll_max_attempts} attempts."
            )
        except Exception as e:
            logger.error(f"Polling error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 4. DOWNLOAD CSV ───────────────────

    def download_report(self, download_url: str) -> str:
        """
        Download report CSV from the temporary Yahoo-provided URL.
        URL is valid for 24 hours after report generation.
        Returns raw CSV text.
        """
        try:
            logger.info("Downloading report CSV from temporary URL...")
            resp = requests.get(download_url, timeout=300)
            if resp.status_code != 200:
                raise Exception(
                    f"Download failed [{resp.status_code}]: {resp.text[:500]}"
                )
            logger.success(f"Downloaded | Size: {len(resp.text):,} chars")
            return resp.text
        except Exception as e:
            logger.error(f"Download error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 5. PARSE CSV → POLARS ─────────────

    def parse_csv_to_df(self, csv_text: str) -> pl.DataFrame:
        """
        Parse raw CSV text into a Polars DataFrame.
        infer_schema_length=0 reads all columns as Utf8 (string).
        Explicit casting happens in the next step.
        """
        try:
            logger.info("Parsing CSV → Polars DataFrame (all Utf8)...")
            df = pl.read_csv(
                io.StringIO(csv_text),
                infer_schema_length=0,          # everything as Utf8 — safe
                null_values=["", "null", "NULL", "None", "N/A"],
            )
            logger.success(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
            logger.info(f"Columns: {df.columns}")
            return df
        except Exception as e:
            logger.error(f"Parse error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 6. CAST TO BQ-COMPATIBLE TYPES ────

    @staticmethod
    def _cast_expression(col_name: str, bq_type: str) -> pl.Expr:
        """
        Return a Polars expression to cast a Utf8 column to the
        target BigQuery-compatible Polars type.
        """
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

        # Fallback — unknown type, keep as Utf8
        logger.warning(f"Unknown BQ type '{bq_type}' for '{col_name}'; keeping Utf8.")
        return pl.col(col_name).cast(pl.Utf8).alias(col_name)

    def cast_df_to_bq_types(
        self, df: pl.DataFrame, schema_fields: list[dict]
    ) -> pl.DataFrame:
        """
        Cast DataFrame columns to BQ-compatible types per schema.
        - Columns NOT in schema → dropped (extras logged)
        - Schema columns NOT in data → warned (not fabricated)
        """
        try:
            logger.info("Casting columns to BQ-compatible types...")
            type_map = {f["name"]: f["type"] for f in schema_fields}
            schema_cols = set(type_map)
            df_cols = set(df.columns)

            missing = schema_cols - df_cols
            extra = df_cols - schema_cols
            if missing:
                logger.warning(f"Schema cols missing in data (skipped): {missing}")
            if extra:
                logger.info(f"Dropping cols not in schema: {extra}")

            # Only keep columns that exist in both data and schema
            keep = [c for c in df.columns if c in schema_cols]
            exprs = [self._cast_expression(c, type_map[c]) for c in keep]
            df = df.select(exprs)

            logger.success(f"Cast complete | {df.shape[0]} rows × {df.shape[1]} cols")
            return df
        except Exception as e:
            logger.error(f"Cast error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 7. SAVE CSV LOCALLY ───────────────

    @staticmethod
    def save_csv_locally(df: pl.DataFrame, output_path: str) -> str:
        """
        Write Polars DataFrame to a local CSV file.
        Used during local development instead of GCS upload.
        Returns the absolute path of the saved file.
        """
        try:
            logger.info(f"Saving DataFrame to local CSV: {output_path}")
            df.write_csv(output_path)
            abs_path = os.path.abspath(output_path)
            logger.success(f"Saved | {df.shape[0]} rows → {abs_path}")
            return abs_path
        except Exception as e:
            logger.error(f"Local save error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 8. CLOSE OF BOOKS CHECK ───────────

    def check_close_of_books(self, advertiser_id: int, day: str) -> dict:
        """
        GET /extreport/cob — verify data is finalized (books closed)
        for a given advertiser + day.

        Args:
            advertiser_id: Yahoo DSP advertiser ID
            day:           Date string in YYYYMMDD format

        Returns:
            {"advertiserTimezone": "...", "dayClosed": "true"/"false"}
        """
        try:
            logger.info(f"COB check → advertiser={advertiser_id}, day={day}")
            token = self.helper.get_access_token()
            headers = self.helper.build_auth_headers(token)

            resp = requests.get(
                self.cob_endpoint,
                params={"advertiserId": advertiser_id, "day": day},
                headers=headers,
                timeout=60,
            )
            if resp.status_code != 200:
                raise Exception(f"COB failed [{resp.status_code}]: {resp.text}")

            body = resp.json()
            logger.success(f"COB result: {body}")
            return body
        except Exception as e:
            logger.error(f"COB error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 9. PIPELINE ORCHESTRATOR ──────────

    def run_pipeline(
        self,
        report_config: dict,
        schema_fields: list[dict],
        local_output_path: str = "output_report.csv",
        # --- GCS/BQ params (commented out for local dev) ---
        # gcs_bucket: str = "",
        # gcs_blob_path: str = "",
        # bq_table_id: str = "",
        # write_disposition: str = "WRITE_APPEND",
    ) -> pl.DataFrame:
        """
        Full pipeline orchestrator (local mode).

        Steps executed:
          1. Validate payload
          2. Create report (POST)
          3. Poll status (GET with backoff)
          4. Download CSV
          5. Parse → Polars DataFrame
          6. Cast types per BQ schema
          7. Save CSV locally

        Returns the final Polars DataFrame for inspection.
        """
        t0 = time.time()
        try:
            logger.info("=" * 60)
            logger.info("🚀  YAHOO DSP REPORTING PIPELINE — START (LOCAL)")
            logger.info("=" * 60)

            # Step 1: Validate payload
            payload = self.build_payload(report_config)

            # Step 2: Create report on Yahoo side
            report_id = self.create_report(payload)

            # Step 3: Poll until report is ready
            status_resp = self.poll_status(report_id)
            url = status_resp.get("url")
            if not url:
                raise Exception("Report succeeded but no download URL returned.")

            # Step 4: Download CSV from temporary URL
            csv_text = self.download_report(url)

            # Step 5: Parse CSV → Polars DataFrame
            df = self.parse_csv_to_df(csv_text)
            if df.is_empty():
                logger.warning("0 rows returned — nothing to process.")
                return df

            # Step 6: Cast columns per BQ schema (drop extras)
            df = self.cast_df_to_bq_types(df, schema_fields)

            # Step 7: Save CSV locally for inspection
            self.save_csv_locally(df, local_output_path)

            # ─────────────────────────────────────────────
            # Step 8: Upload CSV → GCS (COMMENTED FOR LOCAL DEV)
            # ─────────────────────────────────────────────
            # csv_bytes = df.write_csv().encode("utf-8")  # in-memory bytes
            # gcs_uri = self.helper.upload_bytes_to_gcs(
            #     bucket_name=gcs_bucket,
            #     blob_path=gcs_blob_path,
            #     data=csv_bytes,
            # )

            # ─────────────────────────────────────────────
            # Step 9: Load GCS → BigQuery (COMMENTED FOR LOCAL DEV)
            # ─────────────────────────────────────────────
            # self.helper.load_gcs_to_bq(
            #     gcs_uri=gcs_uri,
            #     table_id=bq_table_id,
            #     schema_fields=schema_fields,
            #     write_disposition=write_disposition,
            # )

            elapsed = time.time() - t0
            logger.info("=" * 60)
            logger.success(
                f"✅  PIPELINE COMPLETE | {elapsed:.1f}s | "
                f"{df.shape[0]} rows → {local_output_path}"
            )
            logger.info("=" * 60)

            return df

        except Exception as e:
            logger.error(
                f"❌  PIPELINE FAILED after {time.time() - t0:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            raise