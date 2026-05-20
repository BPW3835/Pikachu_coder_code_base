"""
reporting.py
────────────
Yahoo DSP Reporting API module.
Handles: payload validation (Pydantic), report creation,
status polling, CSV download, Polars transformation,
Close of Books (COB) check, and pipeline orchestration.

GCS/BQ load steps are commented out for local development.
"""

import io
import os
import time
import traceback
import requests
import pendulum
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

    dateTypeId:
        1=Today  2=Yesterday  3=This Week  4=Last 7 Days
        5=Last Week  6=Last 14 Days  7=MTD  8=Last 30 Days
        9=Last Month  10=All Time  11=Custom Range

    intervalTypeId:
        1=Cumulative  2=Day  3=Month  4=Week  5=Hour

    IMPORTANT: intervalTypeId=2 (Day) is what populates the Day column.
    If intervalTypeId=1 (Cumulative), Day will be EMPTY.
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
      1. (Optional) Close of Books check — ensures data is finalized
      2. Validate payload (Pydantic)
      3. POST  /extreport/           → create report
      4. GET   /extreport/{id}       → poll until SUCCESS / FAILED
      5. Download CSV from temp URL
      6. Parse CSV → Polars DataFrame (all Utf8 initially)
      7. Add metadata columns (ingestion_timestamp, lookback_date_range)
      8. Cast columns per BQ schema; drop columns not in schema
      9. Save CSV locally with timestamped filename
      # 10. (COMMENTED) Upload CSV → GCS
      # 11. (COMMENTED) Load GCS → BigQuery
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

    # Human-readable labels for preset date types
    DATE_TYPE_LABELS: dict[int, str] = {
        1: "Today", 2: "Yesterday", 3: "This_Week",
        4: "Last_7_Days", 5: "Last_Week", 6: "Last_14_Days",
        7: "MTD", 8: "Last_30_Days", 9: "Last_Month", 10: "All_Time",
    }

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

            # Shared helper instance (auth, GCS, BQ utilities)
            self.helper = YahooHelpers()

            # Polling tunables for report status check
            self.poll_interval = 15       # seconds — initial wait
            self.poll_max_attempts = 200   # ~20 min ceiling
            self.poll_backoff = 1.2       # multiplicative backoff
            self.poll_max_wait = 120      # cap per-cycle wait in seconds

            logger.success("YahooReporting initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────

    def extract_date_range_label(self, config: dict) -> str:
        """
        Derive a human-readable date range string from report config.
        Used to populate the 'lookback_date_range' metadata column.

        Returns:
            "ALL_TIME"                      for dateTypeId=10
            "2026-04-29 to 2026-05-05"      for dateTypeId=11 (Custom)
            "Yesterday" / "Last_30_Days"    for preset dateTypeIds
        """
        date_type = config.get("dateTypeId")
        if date_type == 10:
            return "ALL_TIME"
        if date_type == 11:
            start = config.get("startDate", "?")[:10]
            end = config.get("endDate", "?")[:10]
            return f"{start} to {end}"
        return self.DATE_TYPE_LABELS.get(date_type, f"dateTypeId_{date_type}")

    def add_metadata_columns(self, df: pl.DataFrame, report_config: dict) -> pl.DataFrame:
        """
        Append two metadata columns to the DataFrame:
          - ingestion_timestamp  : UTC timestamp of when the pipeline ran
          - lookback_date_range  : human-readable extraction window
        """
        try:
            ingestion_ts = pendulum.now("UTC").to_iso8601_string()
            date_range = self.extract_date_range_label(report_config)

            df = df.with_columns(
                pl.lit(ingestion_ts).alias("ingestion_timestamp"),
                pl.lit(date_range).alias("lookback_date_range"),
            )
            logger.info(
                f"Metadata added | ingestion_timestamp={ingestion_ts} "
                f"| lookback_date_range={date_range}"
            )
            return df
        except Exception as e:
            logger.error(f"Metadata column error: {e}\n{traceback.format_exc()}")
            raise

    @staticmethod
    def _generate_output_filename(prefix: str = "yahoo_dsp_report") -> str:
        """
        Generate a timestamped output filename.
        Example: yahoo_dsp_report_2026-05-06_15-30-45.csv
        """
        ts = pendulum.now("UTC").format("YYYY-MM-DD_HH-mm-ss")
        return f"{prefix}_{ts}.csv"

    # ─────────────── 1. VALIDATE PAYLOAD ───────────────

    def build_payload(self, config: dict) -> dict:
        """
        Validate raw config dict through Pydantic models.
        Returns a clean dict ready for the Yahoo API POST body.
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
        POST /extreport/ — submit report creation request.
        Returns the customerReportId (UUID) for polling.
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
        tqdm progress bar for visual feedback.
        Returns full status dict on SUCCESS (contains download URL).
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
                    token = self.helper.get_access_token()
                    headers = self.helper.build_auth_headers(token)

                    resp = requests.get(
                        f"{self.report_endpoint}{report_id}",
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

                    if status == self.SUCCESS:
                        logger.success(
                            f"Report ready | Rows: {body.get('numRows', '?')}"
                        )
                        return body

                    if status == self.FAILED:
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

    # ─────────────── 4. DOWNLOAD CSV ───────────────────

    def download_report(self, download_url: str) -> str:
        """
        Download CSV from Yahoo's temporary URL (valid 24h).
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
        infer_schema_length=0 → all columns read as Utf8 (safe).
        Explicit casting happens later via cast_df_to_bq_types.
        """
        try:
            logger.info("Parsing CSV → Polars DataFrame (all Utf8)...")
            df = pl.read_csv(
                io.StringIO(csv_text),
                infer_schema_length=0,
                null_values=["", "null", "NULL", "None", "N/A"],
            )
            logger.success(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
            logger.info(f"Columns from Yahoo: {df.columns}")
            return df
        except Exception as e:
            logger.error(f"Parse error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 6. CAST TO BQ-COMPATIBLE TYPES ────

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

    def cast_df_to_bq_types(
        self, df: pl.DataFrame, schema_fields: list[dict]
    ) -> pl.DataFrame:
        """
        Cast columns per BQ schema. Extras → dropped. Missing → warned.
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

            keep = [c for c in df.columns if c in schema_cols]
            exprs = [self._cast_expression(c, type_map[c]) for c in keep]
            df = df.select(exprs)

            logger.success(f"Cast complete | {df.shape[0]} rows × {df.shape[1]} cols")
            return df
        except Exception as e:
            logger.error(f"Cast error: {e}\n{traceback.format_exc()}")
            raise

    # ─────────────── 7. CLOSE OF BOOKS ─────────────────

    def check_close_of_books(self, advertiser_id: int, day: str) -> dict:
        """
        GET /extreport/cob — check if data is finalized for
        a given advertiser + day.

        Args:
            advertiser_id:  Yahoo DSP advertiser ID (numeric)
            day:            Date in YYYYMMDD format

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

    def wait_for_cob(
        self,
        advertiser_ids: list[int],
        target_day: str,
        max_retries: int = 12,
        retry_delay_secs: int = 300,
    ) -> bool:
        """
        Wait for Close of Books to be confirmed for ALL advertisers
        on the target day. Retries with fixed delay.

        Yahoo typically closes books within ~6 hours after day end.
        Default: 12 retries × 5 min = 60 min max wait.

        Args:
            advertiser_ids:    List of Yahoo DSP advertiser IDs
            target_day:        YYYYMMDD format
            max_retries:       Max poll attempts
            retry_delay_secs:  Seconds between retries

        Returns:
            True if all closed, False if timed out
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
                        result = self.check_close_of_books(adv_id, target_day)
                        is_closed = (
                            str(result.get("dayClosed", "false")).lower() == "true"
                        )
                        if not is_closed:
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

    # ─────────────── 8. PIPELINE ORCHESTRATOR ──────────

    def run_pipeline(
        self,
        report_config: dict,
        schema_fields: list[dict],
        output_dir: str = ".",
    ) -> pl.DataFrame:
        """
        Full pipeline orchestrator (local mode).

        Flow:
          1. Validate payload
          2. Create report (POST)
          3. Poll status (GET with backoff)
          4. Download CSV
          5. Parse → Polars
          6. Add metadata columns (ingestion_timestamp, lookback_date_range)
          7. Cast types per BQ schema
          8. Save CSV locally (timestamped filename)
          # 9.  (COMMENTED) Upload CSV → GCS
          # 10. (COMMENTED) Load GCS → BigQuery

        Args:
            report_config:  Raw config dict (validated by Pydantic)
            schema_fields:  BQ schema as list of dicts
            output_dir:     Local directory for CSV output

        Returns:
            Final Polars DataFrame for inspection.
        """
        t0 = time.time()
        try:
            logger.info("=" * 60)
            logger.info("🚀  YAHOO DSP REPORTING PIPELINE — START (LOCAL)")
            logger.info("=" * 60)

            # Step 1: Validate payload
            payload = self.build_payload(report_config)

            # Step 2: Create report
            report_id = self.create_report(payload)

            # Step 3: Poll until ready
            status_resp = self.poll_status(report_id)
            url = status_resp.get("url")
            if not url:
                raise Exception("Report succeeded but no download URL returned.")

            # Step 4: Download CSV
            csv_text = self.download_report(url)

            # Step 5: Parse → Polars
            df = self.parse_csv_to_df(csv_text)
            if df.is_empty():
                logger.warning("0 rows returned — nothing to process.")
                return df

            # Step 6: Add metadata columns
            df = self.add_metadata_columns(df, report_config)

            # Step 7: Cast types per BQ schema
            df = self.cast_df_to_bq_types(df, schema_fields)

            # Step 8: Save CSV locally with timestamped filename
            output_filename = self._generate_output_filename()
            output_path = os.path.join(output_dir, output_filename)
            df.write_csv(output_path)
            abs_path = os.path.abspath(output_path)
            logger.success(f"CSV saved → {abs_path}")

            # ─────────────────────────────────────────────
            # Step 9: Upload CSV → GCS (COMMENTED FOR LOCAL DEV)
            # ─────────────────────────────────────────────
            # csv_bytes = df.write_csv().encode("utf-8")
            # gcs_uri = self.helper.upload_bytes_to_gcs(
            #     bucket_name=gcs_bucket,
            #     blob_path=gcs_blob_path,
            #     data=csv_bytes,
            # )

            # ─────────────────────────────────────────────
            # Step 10: Load GCS → BigQuery (COMMENTED FOR LOCAL DEV)
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
                f"{df.shape[0]} rows → {abs_path}"
            )
            logger.info("=" * 60)

            return df

        except Exception as e:
            logger.error(
                f"❌  PIPELINE FAILED after {time.time() - t0:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            raise