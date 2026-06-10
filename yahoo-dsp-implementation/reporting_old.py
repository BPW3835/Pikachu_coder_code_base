"""
reporting.py
────────────
Yahoo DSP Reporting API module.
Handles: payload validation (Pydantic), report creation,
status polling, CSV download, Polars transformation,
and end-to-end pipeline orchestration.
"""

import io
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
    timezone: str = "UTC"
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
        """startDate & endDate are mandatory when dateTypeId = 11."""
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

    Pipeline steps:
      1. Validate payload (Pydantic)
      2. POST  /extreport/           → create report
      3. GET   /extreport/{id}       → poll until SUCCESS / FAILED
      4. Download CSV from temp URL
      5. Parse CSV → Polars DataFrame (all Utf8)
      6. Cast columns per BQ schema; drop extras
      7. Export DF → CSV bytes (in-memory)
      8. Upload CSV → GCS
      9. Load GCS → BigQuery (strict schema)
    """

    # Status constants
    _SUCCESS = "SUCCESS"
    _FAILED = "FAILED"

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
            self.poll_interval = 15       # seconds (initial)
            self.poll_max_attempts = 80   # ~20 min ceiling
            self.poll_backoff = 1.2       # multiplicative backoff
            self.poll_max_wait = 120      # cap per-cycle wait

            logger.success("YahooReporting initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 1. VALIDATE ─────────────────────

    def build_payload(self, config: dict) -> dict:
        """Validate raw config via Pydantic and return a clean dict."""
        try:
            logger.info("Validating report payload...")
            payload = ReportPayload(**config)
            result = payload.model_dump(exclude_none=True)
            logger.success("Payload validated.")
            return result
        except Exception as e:
            logger.error(f"Payload validation failed: {e}")
            raise

    # ───────────────────── 2. CREATE REPORT ────────────────

    def create_report(self, payload: dict) -> str:
        """
        POST /extreport/ — submit report creation request.
        Returns the customerReportId for subsequent polling.
        """
        try:
            token = self.helper.get_access_token()
            headers = self.helper.build_auth_headers(token)

            logger.info("Creating report...")
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

    # ───────────────────── 3. POLL STATUS ──────────────────

    def poll_status(self, report_id: str) -> dict:
        """
        GET /extreport/{customerReportId} — poll with exponential
        backoff + tqdm progress bar until SUCCESS or FAILED.
        Returns the full status response (contains download URL on success).
        """
        try:
            logger.info(f"Polling report {report_id}...")
            wait = self.poll_interval

            with tqdm(
                total=self.poll_max_attempts,
                desc="⏳ Polling",
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
                            f"Attempt {attempt}: HTTP {resp.status_code}"
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
                        raise Exception(f"Report FAILED: {body}")

                    # SUBMITTED / PROCESSING — wait & retry
                    logger.info(
                        f"Attempt {attempt}/{self.poll_max_attempts} "
                        f"| {status} | next in {wait:.0f}s"
                    )
                    time.sleep(wait)
                    wait = min(wait * self.poll_backoff, self.poll_max_wait)

            raise TimeoutError(
                f"Report {report_id} not ready after {self.poll_max_attempts} attempts."
            )
        except Exception as e:
            logger.error(f"Polling error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 4. DOWNLOAD CSV ─────────────────

    def download_report(self, download_url: str) -> str:
        """
        Download the report CSV from the temporary URL.
        Returns raw CSV text.
        """
        try:
            logger.info("Downloading report CSV...")
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

    # ───────────────────── 5. PARSE → POLARS ───────────────

    def parse_csv_to_df(self, csv_text: str) -> pl.DataFrame:
        """
        Parse raw CSV into a Polars DataFrame.
        All columns read as Utf8 (safe; casting happens next step).
        """
        try:
            logger.info("Parsing CSV → Polars DataFrame...")
            df = pl.read_csv(
                io.StringIO(csv_text),
                infer_schema_length=0,          # everything as Utf8
                null_values=["", "null", "NULL", "None", "N/A"],
            )
            logger.success(f"Parsed | {df.shape[0]} rows × {df.shape[1]} cols")
            return df
        except Exception as e:
            logger.error(f"Parse error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 6. CAST TO BQ TYPES ─────────────

    @staticmethod
    def _cast_expressions(
        col_name: str, bq_type: str
    ) -> pl.Expr:
        """
        Return a Polars expression that casts a Utf8 column
        to the target BigQuery-compatible type.
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
                pl.when(
                    pl.col(col_name).str.to_lowercase().is_in(["true", "1"])
                )
                .then(True)
                .when(
                    pl.col(col_name).str.to_lowercase().is_in(["false", "0"])
                )
                .then(False)
                .otherwise(None)
                .alias(col_name)
            )

        if bq_type == "DATE":
            return pl.col(col_name).str.to_date(strict=False).alias(col_name)

        if bq_type in ("DATETIME", "TIMESTAMP"):
            return (
                pl.col(col_name).str.to_datetime(strict=False).alias(col_name)
            )

        # Fallback — keep as string
        logger.warning(
            f"Unknown BQ type '{bq_type}' for '{col_name}'; keeping Utf8."
        )
        return pl.col(col_name).cast(pl.Utf8).alias(col_name)

    def cast_df_to_bq_types(
        self, df: pl.DataFrame, schema_fields: list[dict]
    ) -> pl.DataFrame:
        """
        Cast DataFrame columns per BQ schema.
        Columns NOT in schema → dropped.
        Missing schema columns → warned (not added).
        """
        try:
            logger.info("Casting columns to BQ-compatible types...")
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
            exprs = [self._cast_expressions(c, type_map[c]) for c in keep]
            df = df.select(exprs)

            logger.success(f"Cast complete | {df.shape[0]} rows × {df.shape[1]} cols")
            return df
        except Exception as e:
            logger.error(f"Cast error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 7. DF → CSV BYTES ───────────────

    @staticmethod
    def df_to_csv_bytes(df: pl.DataFrame) -> bytes:
        """Convert Polars DataFrame to CSV bytes (in-memory, no disk I/O)."""
        try:
            logger.info("Exporting DataFrame → CSV bytes...")
            buf = io.BytesIO()
            df.write_csv(buf)
            data = buf.getvalue()
            logger.success(f"CSV ready | {len(data):,} bytes")
            return data
        except Exception as e:
            logger.error(f"CSV export error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────── 8. CLOSE OF BOOKS ───────────────

    def check_close_of_books(
        self, advertiser_id: int, day: str
    ) -> dict:
        """
        GET /extreport/cob — verify books are closed for a
        given advertiser + day (YYYYMMDD format).
        Returns {"advertiserTimezone": "...", "dayClosed": "true"/"false"}.
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

    # ───────────────────── 9. ORCHESTRATOR ─────────────────

    def run_pipeline(
        self,
        report_config: dict,
        schema_fields: list[dict],
        gcs_bucket: str,
        gcs_blob_path: str,
        bq_table_id: str,
        write_disposition: str = "WRITE_APPEND",
    ) -> None:
        """
        Full end-to-end pipeline:
        Validate → Create → Poll → Download → Parse → Cast → CSV → GCS → BQ.

        Args:
            report_config:     Raw config dict (validated by Pydantic).
            schema_fields:     BQ schema as list of dicts.
            gcs_bucket:        Target GCS bucket.
            gcs_blob_path:     Blob path inside the bucket.
            bq_table_id:       Fully-qualified BQ table (project.dataset.table).
            write_disposition: WRITE_APPEND | WRITE_TRUNCATE | WRITE_EMPTY.
        """
        t0 = time.time()
        try:
            logger.info("=" * 60)
            logger.info("🚀  YAHOO DSP REPORTING PIPELINE — START")
            logger.info("=" * 60)

            # 1. Validate
            payload = self.build_payload(report_config)

            # 2. Create report
            report_id = self.create_report(payload)

            # 3. Poll until ready
            status_resp = self.poll_status(report_id)
            url = status_resp.get("url")
            if not url:
                raise Exception("Report succeeded but no download URL returned.")

            # 4. Download CSV
            csv_text = self.download_report(url)

            # 5. Parse → Polars
            df = self.parse_csv_to_df(csv_text)
            if df.is_empty():
                logger.warning("0 rows returned — skipping GCS/BQ load.")
                return

            # 6. Cast types
            df = self.cast_df_to_bq_types(df, schema_fields)

            # 7. DF → CSV bytes
            csv_bytes = self.df_to_csv_bytes(df)

            # 8. Upload → GCS
            gcs_uri = self.helper.upload_bytes_to_gcs(
                bucket_name=gcs_bucket,
                blob_path=gcs_blob_path,
                data=csv_bytes,
            )

            # 9. GCS → BigQuery
            self.helper.load_gcs_to_bq(
                gcs_uri=gcs_uri,
                table_id=bq_table_id,
                schema_fields=schema_fields,
                write_disposition=write_disposition,
            )

            elapsed = time.time() - t0
            logger.info("=" * 60)
            logger.success(
                f"✅  PIPELINE COMPLETE | {elapsed:.1f}s | "
                f"{df.shape[0]} rows → {bq_table_id}"
            )
            logger.info("=" * 60)

        except Exception as e:
            logger.error(
                f"❌  PIPELINE FAILED after {time.time() - t0:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            raise