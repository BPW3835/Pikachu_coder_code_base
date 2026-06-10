"""
helpers.py
──────────
Universal helper class for the Yahoo DSP pipeline.
Contains: Authentication (JWT + OAuth2), GCS operations,
BigQuery operations (dynamic schema evolution), and common utilities.
Reusable across all API modules (reporting, traffic, etc.).
"""

import os
import csv
import json
import time
import base64
import hmac
import hashlib
import requests
import pendulum
import traceback
from loguru import logger
from google.cloud import secretmanager, bigquery, storage
import re

def sanitize_bq_column_name(name: str) -> str:
    """
    Sanitize a column name for BigQuery compatibility.
    BQ allows: letters, numbers, underscores only.
    Must start with letter or underscore. Max 300 chars.
    """
    # Meaningful replacements before generic cleanup
    name = name.replace("%", "_pct")
    name = name.replace("&", "_and_")
    name = name.replace("+", "_plus_")
    # Replace everything else non-alphanumeric with underscore
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    # Must start with letter or underscore
    if name and name[0].isdigit():
        name = f"_{name}"
    return name[:300]

class YahooHelpers:
    """
    Centralised helper providing auth, cloud storage,
    BigQuery load, and utility methods for every Yahoo DSP module.
    """

    def __init__(self):
        self.token_url = "https://id.b2b.yahooincapis.com/zts/v1/oauth2/token"
        self.traffic_base = "https://dspapi.admanagerplus.yahoo.com"
        self.page_limit = 100
        self.max_pages = 500
        self.start_time = str(pendulum.now("America/New_York"))

        self.project_id = os.environ.get("PROJECT_ID", "")
        self.client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
        self.client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()

        # GCS / BQ config from environment
        self.gcs_bucket = os.environ.get("GCS_BUCKET", "")
        self.bq_dataset = os.environ.get("BQ_DATASET", "")

        # Token cache
        self._access_token: str | None = None
        self._token_expiry: float = 0.0

    # ───────────────────────────────────────────────────────
    # AUTHENTICATION
    # ───────────────────────────────────────────────────────

    def _b64url_encode(self, data: bytes) -> str:
        """Base64 URL-safe encode with padding stripped."""
        try:
            return base64.urlsafe_b64encode(data).decode().rstrip("=")
        except Exception as e:
            logger.error(f"Error in _b64url_encode: {e}")
            raise

    def get_secret(self, secret_name: str, project_id: str) -> dict:
        """Retrieve and parse a JSON secret from Google Secret Manager."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": secret_path})
            secret_data = json.loads(response.payload.data.decode("UTF-8"))
            logger.success(f"Secret '{secret_name}' retrieved successfully.")
            return secret_data
        except Exception as e:
            logger.error(f"Error in get_secret: {e}\n{traceback.format_exc()}")
            raise

    def _build_jwt(self, client_id: str, client_secret: str) -> str:
        """Build an HS256 JWT for Yahoo DSP OAuth2 client-credentials grant."""
        try:
            logger.info("Building JWT for authentication...")
            header = {"alg": "HS256", "typ": "JWT"}
            now = int(time.time())
            payload = {
                "aud": "https://id.b2b.yahooincapis.com/zts/v1",
                "iss": f"idb2b.dsp.dspapi.{client_id}",
                "sub": f"idb2b.dsp.dspapi.{client_id}",
                "iat": now,
                "exp": now + 300,
            }
            h_b64 = self._b64url_encode(json.dumps(header).encode())
            p_b64 = self._b64url_encode(json.dumps(payload).encode())
            signing_input = f"{h_b64}.{p_b64}".encode()
            sig = hmac.new(
                client_secret.encode(), signing_input, hashlib.sha256
            ).digest()
            jwt_token = f"{h_b64}.{p_b64}.{self._b64url_encode(sig)}"
            logger.success("JWT built successfully.")
            return jwt_token
        except Exception as e:
            logger.error(f"Error in _build_jwt: {e}\n{traceback.format_exc()}")
            raise

    def get_access_token(self) -> str:
        """Obtain (or return cached) OAuth2 access token."""
        try:
            if self._access_token and time.time() < (self._token_expiry - 30):
                logger.info("Reusing cached access token.")
                return self._access_token

            logger.info("Requesting new access token...")
            jwt = self._build_jwt(self.client_id, self.client_secret)
            resp = requests.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "scope": "api-client",
                    "client_assertion_type": (
                        "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
                    ),
                    "client_assertion": jwt,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                raise Exception(f"Token request failed: {resp.status_code} {resp.text}")

            body = resp.json()
            token = body.get("access_token")
            if not token:
                raise Exception(f"No access_token in response: {resp.text}")

            self._access_token = token
            self._token_expiry = time.time() + body.get("expires_in", 300)
            logger.success("Access token retrieved successfully.")
            return self._access_token
        except Exception as e:
            logger.error(f"Error in get_access_token: {e}\n{traceback.format_exc()}")
            raise

    def build_auth_headers(self, access_token: str) -> dict:
        """Standard Yahoo DSP API request headers."""
        return {
            "Content-Type": "application/json",
            "X-Auth-Method": "OAuth2",
            "X-Auth-Token": access_token,
            "Accept": "application/json",
        }

    # ───────────────────────────────────────────────────────
    # GCS METHODS
    # ───────────────────────────────────────────────────────

    def upload_bytes_to_gcs(self, bucket_name, blob_path, data,
                            content_type="text/csv"):
        """Upload in-memory bytes to GCS. Returns gs:// URI."""
        try:
            gcs_uri = f"gs://{bucket_name}/{blob_path}"
            logger.info(f"Uploading bytes to {gcs_uri} ...")
            client = storage.Client(project=self.project_id)
            blob = client.bucket(bucket_name).blob(blob_path)
            blob.upload_from_string(data, content_type=content_type)
            logger.success(f"Upload complete → {gcs_uri}")
            return gcs_uri
        except Exception as e:
            logger.error(f"Error uploading bytes to GCS: {e}\n{traceback.format_exc()}")
            raise

    def upload_file_to_gcs(self, bucket_name, blob_path, local_filepath):
        """Upload a local file to GCS. Returns gs:// URI."""
        try:
            gcs_uri = f"gs://{bucket_name}/{blob_path}"
            logger.info(f"Uploading {local_filepath} → {gcs_uri} ...")
            client = storage.Client(project=self.project_id)
            blob = client.bucket(bucket_name).blob(blob_path)
            blob.upload_from_filename(local_filepath)
            size_mb = os.path.getsize(local_filepath) / (1024 * 1024)
            os.remove(local_filepath)
            logger.success(f"Upload complete → {gcs_uri} ({size_mb:.2f} MB)")
            return gcs_uri
        except Exception as e:
            logger.error(f"GCS file upload error: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────────────────────────────────────────
    # BIGQUERY METHODS
    # ───────────────────────────────────────────────────────

    def build_bq_schema(self, schema_fields):
        """Convert list of plain dicts into BigQuery SchemaField objects."""
        try:
            bq_schema = [
                bigquery.SchemaField(
                    name=f["name"],
                    field_type=f["type"],
                    mode=f.get("mode", "NULLABLE"),
                    description=f.get("description", ""),
                )
                for f in schema_fields
            ]
            logger.info(f"BQ schema built with {len(bq_schema)} fields.")
            return bq_schema
        except Exception as e:
            logger.error(f"Error building BQ schema: {e}\n{traceback.format_exc()}")
            raise

    # def build_dynamic_schema_from_csv(self, csv_filepath, known_schema):
    #     """
    #     Read CSV headers from file, map to known types.
    #     Unknown columns default to STRING.

    #     Args:
    #         csv_filepath : path to the CSV file
    #         known_schema : list of dicts from build_schema()
    #     Returns:
    #         list of schema dicts covering every CSV column
    #     """
    #     try:
    #         with open(csv_filepath, "r", encoding="utf-8") as f:
    #             reader = csv.reader(f)
    #             headers = next(reader)

    #         known_map = {field["name"]: field for field in known_schema}
    #         dynamic_schema = []

    #         for header in headers:
    #             header = header.strip()
    #             if header in known_map:
    #                 dynamic_schema.append(known_map[header])
    #             else:
    #                 logger.info(f"New column detected in CSV: '{header}' → defaulting to STRING")
    #                 dynamic_schema.append({
    #                     "name": header, "type": "STRING", "mode": "NULLABLE"
    #                 })

    #         logger.info(
    #             f"Dynamic schema: {len(dynamic_schema)} cols "
    #             f"({len(dynamic_schema) - len([h for h in headers if h.strip() in known_map])} new)"
    #         )
    #         return dynamic_schema

    #     except Exception as e:
    #         logger.error(f"Error building dynamic schema: {e}\n{traceback.format_exc()}")
    #         raise
    
    def build_dynamic_schema_from_csv(self, csv_filepath, known_schema):
        try:
            with open(csv_filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)

            # Sanitize both known schema keys and CSV headers
            known_map = {f["name"]: f for f in known_schema}
            dynamic_schema = []

            for header in headers:
                header = header.strip()
                if header in known_map:
                    dynamic_schema.append(known_map[header])
                else:
                    logger.info(f"New column in CSV: '{header}' → STRING")
                    dynamic_schema.append({
                        "name": header, "type": "STRING", "mode": "NULLABLE"
                    })

            logger.info(f"Dynamic schema: {len(dynamic_schema)} cols")
            return dynamic_schema

        except Exception as e:
            logger.error(f"Error building dynamic schema: {e}\n{traceback.format_exc()}")
            raise

    
    
    def ensure_bq_table(self, table_id, schema_fields,
                        partition_field="ingestion_timestamp"):
        try:
            bq_client = bigquery.Client(project=self.project_id)

            # Sanitize all field names for BQ compatibility
            bq_schema = []
            for f in schema_fields:
                safe_name = sanitize_bq_column_name(f["name"])
                field_type = f["type"]
                if f["name"] == partition_field or safe_name == sanitize_bq_column_name(partition_field):
                    field_type = "TIMESTAMP"
                bq_schema.append(bigquery.SchemaField(
                    name=safe_name,
                    field_type=field_type,
                    mode=f.get("mode", "NULLABLE"),
                ))

            safe_partition = sanitize_bq_column_name(partition_field)

            try:
                table = bq_client.get_table(table_id)
                existing_cols = {field.name for field in table.schema}
                new_fields = [f for f in bq_schema if f.name not in existing_cols]

                if new_fields:
                    logger.info(
                        f"Schema evolution on {table_id}: "
                        f"{len(new_fields)} new column(s)"
                    )
                    for field in new_fields:
                        alter_sql = (
                            f"ALTER TABLE `{table_id}` "
                            f"ADD COLUMN IF NOT EXISTS "
                            f"`{field.name}` {field.field_type}"
                        )
                        bq_client.query(alter_sql).result()
                        logger.success(f"  Added column: {field.name} ({field.field_type})")
                else:
                    logger.info(f"Table {table_id} schema is up to date")

            except Exception as get_err:
                if "Not found" in str(get_err):
                    logger.info(f"Table {table_id} not found — creating ...")
                    table = bigquery.Table(table_id, schema=bq_schema)
                    table.time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.DAY,
                        field=safe_partition,
                    )
                    bq_client.create_table(table, exists_ok=True)
                    logger.success(
                        f"Created partitioned table: {table_id} "
                        f"(partition_by=DATE({safe_partition}))"
                    )
                else:
                    raise

        except Exception as e:
            logger.error(f"Error ensuring BQ table: {e}\n{traceback.format_exc()}")
            raise
    
    
    # def delete_bq_rows_for_day(self, table_id, day_str, day_column="Day"):
    #     """
    #     DELETE FROM table WHERE {day_column} = '{day_str}'

    #     Makes the pipeline idempotent — re-running the same date
    #     replaces data instead of duplicating it.

    #     Args:
    #         table_id   : project.dataset.table
    #         day_str    : YYYY-MM-DD
    #         day_column : column name holding the date (default 'Day')
    #     """
    #     try:
    #         bq_client = bigquery.Client(project=self.project_id)

    #         # Check if table exists first — skip delete for brand new tables
    #         try:
    #             bq_client.get_table(table_id)
    #         except Exception as check_err:
    #             if "Not found" in str(check_err):
    #                 logger.info(f"Table {table_id} does not exist yet — skipping delete")
    #                 return 0
    #             raise

    #         sql = (
    #             f"DELETE FROM `{table_id}` "
    #             f"WHERE `{day_column}` = '{day_str}'"
    #         )
    #         logger.info(f"🗑️  Idempotent delete: {table_id} WHERE {day_column} = '{day_str}'")

    #         query_job = bq_client.query(sql)
    #         query_job.result()  # blocks until done

    #         rows_deleted = query_job.num_dml_affected_rows or 0
    #         if rows_deleted > 0:
    #             logger.success(f"🗑️  Deleted {rows_deleted:,} existing rows for {day_str}")
    #         else:
    #             logger.info(f"🗑️  No existing rows for {day_str} — clean insert")

    #         return rows_deleted

    #     except Exception as e:
    #         logger.error(
    #             f"Error deleting rows for {day_str} from {table_id}: "
    #             f"{e}\n{traceback.format_exc()}"
    #         )
    #         raise
    
    def count_bq_rows_for_day(self, table_id, day_str, day_column="Day"):
        """Count existing rows in BQ for a given day. Returns 0 if table doesn't exist."""
        try:
            bq_client = bigquery.Client(project=self.project_id)
            try:
                bq_client.get_table(table_id)
            except Exception as e:
                if "Not found" in str(e):
                    return 0
                raise

            sql = f"SELECT COUNT(*) as cnt FROM `{table_id}` WHERE `{day_column}` = '{day_str}'"
            result = bq_client.query(sql).result()
            count = list(result)[0].cnt
            logger.info(f"📊 Existing rows in {table_id} for {day_str}: {count:,}")
            return count

        except Exception as e:
            logger.error(f"Error counting rows: {e}\n{traceback.format_exc()}")
            return 0

    def safe_idempotent_load(self, gcs_uri, table_id, schema_fields,
                              day_str, new_row_count,
                              day_column="Day",
                              partition_field="ingestion_timestamp"):
        """
        Safe idempotent load: INSERT first, then DELETE old duplicates.

        Flow:
          1. If new_row_count == 0 → skip entirely, preserve existing data
          2. Count existing rows for this day
          3. INSERT new data (WRITE_APPEND)
          4. DELETE old rows (rows where Day = day_str AND they are NOT
             from this load — identified by row count)

        Uses a simple approach: record existing count, load new, delete
        rows beyond the new count by using ingestion_timestamp ordering.
        """
        try:
            bq_client = bigquery.Client(project=self.project_id)

            # ── Guard: don't touch existing data if new data is empty ──
            if new_row_count == 0:
                logger.warning(
                    f"⚠️  SKIPPING load for {table_id} | {day_str} — "
                    f"new data has 0 rows. Existing data preserved."
                )
                return

            # ── Count existing rows before insert ──
            existing_count = self.count_bq_rows_for_day(table_id, day_str, day_column)

            # ── Step 1: INSERT new data (safe — adds rows, never removes) ──
            self.load_gcs_to_bq(
                gcs_uri, table_id, schema_fields,
                write_disposition="WRITE_APPEND",
                partition_field=partition_field,
            )
            logger.success(
                f"✅ Inserted {new_row_count:,} new rows for {day_str}"
            )

            # ── Step 2: DELETE old duplicates (only if there were existing rows) ──
            if existing_count > 0:
                # Delete rows that are NOT from the latest ingestion
                # Latest ingestion = MAX(ingestion_timestamp) for this day
                safe_partition = sanitize_bq_column_name(partition_field)
                safe_day_col = sanitize_bq_column_name(day_column)

                dedup_sql = f"""
                    DELETE FROM `{table_id}`
                    WHERE `{safe_day_col}` = '{day_str}'
                      AND `{safe_partition}` < (
                          SELECT MAX(`{safe_partition}`)
                          FROM `{table_id}`
                          WHERE `{safe_day_col}` = '{day_str}'
                      )
                """
                logger.info(
                    f"🗑️  Deduplicating {table_id} for {day_str} "
                    f"(removing {existing_count:,} old rows) ..."
                )
                job = bq_client.query(dedup_sql)
                job.result()

                rows_deleted = job.num_dml_affected_rows or 0
                logger.success(f"🗑️  Dedup complete: {rows_deleted:,} old rows removed")
            else:
                logger.info(f"🆕 First load for {day_str} — no dedup needed")

        except Exception as e:
            logger.error(
                f"❌ Safe idempotent load failed for {table_id}/{day_str}: "
                f"{e}\n{traceback.format_exc()}"
            )
            # CRITICAL: data is still in GCS — log URI for manual recovery
            logger.critical(
                f"🚨 RECOVERY: Data is safe in GCS at {gcs_uri}. "
                f"Manual BQ load may be needed."
            )
            raise
    
    
    def load_gcs_to_bq(self, gcs_uri, table_id, schema_fields,
                        write_disposition="WRITE_APPEND",
                        skip_leading_rows=1,
                        partition_field="ingestion_timestamp"):
        try:
            logger.info(f"Loading {gcs_uri} → {table_id} ...")
            bq_client = bigquery.Client(project=self.project_id)

            # Sanitize field names to match BQ table schema
            bq_schema = []
            safe_partition = sanitize_bq_column_name(partition_field)
            for f in schema_fields:
                safe_name = sanitize_bq_column_name(f["name"])
                field_type = f["type"]
                if safe_name == safe_partition:
                    field_type = "TIMESTAMP"
                bq_schema.append(bigquery.SchemaField(
                    name=safe_name,
                    field_type=field_type,
                    mode=f.get("mode", "NULLABLE"),
                ))

            job_config = bigquery.LoadJobConfig(
                schema=bq_schema,
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=skip_leading_rows,
                write_disposition=getattr(
                    bigquery.WriteDisposition, write_disposition
                ),
                max_bad_records=0,
                allow_quoted_newlines=True,
            )
            job = bq_client.load_table_from_uri(
                gcs_uri, table_id, job_config=job_config
            )
            job.result()

            dest = bq_client.get_table(table_id)
            logger.success(f"Loaded into {table_id} | Total rows: {dest.num_rows}")

        except Exception as e:
            logger.error(f"Error loading GCS → BQ: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────────────────────────────────────────
    # UTILITY METHODS
    # ───────────────────────────────────────────────────────

    def extract_list(self, payload: dict) -> list:
        """Normalise common Yahoo response shapes into a plain list."""
        if not isinstance(payload, dict):
            return []
        resp = payload.get("response")
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            for key in ("data", "results"):
                if isinstance(resp.get(key), list):
                    return resp[key]
        if isinstance(payload.get("data"), list):
            return payload["data"]
        return []

    