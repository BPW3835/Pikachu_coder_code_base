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

    def build_dynamic_schema_from_csv(self, csv_filepath, known_schema):
        """
        Read CSV headers from file, map to known types.
        Unknown columns default to STRING.

        Args:
            csv_filepath : path to the CSV file
            known_schema : list of dicts from build_schema()
        Returns:
            list of schema dicts covering every CSV column
        """
        try:
            with open(csv_filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)

            known_map = {field["name"]: field for field in known_schema}
            dynamic_schema = []

            for header in headers:
                header = header.strip()
                if header in known_map:
                    dynamic_schema.append(known_map[header])
                else:
                    logger.info(f"New column detected in CSV: '{header}' → defaulting to STRING")
                    dynamic_schema.append({
                        "name": header, "type": "STRING", "mode": "NULLABLE"
                    })

            logger.info(
                f"Dynamic schema: {len(dynamic_schema)} cols "
                f"({len(dynamic_schema) - len([h for h in headers if h.strip() in known_map])} new)"
            )
            return dynamic_schema

        except Exception as e:
            logger.error(f"Error building dynamic schema: {e}\n{traceback.format_exc()}")
            raise

    def ensure_bq_table(self, table_id, schema_fields,
                        partition_field="ingestion_timestamp"):
        """
        CREATE TABLE IF NOT EXISTS with partition by DATE(ingestion_timestamp).
        If table exists, detect new columns via INFORMATION_SCHEMA and
        ALTER TABLE ADD COLUMN for each.

        Args:
            table_id        : project.dataset.table
            schema_fields   : list of dicts (name, type, mode)
            partition_field : column to partition by (overridden to TIMESTAMP)
        """
        try:
            bq_client = bigquery.Client(project=self.project_id)

            # Build BQ schema — override partition field to TIMESTAMP
            bq_schema = []
            for f in schema_fields:
                field_type = f["type"]
                if f["name"] == partition_field:
                    field_type = "TIMESTAMP"
                bq_schema.append(bigquery.SchemaField(
                    name=f["name"],
                    field_type=field_type,
                    mode=f.get("mode", "NULLABLE"),
                ))

            try:
                # ── Table exists → schema evolution ──
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
                    # ── Table does not exist → create with partition ──
                    logger.info(f"Table {table_id} not found — creating ...")
                    table = bigquery.Table(table_id, schema=bq_schema)
                    table.time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.DAY,
                        field=partition_field,
                    )
                    bq_client.create_table(table, exists_ok=True)
                    logger.success(
                        f"Created partitioned table: {table_id} "
                        f"(partition_by=DATE({partition_field}))"
                    )
                else:
                    raise

        except Exception as e:
            logger.error(f"Error ensuring BQ table: {e}\n{traceback.format_exc()}")
            raise

    def load_gcs_to_bq(self, gcs_uri, table_id, schema_fields,
                        write_disposition="WRITE_APPEND",
                        skip_leading_rows=1,
                        partition_field="ingestion_timestamp"):
        """
        Load a CSV from GCS into BigQuery.
        Partition field is overridden to TIMESTAMP in the load schema.
        """
        try:
            logger.info(f"Loading {gcs_uri} → {table_id} ...")
            bq_client = bigquery.Client(project=self.project_id)

            # Build load schema — override partition field
            bq_schema = []
            for f in schema_fields:
                field_type = f["type"]
                if f["name"] == partition_field:
                    field_type = "TIMESTAMP"
                bq_schema.append(bigquery.SchemaField(
                    name=f["name"],
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