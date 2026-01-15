import os
import avro
import random
import datetime
import pytz
import traceback
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import polars as pl
from helpers.log import get_logger, log_execution_time
from google.cloud import bigquery, secretmanager, storage
import time
from fastavro import writer

class Utilities:
    def __init__(self):
        self.logger = get_logger()
        self.bq_client = bigquery.Client()
        self.storage_client = storage.Client()
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.table_id = os.getenv("GCP_BQ_TABLE_ID")
        self.dataset_id = os.getenv("GCP_BQ_DATASET_ID")
        self.query_table_id = os.getenv("GCP_BQ_QUERY_TABLE_ID")
        self.query_dataset_id = os.getenv("GCP_BQ_QUERY_DATASET_ID")
        self.gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
        self.expected_columns = ["mdlzID", "state", "district", "city", "latitude", "longitude"]
        self.avro_to_bq_types = {"string": "STRING","int": "INTEGER","long": "INTEGER","float": "FLOAT","double": "FLOAT","boolean": "BOOLEAN","bytes": "BYTES"}
    

    def normalize_row_to_dict(self, row):
        """
        Normalize a BigQuery Row-like object into {'latitude','longitude','mdlzID'}.
        Works for attribute-style rows and simple positional/tuple rows.
        """
        lat = lng = mid = None

        # attribute-style access: row.latitude
        try:
            if hasattr(row, "latitude") and hasattr(row, "longitude") and hasattr(row, "mdlzID"):
                lat = getattr(row, "latitude")
                lng = getattr(row, "longitude")
                mid = getattr(row, "mdlzID")
                return {"latitude": str(lat) if lat is not None else None,
                        "longitude": str(lng) if lng is not None else None,
                        "mdlzID": mid}
        except Exception:
            pass

        # dict-like access (rare): row['latitude']
        try:
            if isinstance(row, dict):
                lat = row.get("latitude")
                lng = row.get("longitude")
                mid = row.get("mdlzID")
                return {"latitude": str(lat) if lat is not None else None,
                        "longitude": str(lng) if lng is not None else None,
                        "mdlzID": mid}
        except Exception:
            pass

        # positional/tuple access fallback: row[0], row[1], row[2]
        try:
            if len(row) >= 3:
                lat = row[0]
                lng = row[1]
                mid = row[2]
                return {"latitude": str(lat) if lat is not None else None,
                        "longitude": str(lng) if lng is not None else None,
                        "mdlzID": mid}
        except Exception:
            pass

        # give up: return None values so caller can decide
        return {"latitude": None, "longitude": None, "mdlzID": None}
    
    def avro_field_to_bq_schemafield(self, field_def):
        """
        field_def: {"name": "...", "type": ["null","string"], "default": None}
        returns: bigquery.SchemaField(...)
        """
        name = field_def["name"]
        avro_type = field_def["type"]

        # Normalize type: avro_type can be a list (nullable) or a string
        nullable = False
        base_type = None

        # if union includes 'null', treat as nullable and find the other type
        if isinstance(avro_type, list):
            # find first non-null element
            types = list(avro_type)
            if "null" in types:
                nullable = True
                types.remove("null")
            if len(types) == 0:
                # weird: ["null"] only
                base_type = "string"
            else:
                base_type = types[0]  # assume primitive type
        elif isinstance(avro_type, dict):
            # complex type (record) - not supported in this minimal helper
            raise ValueError(f"Complex Avro types not supported yet: {avro_type}")
        else:
            # direct string type
            base_type = avro_type

        # base_type might be a JSON object like {"type":"string"} in some schemas
        if isinstance(base_type, dict):
            base_type = base_type.get("type")

        bq_type = self.avro_to_bq_types.get(base_type)
        if not bq_type:
            raise ValueError(f"Unsupported/unknown Avro base type: {base_type} for field {name}")

        mode = "NULLABLE" if nullable else "REQUIRED"
        return bigquery.SchemaField(name, bq_type, mode=mode)

    @log_execution_time
    def upload_to_gcs(self, chunk_id, file_path):
        try:
            bucket = self.storage_client.get_bucket(self.gcs_bucket_name)
            blob = bucket.blob(f"geo_location_data/avro_files/")
            blob.upload_from_filename(f"{file_path}")
            self.logger.info(f"Uploaded chunk {chunk_id} to GCS")
            return blob.name
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error fetching : {e}")
            raise
    
    @log_execution_time
    def upload_avro_file_from_gcs_to_bigquery_table(self, blob_name, table_id):
        try:
            table_ref = f"{table_id}"
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.AVRO,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            uri = f"gs://{self.gcs_bucket_name}/{blob_name}"
            load_job = self.bq_client.load_table_from_uri(uri,table_ref,job_config=job_config)
            load_job.result()  # Waits for the job to complete.
            self.logger.info(f"Loaded data from {uri} to {table_ref}")
            self.logger.info(f"Total rows loaded: {load_job.output_rows}")
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error fetching : {e}")
            raise
    
    # @log_execution_time
    # def upload_avro_file_from_local_to_bigquery(self, file_path, table_id):
    #     try:
    #         table_ref = f"{table_id}"
    #         job_config = bigquery.LoadJobConfig(
    #             source_format=bigquery.SourceFormat.AVRO,
    #             write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    #         )
    #         with open(file_path, "rb") as avro_file:
    #             load_job = self.bq_client.load_table_from_file(avro_file, table_ref, job_config=job_config)
    #         load_job.result()  # Waits for the job to complete.
    #         self.logger.info(f"Loaded data from {file_path} to {table_ref}")
    #         self.logger.info(f"Total rows loaded: {load_job.output_rows}")
    #     except Exception as e:
    #         traceback.print_exc()
    #         self.logger.error(f"Error fetching : {e}")
    #         raise

    @log_execution_time
    def update_harmonized_table_query(self, table_id):
        try:
            query = f"""MERGE `{self.project_id}.{self.query_dataset_id}.{self.query_table_id}` AS tgt
                        USING `{table_id}` AS src
                        ON tgt.mdlzID = src.mdlzID
                    WHEN MATCHED
                    AND src.city IS NOT NULL
                    AND src.state IS NOT NULL
                    THEN
                    UPDATE SET
                        tgt.city = src.city,
                        tgt.state = src.state,
                        tgt.cde_location_updated_at = CURRENT_TIMESTAMP();"""
            query_job = self.bq_client.query(query)
            query_job.result()  # Waits for the query to finish
            self.logger.info(f"Query executed successfully: {query_job.job_id}")

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error fetching : {e}")
            raise

    
    @log_execution_time
    def delete_temp_table(self, table_id):
        try:
            self.bq_client.delete_table(table_id)
            self.logger.info(f"Table {table_id} deleted successfully")
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error fetching : {e}")
            raise
    
    @log_execution_time
    def upload_avro_file_from_local_to_bigquery(self, file_path, table_id):
        """
        Upload a local Avro file into BigQuery table_id (project.dataset.table).
        table_id must already exist or BigQuery will create based on Avro schema inside the file.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Avro file not found: {file_path}")

            # Use LoadJobConfig for AVRO
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.AVRO,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                # If you want BigQuery to derive schema from AVRO file, do not set schema here.
                # If you want to force a schema, provide job_config.schema = [...]
            )

            with open(file_path, "rb") as avro_file:
                load_job = self.bq_client.load_table_from_file(avro_file, table_id, job_config=job_config)

            load_job.result()  # Waits for the job to complete.

            # load_job has stats in .output_rows (number of rows loaded)
            rows_loaded = getattr(load_job, "output_rows", None)
            self.logger.info(f"Loaded data from {file_path} to {table_id}; rows_loaded={rows_loaded}")
            return rows_loaded

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error uploading avro to BigQuery: {e}")
            raise

    
    @log_execution_time
    def create_avro_schema(self):
        """
        Avro schema used for downstream BigQuery/AVRO compatibility.
        Note: default must be JSON null (Python None), not the string 'null'.
        """
        try:
            avro_schema = {
                "type": "record",
                "name": "geo_location_enriched_data",
                "fields": [
                    {"name": "mdlzID",   "type": ["null", "string"], "default": None},
                    {"name": "state",    "type": ["null", "string"], "default": None},
                    {"name": "district", "type": ["null", "string"], "default": None},
                    {"name": "city",     "type": ["null", "string"], "default": None},
                    {"name": "latitude", "type": ["null", "string"], "default": None},
                    {"name": "longitude","type": ["null", "string"], "default": None}
                ]
            }
            return avro_schema
        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error building avro schema: {e}")
            raise
    
    def polars_schema_for_strings(self):
        """
        Return a Polars-style schema mapping where all expected columns are Utf8 (string).
        """
        return {c: pl.Utf8 for c in self.expected_columns}
    
    def create_temp_table_from_avro(self, avro_schema=None, ttl_minutes=10):
        """
        Create a temporary table (empty) with schema derived from provided avro_schema dict
        (the same dict returned by your create_avro_schema()).
        The table will expire after ttl_minutes.
        Returns the fully qualified table id.
        """
        try:
            # if avro_schema not provided, build from helper
            if avro_schema is None:
                avro_schema = self.create_avro_schema()

            # Validate shape
            fields = avro_schema.get("fields")
            if not fields or not isinstance(fields, list):
                raise ValueError("Invalid avro_schema: missing 'fields' list")

            # Convert to BigQuery SchemaField list
            bq_schema = []
            for f in fields:
                sf = self.avro_field_to_bq_schemafield(f)
                bq_schema.append(sf)

            # Generate unique temp table name and full id
            temp_table_name = f"temp_avro_{random.randint(10000, 99999)}"
            table_id = f"{self.project_id}.{self.dataset_id}.{temp_table_name}"

            # Create table object with schema and expiration
            table = bigquery.Table(table_id, schema=bq_schema)
            expiration_time = datetime.datetime.now(pytz.utc) + datetime.timedelta(minutes=ttl_minutes)
            table.expires = expiration_time

            # Create table (idempotent-ish: will error if name exists; we generate random name)
            created_table = self.bq_client.create_table(table)  # raises on error
            self.logger.info(f"Created temporary table {created_table.full_table_id} expires={created_table.expires}")
            return table_id

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error creating temp table from avro schema: {e}")
            raise

    
    # @log_execution_time
    # def create_temp_table_from_avro(self, avro_schema):
    #     """
    #     Creates a temporary table with a 10 minutes expiration from an Avro file in GCS.
    #     """
        
    #     # Generate a unique table name
    #     temp_table_name = f"temp_avro_{random.randint(10000, 99999)}"
    #     table_id = f"{self.project_id}.{self.dataset_id}.{temp_table_name}"
    #     avro_schema = self.create_avro_schema()

    #     # Configure the load job
    #     job_config = bigquery.LoadJobConfig(
    #         source_format=bigquery.SourceFormat.AVRO,
    #         schema=bigquery.Schema.from_avro_schema(avro_schema), 
    #     )

    #     # Define table expiration (e.g., 10 minutes from now)
    #     expiration_time = datetime.datetime.now(pytz.utc) + datetime.timedelta(minutes=10)
    #     # The table object is used to set the expiration time
    #     table = bigquery.Table(table_id)
    #     table.expires = expiration_time

    #     try:
    #         self.bq_client.create_table(table)
    #         print(f"Created empty temporary table '{table_id}' with expiration set to {expiration_time}")
    #     except Exception as e:
    #         print(f"Table creation failed (might already exist or other error): {e}")

    #     print(f"The table will expire at {expiration_time}")
    #     return table_id
    
    def create_chunks(self, data, size):
        """Yield successive chunks of `size` from `data`."""
        for i in range(0, len(data), size):
            chunk_id = (i // size) + 1
            yield chunk_id, data[i:i + size]


    # @log_execution_time
    # def write_polars_df_to_avro(self, chunk_id, data):
    #     """
    #     Build a Polars DataFrame from `data` (list of dicts), force all expected columns to string (Utf8),
    #     add any missing columns with nulls, reorder columns to expected order, and write Avro.
    #     Returns the output file path.
    #     """
    #     try:
    #         # create df from data (no schema param here)
    #         pl_df = pl.DataFrame(data)

    #         # ensure all expected columns exist; if missing, add them as null column
    #         for col in self.expected_columns:
    #             if col not in pl_df.columns:
    #                 pl_df = pl_df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))

    #         # cast all expected columns to Utf8 (string)
    #         cast_exprs = [pl.col(c).cast(pl.Utf8) for c in self.expected_columns]
    #         pl_df = pl_df.select(cast_exprs)  # also reorders to EXPECTED_COLUMNS

    #         # optionally: if you want to validate avro schema before writing, get it:
    #         # avro_schema = self.create_avro_schema()
    #         # write file name
    #         filename = f"chunk_{chunk_id}_{int(time.time())}.avro"
    #         # write avro with snappy compression
    #         pl_df.write_avro(filename, compression="snappy")
    #         self.logger.info(f"Wrote avro file: {filename} rows={pl_df.height} cols={pl_df.width}")
    #         return filename

    #     except Exception as e:
    #         traceback.print_exc()
    #         self.logger.error(f"Error writing avro: {e}")
    #         raise
    
    @log_execution_time
    def write_fastavro_file(self, chunk_id, data):
        """
        Write enriched_data (list of dicts) to Avro using fastavro.
        """
        try:
            avro_schema = self.create_avro_schema()

            filename = f"chunk_{chunk_id}_{int(time.time())}.avro"

            records = []
            for row in data:
                record = {
                    "mdlzID": row.get("mdlzID"),
                    "state": row.get("state"),
                    "district": row.get("district"),
                    "city": row.get("city"),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                }
                records.append(record)

            with open(filename, "wb") as out:
                writer(out, avro_schema, records)

            self.logger.info(f"Avro written using fastavro: {filename}, rows={len(records)}")
            return filename

        except Exception as e:
            traceback.print_exc()
            self.logger.error(f"Error writing Avro using fastavro: {e}")
            raise
        
