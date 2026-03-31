import uuid
from datetime import date
from google.cloud import bigquery
from google.cloud import storage

# -----------------------------
# CONFIG (edit these)
# -----------------------------
PROJECT_ID = "your-project-id"

# Source (EU)
SRC_DATASET = "your_eu_dataset"
SRC_TABLE = "your_table"  # table name only
SRC_LOCATION = "EU"       # could also be "europe-west1" depending on your dataset location

# Destination (US)
DEST_DATASET = "your_us_dataset"
DEST_TABLE = "your_table"  # can be same or different
DEST_LOCATION = "US"       # could also be "us-central1" depending on your dataset location

# Staging buckets (MUST be in correct locations)
GCS_EU_BUCKET = "your-eu-bucket"  # bucket location EU / europe-westX
GCS_US_BUCKET = "your-us-bucket"  # bucket location US / us-central1 etc.

# Folder/prefix for staging objects
STAGING_PREFIX = "replica_exports/single_table"

# File format
EXPORT_FORMAT = bigquery.DestinationFormat.AVRO  # schema-friendly for BQ
# bigquery.Compression.SNAPPY is supported for AVRO via extract_job_config.compression
EXPORT_COMPRESSION = bigquery.Compression.SNAPPY

# Load behavior (daily replace)
WRITE_DISPOSITION = bigquery.WriteDisposition.WRITE_TRUNCATE


def copy_table_across_locations():
    run_dt = date.today().isoformat()
    run_id = uuid.uuid4().hex[:8]

    bq = bigquery.Client(project=PROJECT_ID)
    gcs = storage.Client(project=PROJECT_ID)

    src_table_id = f"{PROJECT_ID}.{SRC_DATASET}.{SRC_TABLE}"
    dest_table_id = f"{PROJECT_ID}.{DEST_DATASET}.{DEST_TABLE}"

    # 1) Extract EU table -> EU GCS
    eu_uri_prefix = f"gs://{GCS_EU_BUCKET}/{STAGING_PREFIX}/dt={run_dt}/run={run_id}/part-*.avro"
    extract_config = bigquery.job.ExtractJobConfig(
        destination_format=EXPORT_FORMAT,
        compression=EXPORT_COMPRESSION,
        print_header=False,
    )

    print(f"[1/3] Extracting {src_table_id} -> {eu_uri_prefix}")
    extract_job = bq.extract_table(
        source=src_table_id,
        destination_uris=eu_uri_prefix,
        job_config=extract_config,
        location=SRC_LOCATION,  # IMPORTANT: must match the source dataset/table location
    )
    extract_job.result()
    print("[1/3] Extract complete.")

    # List extracted objects in EU bucket (we need concrete object names to copy)
    eu_bucket = gcs.bucket(GCS_EU_BUCKET)
    exported_prefix = f"{STAGING_PREFIX}/dt={run_dt}/run={run_id}/"
    blobs = list(gcs.list_blobs(GCS_EU_BUCKET, prefix=exported_prefix))
    if not blobs:
        raise RuntimeError(f"No exported objects found under gs://{GCS_EU_BUCKET}/{exported_prefix}")

    # 2) Copy EU GCS objects -> US GCS bucket (same object names)
    us_bucket = gcs.bucket(GCS_US_BUCKET)
    copied_uris = []

    print(f"[2/3] Copying {len(blobs)} object(s) EU bucket -> US bucket...")
    for blob in blobs:
        # Copy object to US bucket under same path
        new_blob = eu_bucket.copy_blob(blob, us_bucket, new_name=blob.name)
        copied_uris.append(f"gs://{GCS_US_BUCKET}/{new_blob.name}")

    print("[2/3] GCS cross-region copy complete.")

    # 3) Load US GCS -> US BigQuery
    load_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.AVRO,
        write_disposition=WRITE_DISPOSITION,
        autodetect=True,  # AVRO carries schema; autodetect is typically safe here
    )

    # Use wildcard URI for load (BigQuery supports wildcards for load)
    us_uri_wildcard = f"gs://{GCS_US_BUCKET}/{exported_prefix}part-*.avro"

    print(f"[3/3] Loading {us_uri_wildcard} -> {dest_table_id}")
    load_job = bq.load_table_from_uri(
        source_uris=us_uri_wildcard,
        destination=dest_table_id,
        job_config=load_config,
        location=DEST_LOCATION,  # IMPORTANT: must match destination dataset location
    )
    load_job.result()
    print("[3/3] Load complete.")

    # Optional: verify row count
    dest_tbl = bq.get_table(dest_table_id)
    print(f"✅ Done. Destination rows: {dest_tbl.num_rows}")


if __name__ == "__main__":
    copy_table_across_locations()

