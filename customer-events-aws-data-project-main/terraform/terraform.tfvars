# General Settings
aws_region   = "us-east-1"
project_name = "ecommerce-project"

# S3 Bucket Names (Make sure these are globally unique)
aws_s3_raw_data_bucket_name      = "ecommerce-raw-data-hsnlsk1"
aws_s3_processed_bucket_name     = "ecommerce-processed-data-hsnlsk1"
aws_s3_jar_bucket_name           = "ecommerce-assets-jar-hsnlsk1"
aws_s3_glue_script_bucket_name   = "ecommerce-glue-script-hsnlsk1"
aws_s3_athena_output_bucket_name = "ecommerce-athena-results-hsnlsk1"


# Kinesis Settings
kinesis_stream_name      = "ecommerce-raw-stream"
shard_count              = 1
kinesis_retention_period = 72

# Firehose Settings
firehose_delivery_stream_name = "ecommerce-to-s3-firehose"

# Lambda Settings
firehose_lambda_function_name = "ecommerce-data-transformer"

# Glue Settings
glue_database_name = "ecommerce_data_catalog"
glue_crawler_name  = "ecommerce_raw_crawler"
glue_job_name      = "ecommerce_etl_transform_job"

# Roles
legal_teams = ["legal_team_US", "legal_team_UK"]
data_teams = ["data_team_US", "data_team_UK"]
