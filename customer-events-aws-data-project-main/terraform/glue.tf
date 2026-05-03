# 1. Uploading the Python script from PC to S3
resource "aws_s3_object" "ecommerce_glue_script" {
  bucket = aws_s3_bucket.glue_script.bucket
  key    = "scripts/ecommerce_glue_script.py"
  source = "${path.module}/scripts/ecommerce_glue_script.py"

  # Prevent Terraform from overwriting the script after the first upload
  lifecycle {
    ignore_changes = [
      source,
      etag
    ]
  }
}

# --- UPLOAD JAR FILE TO S3 ---
resource "aws_s3_object" "glue_jar_file" {
  bucket     = aws_s3_bucket.assets_jar.id
  key        = "jars/spark-streaming-kinesis.jar"              # S3 path
  source     = "${path.module}/jars/spark-streaming-kinesis.jar" # Local path
  etag       = filemd5("${path.module}/jars/spark-streaming-kinesis.jar")

  depends_on = [
    aws_s3_bucket.assets_jar,
    aws_s3_bucket_public_access_block.block_jar
  ]
}

# 2. Glue Database
resource "aws_glue_catalog_database" "glue_database" {
  name = var.glue_database_name
}

# 3. Glue Job
resource "aws_glue_job" "ecommerce_glue_job" {
  name              = var.glue_job_name
  role_arn          = aws_iam_role.glue_service_role.arn
  worker_type       = "G.1X"
  number_of_workers = 2
  max_retries       = 0
  timeout           = 60

  command {
    name            = "glueetl"
    script_location = "s3://${aws_s3_object.ecommerce_glue_script.bucket}/${aws_s3_object.ecommerce_glue_script.key}"
    python_version  = "3"
  }

  default_arguments = {
    "--TempDir"                          = "s3://${var.aws_s3_glue_script_bucket_name}/temp/"
    "--job-language"                     = "python"
    "--enable-continuous-cloudwatch-log" = "true"
    "--job-bookmark-option"              = "job-bookmark-enable"
    "--extra-jars"                       = "s3://${aws_s3_bucket.assets_jar.bucket}/jars/spark-streaming-kinesis.jar"
  }
}