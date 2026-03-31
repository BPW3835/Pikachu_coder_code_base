# --- 1. RAW DATA BUCKET ---
resource "aws_s3_bucket" "raw_data" {
  bucket        = var.aws_s3_raw_data_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "block_raw" {
  bucket                  = aws_s3_bucket.raw_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- 2. PROCESSED DATA BUCKET ---
resource "aws_s3_bucket" "processed_data" {
  bucket        = var.aws_s3_processed_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "block_processed" {
  bucket                  = aws_s3_bucket.processed_data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- 3. ASSETS & JAR BUCKET ---
resource "aws_s3_bucket" "assets_jar" {
  bucket        = var.aws_s3_jar_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "block_jar" {
  bucket                  = aws_s3_bucket.assets_jar.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- 4. GLUE SCRIPT BUCKET ---
resource "aws_s3_bucket" "glue_script" {
  bucket        = var.aws_s3_glue_script_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "block_glue" {
  bucket                  = aws_s3_bucket.glue_script.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- 5. ATHENA OUTPUT BUCKET ---
resource "aws_s3_bucket" "athena_output" {
  bucket        = var.aws_s3_athena_output_bucket_name
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "block_athena" {
  bucket                  = aws_s3_bucket.athena_output.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

