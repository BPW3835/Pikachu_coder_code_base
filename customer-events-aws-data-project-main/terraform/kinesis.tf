resource "aws_kinesis_stream" "ecommerce_stream" {
  name             = var.kinesis_stream_name
  shard_count      = var.shard_count
  retention_period = var.kinesis_retention_period
}

resource "aws_kinesis_firehose_delivery_stream" "ecommerce_firehose" {
  name        = var.firehose_delivery_stream_name
  destination = "extended_s3"

  kinesis_source_configuration {
    kinesis_stream_arn = aws_kinesis_stream.ecommerce_stream.arn
    role_arn           = aws_iam_role.firehose_role.arn
  }

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose_role.arn
    bucket_arn = aws_s3_bucket.raw_data.arn
    prefix = "customer_events_raw/year=!{timestamp:YYYY}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/"
    error_output_prefix = "errors/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/hour=!{timestamp:HH}/!{firehose:error-output-type}/"

    buffering_size     = 5
    buffering_interval = 60
    compression_format = "UNCOMPRESSED"
  }
}