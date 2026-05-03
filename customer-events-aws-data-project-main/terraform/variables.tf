variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
}

variable "aws_s3_raw_data_bucket_name" {
  description = "S3 bucket for raw data"
  type        = string
}

variable "aws_s3_processed_bucket_name" {
  description = "S3 bucket for processed data"
  type        = string
}

variable "aws_s3_jar_bucket_name" {
  description = "S3 bucket for jar file"
  type        = string
}

variable "aws_s3_glue_script_bucket_name" {
  description = "S3 bucket for glue script file"
  type        = string
}

variable "aws_s3_athena_output_bucket_name" {
  description = "S3 bucket for athena outputs"
  type        = string
}

variable "kinesis_stream_name" {
  description = "kinesis stream name"
  type        = string
}

variable "shard_count" {
  description = "shard_count"
  type        = number
}

variable "kinesis_retention_period" {
  description = "kinesis_retention_period"
  type        = number
}

variable "firehose_delivery_stream_name" {
  description = "firehose_delivery_stream_name"
  type        = string
}

variable "firehose_lambda_function_name" {
  description = "firehose_lambda_function_name"
  type        = string
}


variable "glue_database_name" {
  description = "glue_database_name"
  type        = string
}


variable "glue_crawler_name" {
  description = "glue_crawler_name"
  type        = string
}

variable "glue_job_name" {
  description = "glue job name"
  type        = string
}

variable "legal_teams" {
  type    = list(string)
  default = ["legal_team_US", "legal_team_UK"]
}

variable "data_teams" {
  type    = list(string)
  default = ["data_team_US", "data_team_UK"]
}