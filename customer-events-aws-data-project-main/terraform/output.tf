# --- 1. S3 BUCKET NAMES ---
output "raw_bucket_name" {
  value       = aws_s3_bucket.raw_data.id
  description = "The name of the Raw Data bucket"
}

output "processed_bucket_name" {
  value       = aws_s3_bucket.processed_data.id
  description = "The name of the Processed Data bucket"
}

# --- 2. ATHENA WORKGROUP NAMES ---
output "athena_workgroups" {
  value = {
    data_team       = aws_athena_workgroup.data_team_workgroup.name
    legal_team = aws_athena_workgroup.legal_team_workgroup.name
  }
}

# --- 3. GLUE DATABASE ---
output "glue_catalog_database_name" {
  value = aws_glue_catalog_database.glue_database.name
}