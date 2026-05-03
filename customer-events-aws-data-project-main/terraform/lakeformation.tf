# --- 1. DATA LAKE ADMIN SETTINGS ---
resource "aws_lakeformation_data_lake_settings" "admin_settings" {
  admins = [
    aws_iam_role.glue_service_role.arn,
    data.aws_caller_identity.current.arn
  ]
}

# --- 2. S3 RESOURCE REGISTRATION ---
resource "aws_lakeformation_resource" "raw_data_registration" {
  arn = aws_s3_bucket.raw_data.arn
}

resource "aws_lakeformation_resource" "processed_data_registration" {
  arn = aws_s3_bucket.processed_data.arn
}

# --- 3. DATA LOCATION PERMISSIONS ---
resource "aws_lakeformation_permissions" "glue_raw_location_access" {
  principal   = aws_iam_role.glue_service_role.arn
  permissions = ["DATA_LOCATION_ACCESS"]

  data_location {
    arn = aws_s3_bucket.raw_data.arn
  }

  depends_on = [aws_lakeformation_resource.raw_data_registration]
}

resource "aws_lakeformation_permissions" "glue_processed_location_access" {
  principal   = aws_iam_role.glue_service_role.arn
  permissions = ["DATA_LOCATION_ACCESS"]

  data_location {
    arn = aws_s3_bucket.processed_data.arn
  }

  depends_on = [aws_lakeformation_resource.processed_data_registration]
}

resource "aws_lakeformation_permissions" "firehose_raw_location_access" {
  principal   = aws_iam_role.firehose_role.arn
  permissions = ["DATA_LOCATION_ACCESS"]

  data_location {
    arn = aws_s3_bucket.raw_data.arn
  }

  depends_on = [aws_lakeformation_resource.raw_data_registration]
}

# --- 4. DATABASE PERMISSIONS ---
resource "aws_lakeformation_permissions" "glue_db_permissions" {
  principal   = aws_iam_role.glue_service_role.arn
  permissions = ["CREATE_TABLE", "ALTER", "DROP", "DESCRIBE"]

  database {
    name = aws_glue_catalog_database.glue_database.name
  }
}

resource "aws_lakeformation_permissions" "firehose_glue_db_access" {
  principal   = aws_iam_role.firehose_role.arn
  permissions = ["DESCRIBE"]

  database {
    name = aws_glue_catalog_database.glue_database.name
  }
}

# --- 5. TEAM PERMISSIONS (DATABASE LEVEL) ---
resource "aws_lakeformation_permissions" "legal_teams_db_access" {
  for_each    = aws_iam_role.legal_team_roles
  principal   = each.value.arn
  permissions = ["DESCRIBE"]

  database {
    name = aws_glue_catalog_database.glue_database.name
  }
}

resource "aws_lakeformation_permissions" "data_teams_db_access" {
  for_each    = aws_iam_role.data_team_roles
  principal   = each.value.arn
  permissions = ["DESCRIBE"]

  database {
    name = aws_glue_catalog_database.glue_database.name
  }
}