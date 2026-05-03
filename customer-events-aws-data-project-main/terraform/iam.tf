# --- 1. FIREHOSE ROLE & POLICIES ---
resource "aws_iam_role" "firehose_role" {
  name = "${var.project_name}-firehose-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "firehose.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "firehose_policy" {
  name = "firehose_delivery_policy"
  role = aws_iam_role.firehose_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:GetBucketLocation",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kinesis:DescribeStream",
          "kinesis:GetShardIterator",
          "kinesis:GetRecords",
          "kinesis:ListShards"
        ]
        Resource = "arn:aws:kinesis:${var.aws_region}:*:stream/${var.kinesis_stream_name}"
      },
      {
        Effect = "Allow"
        Action = [
          "glue:GetTable",
          "glue:GetTableVersion",
          "glue:GetTableVersions",
          "glue:GetDatabase",
          "glue:GetDatabases",
          "glue:GetUserDefinedFunctions"
        ]
        Resource = [
          "arn:aws:glue:${var.aws_region}:*:catalog",
          "arn:aws:glue:${var.aws_region}:*:database/*",
          "arn:aws:glue:${var.aws_region}:*:table/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction",
          "lambda:GetFunctionConfiguration"
        ]
        Resource = "arn:aws:lambda:${var.aws_region}:*:function:${var.firehose_lambda_function_name}:*"
      },
      {
        Effect = "Allow"
        Action = ["lakeformation:GetDataAccess"]
        Resource = "*"
      }
    ]
  })
}

# --- 2. GLUE SERVICE ROLE ---
resource "aws_iam_role" "glue_service_role" {
  name = "${var.project_name}-glue-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "glue.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "glue_service_attach" {
  role       = aws_iam_role.glue_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_s3_access" {
  name = "glue_s3_access_policy"
  role = aws_iam_role.glue_service_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}/*",
          "arn:aws:s3:::${var.aws_s3_processed_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_processed_bucket_name}/*"
          "arn:aws:s3:::${var.aws_s3_jar_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_jar_bucket_name}/*",
          "arn:aws:s3:::${var.aws_s3_glue_script_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_glue_script_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = ["lakeformation:GetDataAccess"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "glue_sns_policy" {
  name = "glue_sns_publish_policy"
  role = aws_iam_role.ecommerce_project_glue_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action   = "sns:Publish"
        Effect   = "Allow"
        Resource = "arn:aws:sns:us-east-1:412381764682:DataQualityReport"
      }
    ]
  })
}

# --- 3. LAMBDA ROLE ---
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- 4. KINESIS PRODUCER POLICY ---
resource "aws_iam_policy" "kinesis_producer_policy" {
  name = "kinesis_producer_policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "kinesis:PutRecord",
        "kinesis:PutRecords",
        "kinesis:DescribeStream"
      ]
      Resource = "arn:aws:kinesis:${var.aws_region}:*:stream/${var.kinesis_stream_name}"
    }]
  })
}

# --- 5. LEGAL TEAMS ---
resource "aws_iam_role" "legal_team_roles" {
  for_each = toset(var.legal_teams)
  name     = "${var.project_name}-${each.key}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = ["athena.amazonaws.com", "lakeformation.amazonaws.com"]
      }
    }]
  })
}

resource "aws_iam_role_policy" "legal_team_access_policy" {
  for_each = aws_iam_role.legal_team_roles
  name     = "${each.key}-limited-policy"
  role     = each.value.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:StopQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:GetWorkGroup"
        ]
        Resource = [aws_athena_workgroup.legal_team_workgroup.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}/*",
          "arn:aws:s3:::${var.aws_s3_processed_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_processed_bucket_name}/*",
          "arn:aws:s3:::${var.aws_s3_athena_output_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_athena_output_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetTable",
          "glue:GetPartitions",
          "lakeformation:GetDataAccess"
        ]
        Resource = "*"
      }
    ]
  })
}

# --- 6. DATA TEAMS ---
resource "aws_iam_role" "data_team_roles" {
  for_each = toset(var.data_teams)
  name     = "${var.project_name}-${each.key}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = ["athena.amazonaws.com", "lakeformation.amazonaws.com"]
      }
    }]
  })
}

resource "aws_iam_role_policy" "data_team_access_policy" {
  for_each = aws_iam_role.data_team_roles
  name     = "${each.key}-limited-policy"
  role     = each.value.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "athena:StartQueryExecution",
          "athena:StopQueryExecution",
          "athena:GetQueryExecution",
          "athena:GetQueryResults",
          "athena:GetWorkGroup"
        ]
        Resource = [aws_athena_workgroup.data_team_workgroup.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:PutObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_raw_data_bucket_name}/*",
          "arn:aws:s3:::${var.aws_s3_processed_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_processed_bucket_name}/*",
          "arn:aws:s3:::${var.aws_s3_athena_output_bucket_name}",
          "arn:aws:s3:::${var.aws_s3_athena_output_bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "glue:GetDatabase",
          "glue:GetTable",
          "glue:GetPartitions",
          "lakeformation:GetDataAccess"
        ]
        Resource = "*"
      }
    ]
  })
}