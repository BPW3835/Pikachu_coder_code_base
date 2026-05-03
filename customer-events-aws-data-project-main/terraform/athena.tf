# --- 1. ATHENA WORKGROUP ---
resource "aws_athena_workgroup" "data_team_workgroup" {
  name = "${var.project_name}_data_team_workgroup"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_output.bucket}/query_results/data_team/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }

}


resource "aws_athena_workgroup" "legal_team_workgroup" {
  name = "${var.project_name}_commercial_team_workgroup"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_output.bucket}/query_results/commercial_team/"

      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }

}
