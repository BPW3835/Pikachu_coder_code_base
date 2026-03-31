# --- DATA TRANSFORMER LAMBDA (INGESTION) ---
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/scripts/firehose_transform.py"
  output_path = "${path.module}/scripts/firehose_transform.zip"
}

resource "aws_lambda_function" "data_transformer" {
  filename      = data.archive_file.lambda_zip.output_path
  function_name = var.firehose_lambda_function_name
  role          = aws_iam_role.lambda_role.arn
  handler       = "firehose_transform.handler" 
  runtime       = "python3.10"
  timeout       = 60
  memory_size   = 128

  lifecycle {
    ignore_changes = [
      filename,
      source_code_hash
    ]
  }
}

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${var.firehose_lambda_function_name}"
  retention_in_days = 7
}

resource "aws_lambda_permission" "allow_firehose" {
  statement_id  = "AllowExecutionFromFirehose" # id of the rule
  principal     = "firehose.amazonaws.com" # firehose can do that
  source_arn    = aws_kinesis_firehose_delivery_stream.ecommerce_firehose.arn # but just this spesific firehose
  action        = "lambda:InvokeFunction" # what firehose can do 
  function_name = aws_lambda_function.data_transformer.function_name # but only with this function
}