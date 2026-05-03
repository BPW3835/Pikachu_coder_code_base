import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *
from awsglue.dynamicframe import DynamicFrame
import boto3

# --- JOB INITIALIZATION ---
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

datasource = glueContext.create_dynamic_frame.from_catalog(
    database = "ecommerce_data_catalog", 
    table_name = "customer_events_raw"
)

df = datasource.toDF()

total_count = df.count()
null_counts = {c: df.filter(col(c).isNull()).count() for c in df.columns}
df_null_counts = spark.createDataFrame([null_counts])

avg_age = df.select(avg("age")).first()[0]
if avg_age is not None:
    df = df.fillna({"age": avg_age})

has_negative_price = df.filter(col("price") < 0).count() > 0
has_underage_users = df.filter(col("age") < 18).count() > 0

errors = []
if has_negative_price:
    errors.append("- Negative prices detected in 'price' column.")
if has_underage_users:
    errors.append("- Users under the age of 18 detected.")
    
if errors:
    sns_client = boto3.client('sns')
    error_report = "\n".join(errors)
    sns_client.publish(
        TopicArn='arn:aws:sns:us-east-1:412381764682:DataQualityReport',
        Subject='Critical Data Quality Alert',
        Message=f'Attention, \n\nThe following data quality issues detected:\n{error_report}'
    )    
    
df = df.withColumn("is_priority", 
                    when((col("event_type") == "purchase") & (col("price") > 1000), True)
                    .otherwise(False))

df = df.withColumn("to_be_blocked", 
                    when(col("age") < 18, True)
                    .otherwise(False))

avg_age = df.select(avg("age")).first()[0]
df = df.fillna({"age": avg_age})

# Partitioning columns for both dataframes
df = df.withColumn("year", year(col("timestamp"))) \
       .withColumn("month", month(col("timestamp"))) \
       .withColumn("day", day(col("timestamp")))

df_null_counts = df_null_counts.withColumn("year", lit(year(current_date()))) \
                               .withColumn("month", lit(month(current_date()))) \
                               .withColumn("day", lit(day(current_date())))

dynamic_df = DynamicFrame.fromDF(df, glueContext, "df")

glueContext.write_dynamic_frame.from_options(
    frame=dynamic_df,
    connection_type="s3",
    connection_options={
        "path": "s3://ecommerce-processed-data-hsnlsk1/customer_events_processed/",
        "partitionKeys": ["year", "month", "day"]
    },
    format="parquet"
)

dynamic_report = DynamicFrame.fromDF(df_null_counts, glueContext, "dynamic_report")
glueContext.write_dynamic_frame.from_options(
    frame = dynamic_report,
    connection_type="s3",
    connection_options = {
        "path": "s3://ecommerce-processed-data-hsnlsk1/data_quality/",
        "partitionKeys": ["year", "month", "day"]
    },
    format = "parquet"
)

job.commit()