"""
 Day 1 – PySpark Scenario-Based Interview Question

You are working as a Data Engineer in a fintech company.

You receive daily transaction logs (CSV) with:

- transaction_id (string)
- user_id (string)
- amount (double)
- transaction_type (string: credit/debit)
- transaction_timestamp (string)

You are asked to:

1. Load the CSV into PySpark with an explicit schema
2. Remove duplicate transactions
3. Filter out records where amount <= 0
4. Convert transaction_timestamp to proper timestamp format
5. Calculate total debit and credit amount per day
6. Store the result in Parquet format partitioned by date

How would you implement this? Attached in Image.

Now the real question:

But senior engineers think about:

- What about corrupt records?
- Should overwrite mode be used in production?
- How do we handle late-arriving data?
- What about small file problems?
- How do we optimize shuffle?
- Is partitioning strategy correct for downstream queries?

Writing Spark code is easy. Designing scalable pipelines is not.

"""


from pyspark import SparkSession
from pyspark.sql.functions import col, sum, to_timestamp, to_date
from pyspark.sql.types import StructType, StructField, StringType, DoubleType


# Create Spark Session
spark = SparkSession.builder.appName("FintechTransactionPipeline").getOrCreate()

# Defining explicit schema 
schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("transaction_timestamp", StringType(), True)
])

# 1. Load the CSV
df = spark.read.option("header", True).schema(schema).csv('path/tp/transactions.csv')


# 2. Remove duplicates
df_dedup = df.dropDuplicates(["transaction_id"])

# 3. Filter Invalid
df_valid = df_dedup.filter(col("amount") > 0)


# 4. convert Timestamp & Extarct Date
df_transformed = df.withColumn("transaction_ts", to_timestamp(col("transaction_timestamp"))).withColumn("transaction_date", to_date(col("transaction_ts")))


# 5. aggregate debit and credit Per Day
daily_summary = df_transformed.groupBy("transaction_date", "transaction_type").agg(sum("amount").alias("total_amount"))

# Write to Parquet Partitioned by date 
daily_summary.write.mode("overwrite").partitionBy("transaction_date").parquet("path/to/output")