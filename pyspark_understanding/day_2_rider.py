"""
    Scenario:

You are a Senior Data Engineer at a ride-sharing company.

You receive ~500GB of trip data daily in Parquet format.

Schema:

- trip_id (string)
- driver_id (string)
- city (string)
- fare_amount (double)
- trip_distance (double)
- trip_timestamp (timestamp)

Business Requirements:

1. Identify the Top 3 drivers per city per day based on total fare.
2. Data may arrive late and may contain duplicate records.
3. Some cities generate 10x more data than others (data skew).
4. Output must support fast downstream analytical queries.
5. The solution must scale reliably as data grows.

How would you implement this in PySpark?

Here’s a production-oriented approach (Attached)

But Strong engineers will ask:

• How do we handle skewed cities?
→ Salting keys before aggregation.

• Should we use row_number or dense_rank?
→ Depends on tie-breaking requirement.

• What about late-arriving data?
→ Incremental processing + watermark strategy.

• Is overwrite safe for production?
→ Partition overwrite mode should be enabled.

• How do we avoid small file problems?
→ Controlled repartitioning and file sizing.

• Why partition by trip_date?
→ Enables partition pruning for downstream queries.

"""








from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, sum, to_date, to_timestamp, row_number
from pyspark.sql.window import Window


spark = SparkSession.builder.appName("RideSharingDriverRanking").getOrCreate()



# 1. Load Parquet Data (columnar format for performance)
df = spark.read.parquet('path/to/trip_data')



# 2. Remove Duplicates (Idempotent Processing)
df_dedup = df.dropDuplicates(["trip_id"])


# 3. Extract Trip-date as a new column
df_with_date = df_dedup.withColumn("trip_date", to_date("trip_timestamp"))


# 4. Aggregate total fare per driver per City per Day
