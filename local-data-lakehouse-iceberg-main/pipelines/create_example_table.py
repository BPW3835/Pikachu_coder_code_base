from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, LongType, TimestampType
from datetime import datetime


def create_table(spark: SparkSession) -> None:
    schema = StructType(
        [
            StructField("id", LongType(), False),
            StructField("name", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("created_datetime", TimestampType(), True),
        ]
    )

    data = [
        (1, "Alice", 30, datetime.now()),
        (2, "Bob", 25, datetime.now()),
        (3, "Charlie", 35, datetime.now()),
    ]

    (
        spark
        .createDataFrame(data, schema)
        .write.format("iceberg")
        .mode("overwrite")
        .saveAsTable("catalog_iceberg.schema_iceberg.table_iceberg")
    )

if __name__ == "__main__":
    spark = SparkSession.builder.appName("SparkSessionExample").getOrCreate()
    create_table(spark)
