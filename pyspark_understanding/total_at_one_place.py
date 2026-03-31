from pyspark.sql import SparkSession 
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType 
from pyspark.sql.window import Window
from pyspark.sql.functions import avg, sum, col, row_number, round
# Create the SparkSession
spark = SparkSession.builder.getOrCreate()


# Define the data
data = [Row(1, "John", 30, "Sales", 50000.0),
Row(2, "Alice", 28, "Marketing", 60000.0),
Row(3, "Bob", 32, "Finance", 55000.0),
Row(4, "Sarah", 29, "Sales", 52000.0),

Row(5, "Mike", 31, "Finance", 58000.0)
]


# Define the schema
schema = StructType([
StructField("id", IntegerType(), nullable=False), 
StructField("name", StringType(), nullable=False), 
StructField("age", IntegerType(), nullable=False), 
StructField("department", StringType(), nullable=False), 
StructField("salary", DoubleType(), nullable=False)
])


# Create the DataFrame
employeeDF = spark.createDataFrame(data, schema)


# Show the DataFrame
employeeDF.show()











# Question 1:Calculate the average salary for each department:




avg_sal_by_dept = employeeDF.groupBy("department").agg(avg("salary").alias("average_salary_by_dept"))

avg_sal_by_dept.show()






# Question 2:Add a new column named "bonus" that is 10% of the salary for all employees.



employeeDF = employeeDF.withColumn("bonus", col("salary") * 0.10)

employeeDF.show()

# Alternative using selectExpr:
# selectExpr allows you to use SQL-like expressions as strings
employeeDF2 = employeeDF.selectExpr("*", "salary * 0.10 as bonus")
employeeDF2.show()

print("\n=== Difference between withColumn and selectExpr ===")
print("1. withColumn: Uses PySpark column objects and functions")
print("2. selectExpr: Uses SQL-like string expressions")
print("3. withColumn: More type-safe and IDE-friendly")
print("4. selectExpr: More flexible for complex expressions")



# Question 3: Group the data by department and find the employee with the highest salary in each department


windowSpec = Window.partitionBy("department").orderBy("salary")

highest_sal_by_dept = employeeDF.withColumn("row_number", row_number().over(windowSpec))

highest_sal = highest_sal_by_dept.filter(highest_sal_by_dept.row_number == 1)

print("Highest sal by dept soltion is")
highest_sal.show()






data = [
    ("2025-01-01", 1000),
    ("2025-01-02", 1200),
    ("2025-01-03", 1100),
    ("2025-01-04", 1300),
    ("2025-01-05", 1400),
    ("2025-01-06", 1250),
    ("2025-01-07", 1350),
    ("2025-01-08", 1450),
    ("2025-01-09", 1500),
    ("2025-01-10", 1600),
]

df = spark.createDataFrame(data, ["date", "visits"])



# Interview Question:
"Calculate a 7-day moving average of website visits."





window_spec = Window.orderBy("date") \
                    .rowsBetween(-6, 0)  # Current row + 6 previous rows = 7 days

df_with_ma = df.withColumn("moving_avg_7day", 
                            round(avg("visits").over(window_spec), 2))

df_with_ma.show()


