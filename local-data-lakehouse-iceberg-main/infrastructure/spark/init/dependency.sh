#!/bin/bash

# =======================================================================
# Load common functions
# =======================================================================
. "/root/spark/common/common.sh"

# =======================================================================
# Prepare target directory for packages
# =======================================================================
target_dir="/root/spark"
if [[ ! -d "${target_dir}/packages" ]]; then
  mkdir -p "${target_dir}/packages"
fi

# =======================================================================
# Download Iceberg Spark Runtime JAR and MD5
# Why?: This is needed for Spark to work with Iceberg tables
# =======================================================================
ICEBERG_SPARK_RUNTIME_JAR="https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-${SPARK_SCALA_VERSION}/${ICEBERG_VERSION}/iceberg-spark-runtime-${SPARK_SCALA_VERSION}-${ICEBERG_VERSION}.jar"
ICEBERG_SPARK_RUNTIME_MD5="${ICEBERG_SPARK_RUNTIME_JAR}.md5"
download_and_verify "${ICEBERG_SPARK_RUNTIME_JAR}" "${ICEBERG_SPARK_RUNTIME_MD5}" ${target_dir}

# =======================================================================
# Download PostgreSQL JAR and MD5
# Why?: This is needed for Spark to connect to REST Catalog that uses PostgreSQL as backend
# =======================================================================
POSTGRESQL_JAR="https://repo1.maven.org/maven2/org/postgresql/postgresql/${POSTGRESQL_JAR_VERSION}/postgresql-${POSTGRESQL_JAR_VERSION}.jar"
POSTGRESQL_MD5="${POSTGRESQL_JAR}.md5"
download_and_verify "${POSTGRESQL_JAR}" "${POSTGRESQL_MD5}" ${target_dir}

# =======================================================================
# Download Hadoop AWS JAR and MD5
# Why?: This is needed for Spark to connect to S3 compatible storage like MinIO
# =======================================================================
HADOOP_AWS_JAR="https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_JAR_VERSION}/hadoop-aws-${HADOOP_AWS_JAR_VERSION}.jar"
HADOOP_AWS_MD5="${HADOOP_AWS_JAR}.md5"
download_and_verify "${HADOOP_AWS_JAR}" "${HADOOP_AWS_MD5}" ${target_dir}

# =======================================================================
# Download AWS Java SDK Bundle JAR and MD5
# Why?: This is needed for Gravitino to connect to S3 compatible storage like MinIO
# =======================================================================
AWS_JAVA_SDK_BUNDLE_JAR="https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_JAVA_SDK_BUNDLE_JAR_VERSION}/aws-java-sdk-bundle-${AWS_JAVA_SDK_BUNDLE_JAR_VERSION}.jar"
AWS_JAVA_SDK_BUNDLE_MD5="${AWS_JAVA_SDK_BUNDLE_JAR}.md5"
download_and_verify "${AWS_JAVA_SDK_BUNDLE_JAR}" "${AWS_JAVA_SDK_BUNDLE_MD5}" ${target_dir}

