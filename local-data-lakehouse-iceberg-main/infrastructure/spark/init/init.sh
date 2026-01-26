#!/bin/bash

# =======================================================================
# Copy all of the JAR files to their respective folders
# =======================================================================
cp ./packages/iceberg-spark-runtime-${SPARK_SCALA_VERSION}-${ICEBERG_VERSION}.jar /root/spark/jars/iceberg-spark-runtime-${SPARK_SCALA_VERSION}-${ICEBERG_VERSION}
cp ./packages/postgresql-${POSTGRESQL_JAR_VERSION}.jar /root/spark/jars/postgresql-${POSTGRESQL_JAR_VERSION}.jar
cp ./packages/hadoop-aws-${HADOOP_AWS_JAR_VERSION}.jar /root/spark/jars/hadoop-aws-${HADOOP_AWS_JAR_VERSION}.jar
cp ./packages/aws-java-sdk-bundle-${AWS_JAVA_SDK_BUNDLE_JAR_VERSION}.jar /root/spark/jars/aws-java-sdk-bundle-${AWS_JAVA_SDK_BUNDLE_JAR_VERSION}.jar

# =======================================================================
# Copy all of the configuration in their respective folders
# =======================================================================
cp /tmp/conf/spark-defaults.conf ${SPARK_HOME}/conf/spark-defaults.conf
cp /tmp/conf/core-site.xml ${SPARK_HOME}/conf/core-site.xml

# =======================================================================
# Create metalake and catalog in Gravitino
# =======================================================================
sh /tmp/common/init_metalake_catalog.sh

# Run spark based on the workload type
SPARK_WORKLOAD=$1
echo "SPARK_WORKLOAD: $SPARK_WORKLOAD"
if [ "$SPARK_WORKLOAD" == "master" ];
then
    start-master.sh -p 7077
elif [ "$SPARK_WORKLOAD" == "worker" ];
then
    start-worker.sh spark://spark-master:7077
elif [ "$SPARK_WORKLOAD" == "history" ]
then
    start-history-server.sh
fi