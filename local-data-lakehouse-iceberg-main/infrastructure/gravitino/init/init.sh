#!/bin/bash
echo "Start to download the jar packages"

# =======================================================================
# Copy all of the JAR files to their respective folders
# =======================================================================
cp "/root/gravitino/packages/hadoop-aws-${HADOOP_AWS_JAR_VERSION}.jar" /root/gravitino/iceberg-rest-server/libs
cp "/root/gravitino/packages/hadoop-aws-${HADOOP_AWS_JAR_VERSION}.jar" /root/gravitino/catalogs/lakehouse-iceberg/libs
cp "/root/gravitino/packages/aws-java-sdk-bundle-${AWS_JAVA_SDK_BUNDLE_JAR_VERSION}.jar" /root/gravitino/iceberg-rest-server/libs
cp "/root/gravitino/packages/aws-java-sdk-bundle-${AWS_JAVA_SDK_BUNDLE_JAR_VERSION}.jar" /root/gravitino/catalogs/lakehouse-iceberg/libs


# =======================================================================
# Copy all of the configuration in their respective folders
# =======================================================================
cp /tmp/conf/gravitino.conf /root/gravitino/conf/gravitino.conf
cp /tmp/conf/core-site.xml /root/gravitino/conf/core-site.xml
cp /tmp/conf/core-site.xml /root/gravitino/iceberg-rest-server/conf/core-site.xml

# =======================================================================
# Start Gravitino server
# =======================================================================
echo "Finish downloading"
echo "Start the Gravitino Server"
/bin/bash /root/gravitino/bin/gravitino.sh start &
sleep 3
tail -f /root/gravitino/logs/gravitino-server.log