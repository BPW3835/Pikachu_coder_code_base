# =======================================================================
# Build layer image from apache/gravitino base image
# =======================================================================
ARG GRAVITINO_IMAGE_TAG
FROM apache/gravitino:${GRAVITINO_IMAGE_TAG} AS gravitino-base
RUN apt-get update

# =======================================================================
# Build layer image from gravitino-base image
# =======================================================================
ARG HADOOP_AWS_JAR_VERSION
ARG AWS_JAVA_SDK_BUNDLE_JAR_VERSION
FROM gravitino-base AS gravitino

# Copy all initialization scripts
# Commmon initialization scripts
COPY ./infrastructure/common/ ./common/
# Gravitino initialization scripts
COPY ./infrastructure/gravitino/init/ ./init/

# Install Gravitino Dependencies
RUN chmod u+x ./init/dependency.sh
RUN ./init/dependency.sh

# Run Gravitino Initialization Script
RUN chmod u+x ./init/init.sh
ENTRYPOINT [ "./init/init.sh" ]