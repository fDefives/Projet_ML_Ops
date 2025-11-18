FROM python:3.11-slim

# Dépendances de base
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    tar \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Installer Java 17 (Temurin)
RUN wget https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.13+11/OpenJDK17U-jdk_x64_linux_hotspot_17.0.13_11.tar.gz \
    && mkdir -p /usr/lib/jvm \
    && tar -xzf OpenJDK17U-jdk_x64_linux_hotspot_17.0.13_11.tar.gz -C /usr/lib/jvm \
    && rm OpenJDK17U-jdk_x64_linux_hotspot_17.0.13_11.tar.gz

ENV JAVA_HOME=/usr/lib/jvm/jdk-17.0.13+11
ENV PATH="$JAVA_HOME/bin:$PATH"

# Installer PySpark et libs images
RUN pip install --no-cache-dir pyspark pillow numpy

WORKDIR /app
COPY segmentation_spark.py /app/segmentation_spark.py

CMD ["python", "segmentation_spark.py"]

