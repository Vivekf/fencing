# Base includes the gcloud CLI so entrypoint.sh can pull the DB from GCS.
FROM google/cloud-sdk:slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

# DB_BUCKET_URI is set at deploy time (e.g. gs://my-project-fencing/fencing.db).
# FENCING_DB_PATH is where entrypoint.sh drops the downloaded copy; data.py reads it.
ENV FENCING_DB_PATH=/data/fencing.db \
    DB_BUCKET_URI=""

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
