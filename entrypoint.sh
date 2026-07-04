#!/usr/bin/env bash
# Runs ONCE when a Cloud Run instance boots: pull the latest DB, then serve.
# Not per-request — the running instance reuses this copy for every request.
set -euo pipefail

mkdir -p "$(dirname "${FENCING_DB_PATH}")"

if [ -n "${DB_BUCKET_URI}" ]; then
  echo "Fetching DB from ${DB_BUCKET_URI} -> ${FENCING_DB_PATH}"
  gcloud storage cp "${DB_BUCKET_URI}" "${FENCING_DB_PATH}"
else
  echo "DB_BUCKET_URI not set; expecting a DB already at ${FENCING_DB_PATH}"
fi

exec streamlit run explorer/app.py \
  --server.port "${PORT:-8080}" \
  --server.address 0.0.0.0 \
  --server.headless true
