#!/usr/bin/env bash
# One-time deploy of the Fencing Explorer to Cloud Run, gated by Google login (IAP).
# No domain / no load balancer: IAP is enabled directly on the *.run.app URL.
# Run the sections top-to-bottom. Re-running the build+deploy section is safe.
set -euo pipefail

# ---- fill these in --------------------------------------------------------
PROJECT=fencing-explorer-app            # gcloud projects list
REGION=us-central1
ALLOWED_USER=vivekf@gmail.com
# ---------------------------------------------------------------------------
BUCKET="gs://${PROJECT}-fencing"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/apps/fencing-explorer:latest"
SERVICE=fencing-explorer

gcloud config set project "$PROJECT"

# 1. Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  storage.googleapis.com cloudbuild.googleapis.com iap.googleapis.com

# 2. Bucket + upload the current DB (this is the app/cron handoff point)
gcloud storage buckets create "$BUCKET" --location="$REGION" || true
gcloud storage cp fencing.db "$BUCKET/fencing.db"

# 3. Artifact Registry repo (ignore error if it already exists)
gcloud artifacts repositories create apps \
  --repository-format=docker --location="$REGION" || true

# 4. Build the image from this folder (.dockerignore keeps the upload small)
gcloud builds submit --tag "$IMAGE"

# 5. Deploy — private, one warm instance so the DB is pulled ~once, not per hit
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" \
  --set-env-vars "DB_BUCKET_URI=${BUCKET}/fencing.db" \
  --memory 4Gi --cpu 2 --min-instances 1 --max-instances 1 \
  --no-allow-unauthenticated

# 6. Let the service's identity read the DB bucket
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${RUN_SA}" --role=roles/storage.objectViewer

# 7. Turn on IAP for the service. If this flag errors on your gcloud version,
#    do it in the console: Cloud Run -> fencing-explorer -> Security -> enable IAP.
#    (You'll be prompted once to configure the OAuth consent screen: pick
#     "External", app name "Fencing Explorer", add ALLOWED_USER as a test user.)
gcloud run services update "$SERVICE" --region "$REGION" --iap

# 8. Let IAP invoke the service
IAP_SA="service-${PROJECT_NUMBER}@gcp-sa-iap.iam.gserviceaccount.com"
gcloud run services add-iam-policy-binding "$SERVICE" --region "$REGION" \
  --member="serviceAccount:${IAP_SA}" --role=roles/run.invoker

# 9. Grant yourself access through IAP
gcloud beta iap web add-iam-policy-binding \
  --resource-type=cloud-run --region="$REGION" --service="$SERVICE" \
  --member="user:${ALLOWED_USER}" --role=roles/iap.httpsResourceAccessor

echo
echo "Done. Your URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
echo "Open it on your phone -> sign in as ${ALLOWED_USER}."
