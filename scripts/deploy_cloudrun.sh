#!/usr/bin/env bash
# Optional cloud deployment: build + deploy to Google Cloud Run (fully managed).
# Prereqs: gcloud CLI authenticated, a GCP project with Cloud Run + Artifact Registry enabled.
set -euo pipefail

PROJECT_ID="${1:?usage: deploy_cloudrun.sh <gcp-project-id> [region]}"
REGION="${2:-europe-west3}"
SERVICE="payback-assistant"

gcloud builds submit --project "$PROJECT_ID" --tag "gcr.io/$PROJECT_ID/$SERVICE" .

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --image "gcr.io/$PROJECT_ID/$SERVICE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --cpu 1 --memory 512Mi \
  --min-instances 0 --max-instances 10 \
  --concurrency 80

echo "Deployed. Test with:"
echo "  curl -X POST \"\$(gcloud run services describe $SERVICE --region $REGION --project $PROJECT_ID --format 'value(status.url)')/assist\" -H 'Content-Type: application/json' -d '{\"query\": \"günstige Windeln\"}'"
