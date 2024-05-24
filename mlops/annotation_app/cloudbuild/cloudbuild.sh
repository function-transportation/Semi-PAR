REGION="us-central1"
PROJECT_ID="camera-app-391105"

gcloud builds submit --region=${REGION} --project=${PROJECT_ID} --config=cloudbuild.yaml ../
