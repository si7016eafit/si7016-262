ENDPOINT_ID="mg-endpoint-3d00f6a8-0674-4044-a5a0-315895c81531"
PROJECT_ID="si7016-20262"
INPUT_DATA_FILE="input.json"

curl \
    -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict" \
    -d "@${INPUT_DATA_FILE}"