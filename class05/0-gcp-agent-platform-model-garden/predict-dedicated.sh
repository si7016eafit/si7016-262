PROJECT_ID="si7016-20262"
ENDPOINT_ID="mg-endpoint-3d00f6a8-0674-4044-a5a0-315895c81531"
LOCATION="us-central1"
DEDICATED_HOST="273686586095304704.us-central1-976152022399.prediction.vertexai.goog"
INPUT_DATA_FILE="input.json"

curl \
    -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://${DEDICATED_HOST}/v1/projects/${PROJECT_ID}/locations/${LOCATION}/endpoints/${ENDPOINT_ID}:predict" \
    -d "@${INPUT_DATA_FILE}"