#!/usr/bin/env bash
# class04c-deploy-gcloud.sh — Lab C (variante FUSIONADA) sin notebook, solo gcloud.
#
# Asume que el modelo ya fue fusionado y subido a GCS por
# class04c-deploy-merged.ipynb (la fusión de pesos LoRA requiere GPU +
# transformers/peft, no tiene sentido como script de gcloud puro). Este
# script cubre solo la parte de infraestructura de Vertex AI: subir el
# modelo al Model Registry, crear el endpoint, desplegar, predecir y limpiar.
#
# Uso:
#   export PROJECT_ID=tu-proyecto-gcp
#   export GCS_MERGED_DIR=gs://tu-bucket/gemma-7b-it-samsum-merged
#   ./class04c-deploy-gcloud.sh upload           # sube el modelo al Model Registry
#   ./class04c-deploy-gcloud.sh create-endpoint   # crea el endpoint (una sola vez)
#   ./class04c-deploy-gcloud.sh deploy            # despliega el modelo en el endpoint
#   ./class04c-deploy-gcloud.sh predict           # una predicción de prueba
#   ./class04c-deploy-gcloud.sh cleanup           # undeploy + borrar endpoint + borrar modelo
#
# Los IDs de modelo/endpoint se guardan en archivos locales (.model_id,
# .endpoint_id, .deployed_model_id) para encadenar los subcomandos sin tener
# que copiarlos a mano.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Exporta PROJECT_ID=tu-proyecto-gcp}"
LOCATION="${LOCATION:-us-central1}"
GCS_MERGED_DIR="${GCS_MERGED_DIR:?Exporta GCS_MERGED_DIR=gs://tu-bucket/gemma-7b-it-samsum-merged}"

MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-gemma-7b-it-samsum-lora-merged}"
ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-${MODEL_DISPLAY_NAME}-endpoint}"

MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-l4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"

# Revisa la versión más reciente en la documentación de Model Garden antes
# de usar esto en producción.
VLLM_DOCKER_URI="${VLLM_DOCKER_URI:-us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20241210_0916_RC00}"

MODEL_ID_FILE=".model_id"
ENDPOINT_ID_FILE=".endpoint_id"
DEPLOYED_MODEL_ID_FILE=".deployed_model_id"

cmd_upload() {
  echo ">> Subiendo modelo fusionado al Model Registry..."
  local vllm_args="--model=${GCS_MERGED_DIR},--tensor-parallel-size=${ACCELERATOR_COUNT},--swap-space=16,--gpu-memory-utilization=0.85,--max-model-len=2048,--dtype=bfloat16,--max-loras=1,--max-cpu-loras=4,--disable-log-stats"

  gcloud ai models upload \
    --project="${PROJECT_ID}" \
    --region="${LOCATION}" \
    --display-name="${MODEL_DISPLAY_NAME}" \
    --container-image-uri="${VLLM_DOCKER_URI}" \
    --container-command="python,-m,vllm.entrypoints.api_server,--host=0.0.0.0,--port=8080" \
    --container-args="${vllm_args}" \
    --container-ports=8080 \
    --container-predict-route=/generate \
    --container-health-route=/ping

  local model_id
  model_id=$(gcloud ai models list \
    --project="${PROJECT_ID}" --region="${LOCATION}" \
    --filter="displayName=${MODEL_DISPLAY_NAME}" \
    --sort-by="~createTime" --limit=1 --format="value(name)")
  echo "${model_id}" > "${MODEL_ID_FILE}"
  echo "Modelo registrado: ${model_id}"
}

cmd_create_endpoint() {
  echo ">> Creando endpoint..."
  gcloud ai endpoints create \
    --project="${PROJECT_ID}" \
    --region="${LOCATION}" \
    --display-name="${ENDPOINT_DISPLAY_NAME}"

  local endpoint_id
  endpoint_id=$(gcloud ai endpoints list \
    --project="${PROJECT_ID}" --region="${LOCATION}" \
    --filter="displayName=${ENDPOINT_DISPLAY_NAME}" \
    --sort-by="~createTime" --limit=1 --format="value(name.basename())")
  echo "${endpoint_id}" > "${ENDPOINT_ID_FILE}"
  echo "Endpoint creado: ${endpoint_id}"
}

cmd_deploy() {
  [ -f "${MODEL_ID_FILE}" ] || { echo "Corre primero: $0 upload"; exit 1; }
  [ -f "${ENDPOINT_ID_FILE}" ] || { echo "Corre primero: $0 create-endpoint"; exit 1; }
  local model_id endpoint_id
  model_id=$(cat "${MODEL_ID_FILE}")
  endpoint_id=$(cat "${ENDPOINT_ID_FILE}")

  echo ">> Desplegando modelo ${model_id} en endpoint ${endpoint_id} (10-20 min)..."
  gcloud ai endpoints deploy-model "${endpoint_id}" \
    --project="${PROJECT_ID}" \
    --region="${LOCATION}" \
    --model="${model_id}" \
    --display-name="${MODEL_DISPLAY_NAME}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
    --min-replica-count=1 \
    --max-replica-count=1

  local deployed_model_id
  deployed_model_id=$(gcloud ai endpoints describe "${endpoint_id}" \
    --project="${PROJECT_ID}" --region="${LOCATION}" \
    --format="value(deployedModels[0].id)")
  echo "${deployed_model_id}" > "${DEPLOYED_MODEL_ID_FILE}"
  echo "Despliegue completo. deployed_model_id=${deployed_model_id}"
}

cmd_predict() {
  [ -f "${ENDPOINT_ID_FILE}" ] || { echo "Corre primero: $0 deploy"; exit 1; }
  local endpoint_id
  endpoint_id=$(cat "${ENDPOINT_ID_FILE}")

  local request_file
  request_file="$(mktemp)"
  cat > "${request_file}" <<JSON
{
  "instances": [
    {
      "prompt": "<start_of_turn>user\nResume el siguiente diálogo en 1-2 frases:\n\nCarlos: ¿Vas a venir a la reunión de las 3pm?\nMarta: Sí, ya salgo. ¿La sala sigue siendo la 402?\nCarlos: Sí, misma sala. Nos vemos ahí.<end_of_turn>\n<start_of_turn>model\n",
      "max_tokens": 64,
      "temperature": 0.0,
      "top_p": 1.0,
      "top_k": -1
    }
  ]
}
JSON

  echo ">> Prediciendo contra el endpoint ${endpoint_id}..."
  gcloud ai endpoints predict "${endpoint_id}" \
    --project="${PROJECT_ID}" \
    --region="${LOCATION}" \
    --json-request="${request_file}"

  rm -f "${request_file}"
}

cmd_cleanup() {
  [ -f "${ENDPOINT_ID_FILE}" ] || { echo "No hay endpoint que limpiar."; exit 0; }
  local endpoint_id
  endpoint_id=$(cat "${ENDPOINT_ID_FILE}")

  if [ -f "${DEPLOYED_MODEL_ID_FILE}" ]; then
    local deployed_model_id
    deployed_model_id=$(cat "${DEPLOYED_MODEL_ID_FILE}")
    echo ">> Retirando el modelo del endpoint..."
    gcloud ai endpoints undeploy-model "${endpoint_id}" \
      --project="${PROJECT_ID}" --region="${LOCATION}" \
      --deployed-model-id="${deployed_model_id}" || true
  fi

  echo ">> Borrando el endpoint..."
  gcloud ai endpoints delete "${endpoint_id}" \
    --project="${PROJECT_ID}" --region="${LOCATION}" --quiet || true

  if [ -f "${MODEL_ID_FILE}" ]; then
    local model_id
    model_id=$(cat "${MODEL_ID_FILE}")
    echo ">> Borrando el modelo del registro..."
    gcloud ai models delete "${model_id}" \
      --project="${PROJECT_ID}" --region="${LOCATION}" --quiet || true
  fi

  rm -f "${MODEL_ID_FILE}" "${ENDPOINT_ID_FILE}" "${DEPLOYED_MODEL_ID_FILE}"
  echo "Limpieza completa. Recuerda borrar también ${GCS_MERGED_DIR} si no lo vas a reutilizar."
}

case "${1:-}" in
  upload)          cmd_upload ;;
  create-endpoint) cmd_create_endpoint ;;
  deploy)          cmd_deploy ;;
  predict)         cmd_predict ;;
  cleanup)         cmd_cleanup ;;
  *)
    echo "Uso: $0 {upload|create-endpoint|deploy|predict|cleanup}"
    exit 1
    ;;
esac
