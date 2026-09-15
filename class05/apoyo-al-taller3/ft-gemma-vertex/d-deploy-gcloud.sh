#!/usr/bin/env bash
# class04d-deploy-gcloud.sh — Lab D (variante LoRA DINÁMICO) sin notebook, solo gcloud.
#
# A diferencia de class04c-deploy-gcloud.sh, aquí NO hace falta fusionar
# nada de antemano: se despliega el modelo BASE (google/gemma-7b-it,
# descargado de Hugging Face dentro del contenedor) con --enable-lora, y el
# adaptador LoRA se pasa en cada predicción como campo "dynamic-lora"
# apuntando al bucket de GCS del entrenamiento.
#
# Uso:
#   export PROJECT_ID=tu-proyecto-gcp
#   export HF_TOKEN=hf_xxx                     # token con acceso a google/gemma-7b-it
#   export GCS_ADAPTER_DIR=gs://tu-bucket/gemma-7b-it-samsum-lora
#   ./class04d-deploy-gcloud.sh upload
#   ./class04d-deploy-gcloud.sh create-endpoint
#   ./class04d-deploy-gcloud.sh deploy
#   ./class04d-deploy-gcloud.sh predict          # con y sin el adaptador
#   ./class04d-deploy-gcloud.sh cleanup
#
# NUNCA hardcodees HF_TOKEN en este archivo -expórtalo como variable de
# entorno antes de correr "upload" (que es el único subcomando que lo usa).

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Exporta PROJECT_ID=tu-proyecto-gcp}"
LOCATION="${LOCATION:-us-central1}"
GCS_ADAPTER_DIR="${GCS_ADAPTER_DIR:?Exporta GCS_ADAPTER_DIR=gs://tu-bucket/gemma-7b-it-samsum-lora}"

BASE_MODEL_ID="${BASE_MODEL_ID:-google/gemma-7b-it}"
MODEL_DISPLAY_NAME="${MODEL_DISPLAY_NAME:-gemma-7b-it-lora-dinamico}"
ENDPOINT_DISPLAY_NAME="${ENDPOINT_DISPLAY_NAME:-${MODEL_DISPLAY_NAME}-endpoint}"

MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-4}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-l4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
MAX_LORAS="${MAX_LORAS:-2}"

VLLM_DOCKER_URI="${VLLM_DOCKER_URI:-us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20241210_0916_RC00}"

MODEL_ID_FILE=".model_id_lora"
ENDPOINT_ID_FILE=".endpoint_id_lora"
DEPLOYED_MODEL_ID_FILE=".deployed_model_id_lora"

cmd_upload() {
  : "${HF_TOKEN:?Exporta HF_TOKEN=hf_xxx (con acceso a ${BASE_MODEL_ID})}"

  echo ">> Subiendo modelo base (con soporte de LoRA dinámico) al Model Registry..."
  local vllm_args="--model=${BASE_MODEL_ID},--tensor-parallel-size=${ACCELERATOR_COUNT},--swap-space=16,--gpu-memory-utilization=0.85,--max-model-len=2048,--dtype=bfloat16,--max-loras=${MAX_LORAS},--max-cpu-loras=4,--disable-log-stats,--enable-lora"

  gcloud ai models upload \
    --project="${PROJECT_ID}" \
    --region="${LOCATION}" \
    --display-name="${MODEL_DISPLAY_NAME}" \
    --container-image-uri="${VLLM_DOCKER_URI}" \
    --container-command="python,-m,vllm.entrypoints.api_server,--host=0.0.0.0,--port=8080" \
    --container-args="${vllm_args}" \
    --container-ports=8080 \
    --container-predict-route=/generate \
    --container-health-route=/ping \
    --container-env-vars="HF_TOKEN=${HF_TOKEN}"

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

# Construye el mismo prompt con la plantilla de chat de Gemma que usan los
# notebooks (aquí a mano, porque este script no tiene el tokenizer de HF).
GEMMA_PROMPT="<start_of_turn>user
Resume el siguiente diálogo en 1-2 frases:

Carlos: ¿Vas a venir a la reunión de las 3pm?
Marta: Sí, ya salgo. ¿La sala sigue siendo la 402?
Carlos: Sí, misma sala. Nos vemos ahí.<end_of_turn>
<start_of_turn>model
"

cmd_predict() {
  [ -f "${ENDPOINT_ID_FILE}" ] || { echo "Corre primero: $0 deploy"; exit 1; }
  local endpoint_id
  endpoint_id=$(cat "${ENDPOINT_ID_FILE}")

  local request_con_lora request_sin_lora
  request_con_lora="$(mktemp)"
  request_sin_lora="$(mktemp)"

  python3 -c "
import json, sys
prompt = sys.argv[1]
adapter = sys.argv[2]
instance = {'prompt': prompt, 'max_tokens': 64, 'temperature': 0.0, 'top_p': 1.0, 'top_k': -1}
instance['dynamic-lora'] = adapter
print(json.dumps({'instances': [instance]}))
" "${GEMMA_PROMPT}" "${GCS_ADAPTER_DIR}" > "${request_con_lora}"

  python3 -c "
import json, sys
prompt = sys.argv[1]
instance = {'prompt': prompt, 'max_tokens': 64, 'temperature': 0.0, 'top_p': 1.0, 'top_k': -1}
print(json.dumps({'instances': [instance]}))
" "${GEMMA_PROMPT}" > "${request_sin_lora}"

  echo ">> Prediciendo CON adaptador LoRA (${GCS_ADAPTER_DIR})..."
  gcloud ai endpoints predict "${endpoint_id}" \
    --project="${PROJECT_ID}" --region="${LOCATION}" \
    --json-request="${request_con_lora}"

  echo ">> Prediciendo SIN adaptador (modelo base)..."
  gcloud ai endpoints predict "${endpoint_id}" \
    --project="${PROJECT_ID}" --region="${LOCATION}" \
    --json-request="${request_sin_lora}"

  rm -f "${request_con_lora}" "${request_sin_lora}"
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
  echo "Limpieza completa."
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
