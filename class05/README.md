# Clase 04 — Despliegue de LLMs abiertos en Google Cloud (SI7016, 2026-2)

Esta carpeta reúne tres laboratorios sobre cómo llevar un LLM abierto a producción en GCP, cada uno con un nivel distinto de control vs. facilidad operativa:

| Lab | Opción de despliegue | Control | Operación | Estado |
|---|---|---|---|---|
| **A** | Vertex AI Model Garden (endpoint gestionado) | Bajo | Mínima — Google gestiona el serving | ✅ Listo |
| **B** | GKE + vLLM (cluster propio) | Alto | Tú gestionas el cluster, autoscaling, actualizaciones | 🔜 Pendiente |
| **C** | Cloud Run con GPU (serverless) | Medio | Serverless, escala a cero, ideal para demos | 🔜 Pendiente |

## Nota sobre el nombre de la plataforma (importante)

El 22 de abril de 2026 Google renombró **Vertex AI** a **Gemini Enterprise Agent Platform** en la consola de Google Cloud. Model Garden sigue llamándose igual, ahora bajo el menú "Models" de esa plataforma. El SDK de Python (`google-cloud-aiplatform`, `import vertexai`), la API REST (`aiplatform.googleapis.com`) y los comandos `gcloud ai ...` **no cambiaron de nombre** — todo el código de estos labs sigue siendo válido, solo cambia la navegación en la consola web. Cada lab lo menciona de nuevo en su propia introducción para que quede claro al leerlo de forma independiente.

## Prerrequisitos comunes a los 3 labs

- Un proyecto de Google Cloud con **facturación habilitada** (el uso de GPU no entra en el free tier).
- `gcloud` CLI instalado y autenticado (`gcloud auth login`, `gcloud auth application-default login`).
- Rol IAM `roles/aiplatform.user` (Lab A) y, para los Labs B/C, roles adicionales de GKE/Cloud Run/Artifact Registry según corresponda.
- Cuota de GPU en la región elegida (por defecto `us-central1`). Si el proyecto es nuevo, puede requerir solicitar cuota de `NVIDIA_L4` o `NVIDIA_T4` desde **IAM & Admin → Cuotas**.

Todos los labs son **pedagógicos**: se validan por sintaxis/formato pero no se ejecutan de punta a punta en este entorno (no hay un proyecto GCP con facturación/cuota de GPU disponible aquí). Antes de usarlos en clase, córrelos una vez con tu propio proyecto.

## Lab A — Vertex AI Model Garden

Archivos:

- `class04a-vertex-model-garden.ipynb` — notebook principal: configuración/auth, exploración del catálogo, despliegue de un modelo abierto (Gemma), inferencia (API nativa + API compatible con OpenAI), bonus conectando el endpoint a un agente LangChain (reusando el patrón de `class03a-langchain-agent.ipynb`), costos/buenas prácticas, limpieza, y ejercicio de 5 puntos.
- `class04a-deploy-gcloud.sh` — la misma secuencia de despliegue/predicción/limpieza, pero por línea de comandos (`gcloud ai model-garden models deploy`), para quienes prefieren scripting a notebook.
- `requirements.txt` — dependencias de Python del Lab A.

## Labs B y C (próximas rondas)

- **Lab B (GKE + vLLM):** cluster GKE (Autopilot o Standard) con node pool de GPU, manifiestos de Kubernetes para vLLM sirviendo un modelo abierto vía API compatible con OpenAI, y comparación de control/latencia/costo contra el Lab A.
- **Lab C (Cloud Run + GPU):** contenedor con vLLM u Ollama desplegado en Cloud Run con GPU L4, aprovechando el escalado a cero para cargas intermitentes — buen punto medio entre el Lab A (gestionado pero siempre encendido) y el Lab B (control total pero siempre operando un cluster).

## Fuentes usadas (verificadas, septiembre 2026)

- SDK `vertexai.model_garden`: https://github.com/googleapis/python-aiplatform/blob/main/vertexai/model_garden/README.md
- Notebook oficial de referencia: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/open-models/get_started_with_model_garden_sdk.ipynb
- `gcloud ai model-garden models deploy` / `gcloud ai endpoints ...`: https://cloud.google.com/sdk/gcloud/reference/ai/
- Rebranding Vertex AI → Gemini Enterprise Agent Platform (22 abr. 2026): https://en.wikipedia.org/wiki/Gemini_Enterprise_Agent_Platform
- Serving de modelos abiertos en GKE con vLLM (preview del Lab B): https://docs.cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm
- Cloud Run con GPUs (preview del Lab C): https://cloud.google.com/blog/products/application-development/run-your-ai-inference-applications-on-cloud-run-with-nvidia-gpus/
