"""
submit_vertex_job_gemma.py — Envía train_gemma.py como un Custom Training Job
de Vertex AI (Agent Platform -> Training), usando la imagen construida con
Dockerfile.gemma (la imagen de Hugging Face SIN modificar, + train_gemma.py).

Es el mismo mecanismo que submit_vertex_job.py (CustomContainerTrainingJob,
por la misma razón: la imagen de Hugging Face no está en la lista blanca de
"autopackaging"), pero apunta a google/gemma-7b-it en vez de Qwen/Qwen3-8B.
Este pipeline es independiente del de Qwen3 — no lo reemplaza, corre en
paralelo con su propia imagen y su propio nombre de job.

REQUISITO IMPORTANTE — Gemma es un modelo "gated" en Hugging Face:
    1. Entra a https://huggingface.co/google/gemma-7b-it con tu cuenta de HF
       y acepta la licencia (botón "Acknowledge license").
    2. Genera un token de acceso en https://huggingface.co/settings/tokens
       (basta con permisos de lectura).
    3. Pásalo con --hf_token al ejecutar este script.
Sin esto, la descarga del modelo falla con un error 401/403 "gated repo".

PASO 0 (una sola vez): construir y subir la imagen
----------------------------------------------------
    # Si ya creaste el repositorio si7016-taller3 para la imagen de Qwen3,
    # NO hace falta crearlo de nuevo — puedes reusarlo con otro tag.
    gcloud artifacts repositories create si7016-taller3 \
        --repository-format=docker --location=us-central1 \
        --description="Imágenes de fine-tuning taller 3"

    gcloud builds submit --tag \
        us-central1-docker.pkg.dev/myproyectsi4002-262/si7016-taller3/gemma-7b-it-qlora:latest .

(Repite el `gcloud builds submit` cada vez que cambies train_gemma.py.
 Nota el `-f Dockerfile.gemma`: por defecto `gcloud builds submit` usa un
 archivo llamado `Dockerfile`, así que hay que indicarle explícitamente
 cuál usar ya que en esta carpeta conviven los dos Dockerfile.)

Requisitos previos generales (una sola vez, iguales a los de Qwen3):
    pip3 install --upgrade google-cloud-aiplatform
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project TU_PROJECT_ID
    gcloud services enable aiplatform.googleapis.com compute.googleapis.com \
        container.googleapis.com containerregistry.googleapis.com \
        cloudbuild.googleapis.com artifactregistry.googleapis.com
    gcloud storage buckets create gs://TU_BUCKET --location=us-central1

Uso:
    python3 submit_vertex_job.py \
        --project myproyectsi4002-262 \
        --bucket si7016emontoya2 \
        --region us-central1 \
        --image_uri us-central1-docker.pkg.dev/myproyectsi4002-262/si7016-taller3/gemma-7b-it-qlora:latest \
        --model_name google/gemma-7b-it \
        --hf_token hf_xxxx
"""

import argparse

from google.cloud import aiplatform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Tu GCP project id")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--bucket", required=True, help="Nombre del bucket de GCS (sin gs://); debe ser REGIONAL y estar en la misma región que --region")
    p.add_argument("--image_uri", required=True,
                    help="Imagen construida en el PASO 0 con Dockerfile.gemma, ej. "
                         "us-central1-docker.pkg.dev/TU_PROJECT_ID/si7016-taller3/gemma-7b-it-qlora:latest")
    p.add_argument("--display_name", default="gemma-7b-it-samsum-qlora-sft")
    p.add_argument("--machine_type", default="g2-standard-12",
                    help="g2-standard-12 + NVIDIA_L4 es un buen default; "
                         "n1-standard-4 + NVIDIA_TESLA_T4 es más barato si tienes cuota de T4 y no de L4")
    p.add_argument("--accelerator_type", default="NVIDIA_L4",
                    choices=["NVIDIA_L4", "NVIDIA_TESLA_T4", "NVIDIA_TESLA_A100"])
    p.add_argument("--accelerator_count", type=int, default=1)
    p.add_argument("--hf_token", required=True,
                    help="Token de Hugging Face con acceso aceptado a google/gemma-7b-it (obligatorio: Gemma es 'gated')")
    p.add_argument("--model_name", default="google/gemma-7b-it",
                    help="Alternativa más liviana si hay poca cuota/memoria: google/gemma-2b-it")
    p.add_argument("--max_steps", type=int, default=-1,
                    help="Deja -1 para entrenar las 3 épocas completas; usa p.ej. 20 para una corrida corta de prueba")
    return p.parse_args()


def main():
    args = parse_args()
    bucket_uri = f"gs://{args.bucket}"

    aiplatform.init(project=args.project, location=args.region, staging_bucket=bucket_uri)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=args.display_name,
        container_uri=args.image_uri,  # imagen (Dockerfile.gemma) con train_gemma.py ya adentro
    )

    output_dir = f"/gcs/{args.bucket}/gemma-7b-it-samsum-lora"

    script_args = [
        f"--model_name={args.model_name}",
        "--dataset_name=knkarthick/samsum",
        f"--output_dir={output_dir}",
        f"--max_steps={args.max_steps}",
    ]

    env_vars = {
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_LOG_LEVEL": "INFO",
        "HF_TOKEN": args.hf_token,
    }

    job.run(
        args=script_args,
        replica_count=1,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        environment_variables=env_vars,
        base_output_dir=f"{bucket_uri}/gemma-7b-it-samsum-lora-job-artifacts",
        sync=True,  # bloquea hasta que el job termine; pon False para lanzarlo y seguir trabajando
    )

    print(f"Job terminado. Adaptadores LoRA en: {bucket_uri}/gemma-7b-it-samsum-lora")


if __name__ == "__main__":
    main()
