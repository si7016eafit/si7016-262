"""
submit_vertex_job.py — Envía train.py como un Custom Training Job de Vertex AI
(la sección "Training" que vive junto a "Models -> Tuning" dentro de Agent
Platform), usando una imagen propia construida sobre el contenedor de
entrenamiento de Hugging Face (que ya trae transformers/peft/trl/bitsandbytes).

IMPORTANTE — por qué usamos CustomContainerTrainingJob y no script_path:
`CustomTrainingJob(script_path=...)` ("autopackaging") solo acepta imágenes
de una lista blanca interna de Vertex AI para "Python package training".
La imagen de Hugging Face no está en esa lista y el job es rechazado con
"image ... is not supported". CustomContainerTrainingJob no tiene esa
restricción: corre cualquier imagen de un registro de contenedores, tal como
la documenta la propia Hugging Face para este mismo caso. Por eso hay que
construir una imagen (con train.py ya adentro) y referenciarla aquí.

PASO 0 (una sola vez): construir y subir la imagen
----------------------------------------------------
    gcloud artifacts repositories create si7016-taller3 \
        --repository-format=docker --location=us-central1 \
        --description="Imágenes de fine-tuning taller 3"

    gcloud builds submit --tag \
        us-central1-docker.pkg.dev/TU_PROJECT_ID/si7016-taller3/qwen3-8b-qlora:latest .

(Repite el `gcloud builds submit` cada vez que cambies train.py.)

Requisitos previos generales (una sola vez):
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
            --image_uri us-central1-docker.pkg.dev/myproyectsi4002-262/si7016-taller3/qwen3-8b-qlora:latest
"""

import argparse

from google.cloud import aiplatform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="Tu GCP project id")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--bucket", required=True, help="Nombre del bucket de GCS (sin gs://)")
    p.add_argument("--image_uri", required=True,
                    help="Imagen construida en el PASO 0, ej. "
                         "us-central1-docker.pkg.dev/TU_PROJECT_ID/si7016-taller3/qwen3-8b-qlora:latest")
    p.add_argument("--display_name", default="qwen3-8b-samsum-qlora-sft")
    p.add_argument("--machine_type", default="g2-standard-12",
                    help="g2-standard-12 + NVIDIA_L4 es un buen default; "
                         "n1-standard-4 + NVIDIA_TESLA_T4 es más barato y ya validado en el notebook (T4, 15.6GB)")
    p.add_argument("--accelerator_type", default="NVIDIA_L4",
                    choices=["NVIDIA_L4", "NVIDIA_TESLA_T4", "NVIDIA_TESLA_A100"])
    p.add_argument("--accelerator_count", type=int, default=1)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--max_steps", type=int, default=-1,
                    help="Deja -1 para entrenar las 3 épocas completas; usa p.ej. 20 para una corrida corta de prueba")
    return p.parse_args()


def main():
    args = parse_args()
    bucket_uri = f"gs://{args.bucket}"

    aiplatform.init(project=args.project, location=args.region, staging_bucket=bucket_uri)

    job = aiplatform.CustomContainerTrainingJob(
        display_name=args.display_name,
        container_uri=args.image_uri,  # tu imagen (Dockerfile) con train.py ya adentro
    )

    output_dir = f"/gcs/{args.bucket}/qwen3-8b-samsum-lora"

    script_args = [
        "--model_name=Qwen/Qwen3-8B",
        "--dataset_name=knkarthick/samsum",
        f"--output_dir={output_dir}",
        f"--max_steps={args.max_steps}",
    ]

    env_vars = {
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_LOG_LEVEL": "INFO",
    }
    if args.hf_token:
        env_vars["HF_TOKEN"] = args.hf_token

    job.run(
        args=script_args,
        replica_count=1,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        environment_variables=env_vars,
        base_output_dir=f"{bucket_uri}/qwen3-8b-samsum-lora-job-artifacts",
        sync=True,  # bloquea hasta que el job termine; pon False para lanzarlo y seguir trabajando
    )

    print(f"Job terminado. Adaptadores LoRA en: {bucket_uri}/qwen3-8b-samsum-lora")


if __name__ == "__main__":
    main()
