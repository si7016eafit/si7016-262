"""
infer_gemma_vertex.py — Probar con ejemplos los adaptadores LoRA entrenados
con un Custom Training Job de Vertex AI / Gemini Enterprise Agent Platform
(Agent Platform -> Models -> Training), a diferencia de infer_gemma.py que
asume que ya entrenaste dentro de esta misma VM+JupyterHub.

La diferencia real frente a infer_gemma.py es una sola: el training job de
Vertex AI no escribe los adaptadores en el disco de esta VM, los escribe en
un bucket de Cloud Storage (típicamente en la carpeta que le pasaste como
`--output_dir` al job, algo como gs://tu-bucket/gemma-7b-it-samsum-lora/).
Este script descarga esa carpeta a un caché local y de ahí en adelante usa
exactamente la misma lógica de generación que infer_gemma.py.

Cómo se descarga:
    Usamos el cliente de Python `google-cloud-storage` en vez de llamar a
    `gcloud storage cp` / `gsutil` por dos razones:
      1. La imagen quay.io/jupyter/tensorflow-notebook:x86_64-cuda-latest no
         trae la CLI de gcloud instalada (sí trae Python).
      2. El cliente de Python toma credenciales automáticamente del servidor
         de metadata de la VM (Application Default Credentials) -no hace
         falta montar ni copiar ningún archivo de service account, siempre
         que la VM tenga permisos de lectura sobre el bucket (rol
         "Storage Object Viewer" o superior en la cuenta de servicio de la VM).

Instalar antes (además de lo que ya instalaste para entrenar/inferir):
    pip install google-cloud-storage

Uso:
    python infer_gemma_vertex.py --gcs_adapter_dir gs://tu-bucket/gemma-7b-it-samsum-lora
    python infer_gemma_vertex.py --gcs_adapter_dir gs://tu-bucket/... --compare_base
    python infer_gemma_vertex.py --gcs_adapter_dir gs://tu-bucket/... --dialogue "A: ...\\nB: ..."

Si ya descargaste antes y el caché local sigue ahí, la segunda corrida NO
vuelve a descargar (a menos que pases --force_download) -así puedes iterar
rápido sin repetir la transferencia desde el bucket.
"""

import argparse
import os


EJEMPLOS = [
    # Los primeros dos son diálogos "de juguete" (no vistos en el dataset de
    # entrenamiento); el tercero es del estilo samsum -mensajes cortos e
    # informales, como los que sí vio en el fine-tuning.
    "Carlos: ¿Vas a venir a la reunión de las 3pm?\n"
    "Marta: Sí, ya salgo. ¿La sala sigue siendo la 402?\n"
    "Carlos: Sí, misma sala. Nos vemos ahí.",

    "Sofía: Se me quedó el cargador en tu casa ayer, ¿lo tienes ahí?\n"
    "Diego: Sí, lo vi en la mesa de la sala. Te lo llevo mañana a la oficina.\n"
    "Sofía: Perfecto, gracias!",

    "Ana: hey are we still on for the gym at 6?\n"
    "Leo: yeah but running a bit late, more like 6:20\n"
    "Ana: no worries, I'll grab a locker and wait",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/gemma-7b-it")
    p.add_argument("--gcs_adapter_dir",
                    default=os.environ.get("GCS_ADAPTER_DIR", "gs://[tu-bucket]/gemma-7b-it-samsum-lora"),
                    help="Carpeta en Cloud Storage donde el Custom Training Job de Vertex AI "
                         "guardó los adaptadores LoRA (el --output_dir que le pasaste al job).")
    p.add_argument("--local_cache_dir",
                    default=os.environ.get("LOCAL_CACHE_DIR", "/home/jovyan/labs/gemma-7b-it-samsum-lora-vertex"),
                    help="Carpeta local (dentro del volumen montado) donde se descarga una copia "
                         "de los adaptadores, para no tener que bajarlos de nuevo cada vez.")
    p.add_argument("--force_download", action="store_true",
                    help="Volver a descargar aunque ya exista una copia en --local_cache_dir "
                         "(útil si volviste a entrenar y el bucket tiene una versión más nueva).")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                    help="Solo necesario si los pesos base de Gemma no quedaron ya en caché "
                         "de una corrida anterior. También puedes exportar HF_TOKEN.")
    p.add_argument("--dialogue", default=None,
                    help="Un diálogo propio para resumir, en vez de los ejemplos por defecto")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--compare_base", action="store_true",
                    help="Además de generar con el modelo afinado, generar con el modelo "
                         "base SIN adaptadores, para comparar el efecto del fine-tuning")
    return p.parse_args()


def download_gcs_dir(gcs_uri, local_dir, force=False):
    """Descarga (recursivamente) el contenido de una carpeta gs://... a
    local_dir, usando el cliente de Python de Cloud Storage. Si local_dir ya
    existe y no está vacía, y force=False, no vuelve a descargar."""
    if not gcs_uri.startswith("gs://"):
        raise SystemExit(
            f"--gcs_adapter_dir debe empezar con 'gs://' (recibí: {gcs_uri!r}). "
            "Copia la ruta exacta que le pasaste como --output_dir al Custom Training Job."
        )

    if os.path.isdir(local_dir) and os.listdir(local_dir) and not force:
        print(f"Ya existe una copia local en {local_dir} (usa --force_download para "
              "volver a bajarla). Saltando descarga.")
        return local_dir

    try:
        from google.cloud import storage
    except ImportError:
        raise SystemExit(
            "Falta el cliente de Cloud Storage. Corre primero:\n\n"
            "    pip install google-cloud-storage\n"
        )

    bucket_name, _, prefix = gcs_uri[len("gs://"):].partition("/")
    prefix = prefix.rstrip("/")

    print(f"Descargando {gcs_uri} -> {local_dir} ...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix + "/" if prefix else prefix))
    # Filtramos "carpetas" vacías (blobs que terminan en "/" y no tienen contenido)
    blobs = [b for b in blobs if not b.name.endswith("/")]

    if not blobs:
        raise SystemExit(
            f"No encontré archivos en {gcs_uri}. Revisa que el Custom Training Job haya "
            "terminado exitosamente y que la ruta (bucket + carpeta) sea exactamente la "
            "que le pasaste como --output_dir / AIP_MODEL_DIR."
        )

    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        rel_path = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
        dest = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(dest) or local_dir, exist_ok=True)
        blob.download_to_filename(dest)
        print(f"  descargado: {rel_path}")

    print(f"Descarga completa: {len(blobs)} archivo(s) en {local_dir}")
    return local_dir


def build_prompt(tokenizer, dialogue):
    messages = [
        {"role": "user", "content": f"Resume el siguiente diálogo en 1-2 frases:\n\n{dialogue}"},
    ]
    # add_generation_prompt=True: agrega el "<start_of_turn>model\n" para que
    # el modelo sepa que le toca generar la respuesta.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, dialogue, max_new_tokens):
    import torch

    prompt = build_prompt(tokenizer, dialogue)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # determinístico, para poder comparar corridas entre sí
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    try:
        import torch
        from huggingface_hub import login
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise SystemExit(
            "Falta una dependencia (" + str(e) + "). Corre primero:\n\n"
            "    pip install -r requirements.txt\n"
        )

    adapter_dir = download_gcs_dir(args.gcs_adapter_dir, args.local_cache_dir, force=args.force_download)

    if not os.listdir(adapter_dir):
        raise SystemExit(f"{adapter_dir} está vacío tras la descarga. Revisa --gcs_adapter_dir.")

    if args.hf_token:
        login(token=args.hf_token)

    print(f"Modelo base: {args.model_name}")
    print(f"Adaptadores (bucket): {args.gcs_adapter_dir}")
    print(f"Adaptadores (caché local): {adapter_dir}")
    print("CUDA disponible:", torch.cuda.is_available())

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # El tokenizer se carga del propio directorio de adaptadores (ahí quedó
    # guardado al final del entrenamiento, ya con pad_token configurado) en
    # vez de descargarlo de nuevo desde el modelo base.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Memoria GPU ocupada tras cargar el modelo base (GB):",
          round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else "N/A")

    finetuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
    finetuned_model.eval()

    dialogues = [args.dialogue] if args.dialogue else EJEMPLOS

    for i, dialogue in enumerate(dialogues, start=1):
        print(f"\n{'=' * 70}\nEjemplo {i}\n{'=' * 70}")
        print("Diálogo:\n" + dialogue)

        resumen_finetuned = generate(finetuned_model, tokenizer, dialogue, args.max_new_tokens)
        print("\n>> Resumen (modelo afinado con LoRA, entrenado en Vertex AI):\n" + resumen_finetuned)

        if args.compare_base:
            # disable_adapter() desactiva temporalmente el LoRA para comparar
            # contra el modelo base "de fábrica", sin tener que cargar dos
            # copias del modelo por separado.
            with finetuned_model.disable_adapter():
                resumen_base = generate(finetuned_model, tokenizer, dialogue, args.max_new_tokens)
            print("\n>> Resumen (modelo base, SIN fine-tuning):\n" + resumen_base)


if __name__ == "__main__":
    main()
