"""
evaluate_gemma_vertex.py — Evaluación cuantitativa (ROUGE + BERTScore) del
modelo Gemma-7b-it afinado con LoRA vía un Custom Training Job de Vertex AI
/ Gemini Enterprise Agent Platform (Agent Platform -> Models -> Training),
contra el modelo base, sobre una muestra real del split de prueba de samsum.

Es exactamente el mismo pipeline de evaluate_gemma.py (pensado para el
entrenamiento hecho dentro de esta VM+JupyterHub); la única diferencia es de
dónde vienen los adaptadores: en vez de asumir que ya están en el disco
local, este script primero los descarga desde Cloud Storage (el bucket que
le pasaste como --output_dir al training job) con el cliente de Python
`google-cloud-storage` -no con `gcloud`/`gsutil`, porque esa CLI no viene
instalada en la imagen quay.io/jupyter/tensorflow-notebook:x86_64-cuda-latest.
El cliente de Python toma credenciales automáticamente del servidor de
metadata de la VM (Application Default Credentials), sin necesidad de montar
ningún archivo de service account -solo que la cuenta de servicio de la VM
tenga permiso de lectura sobre el bucket.

Qué hace:
  1. Descarga (o reusa un caché local de) los adaptadores LoRA desde
     gs://.../gemma-7b-it-samsum-lora.
  2. Toma N ejemplos del split "test" de knkarthick/samsum (con resumen de
     referencia real, escrito por humanos -no visto durante el entrenamiento).
  3. Genera un resumen para cada uno con el modelo AFINADO (LoRA activo) y
     con el modelo BASE (LoRA desactivado vía disable_adapter()).
  4. Compara ambos conjuntos de resúmenes contra la referencia con ROUGE-1/2/
     L/Lsum, BERTScore F1, y longitud promedio.
  5. Imprime una tabla comparativa base vs. afinado, y guarda el detalle por
     ejemplo en un CSV dentro del volumen montado.

Instalar antes (además de lo que ya instalaste para entrenar/inferir):
    pip install evaluate rouge_score bert_score absl-py pandas google-cloud-storage

Uso:
    python evaluate_gemma_vertex.py --gcs_adapter_dir gs://tu-bucket/gemma-7b-it-samsum-lora --n_examples 30
    python evaluate_gemma_vertex.py --gcs_adapter_dir gs://tu-bucket/... --n_examples 100 --skip_bertscore
"""

import argparse
import os


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
                    help="Volver a descargar aunque ya exista una copia en --local_cache_dir.")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--dataset_name", default="knkarthick/samsum")
    p.add_argument("--eval_split", default="test",
                    help="Usamos 'test' (no 'train') a propósito: son ejemplos que el "
                         "modelo NO vio durante el fine-tuning.")
    p.add_argument("--n_examples", type=int, default=30,
                    help="Cuántos ejemplos del split de prueba evaluar. Empieza chico (20-30) "
                         "para iterar rápido; sube esto para un número más confiable.")
    p.add_argument("--seed", type=int, default=42, help="Para que la muestra sea reproducible")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--bertscore_model_type", default="distilbert-base-uncased",
                    help="Modelo chico (268MB) por defecto para no volver a llenar el disco "
                         "-usa 'roberta-large' (~1.4GB) si quieres el score más preciso de la métrica.")
    p.add_argument("--skip_bertscore", action="store_true",
                    help="Solo calcular ROUGE (más rápido, no descarga un modelo BERT adicional)")
    p.add_argument("--output_csv",
                    default=os.environ.get(
                        "EVAL_OUTPUT_CSV", "/home/jovyan/labs/eval_base_vs_finetuned_vertex.csv"
                    ))
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
    blobs = [b for b in blobs if not b.name.endswith("/")]

    if not blobs:
        raise SystemExit(
            f"No encontré archivos en {gcs_uri}. Revisa que el Custom Training Job haya "
            "terminado exitosamente y que la ruta sea exactamente la que le pasaste como "
            "--output_dir / AIP_MODEL_DIR."
        )

    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        rel_path = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
        dest = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(dest) or local_dir, exist_ok=True)
        blob.download_to_filename(dest)

    print(f"Descarga completa: {len(blobs)} archivo(s) en {local_dir}")
    return local_dir


def build_prompt(tokenizer, dialogue):
    messages = [
        {"role": "user", "content": f"Resume el siguiente diálogo en 1-2 frases:\n\n{dialogue}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, dialogue, max_new_tokens):
    import torch

    prompt = build_prompt(tokenizer, dialogue)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()

    try:
        import evaluate
        import pandas as pd
        import torch
        from datasets import load_dataset
        from huggingface_hub import login
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise SystemExit(
            "Falta una dependencia (" + str(e) + "). Corre primero:\n\n"
            "    pip install evaluate rouge_score bert_score absl-py pandas google-cloud-storage\n"
        )

    adapter_dir = download_gcs_dir(args.gcs_adapter_dir, args.local_cache_dir, force=args.force_download)

    if not os.listdir(adapter_dir):
        raise SystemExit(f"{adapter_dir} está vacío tras la descarga. Revisa --gcs_adapter_dir.")

    if args.hf_token:
        login(token=args.hf_token)

    print("CUDA disponible:", torch.cuda.is_available())
    print(f"Adaptadores (bucket): {args.gcs_adapter_dir}")
    print(f"Adaptadores (caché local): {adapter_dir}")

    # --- 1. Cargar modelo base + adaptadores (igual que infer_gemma_vertex.py) --
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    # --- 2. Muestra del split de prueba (con resumen de referencia real) ---
    test_dataset = load_dataset(args.dataset_name, split=args.eval_split)
    test_dataset = test_dataset.shuffle(seed=args.seed).select(
        range(min(args.n_examples, len(test_dataset)))
    )
    print(f"Evaluando sobre {len(test_dataset)} ejemplos de "
          f"{args.dataset_name} ({args.eval_split})")

    # --- 3. Generar con el modelo afinado y con el modelo base -------------
    dialogues, references = [], []
    preds_finetuned, preds_base = [], []

    for i, example in enumerate(test_dataset):
        dialogues.append(example["dialogue"])
        references.append(example["summary"])

        preds_finetuned.append(generate(model, tokenizer, example["dialogue"], args.max_new_tokens))
        with model.disable_adapter():
            preds_base.append(generate(model, tokenizer, example["dialogue"], args.max_new_tokens))

        if (i + 1) % 10 == 0 or (i + 1) == len(test_dataset):
            print(f"  ... {i + 1}/{len(test_dataset)} ejemplos generados")

    # --- 4. Métricas ---------------------------------------------------------
    rouge = evaluate.load("rouge")

    rouge_finetuned = rouge.compute(predictions=preds_finetuned, references=references)
    rouge_base = rouge.compute(predictions=preds_base, references=references)

    rouge_finetuned_per_example = rouge.compute(
        predictions=preds_finetuned, references=references, use_aggregator=False
    )
    rouge_base_per_example = rouge.compute(
        predictions=preds_base, references=references, use_aggregator=False
    )

    resumen = {
        "modelo": ["base (sin LoRA)", "afinado (con LoRA, Vertex AI)"],
        "rouge1": [rouge_base["rouge1"], rouge_finetuned["rouge1"]],
        "rouge2": [rouge_base["rouge2"], rouge_finetuned["rouge2"]],
        "rougeL": [rouge_base["rougeL"], rouge_finetuned["rougeL"]],
        "rougeLsum": [rouge_base["rougeLsum"], rouge_finetuned["rougeLsum"]],
        "longitud_promedio_palabras": [
            sum(len(p.split()) for p in preds_base) / len(preds_base),
            sum(len(p.split()) for p in preds_finetuned) / len(preds_finetuned),
        ],
    }

    if not args.skip_bertscore:
        bertscore = evaluate.load("bertscore")
        bs_finetuned = bertscore.compute(
            predictions=preds_finetuned, references=references,
            lang="en", model_type=args.bertscore_model_type,
        )
        bs_base = bertscore.compute(
            predictions=preds_base, references=references,
            lang="en", model_type=args.bertscore_model_type,
        )
        resumen["bertscore_f1"] = [
            sum(bs_base["f1"]) / len(bs_base["f1"]),
            sum(bs_finetuned["f1"]) / len(bs_finetuned["f1"]),
        ]

    df_resumen = pd.DataFrame(resumen)
    print("\n" + "=" * 70)
    print("RESULTADOS (promedio sobre", len(test_dataset), "ejemplos)")
    print("=" * 70)
    print(df_resumen.to_string(index=False))

    referencia_len = sum(len(r.split()) for r in references) / len(references)
    print(f"\n(longitud promedio de la referencia humana: {referencia_len:.1f} palabras)")

    # --- 5. Guardar el detalle por ejemplo para inspección manual ----------
    df_detalle = pd.DataFrame({
        "dialogo": dialogues,
        "referencia": references,
        "resumen_base": preds_base,
        "resumen_afinado": preds_finetuned,
        "rougeL_base": rouge_base_per_example["rougeL"],
        "rougeL_afinado": rouge_finetuned_per_example["rougeL"],
    })
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_detalle.to_csv(args.output_csv, index=False)
    print(f"\nDetalle por ejemplo guardado en: {args.output_csv}")


if __name__ == "__main__":
    main()
