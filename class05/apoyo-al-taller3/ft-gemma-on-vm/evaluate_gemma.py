"""
evaluate_gemma.py — Evaluación cuantitativa (ROUGE + BERTScore) del modelo
Gemma-7b-it afinado con LoRA (train_gemma.py) contra el modelo base, sobre
una muestra real del split de prueba de samsum -no sobre los ejemplos "de
juguete" de infer_gemma.py, que no tienen resumen de referencia y por lo
tanto no permiten calcular ninguna métrica.

Qué hace:
  1. Toma N ejemplos del split "test" de knkarthick/samsum (con resumen de
     referencia real, escrito por humanos).
  2. Genera un resumen para cada uno con el modelo AFINADO (LoRA activo) y
     con el modelo BASE (LoRA desactivado vía disable_adapter(), sin cargar
     el modelo dos veces).
  3. Compara ambos conjuntos de resúmenes contra la referencia con:
       - ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum (solapamiento de n-gramas)
       - BERTScore F1 (similitud semántica -captura paráfrasis que ROUGE no ve)
       - Longitud promedio del resumen generado (compresión vs. la referencia)
  4. Imprime una tabla comparativa base vs. afinado, y guarda el detalle
     por ejemplo (diálogo, referencia, resumen base, resumen afinado, ROUGE-L
     de cada uno) en un CSV dentro del volumen montado, para inspección manual.

Instalar antes (además de lo que ya instalaste para entrenar/inferir):
    pip install evaluate rouge_score bert_score absl-py pandas

Uso:
    python evaluate_gemma.py --n_examples 30
    python evaluate_gemma.py --n_examples 100 --skip_bertscore   # más rápido, solo ROUGE

Nota de tiempo: generar 2 resúmenes (base + afinado) por ejemplo con un
modelo de 8B en 4-bit toma unos segundos por ejemplo -para 30 ejemplos,
cuenta con varios minutos; para 100+, mejor déjalo correr de fondo.
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/gemma-7b-it")
    p.add_argument("--adapter_dir",
                    default=os.environ.get("OUTPUT_DIR", "/home/jovyan/labs/gemma-7b-it-samsum-lora"))
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
                        "EVAL_OUTPUT_CSV", "/home/jovyan/labs/eval_base_vs_finetuned.csv"
                    ))
    return p.parse_args()


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
            "    pip install evaluate rouge_score bert_score absl-py pandas\n"
        )

    if not os.path.isdir(args.adapter_dir) or not os.listdir(args.adapter_dir):
        raise SystemExit(f"No encuentro adaptadores en {args.adapter_dir}. ¿Ya terminó train_gemma.py?")

    if args.hf_token:
        login(token=args.hf_token)

    print("CUDA disponible:", torch.cuda.is_available())

    # --- 1. Cargar modelo base + adaptadores (igual que infer_gemma.py) -----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
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

    # Versión sin agregar, para poder guardar el ROUGE-L por ejemplo en el CSV.
    rouge_finetuned_per_example = rouge.compute(
        predictions=preds_finetuned, references=references, use_aggregator=False
    )
    rouge_base_per_example = rouge.compute(
        predictions=preds_base, references=references, use_aggregator=False
    )

    resumen = {
        "modelo": ["base (sin LoRA)", "afinado (con LoRA)"],
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
    print(f"\n(longitud promedio de la referencia humana: {referencia_len:.1f} palabras "
          "-para comparar qué tan bien calibrada quedó la longitud del resumen)")

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
    print("(columnas rougeL_base / rougeL_afinado te dejan encontrar los mejores y "
          "peores casos de cada modelo para revisarlos a mano)")


if __name__ == "__main__":
    main()
