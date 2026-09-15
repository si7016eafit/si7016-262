"""
infer_gemma.py — Probar con ejemplos los adaptadores LoRA entrenados por
train_gemma.py, en el mismo contenedor JupyterHub (quay.io/jupyter/tensorflow-notebook:x86_64-cuda-latest).

Carga el modelo base (google/gemma-7b-it) en 4-bit + los adaptadores LoRA
guardados en /home/jovyan/labs/gemma-7b-it-samsum-lora, y genera resúmenes
para unos diálogos de ejemplo -tanto con el modelo afinado como (opcional)
con el modelo base sin adaptadores, para comparar el efecto del fine-tuning.

Uso (terminal del contenedor, después de que train_gemma.py haya terminado):
    python infer_gemma.py
    python infer_gemma.py --compare_base          # además corre el modelo SIN LoRA, para comparar
    python infer_gemma.py --dialogue "A: ...\\nB: ..."   # tu propio diálogo

La versión notebook (infer_gemma.ipynb) hace lo mismo celda por celda, con
una celda extra para escribir tu propio diálogo interactivamente.
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
    p.add_argument("--adapter_dir",
                    default=os.environ.get("OUTPUT_DIR", "/home/jovyan/labs/gemma-7b-it-samsum-lora"),
                    help="Carpeta con los adaptadores LoRA guardados por train_gemma.py")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                    help="Solo necesario si los pesos base de Gemma no quedaron ya en caché "
                         "de la corrida de entrenamiento. También puedes exportar HF_TOKEN.")
    p.add_argument("--dialogue", default=None,
                    help="Un diálogo propio para resumir, en vez de los ejemplos por defecto")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--compare_base", action="store_true",
                    help="Además de generar con el modelo afinado, generar con el modelo "
                         "base SIN adaptadores, para comparar el efecto del fine-tuning")
    return p.parse_args()


def build_prompt(tokenizer, dialogue):
    messages = [
        {"role": "user", "content": f"Resume el siguiente diálogo en 1-2 frases:\n\n{dialogue}"},
    ]
    # add_generation_prompt=True: agrega el "<start_of_turn>model\n" para que
    # el modelo sepa que le toca generar la respuesta (a diferencia de
    # train_gemma.py, que incluye también el turno del assistant porque ahí
    # se usa como texto de entrenamiento completo, no como prompt).
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
    # Solo la parte generada (sin repetir el prompt de entrada)
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
            "Falta una dependencia (" + str(e) + "). Si esto corre en un contenedor nuevo "
            "(no el mismo donde entrenaste), corre primero:\n\n"
            "    pip install -r requirements.txt\n"
        )

    if not os.path.isdir(args.adapter_dir) or not os.listdir(args.adapter_dir):
        raise SystemExit(
            f"No encuentro adaptadores en {args.adapter_dir}. ¿Ya terminó "
            "train_gemma.py? Revisa que haya impreso 'Adaptadores guardados en: ...' "
            "antes de correr esto."
        )

    if args.hf_token:
        login(token=args.hf_token)

    print(f"Modelo base: {args.model_name}")
    print(f"Adaptadores: {args.adapter_dir}")
    print("CUDA disponible:", torch.cuda.is_available())

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # El tokenizer se carga del propio directorio de adaptadores (ahí quedó
    # guardado al final de train_gemma.py, ya con pad_token configurado) en
    # vez de descargarlo de nuevo desde el modelo base.
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Memoria GPU ocupada tras cargar el modelo base (GB):",
          round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else "N/A")

    finetuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    finetuned_model.eval()

    dialogues = [args.dialogue] if args.dialogue else EJEMPLOS

    for i, dialogue in enumerate(dialogues, start=1):
        print(f"\n{'=' * 70}\nEjemplo {i}\n{'=' * 70}")
        print("Diálogo:\n" + dialogue)

        resumen_finetuned = generate(finetuned_model, tokenizer, dialogue, args.max_new_tokens)
        print("\n>> Resumen (modelo afinado con LoRA):\n" + resumen_finetuned)

        if args.compare_base:
            # disable_adapter() desactiva temporalmente el LoRA para comparar
            # contra el modelo base "de fábrica", sin tener que cargar dos
            # copias del modelo por separado.
            with finetuned_model.disable_adapter():
                resumen_base = generate(finetuned_model, tokenizer, dialogue, args.max_new_tokens)
            print("\n>> Resumen (modelo base, SIN fine-tuning):\n" + resumen_base)


if __name__ == "__main__":
    main()
