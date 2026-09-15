"""
train_gemma.py — Mismo pipeline QLoRA/SFT del laboratorio Lecture04b, pero con
google/gemma-7b-it en vez de Qwen3-8B.

Por qué este script existe por separado: Gemma es un modelo de Google que la
imagen base de Hugging Face (transformers 4.42) ya soporta de fábrica -no hace
falta actualizar torch ni transformers como tocó hacer para Qwen3-, así que
este pipeline corre con la imagen "de fábrica" y elimina toda la cadena de
incompatibilidades de versiones que veníamos resolviendo. Es la opción para
GARANTIZAR que el fine-tuning se ejecute de punta a punta sin sorpresas.

Requisito importante: Gemma es un modelo "gated" en Hugging Face -tienes que:
  1. Entrar a https://huggingface.co/google/gemma-7b-it y aceptar la licencia
     con tu cuenta de HF.
  2. Generar un token de acceso (https://huggingface.co/settings/tokens).
  3. Pasarlo al job con --hf_token (ver submit_vertex_job_gemma.py).
Sin esto, la descarga del modelo falla con un error 401/403 "gated repo".

Uso local (para probar antes de enviarlo a GCP):
    python train_gemma.py --output_dir ./gemma-7b-it-samsum-lora --max_steps 20

Uso dentro del job de Vertex AI (lo hace submit_vertex_job_gemma.py por ti):
    python train_gemma.py --output_dir /gcs/<tu-bucket>/gemma-7b-it-samsum-lora
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/gemma-7b-it",
                    help="Alternativa más liviana: google/gemma-2b-it")
    p.add_argument("--dataset_name", default="knkarthick/samsum",
                    help="Alternativa si samsum falla: knkarthick/dialogsum")
    p.add_argument("--train_split", default="train[:2000]")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_train_epochs", type=float, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=-1,
                    help="Útil para pruebas rápidas (ej. 20) antes de correr el job completo")
    p.add_argument("--output_dir", default=os.environ.get("AIP_MODEL_DIR", "./gemma-7b-it-samsum-lora"))
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Modelo base: {args.model_name}")
    print(f"Dataset: {args.dataset_name} ({args.train_split})")
    print(f"Salida de los adaptadores: {args.output_dir}")

    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --- 1. Cargar el modelo base en 4-bit (idéntico al notebook original) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Memoria GPU ocupada tras cargar el modelo (GB):",
          round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else "N/A")

    # --- 2. Adaptadores LoRA (mismos target_modules; Gemma usa la misma
    #         convención de nombres de proyecciones que Llama/Qwen) ---
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 3. Dataset + formateo con chat template ---
    # Nota: a diferencia de Qwen3, la plantilla de chat de Gemma no tiene modo
    # "thinking", así que NO pasamos enable_thinking aquí (evita cualquier
    # duda sobre kwargs específicos de un solo modelo).
    raw_dataset = load_dataset(args.dataset_name, split=args.train_split)

    def format_example(example):
        messages = [
            {"role": "user", "content": f"Resume el siguiente diálogo en 1-2 frases:\n\n{example['dialogue']}"},
            {"role": "assistant", "content": example["summary"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)
    print("Ejemplo formateado:\n", dataset[0]["text"][:400])

    # --- 4. Entrenamiento con SFTTrainer (idéntico al notebook, parametrizado) ---
    local_ckpt_dir = "/tmp/gemma-7b-it-samsum-lora-ckpts"
    sft_config = SFTConfig(
        output_dir=local_ckpt_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
    )
    trainer.train()

    # --- 5. Guardar SOLO los adaptadores LoRA, directo a GCS ---
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adaptadores guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()
