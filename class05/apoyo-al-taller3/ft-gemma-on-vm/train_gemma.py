"""
train_gemma.py — Mismo pipeline QLoRA/SFT del laboratorio Lecture04b, pero con
google/gemma-7b-it en vez de Qwen3-8B.

Por qué este script existe por separado: Gemma es un modelo de Google que la
imagen base de Hugging Face (transformers 4.42) ya soporta de fábrica -no hace
falta actualizar torch ni transformers como tocó hacer para Qwen3-, así que
este pipeline corre con la imagen "de fábrica" y elimina toda la cadena de
incompatibilidades de versiones que veníamos resolviendo. Es la opción para
GARANTIZAR que el fine-tuning se ejecute de punta a punta sin sorpresas.

--------------------------------------------------------------------------
ADAPTADO para correr sobre el contenedor JupyterHub de la clase 04
(quay.io/jupyter/tensorflow-notebook:x86_64-cuda-latest, ver docker-compose.yml).
--------------------------------------------------------------------------

Diferencias frente a la versión pensada para Vertex AI:

1. Esta imagen es la línea "tensorflow-notebook" de Jupyter Docker Stacks:
   trae TensorFlow + CUDA, pero **no** trae PyTorch ni el stack de Hugging
   Face (a diferencia de la línea "pytorch-notebook", que sí lo incluiría).
   Antes de correr este script, instala las dependencias que faltan -el
   usuario `jovyan` tiene permisos de escritura sobre /opt/conda, así que no
   necesitas sudo ni --user-:

       pip install -r requirements.txt

   (o, sin el archivo, sencillamente:
       pip install torch transformers peft trl bitsandbytes accelerate datasets huggingface_hub
   )

2. `AIP_MODEL_DIR` (variable de entorno propia de Vertex AI Training) ya no
   existe en este contenedor, así que `--output_dir` ahora apunta por
   defecto a /home/jovyan/labs/... -la carpeta que el docker-compose.yml de
   este lab monta desde $HOME/si7016-262 del host. Cualquier cosa que
   guardes ahí sobrevive a que borres o reinicies el contenedor.

3. El script original mencionaba pasar el token de Hugging Face con
   `--hf_token` (vía submit_vertex_job_gemma.py) pero nunca lo definía como
   argumento propio -ese wrapper ya no aplica aquí. Se añadió el argumento
   `--hf_token` explícitamente (o exporta la variable de entorno HF_TOKEN, o
   corre `huggingface-cli login` antes de este script) para que la
   autenticación con el modelo "gated" funcione de forma autocontenida.

Requisito importante: Gemma es un modelo "gated" en Hugging Face -tienes que:
  1. Entrar a https://huggingface.co/google/gemma-7b-it y aceptar la licencia
     con tu cuenta de HF.
  2. Generar un token de acceso (https://huggingface.co/settings/tokens).
  3. Pasarlo con --hf_token, o exportar HF_TOKEN, o correr `huggingface-cli
     login` antes de este script.
Sin esto, la descarga del modelo falla con un error 401/403 "gated repo".

Uso dentro del contenedor JupyterHub (terminal de Jupyter, o `docker compose
exec notebook bash`):
    python train_gemma.py --hf_token hf_xxx --max_steps 20

La versión notebook (train_gemma.ipynb) hace exactamente lo mismo celda por
celda, con login interactivo por getpass en vez de --hf_token.
"""

import argparse
import os


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
                    help="Útil para pruebas rápidas (ej. 20) antes de correr el entrenamiento completo")
    p.add_argument("--output_dir",
                    default=os.environ.get(
                        "OUTPUT_DIR", "/home/jovyan/labs/gemma-7b-it-samsum-lora"
                    ),
                    help="Por defecto, dentro del volumen montado del contenedor "
                         "(/home/jovyan/labs -> $HOME/si7016-262 en el host)")
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                    help="Token de Hugging Face con acceso a google/gemma-7b-it. "
                         "También puedes exportar HF_TOKEN, o correr "
                         "`huggingface-cli login` antes de este script.")
    return p.parse_args()


def main():
    args = parse_args()

    # --- 0. Instalar/importar el stack de PyTorch + Hugging Face -------------
    # Se importa aquí (no al tope del archivo) para poder dar un mensaje de
    # error claro si el contenedor todavía no tiene estas librerías instaladas
    # (la imagen tensorflow-notebook no las trae de fábrica).
    try:
        import torch
        from datasets import load_dataset
        from huggingface_hub import login
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            "Falta una dependencia (" + str(e) + "). Esta imagen base "
            "(tensorflow-notebook) no trae el stack de PyTorch/Hugging Face. "
            "Corre primero:\n\n    pip install -r requirements.txt\n"
        )

    if args.hf_token:
        login(token=args.hf_token)
    else:
        print("Aviso: no se dio --hf_token ni HF_TOKEN. Si no corriste "
              "`huggingface-cli login` antes, la descarga de un modelo "
              "gated como Gemma va a fallar con 401/403.")

    print(f"Modelo base: {args.model_name}")
    print(f"Dataset: {args.dataset_name} ({args.train_split})")
    print(f"Salida de los adaptadores: {args.output_dir}")

    print("CUDA disponible:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Aviso: no se detectó GPU. Revisa que el contenedor se haya "
              "levantado con `--gpus all` / `runtime: nvidia` (ver docker-compose.yml "
              "de este lab) y que `nvidia-smi` funcione dentro del contenedor.")

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
    # Los checkpoints intermedios se quedan en el disco efímero del contenedor
    # (no en el volumen montado) -solo los adaptadores finales (paso 5) se
    # guardan en --output_dir, que sí persiste en el host.
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

    # --- 5. Guardar SOLO los adaptadores LoRA, al volumen montado del host ---
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adaptadores guardados en: {args.output_dir}")
    print("(esa carpeta vive dentro del volumen montado -sigue disponible en "
          "$HOME/si7016-262 del host aunque borres el contenedor)")

if __name__ == "__main__":
    main()
