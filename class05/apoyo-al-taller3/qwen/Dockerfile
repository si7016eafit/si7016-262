# Imagen mínima para el Custom Training Job del taller 3 (Qwen3-8B + QLoRA).
#
# Parte del contenedor oficial de Hugging Face para GCP, que ya trae
# transformers/peft/trl/bitsandbytes/accelerate instalados y probados con
# GPU en Vertex AI — solo le agregamos train.py encima.
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310

WORKDIR /trainer
COPY train.py /trainer/train.py

# La imagen base trae transformers 4.42 (ANTERIOR a que se agregara soporte
# para Qwen3) y torch 2.3.0. El `transformers` más reciente sí reconoce
# Qwen3, pero exige torch >= 2.5 — si solo actualizamos transformers, se
# desactiva su integración con torch (por el mismo torch 2.3 "viejo") y
# revienta con NameError: name 'nn' is not defined.
#
# Por eso subimos AMBOS juntos: primero torch a una versión >= 2.5 que siga
# usando CUDA 12.1 (el mismo que ya trae la imagen, para no romper
# bitsandbytes), y luego transformers/accelerate/peft/trl/bitsandbytes.
RUN pip install --no-cache-dir --upgrade \
    "torch>=2.5.0,<2.6.0" --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir --upgrade \
    "transformers>=4.51.0" accelerate peft trl bitsandbytes

# Paquetes que el notebook usa en la Parte 6 (evaluación ROUGE) y que la
# imagen base no trae por defecto.
RUN pip install --no-cache-dir evaluate rouge-score

ENTRYPOINT ["python3", "/trainer/train.py"]
