# Imagen para el Custom Training Job del taller 3 (Gemma-7B-it + QLoRA).
#
# A diferencia del Dockerfile de Qwen3, aquí NO tocamos torch ni transformers:
# Gemma (v1) está soportado por `transformers` desde una versión bastante
# anterior a la que ya trae la imagen base (4.42), así que la imagen "de
# fábrica" del contenedor de Hugging Face para GCP funciona tal cual, sin
# ninguna de las incompatibilidades de versiones que tocó resolver para
# Qwen3. Esa es la garantía: menos cambios = menos superficie de error.
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-pytorch-training-cu121.2-3.transformers.4-42.ubuntu2204.py310

WORKDIR /trainer
COPY train_gemma.py /trainer/train_gemma.py

# Paquetes que el notebook usa en la Parte 6 (evaluación ROUGE) y que la
# imagen base no trae por defecto. Nada más se instala ni se actualiza.
RUN pip install --no-cache-dir evaluate rouge-score

ENTRYPOINT ["python3", "/trainer/train_gemma.py"]
