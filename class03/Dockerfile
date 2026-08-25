FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# 1. Dependencias de sistema
RUN apt update && apt install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Carpeta de trabajo
WORKDIR /workspace

# 3. Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 4. Exponer puerto para Jupyter Notebook
EXPOSE 8888

# 5. Comando por defecto
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
