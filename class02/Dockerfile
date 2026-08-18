FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Actualizar e instalar dependencias
RUN apt update && apt install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    zlib1g-dev \
    python3 \
    python3-pip \
    python3-dev \
    cython3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar Jupyter Notebook y dependencias útiles
RUN pip3 install --upgrade pip && \
    pip3 install notebook jupyterlab matplotlib pandas

# 3. Clonar y compilar KenLM
RUN git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && \
    pip3 install -r requirements.txt
    # cd ../python && pip3 install -r requirements.txt && python3 setup.py install

# 4. Crear carpeta de trabajo
WORKDIR /workspace

# 5. Exponer puerto para Jupyter Notebook
EXPOSE 8888

# 6. Comando por defecto
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
