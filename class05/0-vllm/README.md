# vLLM

### vLLM es una plataforma para uso de modelos abiertos LLM.

### entrar al sitio https://vllm.ai/ y leer que tipo de servicios puedo implementar con esta plataforma.

### permite implementar un servicio similar a chatgpt con una variedad de modelos.

### esta versión está dockerizada, se requiere que instale docker en la máquina, con buena CPU, Memoria y una GPU mínimo Nvidia L4.

### para ejecutar el servicio:

    Ya los alumnos tienen las credenciales para acceso al servicio.

## I. hardware requerido:

    Máquina Virtual en GCP, g2-standard-4 (4 vCPUs, 16 GB Memory), con Sistema Operativo Deep Learning Linux + GPU Nvidia L4

Para crear esta VM debe solicitar incremento de quota, le llegará un email, conteste diciendo que esta VM será utilizada como parte del desarrollo de un curso de applied NLP en el marco de la MCDA, y que requiere realizar actividades de Ejecución de modelos abiertos LLM y fine-tuning, solo para fines académicos.

### acceso remoto y tunnel

Cree una clave SSH para ingresar remotamente a la máquina:

En Mac o Linux, ejecute:

ssh-keygen -t rsa -f ~/.ssh/gcp_key -C username_gcp -b 2048

Luego en la consola de GCP:
Compute Engine -> Metadata -> SSH keys: agrege la clave pública generada (~/.ssh/gcp_key.pub)

Luego ya puede conectarse desde una Mac o Linux, así:

	ssh -i ~/.ssh/gcp_key username_gcp@<ip-publica-vm-gcp>

Si quiere conexión con un tunnel, para acceder al jupyter de la VM en gcp:

	ssh -i ~/.ssh/gcp_key username_gcp@<ip-publica-vm-gcp> -L 8888:localhost:8888

Asi, desde un browser local, puede abrir una conexión http://localhost:8888

## II. instalar docker:

ver: https://docs.docker.com/engine/install/ubuntu/

luego:

verificar: 

sudo systemctl status docker
sudo systemctl enable docker
sudo systemctl start docker

sudo usermod -a -G docker <username_gcp>

### verificar driver nvidia para docker:

### instalar:

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit

    sudo nvidia-smi

    sudo nvidia-ctk --version

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    docker info | grep -i runtime
    # debe aparecer nvidia

## III. Jupyter + TensorFlow

    git clone https://github.com/si7016eafit/si7016-262.git

    cd si7016-262/class04/ollama

    mkdir -p "$HOME/si7016-262"

    sudo docker pull \
        quay.io/jupyter/tensorflow-notebook:x86_64-cuda-latest

    sudo docker run --rm -it \
    --gpus all \
    --shm-size=2g \
    -p 8888:8888 \
    -v "$HOME/si7016-262:/home/jovyan/labs" \
    quay.io/jupyter/tensorflow-notebook:x86_64-cuda-latest

## IV. ejecutar vllm:

1.clone en la máquina el repositorio del curso:

    git clone https://github.com/si7016eafit/si7016-262.git

    cd si7016-262/class05/0-vllm

actualizar el archivo .env copiado a partir de .env.example

    docker compose up -d

2.cree un tunnel desde su máquina local, para conectarse a ollama por el puerto 30000

	ssh -i ~/.ssh/gcp_key username_gcp@<ip-publica-vm-gcp> -L 3000:localhost:3000 -L 11434:localhost:11434

3.abra un navegador en su máquina, y entre a:

    localhost:3000

4.entre a la interfaz de consulta tipo chatgpt, y realice varias consultas.... comparé entre los diferentes modelos. Tome tiempos de respuesta.

# docker+nvidia

1a verificación:

    sudo docker run --rm --gpus all \
        nvidia/cuda:12.6.3-base-ubuntu24.04 \
        nvidia-smi