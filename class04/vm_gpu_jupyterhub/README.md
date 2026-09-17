# JupyterHub sobre una máquina virtual con GPU nvidia L4 en GCP

## esta versión está dockerizada, se requiere que instale docker en la máquina, con buena CPU, Memoria y una GPU mínimo Nvidia L4.

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

verificar: 

    sudo systemctl status docker
    sudo systemctl enable docker
    sudo systemctl start docker

    sudo usermod -a -G docker <username_gcp>

### verificar driver nvidia para docker:

    sudo nvidia-smi

    sudo nvidia-ctk --version

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    docker info | grep -i runtime
    # debe aparecer nvidia

## III. Jupyter + TensorFlow

    git clone https://github.com/si7016eafit/si7016-262.git

    cd si7016-262/class04/vm_gpu_jupyterhub

    docker compose up -d


### IV Abra un navegador en su máquina local (DESPUES DE LANZAR EL TÚNEL), y entre a:

    localhost:8888

