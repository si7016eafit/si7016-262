# vLLM

### vLLM es una plataforma para uso de modelos abiertos LLM.

### entrar al sitio https://vllm.ai/ y leer que tipo de servicios puedo implementar con esta plataforma.

### permite implementar un servicio similar a chatgpt con una variedad de modelos.

### esta versión está dockerizada, se requiere que instale docker en la máquina, con buena CPU, Memoria y una GPU mínimo Nvidia L4.

### Para ejecutar el servicio:

    Ya los alumnos tienen las credenciales para acceso al servicio.

## I. Hardware requerido:

    Máquina Virtual en GCP, g2-standard-4 (4 vCPUs, 16 GB Memory), con Sistema Operativo Deep Learning Linux + GPU Nvidia L4

ESTA MÁQUINA REQUIERE AL MENOS 150 GB DE TAMAÑO EN EL DISCO DURO, SE RECOMIENDA 200 GB. SI ES LA PRIMERA VEZ QUE CREA LA VM REQUIERA DE UNA VEZ LOS 200 GB.

SI YA ESTÁ CREADA LA VM, PUEDE AGRANDAR EL DISCO DURO ASÍ:

    1. entre por la consola de gcp, a la Máquina Virtual, Ir a Storage y editar la configuración y darle EDIT en los 3 puntos (en la parte superior - derecha). Cambie a 200 GB.

    2. Entre a la máquina virtual por SSH y dele los siguientes comandos:

    df -h /
    sudo lsblk
    sudo growpart /dev/nvme0n1 1
    sudo resize2fs /dev/nvme0n1p1
    df -h /

Para crear esta VM debe solicitar incremento de quota, le llegará un email, conteste diciendo que esta VM será utilizada como parte del desarrollo de un curso de applied NLP en el marco de la MCDA, y que requiere realizar actividades de Ejecución de modelos abiertos LLM y fine-tuning, solo para fines académicos.

## II. instalar docker:

ver: https://docs.docker.com/engine/install/ubuntu/

verificar: 

    sudo systemctl status docker
    sudo systemctl enable docker
    sudo systemctl start docker

    sudo usermod -a -G docker <username_gcp>

### instalar:

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit

    sudo nvidia-smi

    sudo nvidia-ctk --version

    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    docker info | grep -i runtime
    # debe aparecer nvidia

## IV. ejecutar vllm:

1.clone en la máquina el repositorio del curso:

    git clone https://github.com/si7016eafit/si7016-262.git

    cd si7016-262/class05/0-vllm

actualizar el archivo .env copiado a partir de .env.example

    docker compose up -d

### acceso remoto y tunnel

* Cree una clave SSH para ingresar remotamente a la máquina:

En Mac o Linux (SOLO UNA VEZ), ejecute:

    ssh-keygen -t rsa -f ~/.ssh/gcp_key -C username_gcp -b 2048

* Luego en la consola de GCP:

    Compute Engine -> Metadata -> SSH keys: agrege la clave pública generada (~/.ssh/gcp_key.pub)

* Luego ya puede conectarse desde una Mac o Linux, así:

	ssh -i ~/.ssh/gcp_key username_gcp@<ip-publica-vm-gcp> -L 3000:localhost:3000 -L 11434:localhost:11434

* abra un navegador en su máquina, y entre a:

    localhost:3000

* entre a la interfaz de consulta tipo chatgpt, y realice varias consultas.... comparé entre los diferentes modelos. Tome tiempos de respuesta.