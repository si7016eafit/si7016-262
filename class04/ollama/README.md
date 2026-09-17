# Ollama

### ollama es una plataforma para uso de modelos abiertos LLM.

### entrar al sitio https://ollama.com y leer que tipo de servicios puedo implementar con esta plataforma.

### permite implementar un servicio similar a chatgpt con una variedad de modelos.

### esta versión está dockerizada, se requiere que instale docker en la máquina, con buena CPU, Memoria y una GPU mínimo Nvidia L4.

### para ejecutar el servicio:

    Ya los alumnos tienen las credenciales para acceso al servicio GCP

## I. hardware requerido:

    Máquina Virtual en GCP, g2-standard-4 (4 vCPUs, 16 GB Memory), con Sistema Operativo Deep Learning Linux + GPU Nvidia L4

Para crear esta VM debe solicitar incremento de quota, le llegará un email, conteste diciendo que esta VM será utilizada como parte del desarrollo de un curso de applied NLP en el marco de la MCDA, y que requiere realizar actividades de Ejecución de modelos abiertos LLM y fine-tuning, solo para fines académicos.

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

## III. ejecutar ollama:

1.clone en la máquina el repositorio del curso:

    git clone https://github.com/si7016eafit/si7016-262.git

    cd si7016-262/class04/ollama

    docker compose up -d

### acceso remoto y tunnel

Cree una clave SSH para ingresar remotamente a la máquina (SOLO UNA VEZ):

    En Mac o Linux, ejecute:

    ssh-keygen -t rsa -f ~/.ssh/gcp_key -C username_gcp -b 2048

    Luego en la consola de GCP:
    Compute Engine -> Metadata -> SSH keys: agrege la clave pública generada (~/.ssh/gcp_key.pub)

Luego ya puede conectarse desde una Mac o Linux, así:

para conectarse a ollama por el puerto 3000 y admin: 11434

	ssh -i ~/.ssh/gcp_key username_gcp@<ip-publica-vm-gcp> -L 3000:localhost:3000 -L 11434:localhost:11434

## Abra un navegador en su máquina, y entre a:

    localhost:3000

### 1. despues que cree una cuenta de administrador, agregue un modelo.

    Settings del usuario (parte superior derecha - icono naranjado)
    Admin Panel
    Settings
    Models
    Manage
    Pull a model from Ollama.com.     (click en la lista de modelos para que los explore)
    <digite algun nombre típico de modelo: mistral, gpt-oss, qwen2.5, deepseek-r1, etc

adicione, al menos 3 modelos.

### 2. entre a la interfaz de consulta tipo chatgpt, y realice varias consultas.... comparé entre los diferentes modelos. Tome tiempos de respuesta.

### 3. Reto: de acuerdo a las características leidas en https://ollama.com, ¿Qué tipo de aplicaciones se pueden realizar en el marco de la materia SI7016?