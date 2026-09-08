# DESPLEGAR MODELOS EN GCP con:

# Agent Platform -> Model Garden -> Deploy Model -> Agent Platform

# CREAR EL MODELO Y ENDPOINT

modelo: qwen2.5-7b-instruct@001

model name: qwen2.5-7b-instruct-si7016

    pip3 install --upgrade google-cloud-aiplatform
    gcloud auth application-default login

    python3 create-model-agent-platform.py

# Agent Platform -> Model Garden -> Deploy Model -> Agent Platform Fast Deployment:

modelo: qwen3-1.7b

model name: qwen_qwen3-1_7b-si7016

    pip3 install --upgrade google-cloud-aiplatform
    gcloud auth application-default login

    python3 create-model-agent-platform-fast.py

# ACCESO COMPARTIDO

## desde shell

    sh predict-shared.sh

# ACCESO HOST DEDICADO

## desde shell

    sh predict-dedicated.sh

## desde python

    pip3 install requests google-auth

    python predict-dedicated.py
