pip3 install --upgrade google-cloud-aiplatform

pip3 install --upgrade huggingface_hub

pip3 install --upgrade setuptools wheel

python3 submit_vertex_job.py --project myproyectsi4002-262 --bucket si7016emontoya --region us-central1

Solicitar la cuota para entrenar modelos en Agent Platform:

    Consola de GCP → IAM y administración → Cuotas y límites del sistema (https://console.cloud.google.com/iam-admin/quotas?project=myproyectsi4002-262)

    Filtra por servicio: Vertex AI API, y busca la métrica: custom_model_training_nvidia_l4_gpus

    Filtra por región: us-central1

    Marca esa fila → Editar cuota → solicita, por ejemplo, 1

    En la justificación, usa algo como: "Actividad académica del curso SI7016 (NLP Aplicado) de la Maestría en Ciencias de Datos, EAFIT — fine-tuning de un modelo abierto con fines educativos."

Autorizar modelo gemma en huggingface:

    * entra a huggingface.co 
    * logueado con la cuenta dueña de ese token, ve a https://huggingface.co/google/gemma-7b-it, y confirma si ves el botón "Acknowledge license" (sin apretar todavía) o si ya dice que lo aceptaste. 
    * El texto del repo dice "Requests are processed immediately" — o sea que en teoría es instantáneo, así que si el botón sigue pidiendo aceptar, ese es el problema completo.