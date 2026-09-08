# pip install requests google-auth

import google.auth
from google.auth.transport.requests import Request
import requests
import json

# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------

PROJECT_ID = "si7016-20262"
LOCATION = "us-central1"

ENDPOINT_ID = "mg-endpoint-3d00f6a8-0674-4044-a5a0-315895c81531"

DEDICATED_HOST = (
    "273686586095304704."
    "us-central1-976152022399."
    "prediction.vertexai.goog"
)

# ---------------------------------------------------------
# Obtener credenciales de Google Cloud
# ---------------------------------------------------------

credentials, project = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

credentials.refresh(Request())

# ---------------------------------------------------------
# URL del Dedicated Endpoint
# ---------------------------------------------------------

url = (
    f"https://{DEDICATED_HOST}"
    f"/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}"
    f"/endpoints/{ENDPOINT_ID}:predict"
)

# ---------------------------------------------------------
# Payload
# Debe coincidir con lo que espera el modelo desplegado.
# ---------------------------------------------------------

with open("input.json", "r") as f:
    payload = json.load(f)

# ---------------------------------------------------------
# Llamar al endpoint
# ---------------------------------------------------------

headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json",
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=120
)

# ---------------------------------------------------------
# Resultado
# ---------------------------------------------------------

print("HTTP status:", response.status_code)

if response.ok:
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print("Error:")
    print(response.text)