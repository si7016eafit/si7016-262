# pip3 install --upgrade google-cloud-aiplatform
# gcloud auth application-default login

import vertexai
from vertexai import model_garden

vertexai.init(project="si7016-20262", location="us-central1")

model = model_garden.OpenModel("hf-qwen/qwen2.5-7b-instruct@001")
endpoint = model.deploy(
  accept_eula=True,
  machine_type="g4-standard-48",
  accelerator_type="NVIDIA_RTX_PRO_6000",
  accelerator_count=1,
  serving_container_image_uri="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20260713_0916_RC01",
  endpoint_display_name="qwen2_5-7b-instruct-mg-one-click-deploy",
  model_display_name="qwen2_5-7b-instruct-si7016",
  use_dedicated_endpoint=True,
  reservation_affinity_type="NO_RESERVATION",
)