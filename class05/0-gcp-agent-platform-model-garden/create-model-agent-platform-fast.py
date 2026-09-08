# pip3 install --upgrade google-cloud-aiplatform
# gcloud auth application-default login

import vertexai
from vertexai import model_garden

vertexai.init(project="si7016-20262", location="us-central1")

model = model_garden.OpenModel("qwen/qwen3@qwen3-1.7b")
endpoint = model.deploy(
  accept_eula=True,
  machine_type="g2-standard-12",
  accelerator_type="NVIDIA_L4",
  accelerator_count=1,
  serving_container_image_uri="us-docker.pkg.dev/deeplearning-platform-release/vertex-model-garden/sglang-serve.cu124.0-4.ubuntu2204.py310:20250428-1803-rc0",
  endpoint_display_name="qwen_qwen3-1_7b-mg-one-click-deploy",
  model_display_name="qwen_qwen3-1_7b-si7016",
  use_dedicated_endpoint=True,
  fast_tryout_enabled=True,
  reservation_affinity_type="NO_RESERVATION",
)