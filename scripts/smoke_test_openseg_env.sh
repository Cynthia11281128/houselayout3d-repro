#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HOUSELAYOUT3D_OPENSEG_PYTHON:-/tmp/tmp_data/miniconda3/envs/houselayout3d-openseg/bin/python}"
export HOUSELAYOUT3D_PROJECT_ROOT="${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" TF_CPP_MIN_LOG_LEVEL=2 "${PYTHON}" - <<'PY'
import os
from pathlib import Path

import tensorflow as tf

root = Path(os.environ["HOUSELAYOUT3D_PROJECT_ROOT"])
gpus = tf.config.list_physical_devices("GPU")
assert gpus
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.saved_model.load(
    str(root / "weights/OpenSeg/exported_model"),
    tags=[tf.saved_model.SERVING],
)
signature = model.signatures["serving_default"]
outputs = signature.structured_outputs
assert outputs["ppixel_ave_feat"].shape == (1, 640, 640, 768)
assert outputs["image_embedding_feat"].shape == (1, 640, 640, 768)
assert signature.structured_input_signature[1]["inp_text_emb"].shape[-1] == 768

print("OpenSeg environment: ok")
print("tensorflow", tf.__version__)
print("outputs", "ppixel_ave_feat", "image_embedding_feat")
PY
