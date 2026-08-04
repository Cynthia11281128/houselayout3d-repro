#!/usr/bin/env bash
set -euo pipefail

PYTHON="${HOUSELAYOUT3D_NERFSTUDIO_PYTHON:-/tmp/tmp_data/miniconda3/envs/nerfstudio/bin/python}"
BIN_DIR="$(dirname "${PYTHON}")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "${PYTHON}" - <<'PY'
import importlib.metadata as metadata

import dn_splatter
import torch
from nerfstudio.plugins.registry import discover_methods
from nerfstudio.plugins.registry_dataparser import discover_dataparsers

assert torch.cuda.is_available()
value = torch.tensor([2.0], device="cuda") ** 2
assert value.item() == 4.0

methods, _ = discover_methods()
parsers, _ = discover_dataparsers()
assert {"dn-splatter", "dn-splatter-big", "ags-mesh"} <= methods.keys()
assert {"coolermap", "mushroom", "normal-nerfstudio", "scannetpp"} <= parsers.keys()
assert metadata.version("dn-splatter") == "0.0.1"

print("nerfstudio environment: ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("dn_splatter", dn_splatter.__file__)
PY

"${BIN_DIR}/gs-mesh" --help >/dev/null
