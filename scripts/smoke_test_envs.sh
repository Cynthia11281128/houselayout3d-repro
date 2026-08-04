#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${PROJECT_ROOT}/scripts/verify_external_revisions.py"
bash "${PROJECT_ROOT}/scripts/verify_weights.sh"
bash "${PROJECT_ROOT}/scripts/smoke_test_nerfstudio_env.sh"
bash "${PROJECT_ROOT}/scripts/smoke_test_layout_env.sh"
bash "${PROJECT_ROOT}/scripts/smoke_test_openseg_env.sh"

"/tmp/tmp_data/miniconda3/envs/nerfstudio/bin/python" -m pip check
"/tmp/tmp_data/miniconda3/envs/houselayout3d-layout/bin/python" -m pip check
"/tmp/tmp_data/miniconda3/envs/houselayout3d-openseg/bin/python" -m pip check

printf 'all HouseLayout3D environments: ok\n'
