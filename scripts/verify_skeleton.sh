#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m houselayout3d inspect-config "${PROJECT_ROOT}/configs/r04_front.yaml"
python -m houselayout3d stages
python -m unittest discover -s "${PROJECT_ROOT}/tests" -v

