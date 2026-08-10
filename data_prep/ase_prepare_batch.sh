#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_ROOT="${REPO_ROOT}/raw_data/aria_ase_train_random10_chunks_seed42/raw"
DEST_ROOT="${REPO_ROOT}/data/aria_ase"
PYTHON_BIN="${PYTHON:-python3}"
CONTINUE_ON_ERROR=0
SINGLE_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  data_prep/ase_prepare_batch.sh [options] [-- single-script-options]

Options:
  --raw-root PATH          Folder containing raw scene subdirectories.
  --dest-root PATH         Folder where converted scenes are written.
  --python PATH            Python executable used to run ase_prepare_single.py.
  --continue-on-error      Continue after a scene fails.
  -h, --help               Show this help.

Any remaining options are passed to ase_prepare_single.py, for example:
  --force
  --dry-run
  --frame-stride 2
  --intrinsics-json /path/to/pinhole_intrinsics.json
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root)
      RAW_ROOT="$2"
      shift 2
      ;;
    --dest-root)
      DEST_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      SINGLE_ARGS+=("$@")
      break
      ;;
    *)
      SINGLE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "ase_prepare_batch: raw root does not exist: ${RAW_ROOT}" >&2
  exit 2
fi

mapfile -t SCENES < <(find "${RAW_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ "${#SCENES[@]}" -eq 0 ]]; then
  echo "ase_prepare_batch: no scene directories found under ${RAW_ROOT}" >&2
  exit 2
fi

success=0
failed=0
for scene_id in "${SCENES[@]}"; do
  source_dir="${RAW_ROOT}/${scene_id}"
  dest_dir="${DEST_ROOT}/${scene_id}"
  echo "[ase_prepare_batch] ${scene_id}: ${source_dir} -> ${dest_dir}"
  if "${PYTHON_BIN}" "${SCRIPT_DIR}/ase_prepare_single.py" \
      --source "${source_dir}" \
      --dest-root "${dest_dir}" \
      "${SINGLE_ARGS[@]}"; then
    success=$((success + 1))
  else
    failed=$((failed + 1))
    if [[ "${CONTINUE_ON_ERROR}" -eq 0 ]]; then
      echo "ase_prepare_batch: failed on scene ${scene_id}" >&2
      exit 1
    fi
  fi
done

echo "[ase_prepare_batch] complete: success=${success} failed=${failed}"
if [[ "${failed}" -gt 0 ]]; then
  exit 1
fi
