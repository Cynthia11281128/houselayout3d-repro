"""Seeded launcher for the byte-preserved unofficial prototype optimizer."""

from __future__ import annotations

import random
import runpy
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] != "--source-script":
        raise SystemExit(
            "usage: python -m src.layout_prototype.prototype_entry "
            "--source-script PATH --random-seed N [source arguments ...]"
        )
    source_script = Path(sys.argv[2]).resolve()
    if sys.argv[3] != "--random-seed" or len(sys.argv) < 5:
        raise SystemExit("--random-seed must follow --source-script PATH")
    random_seed = int(sys.argv[4])
    source_arguments = sys.argv[5:]
    if not source_script.is_file():
        raise SystemExit(f"source script is missing: {source_script}")

    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is required by the prototype optimizer") from error
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    sys.argv = [str(source_script), *source_arguments]
    runpy.run_path(str(source_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
