#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "references" / "external_revisions.json"


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def check(name: str, path: Path, expected: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing external repository: {path}")
    actual = git_head(path)
    if actual != expected:
        raise SystemExit(
            f"revision mismatch for {name}: expected {expected}, got {actual}"
        )
    print(f"{name}: {actual}")


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    external = PROJECT_ROOT / "external"
    for name, expected in lock["repositories"].items():
        check(name, external / name, expected)

    dependencies = external / "superpoint-transformer" / "src" / "dependencies"
    for name, expected in lock["superpoint_transformer_dependencies"].items():
        check(f"superpoint-transformer/{name}", dependencies / name, expected)


if __name__ == "__main__":
    main()
