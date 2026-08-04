#!/usr/bin/env python3
"""Independently verify the formal 08_prototype artifact contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {detail}")


def check_record(record: dict, name: str) -> Path:
    path = Path(record["path"])
    require(path.is_file(), f"missing {name}: {path}")
    require(
        path.stat().st_size == record["size_bytes"],
        f"size mismatch for {name}",
    )
    require(sha256(path) == record["sha256"], f"hash mismatch for {name}")
    return path


def check_mesh(record: dict, name: str) -> dict[str, object]:
    path = check_record(record, name)
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    require(len(vertices) > 0, f"{name} has no vertices")
    require(len(triangles) > 0, f"{name} has no triangles")
    require(np.isfinite(vertices).all(), f"{name} has non-finite vertices")
    require(len(vertices) == record["vertex_count"], f"wrong {name} vertex count")
    require(
        len(triangles) == record["triangle_count"],
        f"wrong {name} triangle count",
    )
    require(
        np.isclose(mesh.get_surface_area(), record["surface_area_square_meters"]),
        f"wrong {name} surface area",
    )
    require(
        np.allclose(vertices.min(axis=0), record["axis_aligned_minimum"]),
        f"wrong {name} minimum bounds",
    )
    require(
        np.allclose(vertices.max(axis=0), record["axis_aligned_maximum"]),
        f"wrong {name} maximum bounds",
    )
    return {
        "path": str(path),
        "vertices": len(vertices),
        "triangles": len(triangles),
        "surface_area_square_meters": float(mesh.get_surface_area()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_root",
        type=Path,
        help="outputs/<scene>/<run-id> directory",
    )
    args = parser.parse_args()
    stage = args.run_root.resolve() / "08_prototype"
    manifest_path = stage / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["stage"] == "08_prototype", "wrong manifest stage")
    require(manifest["status"] == "complete", "stage is not complete")
    require(manifest["return_code"] == 0, "optimizer return code is non-zero")
    require(manifest["algorithm"]["iterations"] == 4000, "wrong iteration count")
    require(
        manifest["algorithm"]["checkpoint_interval"] == 100,
        "wrong checkpoint interval",
    )

    check_record(manifest["inputs"]["prepared_manifest"], "prepared manifest")
    for name, record in manifest["inputs"].items():
        if name != "prepared_manifest":
            check_record(record, f"input {name}")
    for name, record in manifest["source"]["files"].items():
        check_record(record, f"source {name}")
    require(
        manifest["source"]["source_files_modified"] is False,
        "unofficial source is marked modified",
    )

    outputs = manifest["outputs"]
    attempt = Path(outputs["attempt_dir"])
    require(attempt.is_dir(), f"missing completed attempt: {attempt}")
    require(attempt.name.endswith("_complete"), "attempt is not marked complete")
    log_path = check_record(outputs["log"], "optimizer log")
    initial = check_mesh(outputs["initial_mesh"], "initial mesh")
    final = check_mesh(outputs["final_mesh"], "final mesh")
    check_record(outputs["final_model_state"], "final model state")

    expected_steps = list(range(0, 4000, 100))
    mesh_steps = [record["step"] for record in outputs["mesh_checkpoints"]]
    state_steps = [record["step"] for record in outputs["model_state_checkpoints"]]
    require(mesh_steps == expected_steps, "mesh checkpoint steps are incomplete")
    require(state_steps == expected_steps, "state checkpoint steps are incomplete")
    for record in outputs["mesh_checkpoints"]:
        check_record(record, f"mesh checkpoint {record['step']}")
    for record in outputs["model_state_checkpoints"]:
        check_record(record, f"state checkpoint {record['step']}")

    validation = manifest["validation"]
    require(validation and all(validation.values()), "manifest validation failed")
    require(
        outputs["initial_mesh"]["sha256"] != outputs["final_mesh"]["sha256"],
        "initial and final meshes are byte-identical",
    )
    log = log_path.read_text(encoding="utf-8", errors="replace")
    require("Saved checkpoint" in log, "optimizer log has no checkpoint evidence")

    print(
        json.dumps(
            {
                "status": "verified",
                "iterations": 4000,
                "mesh_checkpoint_count": len(mesh_steps),
                "state_checkpoint_count": len(state_steps),
                "initial_mesh": initial,
                "final_mesh": final,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
