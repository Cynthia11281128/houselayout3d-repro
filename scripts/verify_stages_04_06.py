#!/usr/bin/env python3
"""Verify the completed formal 04_mesh through 06_skeleton artifact chain."""

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


def require_hash(record: dict) -> Path:
    path = Path(record["path"])
    assert path.is_file(), path
    assert sha256(path) == record["sha256"], path
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    manifests = {}
    for stage in ("04_mesh", "05_oneformer", "06_skeleton"):
        path = run_dir / stage / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["status"] == "complete", path
        manifests[stage] = manifest

    mesh_manifest = manifests["04_mesh"]
    poisson_path = require_hash(mesh_manifest["outputs"]["poisson_mesh"])
    require_hash(mesh_manifest["outputs"]["oriented_pointcloud"])
    poisson = o3d.io.read_triangle_mesh(str(poisson_path))
    assert len(poisson.vertices) == mesh_manifest["outputs"]["poisson_mesh"]["vertex_count"]
    assert len(poisson.triangles) == mesh_manifest["outputs"]["poisson_mesh"]["triangle_count"]

    oneformer = manifests["05_oneformer"]
    require_hash(oneformer["outputs"]["labels"])
    records_path = require_hash(oneformer["outputs"]["per_image"])
    semantic_records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(semantic_records) == oneformer["validation"]["frame_count"]
    for record in semantic_records:
        assert sha256(Path(record["coco_path"])) == record["coco_sha256"]
        assert sha256(Path(record["layout_path"])) == record["layout_sha256"]

    skeleton = manifests["06_skeleton"]
    for record in skeleton["outputs"]["rendered_depth_records"]:
        require_hash(record)
    array_paths = {
        name: require_hash(record)
        for name, record in skeleton["outputs"]["arrays"].items()
    }
    for record in skeleton["outputs"]["meshes"].values():
        if record.get("path") is not None:
            require_hash(record)
        if record.get("classes_path") is not None:
            class_path = Path(record["classes_path"])
            assert sha256(class_path) == record["classes_sha256"]

    ray_origins = np.load(array_paths["full_ray_origins.npy"], mmap_mode="r")
    ray_dests = np.load(array_paths["full_ray_dests.npy"], mmap_mode="r")
    ray_valid = np.load(array_paths["ray_is_valid.npy"], mmap_mode="r")
    ray_labels = np.load(
        array_paths["hard_labels_simplified_segmentations.npy"], mmap_mode="r"
    )
    vertex_votes = np.load(array_paths["vertex_vote_counts.npy"], mmap_mode="r")
    probabilities = np.load(array_paths["vertex_probabilities.npy"])
    hard = np.load(array_paths["vertex_hard_assignments.npy"])
    expected_rays = skeleton["statistics"]["ray_count"]
    expected_vertices = skeleton["statistics"]["mesh_vertex_count"]
    assert ray_origins.shape == ray_dests.shape == (expected_rays, 3)
    assert ray_valid.shape == ray_labels.shape == (expected_rays,)
    assert vertex_votes.shape == probabilities.shape == (expected_vertices, 9)
    assert hard.shape == (expected_vertices,)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-3)
    assert np.array_equal(probabilities.argmax(axis=1), hard)

    segmentations = [
        np.load(run_dir / "06_skeleton" / "spt" / f"level_{level}_segmentation.npy")
        for level in range(1, 4)
    ]
    for level, segmentation in enumerate(segmentations, start=1):
        assert segmentation.shape == (expected_vertices,)
        assert int(segmentation.max()) + 1 == skeleton["statistics"][
            "superpoint_segment_counts"
        ][level - 1]
    for fine, coarse in zip(segmentations, segmentations[1:]):
        mapping = np.full(int(fine.max()) + 1, -1, dtype=coarse.dtype)
        mapping[fine] = coarse
        assert np.array_equal(mapping[fine], coarse)

    print(
        json.dumps(
            {
                "status": "verified",
                "run_dir": str(run_dir),
                "frames": len(semantic_records),
                "rays": expected_rays,
                "mesh_vertices": expected_vertices,
                "mesh_triangles": skeleton["statistics"]["mesh_triangle_count"],
                "superpoints": skeleton["statistics"]["superpoint_segment_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
