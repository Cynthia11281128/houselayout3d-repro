#!/usr/bin/env python3
"""Independently verify the formal 07_polygon_init artifact contract."""

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
    require(sha256(path) == record["sha256"], f"hash mismatch for {name}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_root",
        type=Path,
        help="outputs/<scene>/<run-id> directory",
    )
    args = parser.parse_args()
    stage = args.run_root.resolve() / "07_polygon_init"
    manifest_path = stage / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["stage"] == "07_polygon_init", "wrong manifest stage")
    require(manifest["status"] == "complete", "stage is not complete")

    outputs = manifest["outputs"]
    rectified_path = check_record(outputs["rectified_mesh"], "rectified mesh")
    component_path = check_record(
        outputs["plane_components_mesh"], "plane-components mesh"
    )
    polygon_path = check_record(outputs["polygon_info"], "polygon info")
    check_record(outputs["polygon_boundaries"], "polygon boundaries")
    candidate_path = check_record(outputs["plane_candidates"], "plane candidates")
    array_paths = {
        name: check_record(record, name)
        for name, record in outputs["arrays"].items()
    }

    mesh = o3d.io.read_triangle_mesh(
        str(rectified_path), enable_post_processing=False
    )
    component_mesh = o3d.io.read_triangle_mesh(
        str(component_path), enable_post_processing=False
    )
    vertices = np.asarray(mesh.vertices)
    require(
        len(vertices) == manifest["statistics"]["structure_vertex_count"],
        "rectified vertex count disagrees with manifest",
    )
    require(
        len(mesh.triangles) == manifest["statistics"]["structure_triangle_count"],
        "rectified triangle count disagrees with manifest",
    )
    require(
        len(component_mesh.triangles)
        == manifest["statistics"]["component_triangle_count"],
        "component triangle count disagrees with manifest",
    )

    polygons = json.loads(polygon_path.read_text(encoding="utf-8"))
    require(
        set(polygons) == {str(index) for index in range(len(polygons))},
        "polygon JSON keys are not consecutive integer strings",
    )
    require(
        len(polygons) == manifest["statistics"]["polygon_count"],
        "polygon count disagrees with manifest",
    )
    required = {"vertices", "contours", "plane_eq", "color"}
    contour_count = 0
    full_contour_vertices = 0
    simplified_contour_vertices = 0
    maximum_residual = 0.0
    for key, polygon in polygons.items():
        require(required <= set(polygon), f"polygon {key} misses optimizer keys")
        require(polygon["id"] == int(key), f"polygon {key} has a wrong id")
        inliers = np.asarray(polygon["vertices"], dtype=np.int64)
        plane = np.asarray(polygon["plane_eq"], dtype=np.float64)
        require(plane.shape == (4,), f"polygon {key} plane shape is invalid")
        require(
            np.isclose(np.linalg.norm(plane[:3]), 1.0),
            f"polygon {key} plane normal is not unit length",
        )
        require(
            len(polygon["contours"]) == len(polygon["contour_masks_rdp"]),
            f"polygon {key} contour/mask counts differ",
        )
        require(
            len(polygon["contours"]) == len(polygon["contour_areas_original"]),
            f"polygon {key} contour/area counts differ",
        )
        inlier_set = set(inliers.tolist())
        require(
            len(inliers) > 0 and inliers.min() >= 0 and inliers.max() < len(vertices),
            f"polygon {key} inlier index is invalid",
        )
        residual = np.abs(vertices[inliers] @ plane[:3] + plane[3])
        maximum_residual = max(maximum_residual, float(residual.max()))
        for contour, mask, area in zip(
            polygon["contours"],
            polygon["contour_masks_rdp"],
            polygon["contour_areas_original"],
        ):
            contour_array = np.asarray(contour, dtype=np.int64)
            mask_array = np.asarray(mask, dtype=bool)
            require(len(contour_array) >= 3, f"polygon {key} has a short contour")
            require(
                len(contour_array) == len(mask_array),
                f"polygon {key} contour/mask lengths differ",
            )
            require(
                int(mask_array.sum()) >= 3,
                f"polygon {key} simplifies below three vertices",
            )
            require(float(area) > 0, f"polygon {key} has non-positive contour area")
            require(
                set(contour_array.tolist()) <= inlier_set,
                f"polygon {key} contour is not a subset of its inliers",
            )
            contour_count += 1
            full_contour_vertices += len(contour_array)
            simplified_contour_vertices += int(mask_array.sum())

    statistics = manifest["statistics"]
    require(contour_count == statistics["contour_count"], "wrong contour count")
    require(
        full_contour_vertices == statistics["full_contour_vertex_count"],
        "wrong full contour vertex count",
    )
    require(
        simplified_contour_vertices
        == statistics["simplified_contour_vertex_count"],
        "wrong simplified contour vertex count",
    )
    require(maximum_residual < 1e-6, "rectified inliers are not on their planes")

    assignments = np.load(array_paths["assigned_plane_ids.npy"])
    residual_mask = np.load(array_paths["residual_unassigned_mask.npy"])
    structure_to_full = np.load(array_paths["structure_to_full_vertex.npy"])
    require(len(assignments) == len(vertices), "assignment length is invalid")
    require(len(residual_mask) == len(vertices), "residual mask length is invalid")
    require(len(structure_to_full) == len(vertices), "full-index map length is invalid")
    require(
        np.array_equal(residual_mask, assignments == -1),
        "residual mask and assignments disagree",
    )
    require(
        int((assignments >= 0).sum()) == statistics["assigned_polygon_vertex_count"],
        "assigned vertex count disagrees with manifest",
    )
    require(
        int((assignments == -2).sum()) == statistics["rejected_vertex_count"],
        "rejected vertex count disagrees with manifest",
    )
    require(
        int(residual_mask.sum()) == statistics["residual_unassigned_vertex_count"],
        "residual vertex count disagrees with manifest",
    )
    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    require(
        len(candidates) == statistics["candidate_plane_count"],
        "candidate count disagrees with manifest",
    )

    print(
        json.dumps(
            {
                "status": "verified",
                "polygon_count": len(polygons),
                "contour_count": contour_count,
                "assigned_vertex_count": int((assignments >= 0).sum()),
                "rejected_vertex_count": int((assignments == -2).sum()),
                "residual_vertex_count": int(residual_mask.sum()),
                "maximum_rectified_plane_residual_meters": maximum_residual,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
