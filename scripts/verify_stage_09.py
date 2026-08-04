#!/usr/bin/env python3
"""Independently verify the formal 09_scene_graph artifact contract."""

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
    require(path.stat().st_size == record["size_bytes"], f"size mismatch for {name}")
    require(sha256(path) == record["sha256"], f"hash mismatch for {name}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    stage = args.run_root.resolve() / "09_scene_graph"
    manifest_path = stage / "manifest.json"
    require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["stage"] == "09_scene_graph", "wrong stage")
    require(manifest["status"] == "complete", "stage is not complete")
    require(manifest["algorithm"]["appendix_sections"] == ["D.1", "D.2", "D.3", "D.4", "D.5"], "wrong Appendix sections")
    configuration = manifest["algorithm"]["configuration"]
    require(configuration["floor_merge_height_meters"] == 0.5, "wrong floor merge threshold")
    require(configuration["ceiling_minimum_clearance_meters"] == 1.0, "wrong ceiling clearance")
    require(configuration["wall_interval_height_meters"] == 2.5, "wrong wall interval")
    require(configuration["bottleneck_widths_meters"] == [2.5, 1.5], "wrong bottleneck widths")
    require(configuration["door_maximum_width_meters"] == 1.5, "wrong door threshold")
    require(configuration["stair_room_maximum_distance_meters"] == 0.5, "wrong stair distance threshold")

    for name, record in manifest["inputs"].items():
        if name == "openseg_model":
            check_record(record["saved_model"], "OpenSeg SavedModel")
        else:
            check_record(record, f"input {name}")
    output_paths = {
        name: check_record(record, f"output {name}")
        for name, record in manifest["outputs"].items()
        if name != "attempt_dir"
    }
    attempt = Path(manifest["outputs"]["attempt_dir"])
    require(attempt.is_dir(), f"missing completed attempt: {attempt}")
    require(attempt.name.endswith("_complete"), "attempt is not marked complete")

    levels = json.loads(output_paths["levels"].read_text(encoding="utf-8"))
    rooms_geojson = json.loads(output_paths["rooms"].read_text(encoding="utf-8"))
    graph = json.loads(output_paths["scene_graph"].read_text(encoding="utf-8"))
    rooms = graph["rooms"]
    edges = graph["edges"]
    room_ids = {str(room["room_id"]) for room in rooms}
    require(len(levels) == manifest["counts"]["levels"], "level count mismatch")
    require(len(rooms) == manifest["counts"]["rooms_before_pruning"], "room count mismatch")
    require(len(rooms_geojson["features"]) == len(rooms), "GeoJSON room count mismatch")
    require(all(feature["geometry"] for feature in rooms_geojson["features"]), "empty room geometry")
    require(all(str(room_id) in room_ids for edge in edges for room_id in edge["room_ids"]), "edge references unknown room")
    require(set(graph["active_room_ids"]).isdisjoint(graph["pruned_room_ids"]), "active and pruned rooms overlap")
    require(set(graph["active_room_ids"]) | set(graph["pruned_room_ids"]) == room_ids, "room activity partition is incomplete")

    room_features = np.load(output_paths["room_semantic_features"], allow_pickle=False)
    text_features = np.load(output_paths["room_text_features"], allow_pickle=False)
    require(room_features.shape == (len(rooms), 768), "wrong room feature shape")
    require(text_features.shape == (15, 768), "wrong room text feature shape")
    require(np.isfinite(room_features).all(), "room features are non-finite")
    require(np.isfinite(text_features).all(), "text features are non-finite")
    openseg = np.load(output_paths["openseg_triangle_features"], allow_pickle=False)
    triangle_features = openseg["triangle_features"]
    triangle_counts = openseg["triangle_feature_counts"]
    require(triangle_features.ndim == 2 and triangle_features.shape[1] == 768, "wrong triangle feature shape")
    require(triangle_counts.shape == (len(triangle_features),), "wrong triangle count shape")
    require(np.isfinite(triangle_features).all(), "triangle features are non-finite")
    require(int(np.count_nonzero(triangle_counts)) == manifest["counts"]["triangles_with_openseg_features"], "OpenSeg coverage mismatch")

    level_room_total = 0
    for level in levels:
        grid_path = attempt / f"{level['level_id']}_grid.npz"
        require(grid_path.is_file(), f"missing level grid: {grid_path}")
        grid = np.load(grid_path, allow_pickle=False)
        labels = grid["labels"]
        level_labels = [int(value) for value in np.unique(labels) if value > 0]
        require(len(level_labels) == len(level["room_ids"]), f"room labels mismatch for {level['level_id']}")
        require(float(grid["resolution_meters"]) == configuration["grid_resolution_meters"], "grid resolution mismatch")
        level_room_total += len(level_labels)
    require(level_room_total == len(rooms), "per-level rooms do not cover graph rooms")

    preview = o3d.io.read_triangle_mesh(str(output_paths["preview_mesh"]))
    require(len(preview.vertices) > 0 and len(preview.triangles) > 0, "preview mesh is empty")
    require(manifest["validation"] and all(manifest["validation"].values()), "manifest validation failed")
    print(
        json.dumps(
            {
                "status": "verified",
                "levels": len(levels),
                "rooms": len(rooms),
                "edges": len(edges),
                "triangles_with_openseg_features": int(np.count_nonzero(triangle_counts)),
                "preview_vertices": len(preview.vertices),
                "preview_triangles": len(preview.triangles),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
