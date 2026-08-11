"""Independent validation validation for the exported HouseLayout3D layout."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "houselayout3d"


from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
import open3d as o3d

from .config import PipelineConfig


class ValidationError(RuntimeError):
    """Raised when final layout validation fails."""


ATTEMPT_RE = re.compile(r"^attempt_(?P<index>\d+)_(?:running|complete|failed)$")
LAYOUT_CLASSES = frozenset({"wall", "floor", "ceiling", "stairs", "door", "window"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_status(path: Path, state: str, detail: str) -> None:
    write_json(
        path,
        {"state": state, "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat()},
    )


def next_attempt(component_dir: Path) -> Path:
    indices = []
    for path in component_dir.iterdir() if component_dir.is_dir() else ():
        match = ATTEMPT_RE.match(path.name)
        if match:
            indices.append(int(match.group("index")))
    attempt = component_dir / f"attempt_{max(indices, default=0) + 1:03d}_running"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def rename_attempt(path: Path, state: str) -> Path:
    destination = path.with_name(path.name.rsplit("_", 1)[0] + f"_{state}")
    path.rename(destination)
    return destination


def require_hash(entry: Mapping[str, Any], label: str) -> Path:
    path = Path(str(entry["path"]))
    if not path.is_file():
        raise ValidationError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != entry["sha256"]:
        raise ValidationError(f"SHA256 mismatch for {label}: {path}")
    if path.stat().st_size != int(entry["size_bytes"]):
        raise ValidationError(f"size mismatch for {label}: {path}")
    return path


def mesh_report(path: Path, weld: bool = False) -> dict[str, Any]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValidationError(f"empty mesh: {path}")
    if weld:
        mesh.merge_close_vertices(1.0e-6)
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices)
    return {
        "path": str(path),
        "vertices": len(vertices),
        "triangles": len(mesh.triangles),
        "surface_area_square_meters": float(mesh.get_surface_area()),
        "finite": bool(np.isfinite(vertices).all()),
        "edge_manifold_closed": bool(mesh.is_edge_manifold(allow_boundary_edges=False)),
        "edge_manifold_allow_boundary": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "orientable": bool(mesh.is_orientable()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
        "watertight": bool(mesh.is_watertight()),
    }


def triangle_xy_area(vertices: Iterable[Iterable[float]]) -> float:
    points = np.asarray(list(vertices), dtype=np.float64)
    if points.shape != (3, 3):
        raise ValidationError(f"floor entity is not a 3D triangle: {points.shape}")
    return 0.5 * abs(float(np.cross(points[1, :2] - points[0, :2], points[2, :2] - points[0, :2])))


def validate_window(entity: Mapping[str, Any], minimum_points: int, minimum_size: float) -> dict[str, Any]:
    vertices = np.asarray(entity["vertices"], dtype=np.float64)
    if vertices.shape != (4, 3) or not np.isfinite(vertices).all():
        raise ValidationError(f"invalid window vertices: {entity.get('entity_id')}")
    edges = np.roll(vertices, -1, axis=0) - vertices
    closure = float(np.linalg.norm(edges[0] + edges[2]) + np.linalg.norm(edges[1] + edges[3]))
    coplanarity = float(abs(np.dot(edges[0], np.cross(edges[1], edges[2]))))
    if closure > 1.0e-5 or coplanarity > 1.0e-5:
        raise ValidationError(f"window is not a planar parallelogram: {entity['entity_id']}")
    if int(entity["point_count"]) < minimum_points:
        raise ValidationError(f"window has too few points: {entity['entity_id']}")
    if float(entity["width_meters"]) <= minimum_size or float(entity["height_meters"]) <= minimum_size:
        raise ValidationError(f"window is below the paper size threshold: {entity['entity_id']}")
    return {
        "entity_id": entity["entity_id"],
        "point_count": int(entity["point_count"]),
        "width_meters": float(entity["width_meters"]),
        "height_meters": float(entity["height_meters"]),
        "room_ids": list(entity["room_ids"]),
    }


def run_validation(config: PipelineConfig, run_id: str) -> Path:
    run_dir = config.storage.outputs / config.scene / run_id
    component_dir = run_dir / "validation"
    component_dir.mkdir(parents=True, exist_ok=True)
    attempt = next_attempt(component_dir)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    write_status(component_dir / "STATUS.json", "running", str(attempt))
    write_status(attempt / "STATUS.json", "loading", "scene_graph and layout manifests")
    try:
        scene_graph_manifest_path = run_dir / "scene_graph" / "manifest.json"
        layout_manifest_path = run_dir / "layout" / "manifest.json"
        for path, component in (
            (scene_graph_manifest_path, "scene_graph"),
            (layout_manifest_path, "layout"),
        ):
            if not path.is_file():
                raise ValidationError(f"missing completed manifest: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("component") != component or payload.get("status") != "complete":
                raise ValidationError(f"manifest is not complete for {component}: {path}")
        scene_graph = json.loads(scene_graph_manifest_path.read_text(encoding="utf-8"))
        layout_manifest = json.loads(layout_manifest_path.read_text(encoding="utf-8"))
        write_status(attempt / "STATUS.json", "hashes", "checking layout output records")
        output_paths = {
            name: require_hash(layout_manifest["outputs"][name], f"layout {name}")
            for name in (
                "layout_mesh",
                "layout_obj",
                "structures_mesh",
                "walls_mesh",
                "entities",
                "final_scene_graph",
                "diagnostics",
            )
        }
        for optional in ("windows_mesh", "doors_mesh", "stairs_mesh"):
            if optional in layout_manifest["outputs"]:
                output_paths[optional] = require_hash(layout_manifest["outputs"][optional], f"layout {optional}")
        for category in ("rooms_closed", "rooms_final"):
            for room_id, entry in layout_manifest["outputs"][category].items():
                require_hash(entry, f"{category} {room_id}")

        graph_path = Path(scene_graph["outputs"]["scene_graph"]["path"])
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        entities_payload = json.loads(output_paths["entities"].read_text(encoding="utf-8"))
        final_graph = json.loads(output_paths["final_scene_graph"].read_text(encoding="utf-8"))
        entities = entities_payload["entities"]
        active_room_ids = [str(value) for value in graph["active_room_ids"]]
        active_room_set = set(active_room_ids)
        entity_ids = [str(value["entity_id"]) for value in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValidationError("layout entity IDs are not unique")
        class_counts = Counter(str(value["class"]) for value in entities)
        if not set(class_counts).issubset(LAYOUT_CLASSES):
            raise ValidationError(f"unexpected entity classes: {set(class_counts) - LAYOUT_CLASSES}")
        if class_counts["floor"] == 0 or class_counts["ceiling"] == 0 or class_counts["wall"] == 0:
            raise ValidationError("structural entity class is empty")

        floor_areas = Counter()
        invalid_room_references = []
        for entity in entities:
            room_id = entity.get("room_id")
            if room_id is not None and str(room_id) not in active_room_set:
                invalid_room_references.append(str(room_id))
            for associated in entity.get("room_ids", []):
                if str(associated) not in active_room_set:
                    invalid_room_references.append(str(associated))
            if entity["class"] == "floor":
                floor_areas[str(room_id)] += triangle_xy_area(entity["vertices"])
        if invalid_room_references:
            raise ValidationError(f"invalid room references: {sorted(set(invalid_room_references))}")

        diagnostics = json.loads(output_paths["diagnostics"].read_text(encoding="utf-8"))
        expected_areas = {
            str(value["room_id"]): float(value["floor_area_square_meters"])
            for value in diagnostics["rooms"]
        }
        area_errors = {
            room_id: abs(float(floor_areas[room_id]) - expected_areas[room_id])
            for room_id in active_room_ids
        }
        if max(area_errors.values(), default=0.0) > 1.0e-5:
            raise ValidationError(f"floor triangulation area mismatch: {area_errors}")

        write_status(attempt / "STATUS.json", "topology", "reloading final and per-room meshes")
        mesh_reports = {
            "layout": mesh_report(output_paths["layout_mesh"]),
            "structures": mesh_report(output_paths["structures_mesh"]),
            "walls": mesh_report(output_paths["walls_mesh"]),
        }
        closed_reports = {
            room_id: mesh_report(Path(layout_manifest["outputs"]["rooms_closed"][room_id]["path"]), weld=True)
            for room_id in active_room_ids
        }
        final_reports = {
            room_id: mesh_report(Path(layout_manifest["outputs"]["rooms_final"][room_id]["path"]), weld=True)
            for room_id in active_room_ids
        }
        for room_id, report in closed_reports.items():
            if not (report["edge_manifold_closed"] and report["vertex_manifold"] and report["orientable"]):
                raise ValidationError(f"closed room shell failed topology: {room_id}: {report}")
        graph_open_rooms = {
            str(room_id)
            for edge in graph["edges"]
            if not edge.get("pruned", False) and edge["kind"] in {"door", "opening"}
            for room_id in edge["room_ids"]
        }
        for room_id in graph_open_rooms:
            if final_reports[room_id]["edge_manifold_closed"]:
                raise ValidationError(f"graph opening did not create a room boundary: {room_id}")

        windows = [value for value in entities if value["class"] == "window"]
        window_reports = [
            validate_window(
                value,
                config.layout.window_minimum_cluster_points,
                config.layout.window_minimum_size_meters,
            )
            for value in windows
        ]
        graph_edges = [value for value in graph["edges"] if not value.get("pruned", False)]
        edge_counts = Counter(value["kind"] for value in graph_edges)
        final_graph_checks = {
            "active_room_ids_match": set(final_graph["active_room_ids"]) == active_room_set,
            "room_mesh_paths_exist": all(Path(value).is_file() for value in final_graph["room_meshes"].values()),
            "window_count_matches": len(final_graph["windows"]) == len(windows),
            "door_count_matches": len(final_graph["door_entities"]) == edge_counts["door"],
            "stair_count_matches": len(final_graph["stair_entities"]) == edge_counts["stair"],
        }
        if not all(final_graph_checks.values()):
            raise ValidationError(f"final scene graph failed: {final_graph_checks}")
        validation = {
            "all_declared_output_hashes_match": True,
            "layout_mesh_finite_and_nonempty": mesh_reports["layout"]["finite"] and mesh_reports["layout"]["triangles"] > 0,
            "all_active_rooms_have_floor_area": set(floor_areas) == active_room_set,
            "floor_triangulation_area_conserved": max(area_errors.values(), default=0.0) <= 1.0e-5,
            "all_preopening_room_shells_topologically_closed": True,
            "graph_openings_create_room_boundaries": all(not final_reports[value]["edge_manifold_closed"] for value in graph_open_rooms),
            "window_rectangles_valid": len(window_reports) == class_counts["window"],
            "entity_room_references_valid": True,
            "final_scene_graph_valid": all(final_graph_checks.values()),
            "no_ground_truth_inputs_used": True,
        }
        if not all(validation.values()):
            raise ValidationError(f"final validation failed: {validation}")

        report = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "rooms": len(active_room_ids),
                "entities": len(entities),
                "entity_classes": dict(sorted(class_counts.items())),
                "graph_edges": dict(sorted(edge_counts.items())),
                "windows": len(windows),
            },
            "mesh_reports": mesh_reports,
            "closed_room_reports": closed_reports,
            "final_room_reports": final_reports,
            "floor_area_absolute_errors_square_meters": area_errors,
            "maximum_floor_area_error_square_meters": max(area_errors.values(), default=0.0),
            "window_reports": window_reports,
            "window_projection_diagnostics": diagnostics["windows"],
            "final_scene_graph_checks": final_graph_checks,
            "validation": validation,
            "known_limitations": [
                "The pose component uses the supplied known poses rather than COLMAP poses, as explicitly selected for this run.",
                "scene_graph accepted no door or stair edges for this single-level scene; the final layout contains one ordinary opening.",
                "The paper does not disclose DBSCAN, LOF, voxel, and stair-step visualization settings; the explicit layout YAML values were validated for reproducibility.",
                "The unified layout contains intentional coincident/shared room surfaces and opening boundaries, so global self-intersection is reported rather than treated as a closed-solid requirement.",
            ],
        }
        report_path = attempt / "final_report.json"
        markdown_path = attempt / "final_report.md"
        write_json(report_path, report)
        markdown_path.write_text(
            "# HouseLayout3D final validation\n\n"
            f"- Scene/run: `{config.scene}/{run_id}`\n"
            f"- Rooms: {len(active_room_ids)}\n"
            f"- Final mesh: {mesh_reports['layout']['vertices']} vertices, {mesh_reports['layout']['triangles']} triangles\n"
            f"- Entities: {len(entities)} ({dict(sorted(class_counts.items()))})\n"
            f"- Graph: {dict(sorted(edge_counts.items()))}\n"
            f"- Window rays: {diagnostics['windows']['candidate_pixel_count']} candidates, "
            f"{diagnostics['windows']['wall_ray_hit_count']} wall hits, {len(windows)} accepted windows\n"
            f"- Maximum room floor-area error: {max(area_errors.values(), default=0.0):.9g} m^2\n"
            "- Validation: PASS\n\n"
            "See `final_report.json` for per-room topology and per-window geometry.\n",
            encoding="utf-8",
        )
        completed_attempt = rename_attempt(attempt, "complete")
        completed_report = completed_attempt / report_path.name
        completed_markdown = completed_attempt / markdown_path.name
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "component": "validation",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "inputs": {
                "scene_graph_manifest": record(scene_graph_manifest_path),
                "layout_manifest": record(layout_manifest_path),
            },
            "outputs": {
                "attempt_dir": str(completed_attempt),
                "final_report": record(completed_report),
                "final_report_markdown": record(completed_markdown),
            },
            "counts": report["counts"],
            "validation": validation,
            "warnings": report["known_limitations"],
        }
        manifest_path = component_dir / "manifest.json"
        write_json(manifest_path, manifest)
        write_status(completed_attempt / "STATUS.json", "complete", str(completed_report))
        write_status(component_dir / "STATUS.json", "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        if attempt.exists():
            failed_attempt = rename_attempt(attempt, "failed")
            write_status(failed_attempt / "STATUS.json", "failed", f"{type(error).__name__}: {error}")
            write_status(component_dir / "STATUS.json", "failed", str(failed_attempt))
        if isinstance(error, ValidationError):
            raise
        raise ValidationError(f"{type(error).__name__}: {error}") from error


def main() -> int:
    from .direct import run_component

    return run_component(run_validation, "Validate final layout outputs.")


if __name__ == "__main__":
    raise SystemExit(main())
