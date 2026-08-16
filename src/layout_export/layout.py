"""Export final single-floor Section 4.4 layout entities and meshes."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.layout_export"

import argparse
import colorsys
import hashlib
import json
import math
import os
import platform
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._core.extrusion import (
    CLASS_COLORS,
    LayoutExportConfig,
    MeshData,
    add_room_shell,
    door_entity,
    load_ceiling_candidates,
    mesh_from_entity,
    stair_entity,
    window_entity,
    write_obj,
    write_ply,
)
from ._core.triangulation import LayoutExportError
from ._core.triangulation import iter_polygons

FALLBACK_COLOR = (230, 38, 219)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def write_status(component_dir: Path, state: str, detail: str = "") -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        component_dir / "STATUS.json",
        {"state": state, "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat()},
    )


def remove_existing_output(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise LayoutExportError(f"cannot overwrite unsupported output path: {path}")


def component_dir(path: Path, name: str) -> Path:
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise LayoutExportError(f"{name} directory is missing: {directory}")
    return directory


def required_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise LayoutExportError(f"{name} is missing: {path}")
    return path


def _class_meshes_for_entities(entities: list[dict[str, Any]], start: int) -> dict[str, MeshData]:
    class_meshes = {name: MeshData.empty() for name in CLASS_COLORS}
    for entity in entities[start:]:
        class_meshes[str(entity["class"])].extend(mesh_from_entity(entity))
    return class_meshes


def _extend_class_meshes(target: dict[str, MeshData], source: dict[str, MeshData]) -> None:
    for class_name, mesh in source.items():
        target[class_name].extend(mesh)


def _candidate_color(polygon_id: int | None) -> tuple[int, int, int]:
    if polygon_id is None:
        return FALLBACK_COLOR
    hue = (0.6180339887498949 * (int(polygon_id) + 1)) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.95)
    return int(round(red * 255)), int(round(green * 255)), int(round(blue * 255))


def _assignment_id(entity: Mapping[str, Any], key: str) -> int | None:
    value = entity.get(key)
    return None if value is None else int(value)


def _lighten(color: tuple[int, int, int], amount: float = 0.38) -> tuple[int, int, int]:
    return tuple(int(round(channel + (255 - channel) * amount)) for channel in color)


def _xy_vertices(entity: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(vertex[0]), float(vertex[1])) for vertex in entity.get("vertices", [])]


def _bounds_from_floor_entities(floor_entities: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, float] | None:
    points = [point for entity in floor_entities for point in _xy_vertices(entity)]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def write_triangulation_assignment_png(
    path: Path,
    floor_entities: Sequence[Mapping[str, Any]],
    *,
    max_image_size: int = 2400,
) -> dict[str, int] | None:
    bounds = _bounds_from_floor_entities(floor_entities)
    if bounds is None:
        return None

    from PIL import Image, ImageDraw

    min_x, min_y, max_x, max_y = bounds
    width_m = max(max_x - min_x, 1.0e-6)
    height_m = max(max_y - min_y, 1.0e-6)
    margin = 32
    drawable = max(max_image_size - 2 * margin, 1)
    scale = min(120.0, drawable / max(width_m, height_m))
    image_width = max(1, int(math.ceil(width_m * scale + 2 * margin)))
    image_height = max(1, int(math.ceil(height_m * scale + 2 * margin)))

    def pixel(point: Sequence[float]) -> tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        return margin + (x - min_x) * scale, image_height - margin - (y - min_y) * scale

    image = Image.new("RGB", (image_width, image_height), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    for entity in floor_entities:
        points = _xy_vertices(entity)
        if len(points) != 3:
            continue
        color = _lighten(_candidate_color(_assignment_id(entity, "assigned_ceiling_polygon_id")))
        draw.polygon([pixel(point) for point in points], fill=color)

    edge_width = max(1, int(round(scale * 0.01)))
    for entity in floor_entities:
        points = _xy_vertices(entity)
        if len(points) != 3:
            continue
        pixels = [pixel(point) for point in points]
        draw.line([*pixels, pixels[0]], fill=(24, 24, 24), width=edge_width)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {"width": image_width, "height": image_height, "triangles": len(floor_entities)}


def write_ceiling_assignment_ply(path: Path, ceiling_entities: Sequence[Mapping[str, Any]]) -> dict[str, int] | None:
    mesh = MeshData.empty()
    for entity in ceiling_entities:
        vertices = entity.get("vertices")
        if not vertices:
            continue
        red, green, blue = _candidate_color(_assignment_id(entity, "source_ceiling_polygon_id"))
        mesh.add_triangle(vertices, (red / 255.0, green / 255.0, blue / 255.0))  # type: ignore[arg-type]
    if not mesh.triangles:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(path, mesh)
    return {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)}


def write_debug_outputs(output: Path, by_class: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    debug_dir = output / "debug"
    records: dict[str, Any] = {}

    triangulation_path = debug_dir / "triangulation_assignment.png"
    triangulation_counts = write_triangulation_assignment_png(triangulation_path, by_class.get("floor", []))
    if triangulation_counts is not None:
        records["triangulation_assignment_preview"] = {
            "path": triangulation_path,
            "counts": triangulation_counts,
        }

    ceiling_path = debug_dir / "ceiling_assignment_by_candidate.ply"
    ceiling_counts = write_ceiling_assignment_ply(ceiling_path, by_class.get("ceiling", []))
    if ceiling_counts is not None:
        records["ceiling_assignment_mesh"] = {
            "path": ceiling_path,
            "counts": ceiling_counts,
        }
    return records


def _write_layout_outputs(
    output: Path,
    graph: dict[str, Any],
    entities: list[dict[str, Any]],
    mesh: MeshData,
    class_meshes: dict[str, MeshData],
    room_diagnostics: list[dict[str, Any]],
    ceiling_candidate_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    entities_dir = output / "entities"
    meshes_dir = output / "meshes"
    layout_path = output / "layout.ply"
    layout_obj_path = output / "layout.obj"
    layout_json_path = output / "layout.json"
    diagnostics_path = output / "diagnostics.json"

    write_ply(layout_path, mesh)
    write_obj(layout_obj_path, mesh)
    by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_COLORS}
    for entity in entities:
        by_class.setdefault(str(entity["class"]), []).append(entity)

    entity_records = {}
    for class_name, values in sorted(by_class.items()):
        path = entities_dir / f"{class_name}.json"
        write_json(path, values)
        entity_records[class_name] = record(path)

    mesh_records = {}
    for class_name, class_mesh in sorted(class_meshes.items()):
        if not class_mesh.triangles:
            continue
        path = meshes_dir / f"{class_name}.ply"
        write_ply(path, class_mesh)
        mesh_records[class_name] = record(path)
    debug_records = {}
    for name, value in write_debug_outputs(output, by_class).items():
        path = Path(value["path"])
        debug_records[name] = {"artifact": record(path), "counts": value["counts"]}

    write_json(
        layout_json_path,
        {
            "schema_version": 1,
            "coordinate_system": "Z-up meters",
            "single_floor_assumption": True,
            "levels": graph["levels"],
            "rooms": graph["rooms"],
            "edges": graph.get("edges", []),
            "entities": entities,
        },
    )
    write_json(
        diagnostics_path,
        {
            "rooms": room_diagnostics,
            "entity_counts": {name: len(values) for name, values in by_class.items()},
            "ceiling_candidate_count": ceiling_candidate_count,
            "used_per_triangle_ceiling_planes": ceiling_candidate_count > 0,
            "mesh_vertices": len(mesh.vertices),
            "mesh_triangles": len(mesh.triangles),
        },
    )
    output_records = {
        "layout_json": record(layout_json_path),
        "layout_mesh": record(layout_path),
        "layout_obj": record(layout_obj_path),
        "diagnostics": record(diagnostics_path),
        "entity_files": entity_records,
        "class_meshes": mesh_records,
        "debug": debug_records,
    }
    return by_class, output_records, record(diagnostics_path)


def run_layout_export(
    scene_graph: Path,
    prototype: Path,
    output: Path,
    config: LayoutExportConfig,
    *,
    overwrite: bool = False,
    command: list[str] | None = None,
) -> Path:
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    output = output.expanduser()
    if output.exists():
        if not overwrite:
            raise LayoutExportError(f"layout output already exists: {output}")
        remove_existing_output(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "entities").mkdir()
    (output / "meshes").mkdir()
    write_status(output, "running", "loading graph")
    try:
        from shapely.geometry import shape

        scene_graph_dir = component_dir(scene_graph, "scene_graph")
        prototype_dir = component_dir(prototype, "prototype")
        graph_path = required_file(scene_graph_dir / "graph.json", "scene graph")
        rooms_path = required_file(scene_graph_dir / "rooms.geojson", "rooms")
        graph = read_json(graph_path)
        room_collection = read_json(rooms_path)
        room_geometries = {str(feature["id"]): shape(feature["geometry"]).buffer(0) 
                           for feature in room_collection["features"]}
        level = graph["levels"][0]
        floor_z = float(level.get("floor_height", level.get("elevation_meters", 0.0)))
        ceiling_z = float(level.get("ceiling_elevation_meters", floor_z + config.default_ceiling_height_meters))
        if ceiling_z <= floor_z:
            ceiling_z = floor_z + config.default_ceiling_height_meters
        ceiling_candidates, ceiling_candidates_path = load_ceiling_candidates(scene_graph_dir, level, floor_z, ceiling_z)
        active_room_ids = [str(value) for value in graph.get("active_room_ids", [room["room_id"] for room in graph["rooms"]])]
        active_edges = [edge for edge in graph.get("edges", []) if not edge.get("pruned", False)]

        mesh = MeshData.empty()
        class_meshes = {name: MeshData.empty() for name in CLASS_COLORS}
        entities: list[dict[str, Any]] = []
        room_diagnostics = []
        for room_id in active_room_ids:
            geometry = room_geometries[room_id]
            room_entity_start = len(entities)
            shell_diagnostics = [
                add_room_shell(mesh, entities, room_id, polygon, floor_z, ceiling_z, ceiling_candidates, active_edges, config)
                for polygon in iter_polygons(geometry)
            ]
            _extend_class_meshes(class_meshes, _class_meshes_for_entities(entities, room_entity_start))
            room_diagnostics.append(
                {
                    "room_id": room_id,
                    "area_square_meters": float(geometry.area),
                    "entity_count": len(entities) - room_entity_start,
                    "ceiling_candidate_count": sum(item["ceiling_candidate_count"] for item in shell_diagnostics),
                    "extrusion_triangle_count": sum(item["triangle_count"] for item in shell_diagnostics),
                    "fallback_triangle_count": sum(item["fallback_triangle_count"] for item in shell_diagnostics),
                    "ceiling_discontinuity_wall_count": sum(item["ceiling_discontinuity_wall_count"] for item in shell_diagnostics),
                }
            )

        for edge in active_edges:
            if edge.get("kind") != "door":
                continue
            door_mesh, entity = door_entity(edge, floor_z, config)
            mesh.extend(door_mesh)
            class_meshes["door"].extend(door_mesh)
            entities.append(entity)
        for region in graph.get("stair_regions", []):
            stair_mesh, entity = stair_entity(region, floor_z)
            mesh.extend(stair_mesh)
            class_meshes["stairs"].extend(stair_mesh)
            entities.append(entity)
        for window in graph.get("windows", []):
            window_mesh, entity = window_entity(window)
            mesh.extend(window_mesh)
            class_meshes["window"].extend(window_mesh)
            entities.append(entity)

        by_class, output_records, _ = _write_layout_outputs(
            output,
            graph,
            entities,
            mesh,
            class_meshes,
            room_diagnostics,
            len(ceiling_candidates),
        )
        validation = {
            "single_level_input": len(graph["levels"]) == 1,
            "active_room_count_positive": len(active_room_ids) > 0,
            "entity_count_positive": len(entities) > 0,
            "mesh_nonempty": len(mesh.vertices) > 0 and len(mesh.triangles) > 0,
            "entity_room_references_valid": all(
                entity.get("room_id") in room_geometries
                for entity in entities
                if "room_id" in entity
            ),
            "no_ground_truth_inputs_used": True,
        }
        if not all(validation.values()):
            raise LayoutExportError(f"layout validation failed: {validation}")

        manifest = {
            "schema_version": 1,
            "component": "layout",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command,
            "inputs": {
                "scene_graph": record(graph_path),
                "rooms": record(rooms_path),
                **({"ceiling_candidates": record(ceiling_candidates_path)} if ceiling_candidates_path is not None else {}),
                "prototype_dir": str(prototype_dir),
            },
            "algorithm": {
                "paper_sections": ["4.4", "Appendix D.6"],
                "single_floor_assumption": True,
                "per_triangle_ceiling_planes": len(ceiling_candidates) > 0,
                "configuration": asdict(config),
            },
            "counts": {
                "rooms": len(active_room_ids),
                "entities": len(entities),
                "walls": len(by_class.get("wall", [])),
                "floors": len(by_class.get("floor", [])),
                "ceilings": len(by_class.get("ceiling", [])),
                "doors": len(by_class.get("door", [])),
                "windows": len(by_class.get("window", [])),
                "stairs": len(by_class.get("stairs", [])),
                "layout_vertices": len(mesh.vertices),
                "layout_triangles": len(mesh.triangles),
                "ceiling_candidates": len(ceiling_candidates),
            },
            "outputs": output_records,
            "validation": validation,
            "warnings": [
                *(
                    []
                    if ceiling_candidates
                    else ["This exporter found no ceiling candidates and used one representative ceiling height from scene_graph."]
                ),
                "Doors/openings are cut by omitting lower wall spans on matched room boundary segments.",
            ],
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
        }
        manifest_path = output / "manifest.json"
        write_json(manifest_path, manifest)
        write_status(output, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        write_status(output, "failed", f"{type(error).__name__}: {error}")
        if isinstance(error, LayoutExportError):
            raise
        raise LayoutExportError(f"{type(error).__name__}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export final single-floor Section 4.4 layout entities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-graph", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-ceiling-height-meters", type=float, default=2.7)
    parser.add_argument("--triangulation-mode", choices=("split_shapely", "cdt"), default="split_shapely")
    parser.add_argument("--door-height-meters", type=float, default=2.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = LayoutExportConfig(
        default_ceiling_height_meters=args.default_ceiling_height_meters,
        triangulation_mode=args.triangulation_mode,
        door_height_meters=args.door_height_meters,
    )
    print(run_layout_export(args.scene_graph, args.prototype, args.output, config, overwrite=args.overwrite, command=os.sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
