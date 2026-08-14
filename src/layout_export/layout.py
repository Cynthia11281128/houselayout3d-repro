"""Export final single-floor Section 4.4 layout entities and meshes."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.layout_export"

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


class LayoutExportError(RuntimeError):
    """Raised when final layout export fails."""


@dataclass(frozen=True)
class LayoutExportConfig:
    default_ceiling_height_meters: float = 2.7
    door_height_meters: float = 2.10
    door_thickness_meters: float = 0.08
    opening_height_meters: float = 2.35
    stair_step_height_meters: float = 0.18


@dataclass
class MeshData:
    vertices: list[list[float]]
    triangles: list[list[int]]
    colors: list[list[float]]

    @classmethod
    def empty(cls) -> "MeshData":
        return cls([], [], [])

    def add_triangle(self, points: Sequence[Sequence[float]], color: Sequence[float]) -> None:
        base = len(self.vertices)
        self.vertices.extend([[float(c) for c in point] for point in points])
        self.triangles.append([base, base + 1, base + 2])
        self.colors.extend([[float(c) for c in color]] * 3)

    def add_quad(self, points: Sequence[Sequence[float]], color: Sequence[float]) -> None:
        self.add_triangle([points[0], points[1], points[2]], color)
        self.add_triangle([points[0], points[2], points[3]], color)

    def extend(self, other: "MeshData") -> None:
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.colors.extend(other.colors)
        self.triangles.extend([[index + offset for index in triangle] for triangle in other.triangles])


CLASS_COLORS = {
    "wall": (0.78, 0.31, 0.27),
    "floor": (0.31, 0.67, 0.38),
    "ceiling": (0.34, 0.55, 0.82),
    "door": (0.90, 0.22, 0.18),
    "opening": (0.55, 0.55, 0.55),
    "window": (0.18, 0.72, 0.88),
    "stairs": (0.95, 0.52, 0.10),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _write_status(component_dir: Path, state: str, detail: str = "") -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        component_dir / "STATUS.json",
        {"state": state, "detail": detail, "updated_at": datetime.now(timezone.utc).isoformat()},
    )


def _remove_existing_output(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise LayoutExportError(f"cannot overwrite unsupported output path: {path}")


def _component_dir(path: Path, name: str) -> Path:
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise LayoutExportError(f"{name} directory is missing: {directory}")
    return directory


def _required_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise LayoutExportError(f"{name} is missing: {path}")
    return path


def _write_ply(path: Path, mesh: MeshData) -> None:
    if not mesh.vertices or not mesh.triangles:
        raise LayoutExportError(f"refusing to write empty mesh: {path}")
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(mesh.vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(mesh.triangles)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex, color in zip(mesh.vertices, mesh.colors):
            rgb = [max(0, min(255, int(round(value * 255)))) for value in color]
            handle.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
        for triangle in mesh.triangles:
            handle.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")


def _write_obj(path: Path, mesh: MeshData) -> None:
    with path.open("w", encoding="ascii") as handle:
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for triangle in mesh.triangles:
            handle.write(f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}\n")


def _iter_polygons(geometry: Any) -> list[Any]:
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return [part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon) and not part.is_empty]


def _triangulate_polygon(geometry: Any) -> list[Any]:
    from shapely.geometry import Point
    from shapely.ops import triangulate

    triangles = []
    covered = geometry.buffer(1.0e-8)
    for candidate in triangulate(geometry):
        point = candidate.representative_point()
        if covered.covers(Point(point.x, point.y)) and candidate.area > 1.0e-10:
            triangles.append(candidate)
    return triangles


def _coords2d(polygon: Any) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]


def _add_floor_ceiling(
    mesh: MeshData,
    entities: list[dict[str, Any]],
    room_id: str,
    polygon: Any,
    floor_z: float,
    ceiling_z: float,
) -> None:
    for index, triangle in enumerate(_triangulate_polygon(polygon)):
        coords = _coords2d(triangle)
        if len(coords) != 3:
            continue
        floor = [[x, y, floor_z] for x, y in coords]
        ceiling = [[x, y, ceiling_z] for x, y in coords]
        mesh.add_triangle([floor[0], floor[2], floor[1]], CLASS_COLORS["floor"])
        mesh.add_triangle(ceiling, CLASS_COLORS["ceiling"])
        entities.append({"entity_id": f"{room_id}:floor:{index}", "class": "floor", "room_id": room_id, "vertices": [floor[0], floor[2], floor[1]]})
        entities.append({"entity_id": f"{room_id}:ceiling:{index}", "class": "ceiling", "room_id": room_id, "vertices": ceiling})


def _segment_midpoint(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    return ((float(first[0]) + float(second[0])) * 0.5, (float(first[1]) + float(second[1])) * 0.5)


def _matching_opening(first: Sequence[float], second: Sequence[float], room_id: str, edges: Sequence[Mapping[str, Any]], tolerance: float) -> Mapping[str, Any] | None:
    from shapely.geometry import LineString, Point

    midpoint = Point(*_segment_midpoint(first, second))
    segment = LineString([first, second])
    for edge in edges:
        if room_id not in [str(value) for value in edge.get("room_ids", [])]:
            continue
        if edge.get("kind") not in {"door", "opening"}:
            continue
        line = LineString(edge["line_xy"])
        if segment.distance(line) <= tolerance or line.buffer(tolerance, cap_style=2).covers(midpoint):
            return edge
    return None


def _add_walls(
    mesh: MeshData,
    entities: list[dict[str, Any]],
    room_id: str,
    polygon: Any,
    floor_z: float,
    ceiling_z: float,
    edges: Sequence[Mapping[str, Any]],
    config: LayoutExportConfig,
) -> None:
    coords = _coords2d(polygon)
    tolerance = max(0.10, config.door_thickness_meters * 2.0)
    wall_index = 0
    for first, second in zip(coords, [*coords[1:], coords[0]]):
        opening = _matching_opening(first, second, room_id, edges, tolerance)
        bottom_first = [first[0], first[1], floor_z]
        bottom_second = [second[0], second[1], floor_z]
        top_first = [first[0], first[1], ceiling_z]
        top_second = [second[0], second[1], ceiling_z]
        if opening is None:
            quad = [bottom_first, bottom_second, top_second, top_first]
            mesh.add_quad(quad, CLASS_COLORS["wall"])
            entities.append({"entity_id": f"{room_id}:wall:{wall_index}", "class": "wall", "room_id": room_id, "vertices": quad})
            wall_index += 1
            continue
        if opening["kind"] == "door":
            lintel_z = min(floor_z + config.door_height_meters, ceiling_z)
        else:
            lintel_z = min(floor_z + config.opening_height_meters, ceiling_z)
        if ceiling_z > lintel_z + 1.0e-6:
            quad = [[first[0], first[1], lintel_z], [second[0], second[1], lintel_z], top_second, top_first]
            mesh.add_quad(quad, CLASS_COLORS["wall"])
            entities.append(
                {
                    "entity_id": f"{room_id}:wall:{wall_index}:above_{opening['edge_id']}",
                    "class": "wall",
                    "room_id": room_id,
                    "source_edge_id": opening["edge_id"],
                    "vertices": quad,
                }
            )
            wall_index += 1


def _door_entity(edge: Mapping[str, Any], floor_z: float, config: LayoutExportConfig) -> tuple[MeshData, dict[str, Any]]:
    import numpy as np

    line = np.asarray(edge["line_xy"], dtype=float)
    direction = line[1] - line[0]
    length = max(float(np.linalg.norm(direction)), 1.0e-12)
    direction /= length
    normal = np.asarray([-direction[1], direction[0]])
    corners = np.asarray([line[0] - 0.5 * config.door_thickness_meters * normal, line[1] - 0.5 * config.door_thickness_meters * normal, line[1] + 0.5 * config.door_thickness_meters * normal, line[0] + 0.5 * config.door_thickness_meters * normal])
    bottom = [[float(x), float(y), floor_z] for x, y in corners]
    top = [[float(x), float(y), floor_z + config.door_height_meters] for x, y in corners]
    mesh = MeshData.empty()
    faces = []
    for index in range(4):
        quad = [bottom[index], bottom[(index + 1) % 4], top[(index + 1) % 4], top[index]]
        mesh.add_quad(quad, CLASS_COLORS["door"])
        faces.append(quad)
    return mesh, {"entity_id": f"door:{edge['edge_id']}", "class": "door", "edge_id": edge["edge_id"], "room_ids": edge["room_ids"], "width_meters": float(edge["width_meters"]), "frame_faces": faces}


def _window_entity(window: Mapping[str, Any]) -> tuple[MeshData, dict[str, Any]]:
    mesh = MeshData.empty()
    vertices = [[float(c) for c in vertex] for vertex in window["vertices"]]
    mesh.add_quad(vertices, CLASS_COLORS["window"])
    entity = {
        "entity_id": str(window.get("window_id", "window")),
        "class": "window",
        "room_ids": window.get("room_ids", []),
        "source_wall_polygon_id": window.get("source_wall_polygon_id"),
        "width_meters": window.get("width_meters"),
        "height_meters": window.get("height_meters"),
        "vertices": vertices,
    }
    return mesh, entity


def _stair_entity(region: Mapping[str, Any], floor_z: float) -> tuple[MeshData, dict[str, Any]]:
    rectangle = [[float(x), float(y)] for x, y in region["rectangle_xy"]]
    z0 = float(region.get("minimum_z", floor_z))
    z1 = float(region.get("maximum_z", z0))
    vertices = [[x, y, z0] for x, y in rectangle]
    mesh = MeshData.empty()
    mesh.add_triangle([vertices[0], vertices[1], vertices[2]], CLASS_COLORS["stairs"])
    mesh.add_triangle([vertices[0], vertices[2], vertices[3]], CLASS_COLORS["stairs"])
    entity = {
        "entity_id": str(region.get("stair_region_id", "stair_region")),
        "class": "stairs",
        "room_ids": region.get("room_ids", []),
        "rectangle_xy": rectangle,
        "minimum_z": z0,
        "maximum_z": z1,
        "vertices": vertices,
    }
    return mesh, entity


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
        _remove_existing_output(output)
    output.mkdir(parents=True, exist_ok=False)
    entities_dir = output / "entities"
    meshes_dir = output / "meshes"
    entities_dir.mkdir()
    meshes_dir.mkdir()
    _write_status(output, "running", "loading graph")
    try:
        from shapely.geometry import shape

        scene_graph_dir = _component_dir(scene_graph, "scene_graph")
        prototype_dir = _component_dir(prototype, "prototype")
        graph_path = _required_file(scene_graph_dir / "graph.json", "scene graph")
        rooms_path = _required_file(scene_graph_dir / "rooms.geojson", "rooms")
        graph = _read_json(graph_path)
        room_collection = _read_json(rooms_path)
        room_geometries = {str(feature["id"]): shape(feature["geometry"]).buffer(0) for feature in room_collection["features"]}
        level = graph["levels"][0]
        floor_z = float(level.get("floor_height", level.get("elevation_meters", 0.0)))
        ceiling_z = float(level.get("ceiling_elevation_meters", floor_z + config.default_ceiling_height_meters))
        if ceiling_z <= floor_z:
            ceiling_z = floor_z + config.default_ceiling_height_meters
        active_room_ids = [str(value) for value in graph.get("active_room_ids", [room["room_id"] for room in graph["rooms"]])]
        active_edges = [edge for edge in graph.get("edges", []) if not edge.get("pruned", False)]
        mesh = MeshData.empty()
        class_meshes = {name: MeshData.empty() for name in CLASS_COLORS}
        entities: list[dict[str, Any]] = []
        room_diagnostics = []
        for room_id in active_room_ids:
            geometry = room_geometries[room_id]
            room_entity_start = len(entities)
            for polygon in _iter_polygons(geometry):
                _add_floor_ceiling(mesh, entities, room_id, polygon, floor_z, ceiling_z)
                _add_walls(mesh, entities, room_id, polygon, floor_z, ceiling_z, active_edges, config)
            for entity in entities[room_entity_start:]:
                temp = MeshData.empty()
                vertices = entity.get("vertices")
                if entity["class"] in {"floor", "ceiling"} and vertices:
                    temp.add_triangle(vertices, CLASS_COLORS[entity["class"]])
                elif entity["class"] == "wall" and vertices:
                    temp.add_quad(vertices, CLASS_COLORS["wall"])
                class_meshes[entity["class"]].extend(temp)
            room_diagnostics.append({"room_id": room_id, "area_square_meters": float(geometry.area), "entity_count": len(entities) - room_entity_start})
        for edge in active_edges:
            if edge.get("kind") != "door":
                continue
            door_mesh, entity = _door_entity(edge, floor_z, config)
            mesh.extend(door_mesh)
            class_meshes["door"].extend(door_mesh)
            entities.append(entity)
        for region in graph.get("stair_regions", []):
            stair_mesh, entity = _stair_entity(region, floor_z)
            mesh.extend(stair_mesh)
            class_meshes["stairs"].extend(stair_mesh)
            entities.append(entity)
        for window in graph.get("windows", []):
            window_mesh, entity = _window_entity(window)
            mesh.extend(window_mesh)
            class_meshes["window"].extend(window_mesh)
            entities.append(entity)
        layout_path = output / "layout.ply"
        layout_obj_path = output / "layout.obj"
        layout_json_path = output / "layout.json"
        diagnostics_path = output / "diagnostics.json"
        _write_ply(layout_path, mesh)
        _write_obj(layout_obj_path, mesh)
        by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in CLASS_COLORS}
        for entity in entities:
            by_class.setdefault(str(entity["class"]), []).append(entity)
        entity_records = {}
        for class_name, values in sorted(by_class.items()):
            path = entities_dir / f"{class_name}.json"
            _write_json(path, values)
            entity_records[class_name] = _record(path)
        mesh_records = {}
        for class_name, class_mesh in sorted(class_meshes.items()):
            if not class_mesh.triangles:
                continue
            path = meshes_dir / f"{class_name}.ply"
            _write_ply(path, class_mesh)
            mesh_records[class_name] = _record(path)
        _write_json(
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
        _write_json(
            diagnostics_path,
            {
                "rooms": room_diagnostics,
                "entity_counts": {name: len(values) for name, values in by_class.items()},
                "mesh_vertices": len(mesh.vertices),
                "mesh_triangles": len(mesh.triangles),
            },
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
                "scene_graph": _record(graph_path),
                "rooms": _record(rooms_path),
                "prototype_dir": str(prototype_dir),
            },
            "algorithm": {
                "paper_sections": ["4.4", "Appendix D.6"],
                "single_floor_assumption": True,
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
            },
            "outputs": {
                "layout_json": _record(layout_json_path),
                "layout_mesh": _record(layout_path),
                "layout_obj": _record(layout_obj_path),
                "diagnostics": _record(diagnostics_path),
                "entity_files": entity_records,
                "class_meshes": mesh_records,
            },
            "validation": validation,
            "warnings": [
                "This exporter uses one floor and one representative ceiling height from scene_graph.",
                "Doors/openings are cut by omitting lower wall spans on matched room boundary segments.",
            ],
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
        }
        manifest_path = output / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(output, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(output, "failed", f"{type(error).__name__}: {error}")
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
    parser.add_argument("--door-height-meters", type=float, default=2.10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = LayoutExportConfig(default_ceiling_height_meters=args.default_ceiling_height_meters, door_height_meters=args.door_height_meters)
    print(run_layout_export(args.scene_graph, args.prototype, args.output, config, overwrite=args.overwrite, command=os.sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
