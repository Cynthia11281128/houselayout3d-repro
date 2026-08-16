"""Ceiling assignment and 3D extrusion for final layout export."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from shapely.geometry import LineString, Point, shape

from .triangulation import LayoutExportError, coords2d, edge_key, point3, triangle_centroid, triangle_edges, triangulate_room_polygon


@dataclass(frozen=True)
class LayoutExportConfig:
    default_ceiling_height_meters: float = 2.7
    triangulation_mode: str = "split_shapely"
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


def write_ply(path: Path, mesh: MeshData) -> None:
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


def write_obj(path: Path, mesh: MeshData) -> None:
    with path.open("w", encoding="ascii") as handle:
        for vertex in mesh.vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for triangle in mesh.triangles:
            handle.write(f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}\n")


def mesh_from_entity(entity: dict[str, object]) -> MeshData:
    mesh = MeshData.empty()
    vertices = entity.get("vertices")
    class_name = str(entity["class"])
    if class_name in {"floor", "ceiling"} and vertices:
        mesh.add_triangle(vertices, CLASS_COLORS[class_name])  # type: ignore[arg-type]
    elif class_name == "wall" and vertices:
        mesh.add_quad(vertices, CLASS_COLORS["wall"])  # type: ignore[arg-type]
    return mesh


@dataclass(frozen=True)
class CeilingCandidate:
    polygon_id: int | None
    plane_eq: tuple[float, float, float, float]
    geometry: Any | None
    mean_elevation_meters: float
    area_square_meters: float
    is_fallback: bool = False


@dataclass(frozen=True)
class AssignedTriangle:
    index: int
    coords: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    floor_vertices: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    ceiling_vertices: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    ceiling: CeilingCandidate
    polygon: Any


def plane_z(plane: Sequence[float], x: float, y: float) -> float:
    a, b, c, d = [float(value) for value in plane]
    if abs(c) <= 1.0e-8:
        raise LayoutExportError(f"cannot evaluate near-vertical ceiling plane: {plane}")
    return float(-(a * x + b * y + d) / c)


def fallback_ceiling(ceiling_z: float) -> CeilingCandidate:
    return CeilingCandidate(
        polygon_id=None,
        plane_eq=(0.0, 0.0, 1.0, -float(ceiling_z)),
        geometry=None,
        mean_elevation_meters=float(ceiling_z),
        area_square_meters=0.0,
        is_fallback=True,
    )


def load_ceiling_candidates(
    scene_graph_dir: Path,
    level: Mapping[str, Any],
    floor_z: float,
    ceiling_z: float,
) -> tuple[list[CeilingCandidate], Path | None]:
    ref = level.get("ceiling_candidates_ref", "ceiling_candidates.json")
    path = scene_graph_dir / str(ref)
    if not path.is_file():
        return [], None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise LayoutExportError(f"ceiling candidates must be a list: {path}")
    candidates = []
    for record in raw:
        if not isinstance(record, Mapping):
            continue
        plane = tuple(float(value) for value in record["plane_eq"])
        if len(plane) != 4 or abs(plane[2]) <= 1.0e-8:
            continue
        geometry = shape(record["geometry"]).buffer(0)
        if geometry.is_empty or geometry.area <= 1.0e-10:
            continue
        mean_elevation = float(record.get("mean_elevation_meters", ceiling_z))
        if mean_elevation <= floor_z + 1.0e-6:
            continue
        candidates.append(
            CeilingCandidate(
                polygon_id=int(record["polygon_id"]),
                plane_eq=plane,
                geometry=geometry,
                mean_elevation_meters=mean_elevation,
                area_square_meters=float(record.get("area_square_meters", geometry.area)),
            )
        )
    return candidates, path


def room_ceiling_candidates(room_polygon: Any, candidates: Sequence[CeilingCandidate]) -> list[CeilingCandidate]:
    scored = []
    for candidate in candidates:
        if candidate.geometry is None:
            continue
        overlap = room_polygon.intersection(candidate.geometry)
        if not overlap.is_empty and overlap.area > 1.0e-8:
            scored.append((float(overlap.area), candidate))
    return [candidate for _, candidate in sorted(scored, key=lambda item: item[0], reverse=True)[:30]]


def assign_ceiling(
    triangle: Any,
    candidates: Sequence[CeilingCandidate],
    fallback: CeilingCandidate,
    floor_z: float,
) -> CeilingCandidate:
    point = triangle.representative_point()
    center = Point(point.x, point.y)
    matches = []
    for candidate in candidates:
        if candidate.geometry is None:
            continue
        if not candidate.geometry.buffer(1.0e-8).covers(center):
            continue
        z = plane_z(candidate.plane_eq, float(point.x), float(point.y))
        if z > floor_z + 1.0e-4:
            matches.append((z, candidate))
    if not matches:
        return fallback
    return min(matches, key=lambda item: item[0])[1]


def propagate_reachable_ceilings(
    triangles: Sequence[AssignedTriangle],
    fallback: CeilingCandidate,
    floor_z: float,
) -> list[AssignedTriangle]:
    if not any(triangle.ceiling.is_fallback for triangle in triangles):
        return list(triangles)
    edge_to_triangles: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = {}
    for triangle in triangles:
        for first, second in triangle_edges(triangle.coords):
            edge_to_triangles.setdefault(edge_key(first, second), []).append(triangle.index)
    neighbors: dict[int, set[int]] = {triangle.index: set() for triangle in triangles}
    for values in edge_to_triangles.values():
        if len(values) != 2:
            continue
        first, second = values
        neighbors[first].add(second)
        neighbors[second].add(first)

    assigned = list(triangles)
    changed = True
    while changed:
        changed = False
        for triangle in list(assigned):
            if not triangle.ceiling.is_fallback:
                continue
            reachable = [
                assigned[index].ceiling
                for index in neighbors[triangle.index]
                if not assigned[index].ceiling.is_fallback
            ]
            if not reachable:
                continue
            x, y = triangle_centroid(triangle.coords)
            valid = [
                (candidate, plane_z(candidate.plane_eq, x, y))
                for candidate in reachable
                if plane_z(candidate.plane_eq, x, y) > floor_z + 1.0e-4
            ]
            if not valid:
                continue
            ceiling = min(valid, key=lambda item: item[1])[0]
            ceiling_vertices = tuple((float(px), float(py), plane_z(ceiling.plane_eq, float(px), float(py))) for px, py in triangle.coords)
            assigned[triangle.index] = replace(triangle, ceiling=ceiling, ceiling_vertices=ceiling_vertices)
            changed = True
    return [
        triangle if not triangle.ceiling.is_fallback else replace(triangle, ceiling=fallback)
        for triangle in assigned
    ]


def segment_midpoint(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    return ((float(first[0]) + float(second[0])) * 0.5, (float(first[1]) + float(second[1])) * 0.5)


def matching_opening(
    first: Sequence[float],
    second: Sequence[float],
    room_id: str,
    edges: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> Mapping[str, Any] | None:
    midpoint = Point(*segment_midpoint(first, second))
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


def add_wall_span(
    mesh: MeshData,
    entities: list[dict[str, Any]],
    room_id: str,
    entity_id: str,
    first: Sequence[float],
    second: Sequence[float],
    bottom_z: float,
    top_first_z: float,
    top_second_z: float,
    source: dict[str, Any] | None = None,
) -> None:
    if max(top_first_z, top_second_z) <= bottom_z + 1.0e-6:
        return
    quad = [
        point3(first, bottom_z),
        point3(second, bottom_z),
        point3(second, top_second_z),
        point3(first, top_first_z),
    ]
    mesh.add_quad(quad, CLASS_COLORS["wall"])
    entity = {"entity_id": entity_id, "class": "wall", "room_id": room_id, "vertices": quad}
    if source:
        entity.update(source)
    entities.append(entity)


def add_boundary_wall(
    mesh: MeshData,
    entities: list[dict[str, Any]],
    room_id: str,
    edge_index: int,
    first: Sequence[float],
    second: Sequence[float],
    ceiling: CeilingCandidate,
    floor_z: float,
    edges: Sequence[Mapping[str, Any]],
    config: LayoutExportConfig,
) -> None:
    top_first_z = plane_z(ceiling.plane_eq, float(first[0]), float(first[1]))
    top_second_z = plane_z(ceiling.plane_eq, float(second[0]), float(second[1]))
    opening = matching_opening(first, second, room_id, edges, max(0.10, config.door_thickness_meters * 2.0))
    source = {"assigned_ceiling_polygon_id": ceiling.polygon_id}
    if opening is None:
        add_wall_span(
            mesh,
            entities,
            room_id,
            f"{room_id}:wall:{edge_index}",
            first,
            second,
            floor_z,
            top_first_z,
            top_second_z,
            source,
        )
        return
    opening_height = config.door_height_meters if opening["kind"] == "door" else config.opening_height_meters
    lintel_z = min(floor_z + opening_height, top_first_z, top_second_z)
    add_wall_span(
        mesh,
        entities,
        room_id,
        f"{room_id}:wall:{edge_index}:above_{opening['edge_id']}",
        first,
        second,
        lintel_z,
        top_first_z,
        top_second_z,
        {**source, "source_edge_id": opening["edge_id"]},
    )


def add_room_shell(
    mesh: MeshData,
    entities: list[dict[str, Any]],
    room_id: str,
    polygon: Any,
    floor_z: float,
    fallback_ceiling_z: float,
    ceiling_candidates: Sequence[CeilingCandidate],
    edges: Sequence[Mapping[str, Any]],
    config: LayoutExportConfig,
) -> dict[str, Any]:
    fallback = fallback_ceiling(fallback_ceiling_z)
    room_candidates = room_ceiling_candidates(polygon, ceiling_candidates)
    assigned: list[AssignedTriangle] = []
    for triangle in triangulate_room_polygon(polygon, room_candidates, config.triangulation_mode):
        coords = coords2d(triangle)
        if len(coords) != 3:
            continue
        ceiling = assign_ceiling(triangle, room_candidates, fallback, floor_z)
        floor_vertices = tuple((float(x), float(y), float(floor_z)) for x, y in coords)
        ceiling_vertices = tuple((float(x), float(y), plane_z(ceiling.plane_eq, float(x), float(y))) for x, y in coords)
        assigned.append(
            AssignedTriangle(
                index=len(assigned),
                coords=tuple(coords),  # type: ignore[arg-type]
                floor_vertices=floor_vertices,  # type: ignore[arg-type]
                ceiling_vertices=ceiling_vertices,  # type: ignore[arg-type]
                ceiling=ceiling,
                polygon=triangle,
            )
        )
    assigned = propagate_reachable_ceilings(assigned, fallback, floor_z)
    fallback_count = sum(triangle.ceiling.is_fallback for triangle in assigned)
    edge_records: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[AssignedTriangle, tuple[float, float], tuple[float, float]]]] = {}
    for triangle in assigned:
        floor = [list(vertex) for vertex in triangle.floor_vertices]
        ceiling = [list(vertex) for vertex in triangle.ceiling_vertices]
        mesh.add_triangle([floor[0], floor[2], floor[1]], CLASS_COLORS["floor"])
        mesh.add_triangle(ceiling, CLASS_COLORS["ceiling"])
        entities.append(
            {
                "entity_id": f"{room_id}:floor:{triangle.index}",
                "class": "floor",
                "room_id": room_id,
                "vertices": [floor[0], floor[2], floor[1]],
                "assigned_ceiling_polygon_id": triangle.ceiling.polygon_id,
            }
        )
        entities.append(
            {
                "entity_id": f"{room_id}:ceiling:{triangle.index}",
                "class": "ceiling",
                "room_id": room_id,
                "vertices": ceiling,
                "source_ceiling_polygon_id": triangle.ceiling.polygon_id,
                "ceiling_fallback": triangle.ceiling.is_fallback,
            }
        )
        for first, second in triangle_edges(triangle.coords):
            edge_records.setdefault(edge_key(first, second), []).append((triangle, first, second))
    wall_index = 0
    discontinuity_count = 0
    boundary = polygon.boundary
    for records in edge_records.values():
        if len(records) == 1:
            triangle, first, second = records[0]
            segment = LineString([first, second])
            if boundary.distance(segment) <= 1.0e-6:
                add_boundary_wall(mesh, entities, room_id, wall_index, first, second, triangle.ceiling, floor_z, edges, config)
                wall_index += 1
            continue
        if len(records) != 2:
            continue
        first_record, second_record = records
        first_triangle, first_a, first_b = first_record
        second_triangle, _, _ = second_record
        first_top_a = plane_z(first_triangle.ceiling.plane_eq, first_a[0], first_a[1])
        first_top_b = plane_z(first_triangle.ceiling.plane_eq, first_b[0], first_b[1])
        second_top_a = plane_z(second_triangle.ceiling.plane_eq, first_a[0], first_a[1])
        second_top_b = plane_z(second_triangle.ceiling.plane_eq, first_b[0], first_b[1])
        if max(abs(first_top_a - second_top_a), abs(first_top_b - second_top_b)) <= 1.0e-4:
            continue
        quad = [
            point3(first_a, first_top_a),
            point3(first_b, first_top_b),
            point3(first_b, second_top_b),
            point3(first_a, second_top_a),
        ]
        mesh.add_quad(quad, CLASS_COLORS["wall"])
        entities.append(
            {
                "entity_id": f"{room_id}:wall:{wall_index}:ceiling_discontinuity",
                "class": "wall",
                "room_id": room_id,
                "vertices": quad,
                "source": "ceiling_discontinuity",
                "ceiling_polygon_ids": [first_triangle.ceiling.polygon_id, second_triangle.ceiling.polygon_id],
            }
        )
        wall_index += 1
        discontinuity_count += 1
    return {
        "ceiling_candidate_count": len(room_candidates),
        "triangle_count": len(assigned),
        "fallback_triangle_count": fallback_count,
        "ceiling_discontinuity_wall_count": discontinuity_count,
    }


def door_entity(edge: Mapping[str, Any], floor_z: float, config: LayoutExportConfig) -> tuple[MeshData, dict[str, Any]]:
    import numpy as np

    line = np.asarray(edge["line_xy"], dtype=float)
    direction = line[1] - line[0]
    length = max(float(np.linalg.norm(direction)), 1.0e-12)
    direction /= length
    normal = np.asarray([-direction[1], direction[0]])
    corners = np.asarray(
        [
            line[0] - 0.5 * config.door_thickness_meters * normal,
            line[1] - 0.5 * config.door_thickness_meters * normal,
            line[1] + 0.5 * config.door_thickness_meters * normal,
            line[0] + 0.5 * config.door_thickness_meters * normal,
        ]
    )
    bottom = [[float(x), float(y), floor_z] for x, y in corners]
    top = [[float(x), float(y), floor_z + config.door_height_meters] for x, y in corners]
    mesh = MeshData.empty()
    faces = []
    for index in range(4):
        quad = [bottom[index], bottom[(index + 1) % 4], top[(index + 1) % 4], top[index]]
        mesh.add_quad(quad, CLASS_COLORS["door"])
        faces.append(quad)
    return mesh, {
        "entity_id": f"door:{edge['edge_id']}",
        "class": "door",
        "edge_id": edge["edge_id"],
        "room_ids": edge["room_ids"],
        "width_meters": float(edge["width_meters"]),
        "frame_faces": faces,
    }


def window_entity(window: Mapping[str, Any]) -> tuple[MeshData, dict[str, Any]]:
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


def stair_entity(region: Mapping[str, Any], floor_z: float) -> tuple[MeshData, dict[str, Any]]:
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
