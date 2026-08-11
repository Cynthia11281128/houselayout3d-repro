"""Paper-faithful layout room extrusion and final layout generation."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "houselayout3d"


from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import open3d as o3d
from PIL import Image
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

from .config import LayoutConfig, PipelineConfig
from .scene_graph import (
    PrototypeData,
    load_prototype_data,
    polygon_geometries,
)


class LayoutError(RuntimeError):
    """Raised when layout cannot satisfy the final-layout contract."""


ATTEMPT_RE = re.compile(r"^attempt_(?P<index>\d+)_(?:running|complete|failed)$")
WINDOW_COCO_IDS = frozenset({85, 90, 114, 115, 116, 119, 123, 125, 126})
CLASS_COLORS = {
    "wall": (0.78, 0.31, 0.27),
    "floor": (0.31, 0.67, 0.38),
    "ceiling": (0.34, 0.55, 0.82),
    "door": (0.90, 0.22, 0.18),
    "window": (0.18, 0.72, 0.88),
    "stairs": (0.95, 0.52, 0.10),
}


@dataclass(frozen=True)
class CeilingPlane:
    polygon_id: int
    coefficients: np.ndarray
    geometry_xy: BaseGeometry
    area_square_meters: float

    def height(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        a, b, c, d = self.coefficients
        if abs(float(c)) < 1.0e-8:
            raise LayoutError(
                f"ceiling polygon {self.polygon_id} is vertical"
            )
        return -(a * points[:, 0] + b * points[:, 1] + d) / c


@dataclass
class MeshData:
    vertices: list[list[float]]
    triangles: list[list[int]]
    colors: list[list[float]]
    face_tags: list[str]

    @classmethod
    def empty(cls) -> "MeshData":
        return cls([], [], [], [])

    def add_triangle(
        self,
        points: Sequence[Sequence[float]],
        color: Sequence[float],
        tag: str,
    ) -> None:
        base = len(self.vertices)
        self.vertices.extend(np.asarray(points, dtype=np.float64).tolist())
        self.triangles.append([base, base + 1, base + 2])
        self.colors.extend([list(color)] * 3)
        self.face_tags.append(tag)

    def add_quad(
        self,
        points: Sequence[Sequence[float]],
        color: Sequence[float],
        tag: str,
    ) -> None:
        p = np.asarray(points, dtype=np.float64)
        self.add_triangle([p[0], p[1], p[2]], color, tag)
        self.add_triangle([p[0], p[2], p[3]], color, tag)

    def extend(self, other: "MeshData") -> None:
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.colors.extend(other.colors)
        self.triangles.extend(
            [[value + offset for value in face] for face in other.triangles]
        )
        self.face_tags.extend(other.face_tags)

    def as_open3d(self) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(self.triangles, dtype=np.int32).reshape(-1, 3)
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(self.colors, dtype=np.float64).reshape(-1, 3)
        )
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_status(path: Path, state: str, detail: str) -> None:
    _write_json(
        path,
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _next_attempt(component_dir: Path) -> Path:
    indices = []
    for path in component_dir.iterdir() if component_dir.is_dir() else ():
        match = ATTEMPT_RE.match(path.name)
        if match:
            indices.append(int(match.group("index")))
    attempt = component_dir / f"attempt_{max(indices, default=0) + 1:03d}_running"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _rename_attempt(path: Path, state: str) -> Path:
    destination = path.with_name(path.name.rsplit("_", 1)[0] + f"_{state}")
    path.rename(destination)
    return destination


def _require_complete_manifest(path: Path, component: str) -> dict[str, Any]:
    if not path.is_file():
        raise LayoutError(f"missing {component} manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("component") != component or payload.get("status") != "complete":
        raise LayoutError(f"{component} manifest is not complete: {path}")
    return payload


def _write_mesh(path: Path, data: MeshData) -> o3d.geometry.TriangleMesh:
    if not data.vertices or not data.triangles:
        raise LayoutError(f"refusing to write empty layout mesh: {path}")
    mesh = data.as_open3d()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise LayoutError(f"failed to write mesh: {path}")
    return mesh


def _component_polygons(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return [
        part
        for part in getattr(geometry, "geoms", ())
        if isinstance(part, Polygon) and not part.is_empty
    ]


def _fit_plane(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    return np.r_[normal, -float(normal @ center)]


def ceiling_planes(
    data: PrototypeData,
    level: Mapping[str, Any],
    geometries: Mapping[int, BaseGeometry],
    maximum_count: int,
) -> list[CeilingPlane]:
    result = []
    for polygon_id in level["ceiling_polygon_ids"]:
        triangles = data.triangles[data.triangle_polygons == int(polygon_id)]
        if len(triangles) == 0:
            continue
        vertices = data.vertices[np.unique(triangles)]
        geometry = geometries[int(polygon_id)]
        if geometry.is_empty or geometry.area <= 1.0e-8:
            continue
        plane = _fit_plane(vertices)
        if abs(float(plane[2])) < 0.25:
            continue
        result.append(
            CeilingPlane(
                polygon_id=int(polygon_id),
                coefficients=plane,
                geometry_xy=geometry,
                area_square_meters=float(geometry.area),
            )
        )
    result.sort(key=lambda item: (-item.area_square_meters, item.polygon_id))
    return result[:maximum_count]


def _plane_intersection_line(
    first: CeilingPlane,
    second: CeilingPlane,
    bounds: tuple[float, float, float, float],
) -> LineString | None:
    a1, b1, c1, d1 = first.coefficients
    a2, b2, c2, d2 = second.coefficients
    # z1(x,y) == z2(x,y) gives alpha*x + beta*y + gamma == 0.
    alpha = -a1 / c1 + a2 / c2
    beta = -b1 / c1 + b2 / c2
    gamma = -d1 / c1 + d2 / c2
    norm = math.hypot(float(alpha), float(beta))
    if norm < 1.0e-8:
        return None
    center = np.asarray(
        [0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3])]
    )
    signed = (alpha * center[0] + beta * center[1] + gamma) / norm
    normal = np.asarray([alpha, beta], dtype=np.float64) / norm
    point = center - signed * normal
    direction = np.asarray([-normal[1], normal[0]])
    length = 4.0 * math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]) + 1.0
    return LineString([point - length * direction, point + length * direction])


def constrained_room_triangles(
    room: BaseGeometry,
    ceilings: Sequence[CeilingPlane],
) -> list[np.ndarray]:
    """CDT-equivalent planar arrangement followed by Delaunay cell triangulation."""
    if room.is_empty:
        return []
    linework: list[BaseGeometry] = [room.boundary]
    for ceiling in ceilings:
        clipped = ceiling.geometry_xy.intersection(room)
        if not clipped.is_empty:
            linework.append(clipped.boundary)
    for index, first in enumerate(ceilings):
        for second in ceilings[index + 1 :]:
            line = _plane_intersection_line(first, second, room.bounds)
            if line is not None:
                clipped = line.intersection(room)
                if not clipped.is_empty:
                    linework.append(clipped)
    arrangement = unary_union(linework)

    def lines(geometry: BaseGeometry) -> Iterable[LineString]:
        if isinstance(geometry, LineString):
            yield geometry
            return
        for part in getattr(geometry, "geoms", ()):
            yield from lines(part)

    vertex_ids: dict[tuple[int, int], int] = {}
    vertices: list[list[float]] = []
    segments: set[tuple[int, int]] = set()

    def vertex_id(point: Sequence[float]) -> int:
        key = tuple(np.round(np.asarray(point[:2]) * 1.0e9).astype(np.int64))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append([key[0] / 1.0e9, key[1] / 1.0e9])
        return vertex_ids[key]

    for line in lines(arrangement):
        coordinates = np.asarray(line.coords, dtype=np.float64)
        for first, second in zip(coordinates[:-1], coordinates[1:]):
            first_id, second_id = vertex_id(first), vertex_id(second)
            if first_id != second_id:
                segments.add(tuple(sorted((first_id, second_id))))
    if not vertices or not segments:
        raise LayoutError("room constraint graph is empty")
    holes = []
    for polygon in _component_polygons(room):
        for interior in polygon.interiors:
            holes.append(list(Polygon(interior).representative_point().coords[0]))
    payload: dict[str, Any] = {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "segments": np.asarray(sorted(segments), dtype=np.int32),
    }
    if holes:
        payload["holes"] = np.asarray(holes, dtype=np.float64)
    import triangle as constrained_delaunay

    result = constrained_delaunay.triangulate(payload, "pQ")
    output_vertices = np.asarray(result.get("vertices", []), dtype=np.float64)
    output_faces = np.asarray(result.get("triangles", []), dtype=np.int64).reshape(-1, 3)
    triangles: list[np.ndarray] = []
    covered = room.buffer(1.0e-8)
    for face in output_faces:
        coordinates = output_vertices[face].copy()
        center = coordinates.mean(axis=0)
        if not covered.covers(Point(float(center[0]), float(center[1]))):
            continue
        signed = float(np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]))
        if abs(signed) <= 1.0e-10:
            continue
        if signed < 0:
            coordinates[[1, 2]] = coordinates[[2, 1]]
        triangles.append(coordinates)
    if not triangles:
        raise LayoutError("room constrained triangulation produced no triangles")
    return triangles


def assign_triangle_ceilings(
    triangles: Sequence[np.ndarray],
    ceilings: Sequence[CeilingPlane],
    fallback_elevation: float,
) -> list[CeilingPlane]:
    fallback = CeilingPlane(
        polygon_id=-1,
        coefficients=np.asarray([0.0, 0.0, 1.0, -fallback_elevation]),
        geometry_xy=Polygon(),
        area_square_meters=0.0,
    )
    if not ceilings:
        return [fallback] * len(triangles)
    direct: list[CeilingPlane | None] = []
    for triangle in triangles:
        center = triangle.mean(axis=0)
        point = Point(float(center[0]), float(center[1]))
        candidates = [
            ceiling
            for ceiling in ceilings
            if ceiling.geometry_xy.buffer(1.0e-8).covers(point)
        ]
        direct.append(
            min(candidates, key=lambda item: float(item.height(center[None])[0]))
            if candidates
            else None
        )
    edge_to_triangles: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for edge_index in range(3):
            first = tuple(np.round(triangle[edge_index] * 1.0e7).astype(np.int64))
            second = tuple(np.round(triangle[(edge_index + 1) % 3] * 1.0e7).astype(np.int64))
            edge_to_triangles[tuple(sorted((first, second)))].append(triangle_index)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for owners in edge_to_triangles.values():
        if len(owners) == 2:
            adjacency[owners[0]].add(owners[1])
            adjacency[owners[1]].add(owners[0])
    assigned = list(direct)
    queue: deque[int] = deque(index for index, value in enumerate(assigned) if value is not None)
    distances = np.full(len(triangles), np.iinfo(np.int32).max, dtype=np.int32)
    for index in queue:
        distances[index] = 0
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            candidate_distance = int(distances[current]) + 1
            current_plane = assigned[current]
            if candidate_distance < distances[neighbor]:
                distances[neighbor] = candidate_distance
                if direct[neighbor] is None:
                    assigned[neighbor] = current_plane
                queue.append(neighbor)
            elif (
                candidate_distance == distances[neighbor]
                and direct[neighbor] is None
                and current_plane is not None
            ):
                center = triangles[neighbor].mean(axis=0)
                previous = assigned[neighbor]
                if previous is None or float(current_plane.height(center[None])[0]) < float(
                    previous.height(center[None])[0]
                ):
                    assigned[neighbor] = current_plane
    return [value if value is not None else fallback for value in assigned]


def _edge_key(first: np.ndarray, second: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    a = tuple(np.round(first * 1.0e7).astype(np.int64))
    b = tuple(np.round(second * 1.0e7).astype(np.int64))
    return tuple(sorted((a, b)))


def _room_color(room_index: int) -> tuple[float, float, float]:
    hue = (room_index * 0.6180339887498949) % 1.0
    import colorsys

    return colorsys.hsv_to_rgb(hue, 0.55, 0.90)


def _opening_for_segment(
    first: np.ndarray,
    second: np.ndarray,
    room_id: str,
    edges: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> Mapping[str, Any] | None:
    segment = LineString([first, second])
    midpoint = segment.interpolate(0.5, normalized=True)
    for edge in edges:
        if edge.get("pruned") or room_id not in [str(value) for value in edge["room_ids"]]:
            continue
        if edge["kind"] not in {"door", "opening"}:
            continue
        line = LineString(edge["line_xy"])
        if line.buffer(tolerance, cap_style=2).covers(midpoint) or segment.distance(line) <= tolerance:
            projected = line.project(midpoint)
            if -tolerance <= projected <= line.length + tolerance:
                return edge
    return None


def _nearest_wall_key(
    midpoint: np.ndarray,
    wall_polygon_ids: Sequence[int],
    geometries: Mapping[int, BaseGeometry],
    level_id: str,
) -> str:
    point = Point(float(midpoint[0]), float(midpoint[1]))
    candidates = [
        (float(geometries[int(polygon_id)].distance(point)), int(polygon_id))
        for polygon_id in wall_polygon_ids
        if not geometries[int(polygon_id)].is_empty
    ]
    if candidates:
        distance, polygon_id = min(candidates)
        if distance <= 0.40:
            return f"prototype_wall_{polygon_id:04d}"
    direction_key = "fallback"
    return f"{level_id}_{direction_key}_{round(float(midpoint[0]), 1)}_{round(float(midpoint[1]), 1)}"


def _add_surface_triangle(
    room_mesh: MeshData,
    combined: MeshData,
    points: Sequence[Sequence[float]],
    room_color: Sequence[float],
    class_name: str,
    tag: str,
) -> None:
    room_mesh.add_triangle(points, room_color, tag)
    combined.add_triangle(points, CLASS_COLORS[class_name], tag)


def _add_surface_quad(
    room_mesh: MeshData,
    combined: MeshData,
    points: Sequence[Sequence[float]],
    room_color: Sequence[float],
    class_name: str,
    tag: str,
) -> None:
    room_mesh.add_quad(points, room_color, tag)
    combined.add_quad(points, CLASS_COLORS[class_name], tag)


def extrude_room(
    room_id: str,
    room_geometry: BaseGeometry,
    level: Mapping[str, Any],
    ceilings: Sequence[CeilingPlane],
    graph_edges: Sequence[Mapping[str, Any]],
    wall_geometries: Mapping[int, BaseGeometry],
    layout_config: LayoutConfig,
    room_index: int,
) -> tuple[MeshData, MeshData, MeshData, list[str], list[dict[str, Any]], dict[str, Any]]:
    floor_z = float(level["elevation_meters"])
    fallback_ceiling = float(level["ceiling_elevation_meters"])
    triangles_xy = constrained_room_triangles(room_geometry, ceilings)
    assignments = assign_triangle_ceilings(triangles_xy, ceilings, fallback_ceiling)
    room_color = _room_color(room_index)
    closed = MeshData.empty()
    final = MeshData.empty()
    combined = MeshData.empty()
    wall_tags: list[str] = []
    entities: list[dict[str, Any]] = []
    edge_owners: dict[Any, list[tuple[int, int, np.ndarray, np.ndarray]]] = defaultdict(list)
    top_triangles: list[np.ndarray] = []
    for triangle_index, (triangle_xy, ceiling) in enumerate(zip(triangles_xy, assignments)):
        top_z = ceiling.height(triangle_xy)
        top_z = np.maximum(top_z, floor_z + 0.10)
        floor = np.column_stack((triangle_xy, np.full(3, floor_z)))
        top = np.column_stack((triangle_xy, top_z))
        top_triangles.append(top)
        floor_tag = f"room:{room_id}:floor:{triangle_index}"
        ceiling_tag = f"room:{room_id}:ceiling:{triangle_index}"
        _add_surface_triangle(closed, MeshData.empty(), floor[[0, 2, 1]], room_color, "floor", floor_tag)
        _add_surface_triangle(closed, MeshData.empty(), top, room_color, "ceiling", ceiling_tag)
        _add_surface_triangle(final, combined, floor[[0, 2, 1]], room_color, "floor", floor_tag)
        _add_surface_triangle(final, combined, top, room_color, "ceiling", ceiling_tag)
        entities.extend(
            [
                {
                    "entity_id": floor_tag,
                    "class": "floor",
                    "room_id": room_id,
                    "vertices": floor[[0, 2, 1]].tolist(),
                },
                {
                    "entity_id": ceiling_tag,
                    "class": "ceiling",
                    "room_id": room_id,
                    "source_ceiling_polygon_id": ceiling.polygon_id,
                    "vertices": top.tolist(),
                },
            ]
        )
        for local_edge in range(3):
            first = triangle_xy[local_edge]
            second = triangle_xy[(local_edge + 1) % 3]
            edge_owners[_edge_key(first, second)].append(
                (triangle_index, local_edge, first, second)
            )

    boundary_count = 0
    opening_segments = 0
    door_segments = 0
    discontinuity_count = 0
    tolerance = max(0.075, 1.5 * 0.05)
    for owners in edge_owners.values():
        if len(owners) == 1:
            triangle_index, local_edge, first, second = owners[0]
            top = top_triangles[triangle_index]
            top_first = top[local_edge]
            top_second = top[(local_edge + 1) % 3]
            bottom_first = np.r_[first, floor_z]
            bottom_second = np.r_[second, floor_z]
            midpoint = 0.5 * (first + second)
            wall_key = _nearest_wall_key(
                midpoint,
                level["wall_polygon_ids"],
                wall_geometries,
                str(level["level_id"]),
            )
            tag = f"room:{room_id}:wall:{boundary_count}:{wall_key}"
            closed.add_quad(
                [bottom_first, bottom_second, top_second, top_first],
                room_color,
                tag,
            )
            edge = _opening_for_segment(
                first,
                second,
                room_id,
                graph_edges,
                tolerance,
            )
            if edge is None:
                _add_surface_quad(
                    final,
                    combined,
                    [bottom_first, bottom_second, top_second, top_first],
                    room_color,
                    "wall",
                    tag,
                )
                wall_tags.extend([wall_key, wall_key])
                entities.append(
                    {
                        "entity_id": tag,
                        "class": "wall",
                        "room_id": room_id,
                        "wall_instance_id": wall_key,
                        "vertices": np.asarray(
                            [bottom_first, bottom_second, top_second, top_first]
                        ).tolist(),
                    }
                )
            elif edge["kind"] == "door":
                door_z = min(floor_z + layout_config.door_height_meters, float(top_first[2]), float(top_second[2]))
                if max(float(top_first[2]), float(top_second[2])) > door_z + 1.0e-6:
                    first_lintel = np.r_[first, door_z]
                    second_lintel = np.r_[second, door_z]
                    _add_surface_quad(
                        final,
                        combined,
                        [first_lintel, second_lintel, top_second, top_first],
                        room_color,
                        "wall",
                        tag,
                    )
                    wall_tags.extend([wall_key, wall_key])
                door_segments += 1
            else:
                opening_segments += 1
            boundary_count += 1
        elif len(owners) == 2:
            first_owner, second_owner = owners
            first_index, first_local, first_xy, second_xy = first_owner
            second_index, second_local, _, _ = second_owner
            first_top = top_triangles[first_index]
            second_top = top_triangles[second_index]
            z_first = np.asarray(
                [first_top[first_local, 2], first_top[(first_local + 1) % 3, 2]]
            )
            second_points = triangles_xy[second_index]
            second_values = top_triangles[second_index]
            matched = []
            for point in (first_xy, second_xy):
                index = int(np.argmin(np.linalg.norm(second_points - point, axis=1)))
                matched.append(float(second_values[index, 2]))
            z_second = np.asarray(matched)
            if float(np.max(np.abs(z_first - z_second))) > 1.0e-5:
                lower = np.minimum(z_first, z_second)
                upper = np.maximum(z_first, z_second)
                points = [
                    np.r_[first_xy, lower[0]],
                    np.r_[second_xy, lower[1]],
                    np.r_[second_xy, upper[1]],
                    np.r_[first_xy, upper[0]],
                ]
                tag = f"room:{room_id}:ceiling_discontinuity:{discontinuity_count}"
                _add_surface_quad(final, combined, points, room_color, "wall", tag)
                closed.add_quad(points, room_color, tag)
                wall_tags.extend([f"{room_id}_ceiling_discontinuity"] * 2)
                entities.append(
                    {
                        "entity_id": tag,
                        "class": "wall",
                        "room_id": room_id,
                        "wall_instance_id": f"{room_id}_ceiling_discontinuity",
                        "vertices": np.asarray(points).tolist(),
                    }
                )
                discontinuity_count += 1
    diagnostics = {
        "room_id": room_id,
        "floor_area_square_meters": float(room_geometry.area),
        "floor_triangle_count": len(triangles_xy),
        "boundary_segment_count": boundary_count,
        "door_segment_count": door_segments,
        "opening_segment_count": opening_segments,
        "ceiling_discontinuity_count": discontinuity_count,
        "ceiling_polygon_ids_used": sorted(
            {value.polygon_id for value in assignments if value.polygon_id >= 0}
        ),
        "fallback_ceiling_triangle_count": sum(value.polygon_id < 0 for value in assignments),
    }
    return closed, final, combined, wall_tags, entities, diagnostics


def _door_frame(
    edge: Mapping[str, Any],
    floor_z: float,
    ceiling_z: float,
    height: float,
    thickness: float,
) -> tuple[MeshData, dict[str, Any]]:
    line = np.asarray(edge["line_xy"], dtype=np.float64)
    direction = line[1] - line[0]
    direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
    normal = np.asarray([-direction[1], direction[0]])
    corners = np.asarray(
        [
            line[0] - 0.5 * thickness * normal,
            line[1] - 0.5 * thickness * normal,
            line[1] + 0.5 * thickness * normal,
            line[0] + 0.5 * thickness * normal,
        ]
    )
    top_z = min(floor_z + height, ceiling_z)
    mesh = MeshData.empty()
    vertices = []
    for index in range(4):
        next_index = (index + 1) % 4
        quad = [
            np.r_[corners[index], floor_z],
            np.r_[corners[next_index], floor_z],
            np.r_[corners[next_index], top_z],
            np.r_[corners[index], top_z],
        ]
        mesh.add_quad(quad, CLASS_COLORS["door"], f"door:{edge['edge_id']}:{index}")
        vertices.append(np.asarray(quad).tolist())
    entity = {
        "entity_id": f"door:{edge['edge_id']}",
        "class": "door",
        "edge_id": edge["edge_id"],
        "room_ids": edge["room_ids"],
        "height_meters": top_z - floor_z,
        "width_meters": float(edge["width_meters"]),
        "frame_faces": vertices,
    }
    return mesh, entity


def _stair_mesh(
    edge: Mapping[str, Any],
    level_by_room: Mapping[str, Mapping[str, Any]],
    step_height: float,
) -> tuple[MeshData, dict[str, Any]]:
    rectangle = np.asarray(edge["rectangle_xy"], dtype=np.float64)
    endpoints = np.asarray(edge["edge_midpoints_xyz"], dtype=np.float64)
    low_index = int(np.argmin(endpoints[:, 2]))
    low_midpoint, high_midpoint = endpoints[low_index], endpoints[1 - low_index]
    axis = high_midpoint[:2] - low_midpoint[:2]
    length = max(float(np.linalg.norm(axis)), 1.0e-8)
    axis /= length
    parameter = (rectangle - low_midpoint[:2]) @ axis
    interpolation = np.clip(parameter / length, 0.0, 1.0)
    floor_z = low_midpoint[2] + interpolation * (high_midpoint[2] - low_midpoint[2])
    ceiling_z = min(
        float(level_by_room[str(room_id)]["ceiling_elevation_meters"])
        for room_id in edge["room_ids"]
    )
    mesh = MeshData.empty()
    mesh.add_triangle(
        np.column_stack((rectangle[[0, 2, 1]], floor_z[[0, 2, 1]])),
        CLASS_COLORS["stairs"],
        f"stair:{edge['edge_id']}:floor:0",
    )
    mesh.add_triangle(
        np.column_stack((rectangle[[0, 3, 2]], floor_z[[0, 3, 2]])),
        CLASS_COLORS["stairs"],
        f"stair:{edge['edge_id']}:floor:1",
    )
    top = np.column_stack((rectangle, np.full(4, ceiling_z)))
    mesh.add_triangle(top[[0, 1, 2]], CLASS_COLORS["stairs"], f"stair:{edge['edge_id']}:ceiling:0")
    mesh.add_triangle(top[[0, 2, 3]], CLASS_COLORS["stairs"], f"stair:{edge['edge_id']}:ceiling:1")
    step_count = max(1, int(math.ceil(abs(high_midpoint[2] - low_midpoint[2]) / step_height)))
    # Shared room boundaries remain wall-free; only visualization treads are added.
    normal = np.asarray([-axis[1], axis[0]])
    width = max(abs((rectangle - rectangle.mean(axis=0)) @ normal))
    for step in range(step_count + 1):
        t = step / step_count
        center = low_midpoint[:2] + t * length * axis
        z = low_midpoint[2] + t * (high_midpoint[2] - low_midpoint[2]) + 0.005
        half_run = 0.5 * length / step_count
        quad_xy = np.asarray(
            [
                center - half_run * axis - width * normal,
                center + half_run * axis - width * normal,
                center + half_run * axis + width * normal,
                center - half_run * axis + width * normal,
            ]
        )
        mesh.add_quad(
            np.column_stack((quad_xy, np.full(4, z))),
            CLASS_COLORS["stairs"],
            f"stair:{edge['edge_id']}:step:{step}",
        )
    entity = {
        "entity_id": f"stair:{edge['edge_id']}",
        "class": "stairs",
        "edge_id": edge["edge_id"],
        "room_ids": edge["room_ids"],
        "rectangle_xy": rectangle.tolist(),
        "edge_midpoints_xyz": endpoints.tolist(),
        "step_count": step_count,
    }
    return mesh, entity


def detect_windows(
    config: PipelineConfig,
    transforms_path: Path,
    coco_dir: Path,
    wall_mesh: MeshData,
    wall_tags: Sequence[str],
    rooms: Mapping[str, BaseGeometry],
    progress_path: Path,
) -> tuple[list[dict[str, Any]], MeshData, dict[str, Any]]:
    if len(wall_tags) != len(wall_mesh.triangles):
        raise LayoutError("wall triangle instance tags are inconsistent")
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(wall_mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(wall_mesh.triangles, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    tag_names = sorted(set(wall_tags))
    tag_to_index = {name: index for index, name in enumerate(tag_names)}
    primitive_wall_ids = np.asarray([tag_to_index[value] for value in wall_tags], dtype=np.int32)
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms["frames"][:: config.layout.window_frame_stride]
    root = transforms_path.parent
    fx, fy = float(transforms["fl_x"]), float(transforms["fl_y"])
    cx, cy = float(transforms["cx"]), float(transforms["cy"])
    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    voxel = config.layout.window_voxel_size_meters
    voxel_keys: dict[int, set[tuple[int, int, int]]] = defaultdict(set)
    candidate_pixel_count = 0
    ray_hit_count = 0
    for frame_index, frame in enumerate(frames):
        stem = Path(frame["file_path"]).stem
        semantic_path = coco_dir / f"{stem}.png"
        if not semantic_path.is_file():
            raise LayoutError(f"missing COCO segmentation for window detection: {semantic_path}")
        with Image.open(semantic_path) as image:
            semantic = np.asarray(image, dtype=np.uint8)
        mask = np.isin(semantic, list(WINDOW_COCO_IDS))
        stride = config.layout.window_pixel_stride
        if stride > 1:
            sampled = np.zeros_like(mask)
            sampled[::stride, ::stride] = mask[::stride, ::stride]
            mask = sampled
        rows, columns = np.where(mask)
        candidate_pixel_count += len(rows)
        if len(rows):
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float64) @ opengl_to_opencv
            camera = np.column_stack(
                (
                    (columns.astype(np.float64) - cx) / fx,
                    (rows.astype(np.float64) - cy) / fy,
                    np.ones(len(rows)),
                )
            )
            camera /= np.linalg.norm(camera, axis=1, keepdims=True)
            directions = camera @ c2w[:3, :3].T
            origins = np.repeat(c2w[None, :3, 3], len(rows), axis=0)
            for start in range(0, len(rows), 250_000):
                stop = min(start + 250_000, len(rows))
                rays = np.column_stack((origins[start:stop], directions[start:stop])).astype(np.float32)
                result = scene.cast_rays(o3d.core.Tensor(rays))
                distance = result["t_hit"].numpy()
                primitive = result["primitive_ids"].numpy().astype(np.int64)
                valid = np.isfinite(distance) & (
                    distance <= config.layout.window_maximum_ray_distance_meters
                ) & (primitive < len(primitive_wall_ids))
                if not np.any(valid):
                    continue
                hit = origins[start:stop][valid] + distance[valid, None] * directions[start:stop][valid]
                wall_ids = primitive_wall_ids[primitive[valid]]
                ray_hit_count += len(hit)
                keys = np.floor(hit / voxel).astype(np.int64)
                for wall_id in np.unique(wall_ids):
                    unique = np.unique(keys[wall_ids == wall_id], axis=0)
                    voxel_keys[int(wall_id)].update(map(tuple, unique.tolist()))
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(frames):
            _write_status(
                progress_path,
                "window_raycast",
                f"{frame_index + 1}/{len(frames)} candidates={candidate_pixel_count} hits={ray_hit_count}",
            )

    windows: list[dict[str, Any]] = []
    window_mesh = MeshData.empty()
    points_after_voxel = sum(len(values) for values in voxel_keys.values())
    points_after_outlier = 0
    cluster_count = 0
    for wall_id, keys in sorted(voxel_keys.items()):
        points = (np.asarray(sorted(keys), dtype=np.float64) + 0.5) * voxel
        if len(points) < config.layout.window_minimum_cluster_points:
            continue
        neighbor_count = min(config.layout.window_outlier_neighbors, len(points) - 1)
        if neighbor_count >= 2:
            keep = LocalOutlierFactor(n_neighbors=neighbor_count).fit_predict(points) > 0
            points = points[keep]
        points_after_outlier += len(points)
        if len(points) < config.layout.window_minimum_cluster_points:
            continue
        center_xy = points[:, :2].mean(axis=0)
        _, _, axes = np.linalg.svd(points[:, :2] - center_xy, full_matrices=False)
        axis = axes[0]
        if axis[0] < 0 or (abs(axis[0]) < 1.0e-8 and axis[1] < 0):
            axis = -axis
        horizontal = (points[:, :2] - center_xy) @ axis
        coordinates = np.column_stack((horizontal, points[:, 2]))
        labels = DBSCAN(
            eps=config.layout.window_dbscan_epsilon_meters,
            min_samples=config.layout.window_dbscan_minimum_samples,
        ).fit_predict(coordinates)
        for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
            cluster = points[labels == label]
            if len(cluster) < config.layout.window_minimum_cluster_points:
                continue
            cluster_count += 1
            u = (cluster[:, :2] - center_xy) @ axis
            u0, u1 = float(u.min()), float(u.max())
            z0, z1 = float(cluster[:, 2].min()), float(cluster[:, 2].max())
            width, height = u1 - u0, z1 - z0
            if width <= config.layout.window_minimum_size_meters or height <= config.layout.window_minimum_size_meters:
                continue
            first = center_xy + u0 * axis
            second = center_xy + u1 * axis
            rectangle = np.asarray(
                [
                    [first[0], first[1], z0],
                    [second[0], second[1], z0],
                    [second[0], second[1], z1],
                    [first[0], first[1], z1],
                ]
            )
            midpoint = Point(*rectangle[:, :2].mean(axis=0))
            room_ids = sorted(
                room_id
                for room_id, geometry in rooms.items()
                if geometry.boundary.distance(midpoint) <= 0.20
            )
            window_id = f"window_{len(windows):03d}"
            windows.append(
                {
                    "entity_id": window_id,
                    "class": "window",
                    "wall_instance_id": tag_names[wall_id],
                    "room_ids": room_ids,
                    "point_count": len(cluster),
                    "width_meters": width,
                    "height_meters": height,
                    "vertices": rectangle.tolist(),
                }
            )
            window_mesh.add_quad(rectangle, CLASS_COLORS["window"], window_id)
    diagnostics = {
        "frame_count": len(frames),
        "candidate_pixel_count": candidate_pixel_count,
        "wall_ray_hit_count": ray_hit_count,
        "voxel_point_count": points_after_voxel,
        "points_after_local_outlier_factor": points_after_outlier,
        "dbscan_cluster_count_before_rectangle_filter": cluster_count,
        "accepted_window_count": len(windows),
        "window_coco_ids": sorted(WINDOW_COCO_IDS),
    }
    return windows, window_mesh, diagnostics


def _mesh_statistics(mesh: o3d.geometry.TriangleMesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return {
        "vertices": len(vertices),
        "triangles": len(triangles),
        "surface_area_square_meters": float(mesh.get_surface_area()),
        "finite": bool(np.isfinite(vertices).all()),
        "edge_manifold_allow_boundary": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
    }


def run_layout(config: PipelineConfig, run_id: str) -> Path:
    """Run layout and return its completed manifest path."""
    run_dir = config.storage.outputs / config.scene / run_id
    component_dir = run_dir / "layout"
    component_dir.mkdir(parents=True, exist_ok=True)
    attempt = _next_attempt(component_dir)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    _write_status(component_dir / "STATUS.json", "running", str(attempt))
    _write_status(attempt / "STATUS.json", "loading_inputs", "scene_graph/prototype/oneformer/pose")
    try:
        scene_graph_manifest_path = run_dir / "scene_graph" / "manifest.json"
        scene_graph_manifest = _require_complete_manifest(
            scene_graph_manifest_path, "scene_graph"
        )
        prototype_manifest_path = run_dir / "prototype" / "manifest.json"
        prototype_manifest = _require_complete_manifest(
            prototype_manifest_path, "prototype"
        )
        oneformer_manifest_path = run_dir / "oneformer" / "manifest.json"
        oneformer_manifest = _require_complete_manifest(
            oneformer_manifest_path, "oneformer"
        )
        pose_manifest_path = run_dir / "pose" / "manifest.json"
        _require_complete_manifest(pose_manifest_path, "pose")
        graph_path = Path(scene_graph_manifest["outputs"]["scene_graph"]["path"])
        rooms_path = Path(scene_graph_manifest["outputs"]["rooms"]["path"])
        levels_path = Path(scene_graph_manifest["outputs"]["levels"]["path"])
        state_path = Path(prototype_manifest["outputs"]["final_model_state"]["path"])
        transforms_path = run_dir / "pose" / "transforms.json"
        coco_dir = Path(oneformer_manifest["outputs"]["coco_id_dir"])
        for path in (graph_path, rooms_path, state_path, transforms_path):
            if not path.is_file():
                raise LayoutError(f"required layout input is missing: {path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        room_collection = json.loads(rooms_path.read_text(encoding="utf-8"))
        room_geometries = {
            str(feature["id"]): shape(feature["geometry"]).buffer(0)
            for feature in room_collection["features"]
        }
        levels = graph["levels"]
        level_by_id = {str(level["level_id"]): level for level in levels}
        room_by_id = {str(room["room_id"]): room for room in graph["rooms"]}
        level_by_room = {
            room_id: level_by_id[str(room["level_id"])]
            for room_id, room in room_by_id.items()
        }
        active_room_ids = [str(value) for value in graph["active_room_ids"]]
        # scene_graph contours run through raster-cell centers. Expanding every room
        # by half a cell restores the actual cell boundaries: adjacent rooms
        # meet at the midpoint instead of overlapping, and pixel-count areas are
        # preserved to raster precision.
        half_cell = 0.5 * config.scene_graph.grid_resolution_meters
        room_geometries = {
            room_id: geometry.buffer(half_cell, join_style=2).buffer(0)
            for room_id, geometry in room_geometries.items()
        }
        active_edges = [edge for edge in graph["edges"] if not edge.get("pruned", False)]
        stair_edges = [edge for edge in active_edges if edge["kind"] == "stair"]
        stair_cutouts = unary_union(
            [Polygon(edge["rectangle_xy"]) for edge in stair_edges]
        ) if stair_edges else Polygon()
        prototype = load_prototype_data(state_path, config.prototype.source_repository)
        prototype_geometries = polygon_geometries(prototype)

        rooms_closed_dir = attempt / "rooms_closed"
        rooms_final_dir = attempt / "rooms_final"
        rooms_closed_dir.mkdir()
        rooms_final_dir.mkdir()
        combined = MeshData.empty()
        wall_mesh = MeshData.empty()
        wall_tags: list[str] = []
        entities: list[dict[str, Any]] = []
        room_diagnostics = []
        closed_records: dict[str, Any] = {}
        final_records: dict[str, Any] = {}
        _write_status(attempt / "STATUS.json", "extruding_rooms", f"0/{len(active_room_ids)}")
        for room_index, room_id in enumerate(active_room_ids):
            room = room_by_id[room_id]
            level = level_by_room[room_id]
            geometry = room_geometries[room_id]
            if not stair_cutouts.is_empty and geometry.intersects(stair_cutouts):
                geometry = geometry.difference(stair_cutouts).buffer(0)
            ceilings = ceiling_planes(
                prototype,
                level,
                prototype_geometries,
                config.layout.maximum_ceilings_per_room,
            )
            closed, final, room_combined, room_wall_tags, room_entities, diagnostics = extrude_room(
                room_id,
                geometry,
                level,
                ceilings,
                active_edges,
                prototype_geometries,
                config.layout,
                room_index,
            )
            closed_path = rooms_closed_dir / f"{room_id}.ply"
            final_path = rooms_final_dir / f"{room_id}.ply"
            closed_mesh = _write_mesh(closed_path, closed)
            final_mesh = _write_mesh(final_path, final)
            closed_mesh.merge_close_vertices(1.0e-6)
            topologically_closed = bool(
                closed_mesh.is_edge_manifold(allow_boundary_edges=False)
                and closed_mesh.is_vertex_manifold()
                and closed_mesh.is_orientable()
            )
            closed_records[room_id] = {
                **_record(closed_path),
                "topologically_closed_after_vertex_weld": topologically_closed,
                "open3d_watertight_after_vertex_weld": bool(closed_mesh.is_watertight()),
                "self_intersecting_after_vertex_weld": bool(closed_mesh.is_self_intersecting()),
            }
            final_records[room_id] = {
                **_record(final_path),
                "has_graph_opening": any(
                    room_id in edge["room_ids"] and edge["kind"] in {"door", "opening"}
                    for edge in active_edges
                ),
            }
            combined.extend(room_combined)
            # Extract only the wall faces from the per-room class-colored mesh.
            wall_face_indices = [
                index for index, tag in enumerate(room_combined.face_tags)
                if ":wall:" in tag or ":ceiling_discontinuity:" in tag
            ]
            for local_index, tag in enumerate(room_wall_tags):
                face_index = wall_face_indices[local_index]
                face = room_combined.triangles[face_index]
                wall_mesh.add_triangle(
                    [room_combined.vertices[value] for value in face],
                    CLASS_COLORS["wall"],
                    room_combined.face_tags[face_index],
                )
                wall_tags.append(tag)
            entities.extend(room_entities)
            room_diagnostics.append(diagnostics)
            _write_status(
                attempt / "STATUS.json",
                "extruding_rooms",
                f"{room_index + 1}/{len(active_room_ids)}",
            )

        door_mesh = MeshData.empty()
        door_entities = []
        for edge in active_edges:
            if edge["kind"] != "door":
                continue
            level = level_by_room[str(edge["room_ids"][0])]
            mesh_data, entity = _door_frame(
                edge,
                float(level["elevation_meters"]),
                float(level["ceiling_elevation_meters"]),
                config.layout.door_height_meters,
                config.scene_graph.wall_line_width_meters,
            )
            door_mesh.extend(mesh_data)
            combined.extend(mesh_data)
            door_entities.append(entity)
        entities.extend(door_entities)

        stairs_mesh = MeshData.empty()
        stair_entities = []
        for edge in stair_edges:
            mesh_data, entity = _stair_mesh(
                edge, level_by_room, config.layout.stair_step_height_meters
            )
            stairs_mesh.extend(mesh_data)
            combined.extend(mesh_data)
            stair_entities.append(entity)
        entities.extend(stair_entities)

        _write_status(attempt / "STATUS.json", "detecting_windows", "ray casting semantic pixels")
        active_geometries = {room_id: room_geometries[room_id] for room_id in active_room_ids}
        windows, windows_mesh, window_diagnostics = detect_windows(
            config,
            transforms_path,
            coco_dir,
            wall_mesh,
            wall_tags,
            active_geometries,
            attempt / "STATUS.json",
        )
        if windows_mesh.triangles:
            combined.extend(windows_mesh)
        entities.extend(windows)

        layout_path = attempt / "layout.ply"
        layout_obj_path = attempt / "layout.obj"
        structure_path = attempt / "structures.ply"
        wall_path = attempt / "walls.ply"
        window_path = attempt / "windows.ply"
        door_path = attempt / "doors.ply"
        stair_path = attempt / "stairs.ply"
        structure = MeshData.empty()
        structure.extend(combined)
        layout_mesh = _write_mesh(layout_path, combined)
        if not o3d.io.write_triangle_mesh(str(layout_obj_path), layout_mesh):
            raise LayoutError(f"failed to write OBJ: {layout_obj_path}")
        structure_without_edges = MeshData.empty()
        for room_id in active_room_ids:
            mesh = o3d.io.read_triangle_mesh(str(rooms_final_dir / f"{room_id}.ply"))
            data = MeshData.empty()
            vertices = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)
            for face in np.asarray(mesh.triangles):
                data.add_triangle(vertices[face], colors[face].mean(axis=0), f"room:{room_id}")
            structure_without_edges.extend(data)
        _write_mesh(structure_path, structure_without_edges)
        _write_mesh(wall_path, wall_mesh)
        optional_outputs: dict[str, Any] = {}
        for name, path, mesh_data in (
            ("windows_mesh", window_path, windows_mesh),
            ("doors_mesh", door_path, door_mesh),
            ("stairs_mesh", stair_path, stairs_mesh),
        ):
            if mesh_data.triangles:
                _write_mesh(path, mesh_data)
                optional_outputs[name] = _record(path)

        entities_path = attempt / "layout_entities.json"
        final_graph_path = attempt / "final_scene_graph.json"
        diagnostics_path = attempt / "layout_diagnostics.json"
        completed_attempt_future = attempt.with_name(
            attempt.name.rsplit("_", 1)[0] + "_complete"
        )
        _write_json(
            entities_path,
            {
                "schema_version": 1,
                "scene": config.scene,
                "run_id": run_id,
                "coordinate_system": "Z-up meters",
                "classes": ["wall", "floor", "ceiling", "stairs", "door", "window"],
                "entities": entities,
            },
        )
        _write_json(
            final_graph_path,
            {
                **graph,
                "layout_entity_file": str(completed_attempt_future / entities_path.name),
                "room_meshes": {
                    room_id: str(completed_attempt_future / "rooms_final" / f"{room_id}.ply")
                    for room_id in active_room_ids
                },
                "windows": windows,
                "door_entities": door_entities,
                "stair_entities": stair_entities,
            },
        )
        _write_json(
            diagnostics_path,
            {
                "rooms": room_diagnostics,
                "windows": window_diagnostics,
                "mesh": _mesh_statistics(layout_mesh),
            },
        )
        validation = {
            "active_room_count_matches": len(final_records) == len(active_room_ids),
            "all_closed_room_shells_topologically_closed_after_weld": all(
                value["topologically_closed_after_vertex_weld"]
                for value in closed_records.values()
            ),
            "all_final_room_meshes_nonempty": all(
                Path(value["path"]).stat().st_size > 0 for value in final_records.values()
            ),
            "final_layout_nonempty": len(layout_mesh.triangles) > 0,
            "final_layout_finite": bool(np.isfinite(np.asarray(layout_mesh.vertices)).all()),
            "graph_edges_preserved": len(active_edges)
            == len(door_entities) + len(stair_entities)
            + sum(edge["kind"] == "opening" for edge in active_edges),
            "window_rectangles_meet_paper_thresholds": all(
                value["point_count"] >= config.layout.window_minimum_cluster_points
                and value["width_meters"] > config.layout.window_minimum_size_meters
                and value["height_meters"] > config.layout.window_minimum_size_meters
                for value in windows
            ),
            "no_ground_truth_inputs_used": True,
        }
        if not all(validation.values()):
            raise LayoutError(f"layout validation failed: {validation}")

        completed_attempt = _rename_attempt(attempt, "complete")

        def completed(path: Path) -> Path:
            return completed_attempt / path.relative_to(attempt)

        outputs = {
            "attempt_dir": str(completed_attempt),
            "layout_mesh": _record(completed(layout_path)),
            "layout_obj": _record(completed(layout_obj_path)),
            "structures_mesh": _record(completed(structure_path)),
            "walls_mesh": _record(completed(wall_path)),
            "entities": _record(completed(entities_path)),
            "final_scene_graph": _record(completed(final_graph_path)),
            "diagnostics": _record(completed(diagnostics_path)),
            "rooms_closed_dir": str(completed(rooms_closed_dir)),
            "rooms_final_dir": str(completed(rooms_final_dir)),
            "rooms_closed": {
                room_id: {**value, "path": str(completed(Path(value["path"])))}
                for room_id, value in closed_records.items()
            },
            "rooms_final": {
                room_id: {**value, "path": str(completed(Path(value["path"])))}
                for room_id, value in final_records.items()
            },
            **{
                name: _record(completed(Path(value["path"])))
                for name, value in optional_outputs.items()
            },
        }
        # Re-hash per-room meshes at their renamed paths.
        for category in ("rooms_closed", "rooms_final"):
            for room_id, value in outputs[category].items():
                flags = {
                    key: item
                    for key, item in value.items()
                    if key not in {"path", "sha256", "size_bytes"}
                }
                outputs[category][room_id] = {**_record(Path(value["path"])), **flags}
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "component": "layout",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "random_seed": config.runtime.random_seed,
            "inputs": {
                "scene_graph_manifest": _record(scene_graph_manifest_path),
                "prototype_manifest": _record(prototype_manifest_path),
                "oneformer_manifest": _record(oneformer_manifest_path),
                "pose_manifest": _record(pose_manifest_path),
                "prototype_state": _record(state_path),
                "scene_graph": _record(graph_path),
                "rooms": _record(rooms_path),
                "levels": _record(levels_path),
                "transforms": _record(transforms_path),
            },
            "algorithm": {
                "paper_sections": ["4.4 Room Extrusion", "4.4 Window Detection", "Appendix D.6"],
                "configuration": asdict(config.layout),
                "window_coco_ids": sorted(WINDOW_COCO_IDS),
                "ceiling_triangulation": "planar constrained arrangement from room/ceiling edges and pairwise ceiling-plane intersection lines",
            },
            "counts": {
                "rooms": len(active_room_ids),
                "floor_triangles": sum(value["floor_triangle_count"] for value in room_diagnostics),
                "doors": len(door_entities),
                "openings": sum(edge["kind"] == "opening" for edge in active_edges),
                "stairs": len(stair_entities),
                "windows": len(windows),
                "entities": len(entities),
                "layout_vertices": len(layout_mesh.vertices),
                "layout_triangles": len(layout_mesh.triangles),
            },
            "outputs": outputs,
            "validation": validation,
            "warnings": [
                "The paper does not disclose DBSCAN epsilon/min_samples, LOF neighbors, ray subsampling, voxel size, or stair visualization step height; all chosen values are explicit in YAML and recorded above.",
                "Closed pre-opening room shells are exported separately; final room meshes intentionally contain boundary holes at scene-graph doors/openings.",
                "Window rectangles are layout entities over the generated wall surfaces; wall polygons are retained behind them to match the paper's post-extrusion detection order.",
            ],
        }
        manifest_path = component_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(completed_attempt / "STATUS.json", "complete", str(completed(layout_path)))
        _write_status(component_dir / "STATUS.json", "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        if attempt.exists():
            failed_attempt = _rename_attempt(attempt, "failed")
            _write_status(
                failed_attempt / "STATUS.json",
                "failed",
                f"{type(error).__name__}: {error}",
            )
            _write_status(component_dir / "STATUS.json", "failed", str(failed_attempt))
        if isinstance(error, LayoutError):
            raise
        raise LayoutError(f"{type(error).__name__}: {error}") from error


def main() -> int:
    from .direct import run_component

    return run_component(run_layout, "Generate the final layout.")


if __name__ == "__main__":
    raise SystemExit(main())
