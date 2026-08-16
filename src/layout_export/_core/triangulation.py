"""2D room triangulation with optional ceiling constraints."""

from __future__ import annotations

from typing import Any, Sequence


class LayoutExportError(RuntimeError):
    """Raised when final layout export fails."""


def iter_polygons(geometry: Any) -> list[Any]:
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [part for part in geometry.geoms if not part.is_empty]
    return [part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon) and not part.is_empty]


def triangulate_polygon(geometry: Any) -> list[Any]:
    from shapely.geometry import Point
    from shapely.ops import triangulate

    triangles = []
    covered = geometry.buffer(1.0e-8)
    for candidate in triangulate(geometry):
        point = candidate.representative_point()
        if covered.covers(Point(point.x, point.y)) and candidate.area > 1.0e-10:
            triangles.append(candidate)
    return triangles


def coords2d(polygon: Any) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in list(polygon.exterior.coords)[:-1]]


def iter_lines(geometry: Any) -> list[Any]:
    from shapely.geometry import LineString, MultiLineString

    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 1.0e-8 else []
    if isinstance(geometry, MultiLineString):
        return [part for part in geometry.geoms if part.length > 1.0e-8]
    lines = []
    for part in getattr(geometry, "geoms", ()):
        lines.extend(iter_lines(part))
    return lines


def split_polygon_by_lines(polygon: Any, lines: Sequence[Any]) -> list[Any]:
    from shapely.ops import split

    pieces = [polygon]
    for line in lines:
        next_pieces = []
        for piece in pieces:
            try:
                result = split(piece, line)
            except Exception:
                next_pieces.append(piece)
                continue
            split_parts = [part.buffer(0) for part in result.geoms if not part.is_empty and part.area > 1.0e-10]
            next_pieces.extend(split_parts or [piece])
        pieces = next_pieces
    return pieces


def triangle_edges(coords: Sequence[Sequence[float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    values = [(float(x), float(y)) for x, y in coords]
    return list(zip(values, [*values[1:], values[0]]))


def triangle_centroid(coords: Sequence[Sequence[float]]) -> tuple[float, float]:
    return (
        sum(float(value[0]) for value in coords) / len(coords),
        sum(float(value[1]) for value in coords) / len(coords),
    )


def edge_key(first: Sequence[float], second: Sequence[float]) -> tuple[tuple[int, int], tuple[int, int]]:
    a = (round(float(first[0]) * 1_000_000), round(float(first[1]) * 1_000_000))
    b = (round(float(second[0]) * 1_000_000), round(float(second[1]) * 1_000_000))
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def point3(xy: Sequence[float], z: float) -> list[float]:
    return [float(xy[0]), float(xy[1]), float(z)]


def plane_intersection_line_xy(first: Any, second: Any, clip: Any) -> list[Any]:
    from shapely.geometry import LineString

    if first.is_fallback or second.is_fallback:
        return []
    a1, b1, c1, d1 = first.plane_eq
    a2, b2, c2, d2 = second.plane_eq
    if abs(c1) <= 1.0e-8 or abs(c2) <= 1.0e-8:
        return []
    alpha = a1 / c1 - a2 / c2
    beta = b1 / c1 - b2 / c2
    gamma = d1 / c1 - d2 / c2
    if abs(alpha) + abs(beta) <= 1.0e-10:
        return []
    min_x, min_y, max_x, max_y = clip.bounds
    span = max(max_x - min_x, max_y - min_y, 1.0) * 4.0
    cx, cy = (min_x + max_x) * 0.5, (min_y + max_y) * 0.5
    if abs(beta) > abs(alpha):
        x0, x1 = cx - span, cx + span
        y0 = -(alpha * x0 + gamma) / beta
        y1 = -(alpha * x1 + gamma) / beta
    else:
        y0, y1 = cy - span, cy + span
        x0 = -(beta * y0 + gamma) / alpha
        x1 = -(beta * y1 + gamma) / alpha
    return iter_lines(LineString([(x0, y0), (x1, y1)]).intersection(clip.buffer(1.0e-8)))


def ceiling_constraint_lines(room_polygon: Any, candidates: Sequence[Any]) -> list[Any]:
    lines = []
    for candidate in candidates:
        if candidate.geometry is None:
            continue
        lines.extend(iter_lines(candidate.geometry.boundary.intersection(room_polygon.buffer(1.0e-8))))
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            if first.geometry is None or second.geometry is None:
                continue
            overlap = room_polygon.intersection(first.geometry).intersection(second.geometry)
            if overlap.is_empty or overlap.area <= 1.0e-8:
                continue
            lines.extend(plane_intersection_line_xy(first, second, overlap))
    return lines


def triangulate_room_split_shapely(room_polygon: Any, candidates: Sequence[Any]) -> list[Any]:
    lines = ceiling_constraint_lines(room_polygon, candidates)
    cells = split_polygon_by_lines(room_polygon, lines) if lines else [room_polygon]
    triangles = []
    for cell in cells:
        triangles.extend(triangulate_polygon(cell))
    return triangles


def _ring_segments(coords: Sequence[Sequence[float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points = [(float(x), float(y)) for x, y in coords]
    return list(zip(points[:-1], points[1:]))


def _line_segments(line: Any) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    coords = [(float(x), float(y)) for x, y in line.coords]
    return list(zip(coords[:-1], coords[1:]))


def triangulate_room_cdt(room_polygon: Any, candidates: Sequence[Any]) -> list[Any]:
    import numpy as np
    import triangle as triangle_lib
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union

    linework = [LineString(room_polygon.exterior.coords)]
    linework.extend(LineString(interior.coords) for interior in room_polygon.interiors)
    linework.extend(ceiling_constraint_lines(room_polygon, candidates))
    noded_lines = iter_lines(unary_union(linework))

    vertices: list[tuple[float, float]] = []
    vertex_index: dict[tuple[int, int], int] = {}
    segments: list[tuple[int, int]] = []

    def add_vertex(point: Sequence[float]) -> int:
        key = (round(float(point[0]) * 1_000_000_000), round(float(point[1]) * 1_000_000_000))
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append((float(point[0]), float(point[1])))
        return vertex_index[key]

    for line in noded_lines:
        for first, second in _line_segments(line):
            if first == second:
                continue
            segments.append((add_vertex(first), add_vertex(second)))

    if len(vertices) < 3 or len(segments) < 3:
        return triangulate_room_split_shapely(room_polygon, candidates)

    holes = []
    for interior in room_polygon.interiors:
        hole = Polygon(interior)
        if not hole.is_empty and hole.area > 1.0e-10:
            point = hole.representative_point()
            holes.append((float(point.x), float(point.y)))

    payload: dict[str, Any] = {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "segments": np.asarray(segments, dtype=np.int32),
    }
    if holes:
        payload["holes"] = np.asarray(holes, dtype=np.float64)
    try:
        result = triangle_lib.triangulate(payload, "pQ")
    except Exception as error:
        raise LayoutExportError(f"CDT triangulation failed: {error}") from error

    result_vertices = result.get("vertices")
    result_triangles = result.get("triangles")
    if result_vertices is None or result_triangles is None:
        raise LayoutExportError("CDT triangulation produced no triangles")

    covered = room_polygon.buffer(1.0e-8)
    triangles = []
    for triangle in result_triangles:
        polygon = Polygon(result_vertices[triangle]).buffer(0)
        if polygon.is_empty or polygon.area <= 1.0e-10:
            continue
        point = polygon.representative_point()
        if covered.covers(Point(point.x, point.y)):
            triangles.append(polygon)
    return triangles


def triangulate_room_polygon(room_polygon: Any, candidates: Sequence[Any], mode: str = "split_shapely") -> list[Any]:
    if mode == "split_shapely":
        return triangulate_room_split_shapely(room_polygon, candidates)
    if mode == "cdt":
        return triangulate_room_cdt(room_polygon, candidates)
    raise LayoutExportError(f"unsupported triangulation mode: {mode}")
