"""Construct a single-floor Section 4.4 scene graph from a fitted prototype."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.scene_graph"

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .meshes import write_ceiling_candidate_ply


class SceneGraphError(RuntimeError):
    """Raised when scene graph construction fails."""


@dataclass(frozen=True)
class SceneGraphConfig:
    grid_resolution_meters: float = 0.05
    wall_line_width_meters: float = 0.12
    wall_interval_height_meters: float = 2.5
    ceiling_minimum_clearance_meters: float = 1.0
    door_maximum_width_meters: float = 1.5
    bottleneck_widths_meters: tuple[float, float] = (2.5, 1.5)
    minimum_seed_area_square_meters: float = 0.5
    minimum_room_area_square_meters: float = 1.0
    stair_minimum_triangles: int = 10
    stair_room_maximum_distance_meters: float = 0.75
    window_wall_distance_meters: float = 0.35
    window_dbscan_epsilon_meters: float = 0.35
    window_dbscan_minimum_samples: int = 10
    window_minimum_size_meters: float = 0.30
    floor_height: float | None = None


@dataclass(frozen=True)
class PrototypeData:
    vertices: Any
    triangles: Any
    triangle_polygons: Any
    polygon_classes: Any
    plane_eqs: Any
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class Grid2D:
    origin_xy: Any
    resolution_meters: float
    shape: tuple[int, int]

    def pixel_points(self, xy: Any) -> Any:
        import numpy as np

        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        columns = np.floor((points[:, 0] - self.origin_xy[0]) / self.resolution_meters)
        rows = np.floor((points[:, 1] - self.origin_xy[1]) / self.resolution_meters)
        return np.column_stack((columns, rows)).astype(np.int32)

    def world_centers(self, rows: Any, columns: Any) -> Any:
        import numpy as np

        x = self.origin_xy[0] + (np.asarray(columns, dtype=np.float64) + 0.5) * self.resolution_meters
        y = self.origin_xy[1] + (np.asarray(rows, dtype=np.float64) + 0.5) * self.resolution_meters
        return np.column_stack((x, y))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size}


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
        raise SceneGraphError(f"cannot overwrite unsupported output path: {path}")


def _component_dir(path: Path, name: str) -> Path:
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise SceneGraphError(f"{name} directory is missing: {directory}")
    return directory


def _required_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SceneGraphError(f"{name} is missing: {path}")
    return path


def _class_names_from_state(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return tuple(str(value) for value in raw)


def _ensure_unofficial_import_path() -> None:
    source_repo = Path("MultiFloor3D-unofficial").expanduser().resolve()
    if source_repo.is_dir():
        for path in (source_repo, source_repo / "mesh_fitting_3D"):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)


def _fit_plane(points: Any) -> Any:
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, right = np.linalg.svd(points - center, full_matrices=False)
    normal = right[-1]
    length = float(np.linalg.norm(normal))
    if length <= 1.0e-12:
        return np.asarray([0.0, 0.0, 1.0, -float(center[2])], dtype=np.float64)
    normal /= length
    if normal[2] < 0:
        normal = -normal
    return np.concatenate((normal, [-float(normal @ center)]))


def load_prototype_data(state_path: Path) -> PrototypeData:
    import numpy as np
    import torch

    _ensure_unofficial_import_path()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise SceneGraphError(f"prototype state is not a mapping: {state_path}")
    required = ("triangles", "triangle_polygons", "polygon_classes", "vertices.vertices")
    missing = [key for key in required if key not in state]
    if missing:
        raise SceneGraphError(f"prototype state is missing keys: {missing}")
    vertices = state["vertices.vertices"].detach().cpu().numpy().astype(np.float64)
    triangles = state["triangles"].detach().cpu().numpy().astype(np.int64)
    triangle_polygons = state["triangle_polygons"].detach().cpu().numpy().astype(np.int64)
    polygon_classes = state["polygon_classes"].detach().cpu().numpy().astype(np.int64)
    if "vertices.planes" in state:
        plane_eqs = state["vertices.planes"].detach().cpu().numpy().astype(np.float64)
    else:
        plane_eqs = np.zeros((len(polygon_classes), 4), dtype=np.float64)
        for polygon_id in range(len(polygon_classes)):
            polygon_triangles = triangles[triangle_polygons == polygon_id]
            if len(polygon_triangles) == 0:
                plane_eqs[polygon_id] = np.asarray([0.0, 0.0, 1.0, 0.0])
                continue
            plane_eqs[polygon_id] = _fit_plane(vertices[np.unique(polygon_triangles)])
    data = PrototypeData(
        vertices=vertices,
        triangles=triangles,
        triangle_polygons=triangle_polygons,
        polygon_classes=polygon_classes,
        plane_eqs=plane_eqs,
        class_names=_class_names_from_state(state.get("class_names")),
    )
    if len(data.vertices) == 0 or len(data.triangles) == 0 or len(data.polygon_classes) == 0:
        raise SceneGraphError("prototype geometry is empty")
    if not np.isfinite(data.vertices).all():
        raise SceneGraphError("prototype vertices contain non-finite values")
    if data.plane_eqs.shape != (len(data.polygon_classes), 4):
        raise SceneGraphError(f"prototype plane equations have invalid shape: {data.plane_eqs.shape}")
    if not np.isfinite(data.plane_eqs).all():
        raise SceneGraphError("prototype plane equations contain non-finite values")
    return data


def semantic_polygon_ids(data: PrototypeData, name: str) -> list[int]:
    if name not in data.class_names:
        return []
    class_id = data.class_names.index(name)
    return [int(value) for value in (data.polygon_classes == class_id).nonzero()[0]]


def polygon_geometries(data: PrototypeData) -> dict[int, Any]:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    geometries: dict[int, Any] = {}
    for polygon_id in range(len(data.polygon_classes)):
        pieces = []
        for triangle in data.triangles[data.triangle_polygons == polygon_id]:
            geometry = Polygon(data.vertices[triangle, :2])
            if geometry.is_valid and geometry.area > 1.0e-10:
                pieces.append(geometry)
        geometries[polygon_id] = unary_union(pieces).buffer(0) if pieces else Polygon()
    return geometries


def polygon_elevation(data: PrototypeData, polygon_id: int) -> float:
    import numpy as np

    triangles = data.triangles[data.triangle_polygons == polygon_id]
    if len(triangles) == 0:
        return float("nan")
    return float(np.mean(data.vertices[np.unique(triangles), 2]))


def polygon_height_interval(data: PrototypeData, polygon_id: int) -> tuple[float, float]:
    import numpy as np

    triangles = data.triangles[data.triangle_polygons == polygon_id]
    vertices = data.vertices[np.unique(triangles), 2]
    return float(vertices.min()), float(vertices.max())


def weighted_elevation(polygon_ids: Sequence[int], elevations: Mapping[int, float], geometries: Mapping[int, Any]) -> float:
    import numpy as np

    if not polygon_ids:
        raise SceneGraphError("cannot compute elevation without polygons")
    weights = np.asarray([max(float(geometries[value].area), 1.0e-6) for value in polygon_ids], dtype=np.float64)
    values = np.asarray([elevations[value] for value in polygon_ids], dtype=np.float64)
    return float(np.average(values, weights=weights))


def ceiling_candidate_records(
    data: PrototypeData,
    polygon_ids: Sequence[int],
    geometries: Mapping[int, Any],
    elevations: Mapping[int, float],
    level_id: str,
) -> list[dict[str, Any]]:
    from shapely.geometry import mapping

    records = []
    for polygon_id in polygon_ids:
        geometry = geometries[int(polygon_id)]
        if geometry.is_empty or geometry.area <= 1.0e-10:
            continue
        plane = data.plane_eqs[int(polygon_id)].astype(float)
        normal_length = float((plane[:3] ** 2).sum() ** 0.5)
        if normal_length <= 1.0e-12:
            continue
        plane = plane / normal_length
        records.append(
            {
                "polygon_id": int(polygon_id),
                "level_id": level_id,
                "plane_eq": plane.tolist(),
                "mean_elevation_meters": float(elevations[int(polygon_id)]),
                "area_square_meters": float(geometry.area),
                "geometry": mapping(geometry),
            }
        )
    return records


def build_grid(geometry: Any, resolution: float, padding: int = 10) -> Grid2D:
    import numpy as np

    min_x, min_y, max_x, max_y = geometry.bounds
    margin = padding * resolution
    origin = np.asarray([min_x - margin, min_y - margin], dtype=np.float64)
    width = int(math.ceil((max_x - min_x + 2 * margin) / resolution)) + 1
    height = int(math.ceil((max_y - min_y + 2 * margin) / resolution)) + 1
    if width <= 0 or height <= 0 or width * height > 4_000_000:
        raise SceneGraphError(f"invalid scene graph raster size: {height}x{width}")
    return Grid2D(origin, resolution, (height, width))


def rasterize_geometry(geometry: Any, grid: Grid2D) -> Any:
    import numpy as np
    from shapely import contains_xy

    rows, columns = np.indices(grid.shape)
    points = grid.world_centers(rows.ravel(), columns.ravel())
    mask = contains_xy(geometry, points[:, 0], points[:, 1])
    return mask.reshape(grid.shape)


def rasterize_wall_polygons(data: PrototypeData, polygon_ids: Sequence[int], grid: Grid2D, line_width_meters: float) -> Any:
    import cv2
    import numpy as np

    raster = np.zeros(grid.shape, dtype=np.uint8)
    thickness = max(1, int(round(line_width_meters / grid.resolution_meters)))
    selected = set(int(value) for value in polygon_ids)
    for triangle, polygon_id in zip(data.triangles, data.triangle_polygons):
        if int(polygon_id) not in selected:
            continue
        pixels = grid.pixel_points(data.vertices[triangle, :2])
        cv2.fillConvexPoly(raster, pixels, 255, lineType=cv2.LINE_8)
        cv2.polylines(raster, [pixels], True, 255, thickness=thickness, lineType=cv2.LINE_8)
    if int(raster.max()) == 0:
        return raster.astype(bool)
    blurred = cv2.GaussianBlur(raster, (5, 5), 1)
    _, walls = cv2.threshold(blurred, 0.25 * float(blurred.max()), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)


def multi_scale_room_watershed(
    free_space: Any,
    resolution_meters: float,
    bottleneck_widths_meters: tuple[float, float],
    minimum_seed_area_square_meters: float,
    minimum_room_area_square_meters: float,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import cv2
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import watershed

    free = np.asarray(free_space, dtype=bool)
    distance = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float32)
    distance *= resolution_meters
    marker_image = np.zeros(free.shape, dtype=np.int32)
    marker_count = 0
    minimum_seed_pixels = max(1, int(math.ceil(minimum_seed_area_square_meters / resolution_meters**2)))
    scale_stats = []
    for bottleneck_width in bottleneck_widths_meters:
        seed_mask = free & (distance >= bottleneck_width / 2.0)
        components, component_count = ndimage.label(seed_mask)
        accepted = 0
        for component_id in range(1, component_count + 1):
            component = components == component_id
            if int(np.count_nonzero(component)) < minimum_seed_pixels:
                continue
            overlapping = [int(value) for value in np.unique(marker_image[component]) if value > 0]
            if not overlapping:
                marker_count += 1
                marker_image[component] = marker_count
                accepted += 1
            elif len(overlapping) == 1:
                marker_image[component] = overlapping[0]
        scale_stats.append(
            {
                "bottleneck_width_meters": bottleneck_width,
                "raw_component_count": int(component_count),
                "new_marker_count": accepted,
            }
        )
    if marker_count == 0:
        labels, component_count = ndimage.label(free)
        labels = labels.astype(np.int32)
        stats = {"fallback_connected_components": True, "room_count": int(component_count), "scales": scale_stats}
        return labels, distance, marker_image, stats
    labels = watershed(-distance, marker_image, mask=free).astype(np.int32)
    minimum_room_pixels = max(1, int(math.ceil(minimum_room_area_square_meters / resolution_meters**2)))
    stable = np.zeros_like(labels)
    kept = []
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        if int(np.count_nonzero(labels == label)) >= minimum_room_pixels:
            kept.append(label)
    for new_label, old_label in enumerate(kept, start=1):
        stable[labels == old_label] = new_label
    if not kept:
        stable[free] = 1
        kept = [1]
    stats = {
        "fallback_connected_components": False,
        "scales": scale_stats,
        "marker_count": marker_count,
        "room_count": len(kept),
        "minimum_seed_area_square_meters": minimum_seed_area_square_meters,
        "minimum_room_area_square_meters": minimum_room_area_square_meters,
    }
    return stable, distance, marker_image, stats


def mask_geometry(mask: Any, grid: Grid2D) -> Any:
    import cv2
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    contours, hierarchy = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return Polygon()
    hierarchy = hierarchy[0]
    polygons = []

    def world(contour: Any) -> list[tuple[float, float]]:
        pixels = contour[:, 0, :]
        points = grid.world_centers(pixels[:, 1], pixels[:, 0])
        return [tuple(value) for value in points]

    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1 or len(contour) < 3:
            continue
        holes = []
        child = hierarchy[index][2]
        while child != -1:
            if len(contours[child]) >= 3:
                holes.append(world(contours[child]))
            child = hierarchy[child][0]
        polygon = Polygon(world(contour), holes).buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)
    return unary_union(polygons).buffer(0) if polygons else Polygon()


def opening_edges(labels: Any, grid: Grid2D, door_maximum_width_meters: float, room_id_by_label: Mapping[int, str]) -> list[dict[str, Any]]:
    import numpy as np

    boundary_points: dict[tuple[int, int], list[Any]] = defaultdict(list)
    left, right = labels[:, :-1], labels[:, 1:]
    rows, columns = np.where((left > 0) & (right > 0) & (left != right))
    for row, column in zip(rows, columns):
        pair = tuple(sorted((int(left[row, column]), int(right[row, column]))))
        boundary_points[pair].append(grid.world_centers(np.asarray([row]), np.asarray([column + 0.5]))[0])
    bottom, top = labels[:-1, :], labels[1:, :]
    rows, columns = np.where((bottom > 0) & (top > 0) & (bottom != top))
    for row, column in zip(rows, columns):
        pair = tuple(sorted((int(bottom[row, column]), int(top[row, column]))))
        boundary_points[pair].append(grid.world_centers(np.asarray([row + 0.5]), np.asarray([column]))[0])
    edges = []
    for edge_index, (pair, raw_points) in enumerate(sorted(boundary_points.items())):
        if pair[0] not in room_id_by_label or pair[1] not in room_id_by_label:
            continue
        points = np.asarray(raw_points, dtype=np.float64)
        center = points.mean(axis=0)
        if len(points) > 1:
            _, _, axes = np.linalg.svd(points - center, full_matrices=False)
            axis = axes[0]
            parameters = (points - center) @ axis
            first = center + axis * float(parameters.min())
            second = center + axis * float(parameters.max())
            width = max(float(parameters.max() - parameters.min()), grid.resolution_meters)
        else:
            first = second = center
            width = grid.resolution_meters
        edges.append(
            {
                "edge_id": f"opening_{edge_index:03d}",
                "kind": "door" if width < door_maximum_width_meters else "opening",
                "room_ids": [room_id_by_label[pair[0]], room_id_by_label[pair[1]]],
                "width_meters": width,
                "line_xy": [first.tolist(), second.tolist()],
                "boundary_sample_count": len(points),
            }
        )
    return edges


def _write_label_preview(path: Path, labels: Any) -> None:
    import cv2
    import numpy as np

    color = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        hue = int(round(179 * ((0.6180339887498949 * label) % 1.0)))
        color[labels == label] = cv2.cvtColor(np.uint8([[[hue, 180, 242]]]), cv2.COLOR_HSV2BGR)[0, 0]
    cv2.imwrite(str(path), color)


def _geojson_feature_collection(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def _polygon_feature(identifier: str, geometry: Any, properties: Mapping[str, Any]) -> dict[str, Any]:
    from shapely.geometry import mapping

    return {"type": "Feature", "id": identifier, "geometry": mapping(geometry), "properties": dict(properties)}


def _load_stair_regions(stair_mesh_path: Path | None, rooms: Sequence[dict[str, Any]], room_geometries: Mapping[str, Any], config: SceneGraphConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    from shapely.geometry import MultiPoint, Point, Polygon

    if stair_mesh_path is None or not stair_mesh_path.is_file():
        return [], [{"status": "missing_stair_mesh"}]
    try:
        import open3d as o3d
    except ImportError:
        return [], [{"status": "open3d_unavailable"}]
    mesh = o3d.io.read_triangle_mesh(str(stair_mesh_path), enable_post_processing=False)
    if len(mesh.triangles) == 0:
        return [], [{"status": "empty_stair_mesh"}]
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    clusters, counts, areas = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    regions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for component_id, triangle_count in enumerate(counts):
        record: dict[str, Any] = {
            "component_id": int(component_id),
            "triangle_count": int(triangle_count),
            "surface_area_square_meters": float(areas[component_id]),
        }
        if int(triangle_count) < config.stair_minimum_triangles:
            record["status"] = "rejected_too_few_triangles"
            diagnostics.append(record)
            continue
        component_vertices = vertices[np.unique(triangles[clusters == component_id])]
        rectangle = MultiPoint(component_vertices[:, :2]).minimum_rotated_rectangle
        if not isinstance(rectangle, Polygon) or rectangle.area <= 1.0e-8:
            record["status"] = "rejected_degenerate_rectangle"
            diagnostics.append(record)
            continue
        coords = np.asarray(rectangle.exterior.coords[:-1], dtype=np.float64)
        centroid = Point(float(coords[:, 0].mean()), float(coords[:, 1].mean()))
        room_distances = [(room_geometries[str(room["room_id"])].distance(centroid), str(room["room_id"])) for room in rooms]
        assigned = [room_id for distance, room_id in room_distances if distance <= config.stair_room_maximum_distance_meters]
        if not assigned and room_distances:
            assigned = [min(room_distances)[1]]
        region = {
            "stair_region_id": f"stair_region_{len(regions):03d}",
            "level_id": "level_0",
            "room_ids": sorted(set(assigned)),
            "rectangle_xy": coords.tolist(),
            "minimum_z": float(component_vertices[:, 2].min()),
            "maximum_z": float(component_vertices[:, 2].max()),
            "source_component_id": int(component_id),
        }
        regions.append(region)
        record.update({"status": "accepted", "assigned_room_ids": region["room_ids"]})
        diagnostics.append(record)
    return regions, diagnostics


def _load_window_candidates(
    skeleton_dir: Path,
    wall_ids: Sequence[int],
    wall_geometries: Mapping[int, Any],
    room_geometries: Mapping[str, Any],
    floor_height: float,
    ceiling_height: float,
    config: SceneGraphConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    from shapely.geometry import Point

    try:
        import open3d as o3d
        from sklearn.cluster import DBSCAN
    except ImportError:
        return [], {"status": "dependencies_unavailable"}
    mesh_path = skeleton_dir / "semantic_mesh.ply"
    labels_path = skeleton_dir / "vertex_hard_assignments.npy"
    names_path = skeleton_dir / "simplified_segmentation_labels.npy"
    if not mesh_path.is_file() or not labels_path.is_file() or not names_path.is_file():
        return [], {"status": "missing_skeleton_semantic_artifacts"}
    names = tuple(str(value) for value in np.load(names_path, allow_pickle=False))
    wanted = [name for name in ("inaccurate_window", "inaccurate_outdoor") if name in names]
    if not wanted:
        return [], {"status": "no_window_evidence_labels", "label_names": list(names)}
    labels = np.load(labels_path, allow_pickle=False)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    if len(vertices) != len(labels):
        return [], {"status": "semantic_mesh_label_length_mismatch"}
    mask = np.isin(labels, [names.index(name) for name in wanted])
    points = vertices[mask]
    points = points[(points[:, 2] >= floor_height) & (points[:, 2] <= ceiling_height)]
    windows: list[dict[str, Any]] = []
    diagnostics = {"status": "complete", "evidence_point_count": int(len(points)), "wall_records": []}
    if len(points) == 0:
        return windows, diagnostics
    for wall_id in wall_ids:
        geometry = wall_geometries[int(wall_id)]
        if geometry.is_empty:
            continue
        distances = np.asarray([geometry.distance(Point(float(x), float(y))) for x, y in points[:, :2]])
        wall_points = points[distances <= config.window_wall_distance_meters]
        record = {"wall_polygon_id": int(wall_id), "candidate_point_count": int(len(wall_points))}
        if len(wall_points) < config.window_dbscan_minimum_samples:
            record["accepted_windows"] = 0
            diagnostics["wall_records"].append(record)
            continue
        center_xy = wall_points[:, :2].mean(axis=0)
        _, _, axes = np.linalg.svd(wall_points[:, :2] - center_xy, full_matrices=False)
        axis = axes[0]
        u = (wall_points[:, :2] - center_xy) @ axis
        clustered = np.column_stack((u, wall_points[:, 2]))
        labels_db = DBSCAN(eps=config.window_dbscan_epsilon_meters, min_samples=config.window_dbscan_minimum_samples).fit_predict(clustered)
        accepted = 0
        for label in sorted(int(value) for value in np.unique(labels_db) if value >= 0):
            cluster = wall_points[labels_db == label]
            if len(cluster) < config.window_dbscan_minimum_samples:
                continue
            cluster_u = (cluster[:, :2] - center_xy) @ axis
            u0, u1 = float(cluster_u.min()), float(cluster_u.max())
            z0, z1 = float(cluster[:, 2].min()), float(cluster[:, 2].max())
            width, height = u1 - u0, z1 - z0
            if width <= config.window_minimum_size_meters or height <= config.window_minimum_size_meters:
                continue
            first = center_xy + u0 * axis
            second = center_xy + u1 * axis
            rectangle = np.asarray([[first[0], first[1], z0], [second[0], second[1], z0], [second[0], second[1], z1], [first[0], first[1], z1]])
            midpoint = Point(float(rectangle[:, 0].mean()), float(rectangle[:, 1].mean()))
            room_ids = sorted(room_id for room_id, room_geometry in room_geometries.items() if room_geometry.boundary.distance(midpoint) <= config.window_wall_distance_meters)
            windows.append(
                {
                    "window_id": f"window_{len(windows):03d}",
                    "level_id": "level_0",
                    "source_wall_polygon_id": int(wall_id),
                    "room_ids": room_ids,
                    "point_count": int(len(cluster)),
                    "width_meters": width,
                    "height_meters": height,
                    "vertices": rectangle.tolist(),
                }
            )
            accepted += 1
        record["accepted_windows"] = accepted
        diagnostics["wall_records"].append(record)
    diagnostics["accepted_window_count"] = len(windows)
    return windows, diagnostics


def run_scene_graph(
    prototype: Path,
    skeleton: Path,
    output: Path,
    config: SceneGraphConfig,
    *,
    overwrite: bool = False,
    command: list[str] | None = None,
) -> Path:
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    output = output.expanduser()
    if output.exists():
        if not overwrite:
            raise SceneGraphError(f"scene_graph output already exists: {output}")
        _remove_existing_output(output)
    output.mkdir(parents=True, exist_ok=False)
    debug_dir = output / "debug"
    debug_dir.mkdir()
    _write_status(output, "running", "loading prototype")
    try:
        import numpy as np
        from shapely.geometry import LineString, mapping
        from shapely.ops import unary_union

        prototype_dir = _component_dir(prototype, "prototype")
        skeleton_dir = _component_dir(skeleton, "skeleton")
        state_path = _required_file(prototype_dir / "polygon_set_3d.pt", "prototype state")
        mesh_path = _required_file(prototype_dir / "fitted_mesh.ply", "prototype mesh")
        data = load_prototype_data(state_path)
        geometries = polygon_geometries(data)
        elevations = {polygon_id: polygon_elevation(data, polygon_id) for polygon_id in range(len(data.polygon_classes))}
        floor_ids = semantic_polygon_ids(data, "floor")
        ceiling_ids = semantic_polygon_ids(data, "ceiling")
        wall_ids = semantic_polygon_ids(data, "wall")
        if not floor_ids:
            raise SceneGraphError("prototype contains no floor polygons")
        floor_height = config.floor_height if config.floor_height is not None else weighted_elevation(floor_ids, elevations, geometries)
        floor_height_clusters = sorted(float(elevations[polygon_id]) for polygon_id in floor_ids)
        assigned_ceilings = [polygon_id for polygon_id in ceiling_ids if elevations[polygon_id] - floor_height >= config.ceiling_minimum_clearance_meters]
        footprint_sources = [*floor_ids, *assigned_ceilings]
        footprint = unary_union([geometries[polygon_id] for polygon_id in footprint_sources]).buffer(0)
        if footprint.is_empty:
            raise SceneGraphError("single-floor footprint is empty")
        ceiling_height = (
            weighted_elevation(assigned_ceilings, elevations, geometries)
            if assigned_ceilings
            else floor_height + config.wall_interval_height_meters
        )
        
        ceiling_candidates = ceiling_candidate_records(data, assigned_ceilings, 
                                                       geometries, elevations, "level_0")
        
        selected_walls = []
        for polygon_id in wall_ids:
            minimum_z, maximum_z = polygon_height_interval(data, polygon_id)
            if maximum_z >= floor_height and minimum_z <= floor_height + config.wall_interval_height_meters and geometries[polygon_id].intersects(footprint):
                selected_walls.append(polygon_id)
        grid = build_grid(footprint, config.grid_resolution_meters)
        support = rasterize_geometry(footprint, grid)
        walls = rasterize_wall_polygons(data, selected_walls, grid, config.wall_line_width_meters) & support
        free_space = support & ~walls
        if int(np.count_nonzero(free_space)) == 0:
            raise SceneGraphError("room segmentation has no free-space pixels")
        labels, distance, markers, watershed_stats = multi_scale_room_watershed(
            free_space,
            grid.resolution_meters,
            config.bottleneck_widths_meters,
            config.minimum_seed_area_square_meters,
            config.minimum_room_area_square_meters,
        )
        rooms = []
        room_id_by_label: dict[int, str] = {}
        room_geometries: dict[str, Any] = {}
        for room_label in sorted(int(value) for value in np.unique(labels) if value > 0):
            geometry = mask_geometry(labels == room_label, grid)
            if geometry.is_empty:
                continue
            room_id = f"level_0_room_{room_label - 1:03d}"
            room_id_by_label[room_label] = room_id
            room_geometries[room_id] = geometry
            rooms.append(
                {
                    "room_id": room_id,
                    "level_id": "level_0",
                    "room_label": room_label,
                    "area_square_meters": float(geometry.area),
                    "centroid_xy": list(geometry.centroid.coords)[0],
                    "semantic_type": "unknown",
                    "semantic_score": None,
                    "pruned": False,
                }
            )
        if not rooms:
            raise SceneGraphError("room segmentation produced no rooms")
        edges = opening_edges(labels, grid, config.door_maximum_width_meters, room_id_by_label)
        for edge in edges:
            edge["level_id"] = "level_0"
        stair_mesh_path = None
        if (skeleton_dir / "stair_mesh.ply").is_file():
            stair_mesh_path = (skeleton_dir / "stair_mesh.ply").resolve()
        stair_regions, stair_diagnostics = _load_stair_regions(stair_mesh_path, rooms, room_geometries, config)
        windows, window_diagnostics = _load_window_candidates(
            skeleton_dir,
            selected_walls,
            geometries,
            room_geometries,
            floor_height,
            ceiling_height,
            config,
        )
        level = {
            "level_id": "level_0",
            "floor_height": floor_height,
            "elevation_meters": floor_height,
            "ceiling_elevation_meters": ceiling_height,
            "floorplan_ref": "floorplan.geojson",
            "ceiling_candidates_ref": "ceiling_candidates.json",
            "floor_polygon_ids": floor_ids,
            "ceiling_polygon_ids": assigned_ceilings,
            "wall_polygon_ids": selected_walls,
            "room_ids": [room["room_id"] for room in rooms],
            "footprint_area_square_meters": float(footprint.area),
            "single_floor_assumption": True,
        }
        graph = {
            "schema_version": 1,
            "coordinate_system": "Z-up meters; room geometry is world XY",
            "single_floor_assumption": True,
            "levels": [level],
            "rooms": rooms,
            "edges": edges,
            "active_room_ids": [room["room_id"] for room in rooms],
            "pruned_room_ids": [],
            "windows": windows,
            "stair_regions": stair_regions,
        }
        floorplan_path = output / "floorplan.geojson"
        rooms_path = output / "rooms.geojson"
        openings_path = output / "openings.geojson"
        graph_path = output / "graph.json"
        ceiling_candidates_path = output / "ceiling_candidates.json"
        ceiling_candidates_mesh_path = output / "ceiling_candidates.ply"
        windows_path = output / "windows.json"
        stairs_path = output / "stairs.json"
        diagnostics_path = output / "diagnostics.json"
        _write_json(floorplan_path, _geojson_feature_collection([_polygon_feature("level_0", footprint, level)]))
        _write_json(
            rooms_path,
            _geojson_feature_collection([_polygon_feature(str(room["room_id"]), room_geometries[str(room["room_id"])], room) for room in rooms]),
        )
        _write_json(
            openings_path,
            _geojson_feature_collection(
                [
                    {
                        "type": "Feature",
                        "id": edge["edge_id"],
                        "geometry": mapping(LineString(edge["line_xy"])),
                        "properties": edge,
                    }
                    for edge in edges
                ]
            ),
        )
        _write_json(graph_path, graph)
        _write_json(ceiling_candidates_path, ceiling_candidates)
        ceiling_candidate_mesh_counts = write_ceiling_candidate_ply(
            ceiling_candidates_mesh_path,
            data.vertices,
            data.triangles,
            data.triangle_polygons,
            assigned_ceilings,
        )
        _write_json(windows_path, windows)
        _write_json(stairs_path, stair_regions)
        _write_json(
            diagnostics_path,
            {
                "grid_shape": list(grid.shape),
                "grid_origin_xy": grid.origin_xy.tolist(),
                "grid_resolution_meters": grid.resolution_meters,
                "wall_pixel_count": int(np.count_nonzero(walls)),
                "support_pixel_count": int(np.count_nonzero(support)),
                "watershed": watershed_stats,
                "stair_detection": stair_diagnostics,
                "window_detection": window_diagnostics,
                "floor_height_clusters": floor_height_clusters,
                "ceiling_candidate_count": len(ceiling_candidates),
            },
        )
        np.savez_compressed(
            output / "room_grid.npz",
            labels=labels,
            distance_meters=distance,
            markers=markers,
            walls=walls.astype(np.uint8),
            support=support.astype(np.uint8),
            origin_xy=grid.origin_xy,
            resolution_meters=np.asarray(grid.resolution_meters),
        )
        _write_label_preview(debug_dir / "room_segmentation.png", labels)
        validation = {
            "single_level_declared": len(graph["levels"]) == 1 and graph["levels"][0]["level_id"] == "level_0",
            "floorplan_nonempty": not footprint.is_empty and footprint.area > 0,
            "room_count_positive": len(rooms) > 0,
            "all_room_geometries_nonempty": all(not geometry.is_empty for geometry in room_geometries.values()),
            "edge_room_references_valid": all(all(room_id in room_geometries for room_id in edge["room_ids"]) for edge in edges),
            "windows_reference_known_rooms_when_present": all(all(room_id in room_geometries for room_id in window.get("room_ids", [])) for window in windows),
            "ceiling_candidates_reference_known_polygons": all(int(candidate["polygon_id"]) in assigned_ceilings for candidate in ceiling_candidates),
            "no_ground_truth_inputs_used": True,
        }
        if not all(validation.values()):
            raise SceneGraphError(f"scene graph validation failed: {validation}")
        manifest = {
            "schema_version": 1,
            "component": "scene_graph",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command,
            "inputs": {
                "prototype_state": _record(state_path),
                "prototype_mesh": _record(mesh_path),
                "skeleton_dir": str(skeleton_dir),
                **({"stair_mesh": _record(stair_mesh_path)} if stair_mesh_path is not None else {}),
            },
            "algorithm": {
                "paper_sections": ["4.4", "Appendix D.2", "Appendix D.3", "Appendix D.5"],
                "skipped_sections": ["Appendix D.1 floor identification", "Appendix D.4 OpenSeg/CLIP room classification"],
                "single_floor_assumption": True,
                "configuration": asdict(config),
            },
            "counts": {
                "levels": 1,
                "rooms": len(rooms),
                "doors": sum(edge["kind"] == "door" for edge in edges),
                "openings": sum(edge["kind"] == "opening" for edge in edges),
                "stair_regions": len(stair_regions),
                "windows": len(windows),
                "floor_polygons": len(floor_ids),
                "ceiling_polygons": len(assigned_ceilings),
                "ceiling_candidates": len(ceiling_candidates),
                **(
                    {
                        "ceiling_candidate_mesh_vertices": ceiling_candidate_mesh_counts["vertices"],
                        "ceiling_candidate_mesh_triangles": ceiling_candidate_mesh_counts["triangles"],
                    }
                    if ceiling_candidate_mesh_counts is not None
                    else {}
                ),
                "wall_polygons": len(selected_walls),
            },
            "outputs": {
                "graph": _record(graph_path),
                "ceiling_candidates": _record(ceiling_candidates_path),
                **(
                    {"ceiling_candidates_mesh": _record(ceiling_candidates_mesh_path)}
                    if ceiling_candidate_mesh_counts is not None
                    else {}
                ),
                "floorplan": _record(floorplan_path),
                "rooms": _record(rooms_path),
                "openings": _record(openings_path),
                "windows": _record(windows_path),
                "stairs": _record(stairs_path),
                "diagnostics": _record(diagnostics_path),
                "room_grid": _record(output / "room_grid.npz"),
                "room_segmentation_preview": _record(debug_dir / "room_segmentation.png"),
            },
            "validation": validation,
            "warnings": [
                "This implementation assumes a single floor and does not run Appendix D.1 floor identification.",
                "Room classification defaults to unknown; OpenSeg/CLIP pruning is intentionally not required for the first active 4.4 path.",
                "Window detection uses skeleton semantic evidence when available and otherwise produces an empty window list.",
            ],
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
        }
        manifest_path = output / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(output, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(output, "failed", f"{type(error).__name__}: {error}")
        if isinstance(error, SceneGraphError):
            raise
        raise SceneGraphError(f"{type(error).__name__}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construct a single-floor Section 4.4 scene graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-resolution-meters", type=float, default=0.05)
    parser.add_argument("--wall-line-width-meters", type=float, default=0.12)
    parser.add_argument("--floor-height", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = SceneGraphConfig(
        grid_resolution_meters=args.grid_resolution_meters,
        wall_line_width_meters=args.wall_line_width_meters,
        floor_height=args.floor_height,
    )
    print(run_scene_graph(args.prototype, args.skeleton, args.output, config, overwrite=args.overwrite, command=os.sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
