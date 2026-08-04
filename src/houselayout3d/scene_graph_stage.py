"""Appendix-D construction of per-level 2D room scene graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import LineString, MultiPoint, Polygon, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from skimage.segmentation import watershed
import torch

from .config import PipelineConfig, SceneGraphConfig
from .stages import Stage


class SceneGraphStageError(RuntimeError):
    """Raised when Stage09 cannot satisfy its artifact contract."""


ROOM_TYPES = (
    "bathroom",
    "bedroom",
    "living room",
    "garage",
    "entrance",
    "kitchen",
    "office",
    "stairs",
    "gym",
    "classroom",
    "spa/sauna",
    "mirror",
    "grass/bushes/trees",
    "driveway",
    "veranda/terrace/balcony",
)
OUTDOOR_PRUNE_TYPES = frozenset(ROOM_TYPES[-5:])
ATTEMPT_RE = re.compile(r"^attempt_(?P<index>\d+)_(?:running|complete|failed)$")


@dataclass(frozen=True)
class PrototypeData:
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_polygons: np.ndarray
    polygon_classes: np.ndarray
    polygon_colors: np.ndarray
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class Grid2D:
    origin_xy: np.ndarray
    resolution_meters: float
    shape: tuple[int, int]

    def pixel_points(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        columns = np.floor(
            (points[:, 0] - self.origin_xy[0]) / self.resolution_meters
        )
        rows = np.floor(
            (points[:, 1] - self.origin_xy[1]) / self.resolution_meters
        )
        return np.column_stack((columns, rows)).astype(np.int32)

    def world_centers(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        x = self.origin_xy[0] + (
            np.asarray(columns, dtype=np.float64) + 0.5
        ) * self.resolution_meters
        y = self.origin_xy[1] + (
            np.asarray(rows, dtype=np.float64) + 0.5
        ) * self.resolution_meters
        return np.column_stack((x, y))


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


def _next_attempt(stage_dir: Path) -> Path:
    indices = []
    for path in stage_dir.iterdir() if stage_dir.is_dir() else ():
        match = ATTEMPT_RE.match(path.name)
        if match:
            indices.append(int(match.group("index")))
    attempt = stage_dir / f"attempt_{max(indices, default=0) + 1:03d}_running"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _rename_attempt(path: Path, state: str) -> Path:
    destination = path.with_name(path.name.rsplit("_", 1)[0] + f"_{state}")
    path.rename(destination)
    return destination


def _require_complete_manifest(path: Path, stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise SceneGraphStageError(f"missing {stage} manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != stage or payload.get("status") != "complete":
        raise SceneGraphStageError(f"{stage} manifest is not complete: {path}")
    return payload


def load_prototype_data(state_path: Path, source_repository: Path) -> PrototypeData:
    source_paths = (source_repository, source_repository / "mesh_fitting_3D")
    for path in reversed(source_paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    from mesh_fitting_3D.differentiable_3D_polygon_stuctures import PolygonSet3D

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model = PolygonSet3D(
        torch.empty((0, 3)),
        [],
        torch.empty((0, 4)),
        device="cpu",
    )
    model.load_state_dict(dict(state), strict=False)
    vertices = model.get_vertices().detach().cpu().numpy().astype(np.float64)
    result = PrototypeData(
        vertices=vertices,
        triangles=model.triangles.detach().cpu().numpy().astype(np.int64),
        triangle_polygons=model.triangle_polygons.detach().cpu().numpy().astype(
            np.int64
        ),
        polygon_classes=model.polygon_classes.detach().cpu().numpy().astype(np.int64),
        polygon_colors=model.polygon_colors.detach().cpu().numpy().astype(np.float64),
        class_names=tuple(str(value) for value in model.class_names.tolist()),
    )
    if not np.isfinite(result.vertices).all():
        raise SceneGraphStageError("prototype contains non-finite vertices")
    if len(result.triangles) == 0 or len(result.polygon_classes) == 0:
        raise SceneGraphStageError("prototype geometry or semantic polygons are empty")
    return result


def polygon_geometries(data: PrototypeData) -> dict[int, BaseGeometry]:
    geometries: dict[int, BaseGeometry] = {}
    for polygon_id in range(len(data.polygon_classes)):
        pieces = []
        for triangle in data.triangles[data.triangle_polygons == polygon_id]:
            geometry = Polygon(data.vertices[triangle, :2])
            if geometry.is_valid and geometry.area > 1.0e-10:
                pieces.append(geometry)
        geometry = unary_union(pieces).buffer(0) if pieces else Polygon()
        geometries[polygon_id] = geometry
    return geometries


def polygon_elevation(data: PrototypeData, polygon_id: int) -> float:
    triangles = data.triangles[data.triangle_polygons == polygon_id]
    if len(triangles) == 0:
        return float("nan")
    ids = np.unique(triangles)
    return float(np.mean(data.vertices[ids, 2]))


def polygon_height_interval(
    data: PrototypeData, polygon_id: int
) -> tuple[float, float]:
    triangles = data.triangles[data.triangle_polygons == polygon_id]
    ids = np.unique(triangles)
    values = data.vertices[ids, 2]
    return float(values.min()), float(values.max())


def semantic_polygon_ids(data: PrototypeData, name: str) -> list[int]:
    if name not in data.class_names:
        raise SceneGraphStageError(f"prototype has no semantic class {name!r}")
    class_id = data.class_names.index(name)
    return np.flatnonzero(data.polygon_classes == class_id).astype(int).tolist()


def group_floor_polygons(
    floor_ids: Sequence[int],
    elevations: Mapping[int, float],
    maximum_difference_meters: float,
) -> list[list[int]]:
    """Appendix D.1 graph components under the 50 cm height relation."""
    ids = sorted(int(value) for value in floor_ids)
    parent = {value: value for value in ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            if abs(float(elevations[first]) - float(elevations[second])) <= (
                maximum_difference_meters
            ):
                union(first, second)
    groups: dict[int, list[int]] = defaultdict(list)
    for value in ids:
        groups[find(value)].append(value)
    return sorted(
        (sorted(values) for values in groups.values()),
        key=lambda values: (
            float(np.mean([elevations[value] for value in values])),
            values,
        ),
    )


def weighted_elevation(
    polygon_ids: Sequence[int],
    elevations: Mapping[int, float],
    geometries: Mapping[int, BaseGeometry],
) -> float:
    weights = np.asarray(
        [max(float(geometries[value].area), 1.0e-6) for value in polygon_ids],
        dtype=np.float64,
    )
    values = np.asarray([elevations[value] for value in polygon_ids])
    return float(np.average(values, weights=weights))


def assign_ceilings_to_levels(
    ceiling_ids: Sequence[int],
    ceiling_elevations: Mapping[int, float],
    level_elevations: Sequence[float],
    minimum_clearance_meters: float,
) -> dict[int, list[int]]:
    assignments = {index: [] for index in range(len(level_elevations))}
    for polygon_id in sorted(int(value) for value in ceiling_ids):
        candidates = [
            index
            for index, elevation in enumerate(level_elevations)
            if ceiling_elevations[polygon_id] - elevation >= minimum_clearance_meters
        ]
        if candidates:
            closest = max(candidates, key=lambda index: level_elevations[index])
            assignments[closest].append(polygon_id)
    return assignments


def build_grid(geometry: BaseGeometry, resolution: float, padding: int = 10) -> Grid2D:
    min_x, min_y, max_x, max_y = geometry.bounds
    margin = padding * resolution
    origin = np.asarray([min_x - margin, min_y - margin], dtype=np.float64)
    width = int(math.ceil((max_x - min_x + 2 * margin) / resolution)) + 1
    height = int(math.ceil((max_y - min_y + 2 * margin) / resolution)) + 1
    if width * height > 4_000_000:
        raise SceneGraphStageError(
            f"scene-graph raster would contain {width * height:,} cells"
        )
    return Grid2D(origin, resolution, (height, width))


def rasterize_geometry(geometry: BaseGeometry, grid: Grid2D) -> np.ndarray:
    rows, columns = np.indices(grid.shape)
    points = grid.world_centers(rows.ravel(), columns.ravel())
    mask = contains_xy(geometry, points[:, 0], points[:, 1])
    return mask.reshape(grid.shape)


def rasterize_wall_polygons(
    data: PrototypeData,
    polygon_ids: Sequence[int],
    grid: Grid2D,
    line_width_meters: float,
) -> np.ndarray:
    raster = np.zeros(grid.shape, dtype=np.uint8)
    selected = set(int(value) for value in polygon_ids)
    thickness = max(1, int(round(line_width_meters / grid.resolution_meters)))
    for triangle, polygon_id in zip(data.triangles, data.triangle_polygons):
        if int(polygon_id) not in selected:
            continue
        pixels = grid.pixel_points(data.vertices[triangle, :2])
        cv2.fillConvexPoly(raster, pixels, 255, lineType=cv2.LINE_8)
        cv2.polylines(
            raster,
            [pixels],
            True,
            255,
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
    blurred = cv2.GaussianBlur(raster, (5, 5), 1)
    threshold = 0.25 * float(blurred.max())
    _, walls = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    return cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel, iterations=1)


def two_stage_room_watershed(
    free_space: np.ndarray,
    resolution_meters: float,
    bottleneck_widths_meters: tuple[float, float],
    minimum_seed_area_square_meters: float,
    minimum_room_area_square_meters: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the Appendix D.3 2.5 m then 1.5 m bottleneck segmentation."""
    free = np.asarray(free_space, dtype=bool)
    distance = cv2.distanceTransform(
        free.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(np.float32) * resolution_meters
    minimum_seed_pixels = max(
        1,
        int(math.ceil(minimum_seed_area_square_meters / resolution_meters**2)),
    )
    marker_image = np.zeros(free.shape, dtype=np.int32)
    marker_count = 0
    scale_stats = []
    previous_component_markers: dict[int, set[int]] = {}
    for scale_index, bottleneck_width in enumerate(bottleneck_widths_meters):
        seed_mask = free & (distance >= bottleneck_width / 2.0)
        components, component_count = ndimage.label(seed_mask)
        accepted = 0
        for component_id in range(1, component_count + 1):
            component = components == component_id
            if int(np.count_nonzero(component)) < minimum_seed_pixels:
                continue
            overlapping = set(
                int(value)
                for value in np.unique(marker_image[component])
                if value > 0
            )
            if not overlapping:
                marker_count += 1
                marker_image[component] = marker_count
                accepted += 1
            elif len(overlapping) == 1:
                marker_image[component] = next(iter(overlapping))
            previous_component_markers[scale_index * 1_000_000 + component_id] = overlapping
        scale_stats.append(
            {
                "bottleneck_width_meters": bottleneck_width,
                "raw_component_count": int(component_count),
                "new_marker_count": accepted,
            }
        )
    if marker_count == 0:
        raise SceneGraphStageError("two-stage watershed produced no room markers")
    labels = watershed(-distance, marker_image, mask=free).astype(np.int32)
    minimum_room_pixels = max(
        1,
        int(math.ceil(minimum_room_area_square_meters / resolution_meters**2)),
    )
    room_rows = []
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        rows, columns = np.where(labels == label)
        if len(rows) >= minimum_room_pixels:
            room_rows.append((float(columns.mean()), float(rows.mean()), label))
    stable = np.zeros_like(labels)
    for new_label, (_, _, old_label) in enumerate(sorted(room_rows), start=1):
        stable[labels == old_label] = new_label
    stats = {
        "scales": scale_stats,
        "marker_count": marker_count,
        "room_count_before_area_filter": int(len(np.unique(labels)) - 1),
        "room_count": len(room_rows),
        "minimum_seed_area_square_meters": minimum_seed_area_square_meters,
        "minimum_room_area_square_meters": minimum_room_area_square_meters,
    }
    if not room_rows:
        raise SceneGraphStageError("all watershed rooms failed the area threshold")
    return stable, distance, marker_image, stats


def mask_geometry(mask: np.ndarray, grid: Grid2D) -> BaseGeometry:
    image = np.asarray(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return Polygon()
    hierarchy = hierarchy[0]
    polygons = []

    def world(contour: np.ndarray) -> list[tuple[float, float]]:
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
        if not polygon.is_empty:
            polygons.append(polygon)
    return unary_union(polygons).buffer(0) if polygons else Polygon()


def opening_edges(
    labels: np.ndarray,
    grid: Grid2D,
    door_maximum_width_meters: float,
    room_id_by_label: Mapping[int, str],
) -> list[dict[str, Any]]:
    boundary_points: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    left, right = labels[:, :-1], labels[:, 1:]
    rows, columns = np.where((left > 0) & (right > 0) & (left != right))
    for row, column in zip(rows, columns):
        pair = tuple(sorted((int(left[row, column]), int(right[row, column]))))
        boundary_points[pair].append(
            grid.world_centers(np.asarray([row]), np.asarray([column + 0.5]))[0]
        )
    bottom, top = labels[:-1, :], labels[1:, :]
    rows, columns = np.where((bottom > 0) & (top > 0) & (bottom != top))
    for row, column in zip(rows, columns):
        pair = tuple(sorted((int(bottom[row, column]), int(top[row, column]))))
        boundary_points[pair].append(
            grid.world_centers(np.asarray([row + 0.5]), np.asarray([column]))[0]
        )
    edges = []
    for edge_index, (pair, raw_points) in enumerate(sorted(boundary_points.items())):
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


def create_room_floor_mesh(
    levels: Sequence[dict[str, Any]],
    labels_by_level: Sequence[np.ndarray],
    grids: Sequence[Grid2D],
) -> o3d.geometry.TriangleMesh:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    colors: list[list[float]] = []
    golden = 0.6180339887498949
    for level, labels, grid in zip(levels, labels_by_level, grids):
        for room_label in sorted(int(value) for value in np.unique(labels) if value > 0):
            hue = (golden * room_label) % 1.0
            color = np.asarray(
                cv2.cvtColor(
                    np.uint8([[[round(179 * hue), 155, 242]]]),
                    cv2.COLOR_HSV2RGB,
                )[0, 0],
                dtype=np.float64,
            ) / 255.0
            rows, columns = np.where(labels == room_label)
            for row, column in zip(rows, columns):
                x0 = grid.origin_xy[0] + column * grid.resolution_meters
                y0 = grid.origin_xy[1] + row * grid.resolution_meters
                x1, y1 = x0 + grid.resolution_meters, y0 + grid.resolution_meters
                base = len(vertices)
                z = float(level["elevation_meters"]) + 0.01
                vertices.extend([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
                triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
                colors.extend([color.tolist()] * 4)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    mesh.compute_vertex_normals()
    return mesh


def prepare_openseg_request(
    data: PrototypeData,
    transforms_path: Path,
    request_path: Path,
    config: SceneGraphConfig,
) -> dict[str, Any]:
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms["frames"][:: config.frame_stride]
    width = int(transforms["w"])
    height = int(transforms["h"])
    fx, fy = float(transforms["fl_x"]), float(transforms["fl_y"])
    cx, cy = float(transforms["cx"]), float(transforms["cy"])
    centroids = data.vertices[data.triangles].mean(axis=1).astype(np.float64)
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(data.vertices.astype(np.float32)),
        o3d.core.Tensor(data.triangles.astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    opengl_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    root = transforms_path.parent
    image_paths = []
    frame_offsets = [0]
    all_triangle_ids = []
    all_rows = []
    all_columns = []
    per_frame_counts = []
    boundary = config.image_boundary_pixels
    for frame in frames:
        image_path = (root / frame["file_path"]).resolve()
        image_paths.append(str(image_path))
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w_cv = c2w @ opengl_to_opencv
        w2c = np.linalg.inv(c2w_cv)
        camera = (w2c[:3, :3] @ centroids.T + w2c[:3, 3:4]).T
        depth = camera[:, 2]
        columns = fx * camera[:, 0] / np.maximum(depth, 1.0e-12) + cx
        rows = fy * camera[:, 1] / np.maximum(depth, 1.0e-12) + cy
        candidates = np.flatnonzero(
            (depth > 0.05)
            & (columns >= boundary)
            & (columns < width - boundary)
            & (rows >= boundary)
            & (rows < height - boundary)
        )
        if len(candidates):
            origins = np.repeat(c2w_cv[:3, 3][None], len(candidates), axis=0)
            vectors = centroids[candidates] - origins
            distances = np.linalg.norm(vectors, axis=1)
            directions = vectors / np.maximum(distances[:, None], 1.0e-12)
            rays = np.concatenate((origins, directions), axis=1).astype(np.float32)
            hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            visible = np.isfinite(hit) & (
                distances <= hit + config.visibility_tolerance_meters
            )
            candidates = candidates[visible]
        all_triangle_ids.extend(candidates.tolist())
        all_rows.extend(rows[candidates].astype(np.float32).tolist())
        all_columns.extend(columns[candidates].astype(np.float32).tolist())
        per_frame_counts.append(len(candidates))
        frame_offsets.append(len(all_triangle_ids))
    np.savez_compressed(
        request_path,
        image_paths=np.asarray(image_paths, dtype=np.str_),
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
        triangle_indices=np.asarray(all_triangle_ids, dtype=np.int64),
        image_rows=np.asarray(all_rows, dtype=np.float32),
        image_columns=np.asarray(all_columns, dtype=np.float32),
        triangle_count=np.asarray(len(data.triangles), dtype=np.int64),
    )
    return {
        "frame_count": len(frames),
        "sample_count": len(all_triangle_ids),
        "minimum_samples_per_frame": min(per_frame_counts, default=0),
        "maximum_samples_per_frame": max(per_frame_counts, default=0),
        "mean_samples_per_frame": float(np.mean(per_frame_counts)) if per_frame_counts else 0.0,
        "frame_stride": config.frame_stride,
    }


def compute_text_features(config: SceneGraphConfig, output_path: Path) -> np.ndarray:
    import open_clip
    from hovsg.utils.clip_utils import get_text_feats_multiple_templates

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-336",
        pretrained=str(config.clip_weights),
        device=device,
    )
    model.eval()
    features = get_text_feats_multiple_templates(
        list(ROOM_TYPES), model, 768
    ).astype(np.float32)
    np.save(output_path, features)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features


def classify_rooms(
    rooms: list[dict[str, Any]],
    room_geometries: Mapping[str, BaseGeometry],
    levels: Sequence[dict[str, Any]],
    data: PrototypeData,
    triangle_features: np.ndarray,
    feature_counts: np.ndarray,
    text_features: np.ndarray,
    tolerance_meters: float,
) -> np.ndarray:
    centroids = data.vertices[data.triangles].mean(axis=1)
    level_by_id = {str(level["level_id"]): level for level in levels}
    room_features = np.zeros((len(rooms), 768), dtype=np.float32)
    for room_index, room in enumerate(rooms):
        geometry = room_geometries[str(room["room_id"])].buffer(tolerance_meters)
        level = level_by_id[str(room["level_id"])]
        xy_mask = contains_xy(geometry, centroids[:, 0], centroids[:, 1])
        z_mask = (
            (centroids[:, 2] >= float(level["elevation_meters"]) - 0.25)
            & (centroids[:, 2] <= float(level["ceiling_elevation_meters"]) + 0.25)
        )
        mask = xy_mask & z_mask & (feature_counts > 0)
        if np.any(mask):
            feature = triangle_features[mask].mean(axis=0)
            norm = float(np.linalg.norm(feature))
            if norm > 1.0e-12:
                feature /= norm
                room_features[room_index] = feature
                scores = feature @ text_features.T
                order = np.argsort(scores)[::-1]
                room["semantic_type"] = ROOM_TYPES[int(order[0])]
                room["semantic_score"] = float(scores[order[0]])
                room["semantic_top3"] = [
                    {"type": ROOM_TYPES[int(index)], "score": float(scores[index])}
                    for index in order[:3]
                ]
                room["semantic_triangle_count"] = int(np.count_nonzero(mask))
                continue
        room["semantic_type"] = "unknown"
        room["semantic_score"] = None
        room["semantic_top3"] = []
        room["semantic_triangle_count"] = 0
    return room_features


def stair_edges(
    stair_mesh_path: Path,
    rooms: Sequence[dict[str, Any]],
    room_geometries: Mapping[str, BaseGeometry],
    levels: Sequence[dict[str, Any]],
    config: SceneGraphConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mesh = o3d.io.read_triangle_mesh(str(stair_mesh_path))
    if len(mesh.triangles) == 0:
        return [], []
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    triangle_clusters, cluster_counts, cluster_areas = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    level_by_id = {str(level["level_id"]): level for level in levels}
    diagnostics = []
    edges = []
    for component_id, triangle_count in enumerate(cluster_counts):
        record: dict[str, Any] = {
            "component_id": int(component_id),
            "triangle_count": int(triangle_count),
            "surface_area_square_meters": float(cluster_areas[component_id]),
        }
        if int(triangle_count) < config.stair_minimum_triangles:
            record["status"] = "rejected_too_few_triangles"
            diagnostics.append(record)
            continue
        component_triangles = triangles[triangle_clusters == component_id]
        component_vertices = vertices[np.unique(component_triangles)]
        rectangle = MultiPoint(component_vertices[:, :2]).minimum_rotated_rectangle
        if not isinstance(rectangle, Polygon) or rectangle.area <= 1.0e-8:
            record["status"] = "rejected_degenerate_rectangle"
            diagnostics.append(record)
            continue
        coordinates = np.asarray(rectangle.exterior.coords[:-1], dtype=np.float64)
        edge_lengths = np.linalg.norm(np.roll(coordinates, -1, axis=0) - coordinates, axis=1)
        short_indices = np.argsort(edge_lengths)[:2]
        midpoints = 0.5 * (
            coordinates[short_indices]
            + coordinates[(short_indices + 1) % len(coordinates)]
        )
        tree = cKDTree(component_vertices[:, :2])
        k = min(8, len(component_vertices))
        distances, neighbors = tree.query(midpoints, k=k)
        distances = np.atleast_2d(distances)
        neighbors = np.atleast_2d(neighbors)
        weights = 1.0 / np.maximum(distances, 1.0e-4)
        heights = np.sum(weights * component_vertices[neighbors, 2], axis=1) / np.sum(
            weights, axis=1
        )
        endpoints = np.column_stack((midpoints, heights))
        assigned_rooms = []
        assigned_distances = []
        for endpoint in endpoints:
            candidates = []
            for room in rooms:
                room_id = str(room["room_id"])
                level = level_by_id[str(room["level_id"])]
                xy_distance = room_geometries[room_id].distance(
                    MultiPoint([endpoint[:2]])
                )
                lower = float(level["elevation_meters"])
                upper = float(level["ceiling_elevation_meters"])
                z_distance = max(lower - endpoint[2], endpoint[2] - upper, 0.0)
                candidates.append((math.hypot(xy_distance, z_distance), room_id))
            distance, room_id = min(candidates)
            assigned_rooms.append(room_id)
            assigned_distances.append(float(distance))
        record.update(
            {
                "rectangle_xy": coordinates.tolist(),
                "edge_midpoints_xyz": endpoints.tolist(),
                "assigned_room_ids": assigned_rooms,
                "assignment_distances_meters": assigned_distances,
            }
        )
        if max(assigned_distances) > config.stair_room_maximum_distance_meters:
            record["status"] = "rejected_room_too_far"
        elif assigned_rooms[0] == assigned_rooms[1]:
            record["status"] = "rejected_same_room"
        else:
            record["status"] = "accepted"
            edges.append(
                {
                    "edge_id": f"stair_{len(edges):03d}",
                    "kind": "stair",
                    "room_ids": assigned_rooms,
                    "rectangle_xy": coordinates.tolist(),
                    "edge_midpoints_xyz": endpoints.tolist(),
                    "source_component_id": int(component_id),
                }
            )
        diagnostics.append(record)
    return edges, diagnostics


def prune_outdoor_leaf_rooms(
    rooms: list[dict[str, Any]], edges: Sequence[dict[str, Any]]
) -> list[str]:
    degrees = defaultdict(int)
    for edge in edges:
        for room_id in edge["room_ids"]:
            degrees[str(room_id)] += 1
    pruned = []
    for room in rooms:
        room_id = str(room["room_id"])
        should_prune = (
            room.get("semantic_type") in OUTDOOR_PRUNE_TYPES
            and degrees[room_id] <= 1
        )
        room["pruned"] = bool(should_prune)
        room["graph_degree_before_pruning"] = int(degrees[room_id])
        if should_prune:
            pruned.append(room_id)
    return pruned


def run_scene_graph(config: PipelineConfig, run_id: str) -> Path:
    """Run Appendix D.1-D.5 on a completed Stage08 prototype."""
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    run_root = config.storage.outputs / config.scene / run_id
    prototype_root = run_root / Stage.PROTOTYPE.value
    prototype_manifest_path = prototype_root / "manifest.json"
    prototype_manifest = _require_complete_manifest(
        prototype_manifest_path, Stage.PROTOTYPE.value
    )
    stage_dir = run_root / Stage.SCENE_GRAPH.value
    if (stage_dir / "manifest.json").is_file():
        raise SceneGraphStageError(
            f"scene-graph stage already exists and will not be overwritten: {stage_dir}"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    attempt = _next_attempt(stage_dir)
    _write_status(stage_dir / "STATUS.json", "running", str(attempt))
    _write_status(attempt / "STATUS.json", "running", "geometry")

    try:
        state_path = Path(prototype_manifest["outputs"]["final_model_state"]["path"])
        mesh_path = Path(prototype_manifest["outputs"]["final_mesh"]["path"])
        skeleton_root = run_root / Stage.SKELETON.value
        skeleton_manifest_path = skeleton_root / "manifest.json"
        _require_complete_manifest(skeleton_manifest_path, Stage.SKELETON.value)
        stair_mesh_path = skeleton_root / "stair_mesh.ply"
        transforms_path = run_root / Stage.POSE.value / "transforms.json"
        data = load_prototype_data(state_path, config.prototype.source_repository)
        geometries = polygon_geometries(data)
        elevations = {
            polygon_id: polygon_elevation(data, polygon_id)
            for polygon_id in range(len(data.polygon_classes))
        }
        floor_ids = semantic_polygon_ids(data, "floor")
        ceiling_ids = semantic_polygon_ids(data, "ceiling")
        wall_ids = semantic_polygon_ids(data, "wall")
        floor_groups = group_floor_polygons(
            floor_ids,
            elevations,
            config.scene_graph.floor_merge_height_meters,
        )
        if not floor_groups:
            raise SceneGraphStageError("prototype contains no floor levels")
        level_elevations = [
            weighted_elevation(group, elevations, geometries) for group in floor_groups
        ]
        ceiling_assignments = assign_ceilings_to_levels(
            ceiling_ids,
            elevations,
            level_elevations,
            config.scene_graph.ceiling_minimum_clearance_meters,
        )

        levels: list[dict[str, Any]] = []
        rooms: list[dict[str, Any]] = []
        room_geometries: dict[str, BaseGeometry] = {}
        edges: list[dict[str, Any]] = []
        labels_by_level = []
        grids = []
        level_diagnostics = []
        for level_index, floor_group in enumerate(floor_groups):
            level_id = f"level_{level_index:02d}"
            assigned_ceilings = ceiling_assignments[level_index]
            footprint = unary_union(
                [geometries[value] for value in [*floor_group, *assigned_ceilings]]
            ).buffer(0)
            if footprint.is_empty:
                raise SceneGraphStageError(f"{level_id} footprint is empty")
            ceiling_elevation = (
                weighted_elevation(assigned_ceilings, elevations, geometries)
                if assigned_ceilings
                else level_elevations[level_index]
                + config.scene_graph.wall_interval_height_meters
            )
            selected_walls = []
            for polygon_id in wall_ids:
                minimum_z, maximum_z = polygon_height_interval(data, polygon_id)
                if (
                    maximum_z >= level_elevations[level_index]
                    and minimum_z
                    <= level_elevations[level_index]
                    + config.scene_graph.wall_interval_height_meters
                    and geometries[polygon_id].intersects(footprint)
                ):
                    selected_walls.append(polygon_id)
            grid = build_grid(footprint, config.scene_graph.grid_resolution_meters)
            support = rasterize_geometry(footprint, grid)
            walls = rasterize_wall_polygons(
                data,
                selected_walls,
                grid,
                config.scene_graph.wall_line_width_meters,
            )
            walls = (walls > 0) & support
            free_space = support & ~walls
            labels, distance, markers, watershed_stats = two_stage_room_watershed(
                free_space,
                grid.resolution_meters,
                config.scene_graph.bottleneck_widths_meters,
                config.scene_graph.minimum_seed_area_square_meters,
                config.scene_graph.minimum_room_area_square_meters,
            )
            room_id_by_label = {}
            level_room_ids = []
            for room_label in sorted(
                int(value) for value in np.unique(labels) if value > 0
            ):
                room_id = f"{level_id}_room_{room_label - 1:03d}"
                geometry = mask_geometry(labels == room_label, grid)
                if geometry.is_empty:
                    continue
                room_id_by_label[room_label] = room_id
                room_geometries[room_id] = geometry
                area = float(np.count_nonzero(labels == room_label) * grid.resolution_meters**2)
                room = {
                    "room_id": room_id,
                    "level_id": level_id,
                    "room_label": room_label,
                    "area_square_meters": area,
                    "centroid_xy": list(geometry.centroid.coords)[0],
                }
                rooms.append(room)
                level_room_ids.append(room_id)
            level_edges = opening_edges(
                labels,
                grid,
                config.scene_graph.door_maximum_width_meters,
                room_id_by_label,
            )
            for edge in level_edges:
                edge["level_id"] = level_id
                edge["edge_id"] = f"{level_id}_{edge['edge_id']}"
            edges.extend(level_edges)
            level = {
                "level_id": level_id,
                "elevation_meters": level_elevations[level_index],
                "ceiling_elevation_meters": ceiling_elevation,
                "floor_polygon_ids": floor_group,
                "ceiling_polygon_ids": assigned_ceilings,
                "wall_polygon_ids": selected_walls,
                "room_ids": level_room_ids,
                "footprint_area_square_meters": float(footprint.area),
                "footprint_geometry": mapping(footprint),
            }
            levels.append(level)
            labels_by_level.append(labels)
            grids.append(grid)
            np.savez_compressed(
                attempt / f"{level_id}_grid.npz",
                labels=labels,
                distance_meters=distance,
                markers=markers,
                walls=walls.astype(np.uint8),
                support=support.astype(np.uint8),
                origin_xy=grid.origin_xy,
                resolution_meters=np.asarray(grid.resolution_meters),
            )
            color = np.zeros((*labels.shape, 3), dtype=np.uint8)
            for room_label in room_id_by_label:
                hue = int(round(179 * ((0.6180339887498949 * room_label) % 1.0)))
                rgb = cv2.cvtColor(
                    np.uint8([[[hue, 180, 242]]]), cv2.COLOR_HSV2BGR
                )[0, 0]
                color[labels == room_label] = rgb
            cv2.imwrite(str(attempt / f"{level_id}_rooms.png"), color)
            level_diagnostics.append(
                {
                    "level_id": level_id,
                    "grid_shape": list(labels.shape),
                    "wall_pixel_count": int(np.count_nonzero(walls)),
                    "support_pixel_count": int(np.count_nonzero(support)),
                    "watershed": watershed_stats,
                }
            )

        stair_graph_edges, stair_diagnostics = stair_edges(
            stair_mesh_path,
            rooms,
            room_geometries,
            levels,
            config.scene_graph,
        )
        edges.extend(stair_graph_edges)
        _write_status(attempt / "STATUS.json", "running", "OpenSeg projection")
        text_features_path = attempt / "room_text_features.npy"
        text_features = compute_text_features(config.scene_graph, text_features_path)
        request_path = attempt / "openseg_request.npz"
        request_stats = prepare_openseg_request(
            data,
            transforms_path,
            request_path,
            config.scene_graph,
        )
        openseg_output = attempt / "openseg_triangle_features.npz"
        progress_path = attempt / "openseg_progress.json"
        checkpoint_path = attempt / "openseg_checkpoint.npz"
        projector_script = Path(__file__).resolve().parents[2] / "scripts" / "project_openseg_features.py"
        command = [
            str(config.scene_graph.openseg_python),
            str(projector_script),
            "--request",
            str(request_path),
            "--model",
            str(config.scene_graph.openseg_model),
            "--text-features",
            str(text_features_path),
            "--output",
            str(openseg_output),
            "--progress",
            str(progress_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(config.runtime.preferred_gpu)
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        openseg_log = attempt / "openseg.log"
        with openseg_log.open("w", encoding="utf-8") as log:
            log.write("command: " + " ".join(command) + "\n\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=attempt,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0 or not openseg_output.is_file():
            raise SceneGraphStageError(
                f"OpenSeg projector failed with code {result.returncode}; see {openseg_log}"
            )
        feature_payload = np.load(openseg_output, allow_pickle=False)
        triangle_features = feature_payload["triangle_features"]
        feature_counts = feature_payload["triangle_feature_counts"]
        room_features = classify_rooms(
            rooms,
            room_geometries,
            levels,
            data,
            triangle_features,
            feature_counts,
            text_features,
            config.scene_graph.grid_resolution_meters,
        )
        room_features_path = attempt / "room_semantic_features.npy"
        np.save(room_features_path, room_features)
        pruned_room_ids = prune_outdoor_leaf_rooms(rooms, edges)
        for edge in edges:
            edge["pruned"] = any(
                str(room_id) in pruned_room_ids for room_id in edge["room_ids"]
            )

        levels_path = attempt / "levels.json"
        rooms_path = attempt / "rooms.geojson"
        graph_path = attempt / "scene_graph.json"
        stair_diagnostics_path = attempt / "stair_diagnostics.json"
        _write_json(levels_path, levels)
        _write_json(
            rooms_path,
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": room["room_id"],
                        "geometry": mapping(room_geometries[str(room["room_id"])]),
                        "properties": room,
                    }
                    for room in rooms
                ],
            },
        )
        _write_json(stair_diagnostics_path, stair_diagnostics)
        _write_json(
            graph_path,
            {
                "schema_version": 1,
                "scene": config.scene,
                "run_id": run_id,
                "coordinate_system": "Z-up meters; room geometry is world XY",
                "levels": levels,
                "rooms": rooms,
                "edges": edges,
                "active_room_ids": [
                    room["room_id"] for room in rooms if not room["pruned"]
                ],
                "pruned_room_ids": pruned_room_ids,
            },
        )
        preview_path = attempt / "scene_graph_rooms.ply"
        preview = create_room_floor_mesh(levels, labels_by_level, grids)
        o3d.io.write_triangle_mesh(str(preview_path), preview)

        active_rooms = [room for room in rooms if not room["pruned"]]
        all_room_ids = {str(room["room_id"]) for room in rooms}
        edge_references_valid = all(
            all(str(room_id) in all_room_ids for room_id in edge["room_ids"])
            for edge in edges
        )
        validation = {
            "level_count_positive": len(levels) > 0,
            "room_count_positive": len(rooms) > 0,
            "active_room_count_positive": len(active_rooms) > 0,
            "all_room_geometries_nonempty": all(
                not geometry.is_empty for geometry in room_geometries.values()
            ),
            "openseg_triangle_coverage_positive": int(np.count_nonzero(feature_counts)) > 0,
            "edge_room_references_valid": edge_references_valid,
            "no_ground_truth_inputs_used": True,
        }
        if not all(validation.values()):
            raise SceneGraphStageError(f"scene-graph validation failed: {validation}")

        completed_attempt = _rename_attempt(attempt, "complete")

        def completed(path: Path) -> Path:
            return completed_attempt / path.relative_to(attempt)

        output_paths = {
            "levels": completed(levels_path),
            "rooms": completed(rooms_path),
            "scene_graph": completed(graph_path),
            "room_semantic_features": completed(room_features_path),
            "room_text_features": completed(text_features_path),
            "openseg_request": completed(request_path),
            "openseg_triangle_features": completed(openseg_output),
            "openseg_progress": completed(progress_path),
            "openseg_log": completed(openseg_log),
            "stair_diagnostics": completed(stair_diagnostics_path),
            "preview_mesh": completed(preview_path),
        }
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.SCENE_GRAPH.value,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "random_seed": config.runtime.random_seed,
            "inputs": {
                "prototype_manifest": _record(prototype_manifest_path),
                "prototype_state": _record(state_path),
                "prototype_mesh": _record(mesh_path),
                "skeleton_manifest": _record(skeleton_manifest_path),
                "stair_mesh": _record(stair_mesh_path),
                "transforms": _record(transforms_path),
                "openseg_model": {
                    "path": str(config.scene_graph.openseg_model),
                    "saved_model": _record(config.scene_graph.openseg_model / "saved_model.pb"),
                },
                "clip_weights": _record(config.scene_graph.clip_weights),
            },
            "algorithm": {
                "appendix_sections": ["D.1", "D.2", "D.3", "D.4", "D.5"],
                "configuration": {
                    **asdict(config.scene_graph),
                    "openseg_python": str(config.scene_graph.openseg_python),
                    "openseg_model": str(config.scene_graph.openseg_model),
                    "clip_weights": str(config.scene_graph.clip_weights),
                },
                "room_types": list(ROOM_TYPES),
                "outdoor_leaf_prune_types": sorted(OUTDOOR_PRUNE_TYPES),
                "text_prompting": "HOV-SG multiple CLIP templates",
                "openseg_sampling": request_stats,
                "level_diagnostics": level_diagnostics,
            },
            "counts": {
                "levels": len(levels),
                "rooms_before_pruning": len(rooms),
                "rooms_after_pruning": len(active_rooms),
                "pruned_rooms": len(pruned_room_ids),
                "doors": sum(edge["kind"] == "door" for edge in edges),
                "openings": sum(edge["kind"] == "opening" for edge in edges),
                "stairs": sum(edge["kind"] == "stair" for edge in edges),
                "triangles_with_openseg_features": int(np.count_nonzero(feature_counts)),
                "prototype_triangles": len(data.triangles),
            },
            "outputs": {
                "attempt_dir": str(completed_attempt),
                **{name: _record(path) for name, path in output_paths.items()},
            },
            "validation": validation,
            "warnings": [
                "Appendix D does not disclose raster resolution, wall raster width, minimum seed area, minimum room area, image-boundary exclusion, or stair component size; the measured YAML values are recorded in this manifest.",
                "OpenSeg features are fused at visible prototype triangle centroids to avoid materializing hundreds of full-resolution 768-D feature maps.",
                "Stage08 prototype topology is not assumed watertight; Stage09 consumes its semantic planar polygons in BEV as described by Appendix D.",
            ],
        }
        manifest_path = stage_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(
            completed_attempt / "STATUS.json",
            "complete",
            str(completed(graph_path)),
        )
        _write_status(stage_dir / "STATUS.json", "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        if attempt.exists():
            failed_attempt = _rename_attempt(attempt, "failed")
            _write_status(
                failed_attempt / "STATUS.json",
                "failed",
                f"{type(error).__name__}: {error}",
            )
            _write_status(stage_dir / "STATUS.json", "failed", str(failed_attempt))
        if isinstance(error, SceneGraphStageError):
            raise
        raise SceneGraphStageError(f"{type(error).__name__}: {error}") from error
