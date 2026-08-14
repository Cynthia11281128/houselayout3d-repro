"""Appendix-A Algorithm 1 plane fitting and polygon initialization."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.prototype_fitting"

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.layout_skeleton.labels import LAYOUT_LABELS, LAYOUT_PALETTE


class PolygonInitError(RuntimeError):
    """Raised when Algorithm 1 cannot produce a valid polygon artifact."""


@dataclass(frozen=True)
class PolygonInitConfig:
    skeleton: Path
    output: Path
    superpoint_level: int = 3
    plane_distance_threshold_meters: float = 0.04
    minimum_unassigned_vertices: int = 100
    ransac_iterations: int = 256
    rdp_epsilon_meters: float = 0.03
    random_seed: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_status(component_dir: Path, state: str, detail: str = "") -> None:
    component_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        component_dir / "STATUS.json",
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    with (component_dir / "polygon_init.log").open("a", encoding="utf-8") as log:
        log.write(f"{datetime.now(timezone.utc).isoformat()} {state} {detail}\n")


def _resolve_component_artifact(manifest_path: Path, path_text: str, component: str) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    parts = path.parts
    if component in parts:
        suffix = Path(*parts[parts.index(component) + 1 :])
        candidate = manifest_path.parent / suffix
        if candidate.is_file():
            return candidate
    if not path.is_absolute():
        candidate = manifest_path.parent / path
        if candidate.is_file():
            return candidate
    return path


def _component_manifest(path_or_dir: Path, name: str) -> Path:
    path = path_or_dir.expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise PolygonInitError(f"{name} manifest is missing: {path}")
    return path


def _verify_record(manifest_path: Path, record: dict[str, Any], name: str, component: str) -> Path:
    path = _resolve_component_artifact(manifest_path, record["path"], component)
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise PolygonInitError(f"{name} hash mismatch: {path}")
    return path


def _debug_color_from_id(identifier: int) -> np.ndarray:
    value = ((int(identifier) + 1) * 2_654_435_761) & 0xFFFFFF
    return np.asarray(
        [(value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF],
        dtype=np.float64,
    ) / 255.0


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float,
    iterations: int,
    rng: np.random.Generator,
    preferred_normal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``n dot x + d = 0`` and return the refined seed inlier mask."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise PolygonInitError("RANSAC needs at least three 3D points")
    if threshold <= 0 or iterations <= 0:
        raise PolygonInitError("RANSAC threshold and iterations must be positive")

    best_mask: np.ndarray | None = None
    best_count = -1
    best_residual = np.inf
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            continue
        normal /= length
        offset = -float(normal @ sample[0])
        distances = np.abs(points @ normal + offset)
        mask = distances <= threshold
        count = int(mask.sum())
        residual = float(distances[mask].mean()) if count else np.inf
        if count > best_count or (count == best_count and residual < best_residual):
            best_mask = mask
            best_count = count
            best_residual = residual
    if best_mask is None or best_count < 3:
        raise PolygonInitError("RANSAC found no non-degenerate plane")

    normal = np.zeros(3, dtype=np.float64)
    center = np.zeros(3, dtype=np.float64)
    mask = best_mask
    for _ in range(2):
        center = points[mask].mean(axis=0)
        _, _, right = np.linalg.svd(points[mask] - center, full_matrices=False)
        normal = right[-1]
        normal /= np.linalg.norm(normal)
        offset = -float(normal @ center)
        refined = np.abs(points @ normal + offset) <= threshold
        if int(refined.sum()) < 3:
            break
        mask = refined

    if preferred_normal is not None:
        preferred = np.asarray(preferred_normal, dtype=np.float64)
        if np.linalg.norm(preferred) > 1e-12 and float(normal @ preferred) < 0:
            normal = -normal
    offset = -float(normal @ center)
    return np.concatenate((normal, [offset])), mask


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic orthonormal 2D basis for a plane normal."""

    normal = np.asarray(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    return first, second


def project_to_plane_2d(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    first, second = plane_basis(np.asarray(plane)[:3])
    points = np.asarray(points, dtype=np.float64)
    return np.column_stack((points @ first, points @ second))


def polygon_area(points_2d: np.ndarray) -> float:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    return 0.5 * float(
        np.sum(
            points_2d[:, 0] * np.roll(points_2d[:, 1], -1)
            - points_2d[:, 1] * np.roll(points_2d[:, 0], -1)
        )
    )


def extract_boundary_loops(
    component_triangles: np.ndarray,
    vertices: np.ndarray | None = None,
    plane: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Extract closed triangle-boundary cycles using mesh vertex indices."""

    triangles = np.asarray(component_triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
        return [], np.empty((0, 2), dtype=np.int64), 0
    directed = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
    )
    undirected = np.sort(directed, axis=1)
    unique, first, counts = np.unique(
        undirected, axis=0, return_index=True, return_counts=True
    )
    boundary = directed[first[counts == 1]]
    if len(boundary) == 0:
        return [], boundary, 0

    adjacency: dict[int, set[int]] = defaultdict(set)
    for first_vertex, second_vertex in boundary:
        adjacency[int(first_vertex)].add(int(second_vertex))
        adjacency[int(second_vertex)].add(int(first_vertex))
    branch_count = sum(len(neighbors) != 2 for neighbors in adjacency.values())

    if branch_count:
        if vertices is None or plane is None:
            return [], boundary, branch_count
        coordinates = project_to_plane_2d(np.asarray(vertices), np.asarray(plane))
        ordered_neighbors: dict[int, list[int]] = {}
        for vertex, neighbors in adjacency.items():
            center = coordinates[vertex]
            ordered_neighbors[vertex] = sorted(
                neighbors,
                key=lambda neighbor: float(
                    np.arctan2(
                        coordinates[neighbor, 1] - center[1],
                        coordinates[neighbor, 0] - center[0],
                    )
                ),
            )
        visited: set[tuple[int, int]] = set()
        cycles: dict[tuple[int, ...], np.ndarray] = {}
        directed_edges = [
            (first_vertex, second_vertex)
            for first_vertex, neighbors in adjacency.items()
            for second_vertex in neighbors
        ]
        for initial in directed_edges:
            if initial in visited:
                continue
            current = initial
            path: list[int] = []
            local: set[tuple[int, int]] = set()
            closed = False
            while current not in local and current not in visited:
                local.add(current)
                visited.add(current)
                incoming, at_vertex = current
                path.append(incoming)
                neighbors = ordered_neighbors[at_vertex]
                incoming_position = neighbors.index(incoming)
                following = neighbors[(incoming_position - 1) % len(neighbors)]
                current = (at_vertex, following)
                if current == initial:
                    closed = True
                    break
            if not closed or len(path) < 3 or len(set(path)) != len(path):
                continue
            minimum_position = int(np.argmin(path))
            forward = tuple(path[minimum_position:] + path[:minimum_position])
            reversed_path = list(reversed(path))
            reverse_minimum = int(np.argmin(reversed_path))
            reverse = tuple(
                reversed_path[reverse_minimum:] + reversed_path[:reverse_minimum]
            )
            canonical = min(forward, reverse)
            cycles[canonical] = np.asarray(path, dtype=np.int64)
        return list(cycles.values()), boundary, branch_count

    remaining = {
        (min(int(first_vertex), int(second_vertex)), max(int(first_vertex), int(second_vertex)))
        for first_vertex, second_vertex in boundary
    }
    loops: list[np.ndarray] = []
    while remaining:
        start, current = min(remaining)
        remaining.remove((start, current))
        loop = [start]
        previous = start
        while current != start:
            loop.append(current)
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if (
                    min(current, neighbor),
                    max(current, neighbor),
                )
                in remaining
            ]
            if len(candidates) != 1:
                return [], boundary, branch_count + 1
            following = candidates[0]
            remaining.remove((min(current, following), max(current, following)))
            previous, current = current, following
            if len(loop) > len(boundary):
                return [], boundary, branch_count + 1
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops, boundary, branch_count


def group_boundary_contours(
    loops: list[np.ndarray],
    vertices: np.ndarray,
    plane: np.ndarray,
) -> list[list[int]]:
    """Group possibly disjoint boundaries into outer contours and direct holes."""

    if not loops:
        return []
    projected = [project_to_plane_2d(vertices[loop], plane) for loop in loops]
    areas = np.asarray([abs(polygon_area(points)) for points in projected])
    parents = np.full(len(loops), -1, dtype=np.int64)
    for child, child_points in enumerate(projected):
        candidates: list[int] = []
        stride = max(1, len(child_points) // 16)
        representatives = child_points[::stride]
        for candidate, candidate_points in enumerate(projected):
            if areas[candidate] <= areas[child]:
                continue
            contained = sum(
                _point_in_polygon(point, candidate_points)
                for point in representatives
            )
            if contained > len(representatives) // 2:
                candidates.append(candidate)
        if candidates:
            parents[child] = min(candidates, key=lambda index: areas[index])

    depths = np.zeros(len(loops), dtype=np.int64)
    for index in range(len(loops)):
        seen: set[int] = set()
        parent = int(parents[index])
        while parent >= 0:
            if parent in seen:
                raise PolygonInitError("cyclic contour nesting detected")
            seen.add(parent)
            depths[index] += 1
            parent = int(parents[parent])

    groups: list[list[int]] = []
    for outer in np.flatnonzero(depths % 2 == 0):
        holes = [
            index
            for index in range(len(loops))
            if parents[index] == outer and depths[index] == depths[outer] + 1
        ]
        groups.append([int(outer), *holes])
    return sorted(groups, key=lambda group: areas[group[0]], reverse=True)


def _rdp_open_mask(points: np.ndarray, epsilon: float) -> np.ndarray:
    mask = np.zeros(len(points), dtype=bool)
    mask[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, stop = stack.pop()
        segment = points[stop] - points[start]
        length = float(np.linalg.norm(segment))
        middle = points[start + 1 : stop]
        if len(middle) == 0:
            continue
        if length <= 1e-12:
            distances = np.linalg.norm(middle - points[start], axis=1)
        else:
            offsets = middle - points[start]
            projection = np.outer(offsets @ segment / (length * length), segment)
            distances = np.linalg.norm(offsets - projection, axis=1)
        local = int(np.argmax(distances))
        if float(distances[local]) > epsilon:
            index = start + 1 + local
            mask[index] = True
            stack.extend(((start, index), (index, stop)))
    return mask


def closed_rdp_mask(points_2d: np.ndarray, epsilon: float) -> np.ndarray:
    """RDP-simplify a closed contour without privileging an artificial seam."""

    points = np.asarray(points_2d, dtype=np.float64)
    if len(points) <= 3:
        return np.ones(len(points), dtype=bool)
    first = 0
    second = int(np.argmax(np.linalg.norm(points - points[first], axis=1)))
    if second in {0, len(points) - 1}:
        second = len(points) // 2
    mask = np.zeros(len(points), dtype=bool)
    mask[: second + 1] |= _rdp_open_mask(points[: second + 1], epsilon)
    wrapped_indices = np.concatenate((np.arange(second, len(points)), [0]))
    wrapped_mask = _rdp_open_mask(points[wrapped_indices], epsilon)
    mask[wrapped_indices[wrapped_mask]] = True
    if int(mask.sum()) < 3:
        mask[:] = True
    return mask


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = np.asarray(point, dtype=np.float64)
    vertices = np.asarray(polygon, dtype=np.float64)
    inside = False
    previous = len(vertices) - 1
    for current in range(len(vertices)):
        x1, y1 = vertices[current]
        x2, y2 = vertices[previous]
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _verify_inputs(config: PolygonInitConfig) -> dict[str, Any]:
    manifest_path = _component_manifest(config.skeleton, "skeleton")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise PolygonInitError("skeleton manifest is not complete")

    outputs = manifest["outputs"]
    structure_record = outputs["meshes"]["structure"]
    full_record = outputs["meshes"]["semantic_mesh"]
    hard_record = outputs["arrays"]["vertex_hard_assignments.npy"]
    labels_record = outputs["arrays"]["simplified_segmentation_labels.npy"]
    records = {
        "structure_mesh": _verify_record(manifest_path, structure_record, "structure mesh", "skeleton"),
        "semantic_mesh": _verify_record(manifest_path, full_record, "semantic mesh", "skeleton"),
        "hard_labels": _verify_record(manifest_path, hard_record, "vertex hard assignments", "skeleton"),
        "label_names": _verify_record(manifest_path, labels_record, "semantic label names", "skeleton"),
    }
    if structure_record.get("classes_path"):
        class_record = {
            "path": structure_record["classes_path"],
            "sha256": structure_record["classes_sha256"],
        }
        records["structure_classes"] = _verify_record(
            manifest_path, class_record, "structure class probabilities", "skeleton"
        )
    segmentation = manifest_path.parent / "spt" / f"level_{config.superpoint_level}_segmentation.npy"
    if not segmentation.is_file():
        raise PolygonInitError(f"superpoint segmentation is missing: {segmentation}")
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "structure_mesh": records["structure_mesh"],
        "semantic_mesh": records["semantic_mesh"],
        "hard_labels": records["hard_labels"],
        "label_names": records["label_names"],
        "structure_classes": records.get("structure_classes"),
        "segmentation": segmentation,
    }


def _mesh_artifact(path: Path, mesh: Any) -> dict[str, Any]:
    import open3d as o3d

    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise PolygonInitError(f"failed to write mesh: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
    }


def _candidate_faces(
    component: np.ndarray,
    triangles: np.ndarray,
    vertex_faces: Any,
    component_mask: np.ndarray,
) -> np.ndarray:
    candidate_ids = np.unique(vertex_faces[component].indices)
    component_mask[component] = True
    selected = candidate_ids[component_mask[triangles[candidate_ids]].all(axis=1)]
    component_mask[component] = False
    return triangles[selected]


def _convex_contour_vertices(
    contours: list[np.ndarray],
    vertices: np.ndarray,
    adjacency: Any,
    plane: np.ndarray,
    threshold: float = 0.05,
) -> list[int]:
    convex: set[int] = set()
    for contour in contours:
        for vertex in contour:
            neighbors = adjacency.indices[
                adjacency.indptr[vertex] : adjacency.indptr[vertex + 1]
            ]
            if len(neighbors) and np.any(
                vertices[neighbors] @ plane[:3] + plane[3] < -(threshold**2)
            ):
                convex.add(int(vertex))
    return sorted(convex)


def validate_full_mesh_arrays(
    full_vertex_count: int,
    hard_labels: np.ndarray,
    segmentation: np.ndarray,
) -> None:
    if len(hard_labels) != full_vertex_count or len(segmentation) != full_vertex_count:
        raise PolygonInitError("full-mesh arrays do not match the semantic mesh")


def run_polygon_init(config: PolygonInitConfig, command: list[str] | None = None) -> Path:
    """Run the Appendix Algorithm 1 initialization on the structural skeleton."""

    inputs = _verify_inputs(config)
    component_dir = config.output.expanduser()
    if component_dir.exists():
        raise PolygonInitError(
            f"polygon-init component already exists and will not be overwritten: {component_dir}"
        )
    component_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    _write_status(component_dir, "loading", "skeleton structure and superpoints")

    try:
        try:
            import open3d as o3d
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError as error:
            raise PolygonInitError("Open3D and SciPy are required") from error

        structure_mesh = o3d.io.read_triangle_mesh(
            str(inputs["structure_mesh"]), enable_post_processing=False
        )
        semantic_mesh = o3d.io.read_triangle_mesh(
            str(inputs["semantic_mesh"]), enable_post_processing=False
        )
        if len(structure_mesh.vertices) == 0 or len(structure_mesh.triangles) == 0:
            raise PolygonInitError("skeleton structure mesh is empty")
        structure_mesh.compute_vertex_normals()
        vertices = np.asarray(structure_mesh.vertices).astype(np.float64)
        triangles = np.asarray(structure_mesh.triangles).astype(np.int64)
        normals = np.asarray(structure_mesh.vertex_normals).astype(np.float64)
        full_vertices = np.asarray(semantic_mesh.vertices).astype(np.float64)
        hard_labels = np.load(inputs["hard_labels"])
        label_names = tuple(str(value) for value in np.load(inputs["label_names"]))
        segmentation = np.load(inputs["segmentation"]).astype(np.int64)
        if label_names != LAYOUT_LABELS:
            raise PolygonInitError("skeleton semantic label names changed")
        validate_full_mesh_arrays(len(full_vertices), hard_labels, segmentation)
        structure_to_full = np.flatnonzero(np.isin(hard_labels, [0, 1, 2, 3]))
        if len(structure_to_full) != len(vertices):
            raise PolygonInitError("structural label mask does not match structure mesh")
        if not np.allclose(vertices, full_vertices[structure_to_full], atol=1e-6):
            raise PolygonInitError("structure mesh vertex order is not the filtered full order")
        structure_labels = hard_labels[structure_to_full].astype(np.uint8)
        structure_segments = segmentation[structure_to_full]

        edges = np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])
        )
        row = np.concatenate((edges[:, 0], edges[:, 1]))
        column = np.concatenate((edges[:, 1], edges[:, 0]))
        adjacency = coo_matrix(
            (np.ones(len(row), dtype=np.uint8), (row, column)),
            shape=(len(vertices), len(vertices)),
        ).tocsr()
        adjacency.sum_duplicates()
        vertex_faces = coo_matrix(
            (
                np.ones(triangles.size, dtype=np.uint8),
                (
                    triangles.ravel(),
                    np.repeat(np.arange(len(triangles)), 3),
                ),
            ),
            shape=(len(vertices), len(triangles)),
        ).tocsr()
        component_mask = np.zeros(len(vertices), dtype=bool)
        unassigned = np.ones(len(vertices), dtype=bool)
        rectified_vertices = vertices.copy()
        plane_assignments = np.full(len(vertices), -1, dtype=np.int32)
        rng = np.random.default_rng(config.random_seed)
        polygon_info: dict[str, dict[str, Any]] = {}
        candidate_records: list[dict[str, Any]] = []
        rejected_components: list[dict[str, Any]] = []
        all_boundary_edges: list[np.ndarray] = []

        _write_status(
            component_dir,
            "fitting_planes",
            f"vertices={len(vertices)} K={config.minimum_unassigned_vertices}",
        )
        candidate_index = 0
        while True:
            counts = np.bincount(
                structure_segments[unassigned],
                minlength=int(structure_segments.max()) + 1,
            )
            seed_segment = int(np.argmax(counts))
            seed_count = int(counts[seed_segment])
            if seed_count <= config.minimum_unassigned_vertices:
                break
            seed_vertices = np.flatnonzero(
                unassigned & (structure_segments == seed_segment)
            )
            preferred_normal = normals[seed_vertices].mean(axis=0)
            plane, seed_inliers = fit_plane_ransac(
                vertices[seed_vertices],
                config.plane_distance_threshold_meters,
                config.ransac_iterations,
                rng,
                preferred_normal=preferred_normal,
            )
            distances = np.abs(vertices @ plane[:3] + plane[3])
            global_inliers = np.flatnonzero(
                unassigned
                & (distances <= config.plane_distance_threshold_meters)
            )
            if len(global_inliers) == 0:
                raise PolygonInitError("RANSAC plane has no global inliers")
            subgraph = adjacency[global_inliers][:, global_inliers]
            _, component_labels = connected_components(
                subgraph, directed=False, return_labels=True
            )
            seed_global_inliers = seed_vertices[
                distances[seed_vertices]
                <= config.plane_distance_threshold_meters
            ]
            seed_positions = np.searchsorted(global_inliers, seed_global_inliers)
            overlap = np.bincount(component_labels[seed_positions])
            best_component = int(np.argmax(overlap))
            component = global_inliers[component_labels == best_component]
            unassigned[component] = False

            component_triangles = _candidate_faces(
                component, triangles, vertex_faces, component_mask
            )
            loops, boundary_edges, branch_count = extract_boundary_loops(
                component_triangles, vertices=vertices, plane=plane
            )
            record: dict[str, Any] = {
                "candidate": candidate_index,
                "seed_superpoint_id": seed_segment,
                "seed_unassigned_vertex_count": seed_count,
                "seed_ransac_inlier_count": int(seed_inliers.sum()),
                "global_inlier_count": len(global_inliers),
                "selected_component_vertex_count": len(component),
                "selected_component_triangle_count": len(component_triangles),
                "seed_component_overlap": int(overlap[best_component]),
                "boundary_edge_count": len(boundary_edges),
                "boundary_branch_vertex_count": branch_count,
            }
            rejection: str | None = None
            if len(component_triangles) == 0:
                rejection = "component_has_no_complete_triangles"
            elif not loops:
                rejection = "component_has_no_closed_boundary"

            contour_groups: list[list[int]] = []
            if rejection is None:
                contour_groups = group_boundary_contours(loops, vertices, plane)
                if not contour_groups:
                    rejection = "component_has_no_outer_boundary"

            if rejection is not None:
                plane_assignments[component] = -2
                record["accepted"] = False
                record["rejection"] = rejection
                rejected_components.append(record.copy())
            else:
                signed = vertices[component] @ plane[:3] + plane[3]
                rectified_vertices[component] = (
                    vertices[component] - signed[:, None] * plane[None, :3]
                )
                class_histogram = np.bincount(
                    structure_labels[component], minlength=len(LAYOUT_LABELS)
                )
                class_id = int(np.argmax(class_histogram))
                polygon_ids: list[int] = []
                contour_count = 0
                full_count = 0
                simplified_count = 0
                outer_areas: list[float] = []
                for group in contour_groups:
                    group_loops = [loops[index] for index in group]
                    projected_loops = [
                        project_to_plane_2d(vertices[loop], plane)
                        for loop in group_loops
                    ]
                    areas = [
                        abs(polygon_area(points)) for points in projected_loops
                    ]
                    contour_masks = [
                        closed_rdp_mask(
                            points, config.rdp_epsilon_meters
                        )
                        for points in projected_loops
                    ]
                    contours = [loop.tolist() for loop in group_loops]
                    polygon_id = len(polygon_info)
                    polygon_ids.append(polygon_id)
                    polygon_info[str(polygon_id)] = {
                        "id": polygon_id,
                        "vertices": component.tolist(),
                        "contours": contours,
                        "contour_areas_original": [float(area) for area in areas],
                        "contour_masks_rdp": [
                            mask.tolist() for mask in contour_masks
                        ],
                        "plane_eq": plane.tolist(),
                        "color": (
                            LAYOUT_PALETTE[class_id].astype(np.float64) / 255.0
                        ).tolist(),
                        "class": LAYOUT_LABELS[class_id],
                        "class_id": class_id,
                        "semantic_vertex_histogram": {
                            LAYOUT_LABELS[index]: int(value)
                            for index, value in enumerate(class_histogram)
                        },
                        "source_superpoint_id": seed_segment,
                        "convex_edges": _convex_contour_vertices(
                            group_loops, vertices, adjacency, plane
                        ),
                        "shared_edges": {},
                    }
                    all_boundary_edges.extend(
                        np.column_stack((loop, np.roll(loop, -1)))
                        for loop in group_loops
                    )
                    contour_count += len(contours)
                    full_count += sum(len(contour) for contour in contours)
                    simplified_count += sum(mask.sum() for mask in contour_masks)
                    outer_areas.append(float(areas[0]))
                plane_assignments[component] = polygon_ids[0]
                record.update(
                    {
                        "accepted": True,
                        "polygon_ids": polygon_ids,
                        "polygon_count": len(polygon_ids),
                        "contour_count": contour_count,
                        "full_contour_vertex_count": int(full_count),
                        "simplified_contour_vertex_count": int(simplified_count),
                        "outer_areas_square_meters": outer_areas,
                        "non_manifold_boundary_recovered": branch_count > 0,
                    }
                )
            candidate_records.append(record)
            candidate_index += 1
            if candidate_index % 10 == 0:
                _write_status(
                    component_dir,
                    "fitting_planes",
                    f"candidates={candidate_index} polygons={len(polygon_info)} remaining={int(unassigned.sum())}",
                )

        if not polygon_info:
            raise PolygonInitError("Algorithm 1 produced no valid polygons")
        _write_status(
            component_dir,
            "writing_artifacts",
            f"polygons={len(polygon_info)} residual={int(unassigned.sum())}",
        )
        np.save(component_dir / "structure_to_full_vertex.npy", structure_to_full)
        np.save(component_dir / "assigned_plane_ids.npy", plane_assignments)
        np.save(component_dir / "residual_unassigned_mask.npy", unassigned)
        _write_json(component_dir / "polygon_info.json", polygon_info)
        with (component_dir / "plane_candidates.jsonl").open("w", encoding="utf-8") as handle:
            for record in candidate_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        with (component_dir / "rejected_components.jsonl").open("w", encoding="utf-8") as handle:
            for record in rejected_components:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        rectified_mesh = o3d.geometry.TriangleMesh(structure_mesh)
        rectified_mesh.vertices = o3d.utility.Vector3dVector(rectified_vertices)
        rectified_mesh.compute_vertex_normals()
        rectified_record = _mesh_artifact(component_dir / "rectified_mesh.ply", rectified_mesh)

        same_plane = (
            (plane_assignments[triangles[:, 0]] >= 0)
            & (plane_assignments[triangles[:, 0]] == plane_assignments[triangles[:, 1]])
            & (plane_assignments[triangles[:, 0]] == plane_assignments[triangles[:, 2]])
        )
        component_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(rectified_vertices),
            triangles=o3d.utility.Vector3iVector(triangles[same_plane]),
        )
        component_colors = np.full((len(vertices), 3), 0.35, dtype=np.float64)
        accepted = plane_assignments >= 0
        for polygon_id in polygon_info:
            mask = plane_assignments == int(polygon_id)
            component_colors[mask] = _debug_color_from_id(int(polygon_id))
        component_mesh.vertex_colors = o3d.utility.Vector3dVector(component_colors)
        component_mesh.compute_vertex_normals()
        component_record = _mesh_artifact(
            component_dir / "plane_components_mesh.ply", component_mesh
        )

        boundary_lines = (
            np.concatenate(all_boundary_edges)
            if all_boundary_edges
            else np.empty((0, 2), dtype=np.int64)
        )
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(rectified_vertices),
            lines=o3d.utility.Vector2iVector(boundary_lines),
        )
        if len(boundary_lines):
            line_set.colors = o3d.utility.Vector3dVector(
                np.full((len(boundary_lines), 3), [1.0, 0.85, 0.1])
            )
        boundary_path = component_dir / "polygon_boundaries.ply"
        if not o3d.io.write_line_set(str(boundary_path), line_set, write_ascii=False):
            raise PolygonInitError("failed to write polygon boundary line set")

        polygon_path = component_dir / "polygon_info.json"
        array_names = (
            "structure_to_full_vertex.npy",
            "assigned_plane_ids.npy",
            "residual_unassigned_mask.npy",
        )
        contour_count = sum(len(polygon["contours"]) for polygon in polygon_info.values())
        full_contour_vertices = sum(
            len(contour)
            for polygon in polygon_info.values()
            for contour in polygon["contours"]
        )
        simplified_contour_vertices = sum(
            sum(mask)
            for polygon in polygon_info.values()
            for mask in polygon["contour_masks_rdp"]
        )
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "component": "polygon_init",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": config.random_seed,
            "inputs": {
                "skeleton_manifest": {
                    "path": str(inputs["manifest_path"]),
                    "sha256": _sha256(inputs["manifest_path"]),
                },
                "structure_mesh": {
                    "path": str(inputs["structure_mesh"]),
                    "sha256": _sha256(inputs["structure_mesh"]),
                },
                "structure_classes": (
                    None
                    if inputs["structure_classes"] is None
                    else {
                        "path": str(inputs["structure_classes"]),
                        "sha256": _sha256(inputs["structure_classes"]),
                    }
                ),
                "superpoint_segmentation": {
                    "path": str(inputs["segmentation"]),
                    "sha256": _sha256(inputs["segmentation"]),
                },
                "vertex_hard_assignments": {
                    "path": str(inputs["hard_labels"]),
                    "sha256": _sha256(inputs["hard_labels"]),
                },
            },
            "algorithm": {
                "reference": "Appendix A, Section C.1, Algorithm 1",
                "superpoint_level": config.superpoint_level,
                "minimum_unassigned_vertices_K": config.minimum_unassigned_vertices,
                "plane_distance_threshold_meters": config.plane_distance_threshold_meters,
                "ransac_iterations": config.ransac_iterations,
                "rdp_epsilon_meters": config.rdp_epsilon_meters,
                "component_rule": "mesh-edge connected global plane inliers with maximum seed-superpoint overlap",
                "boundary_rule": "edges incident to exactly one selected-component triangle",
            },
            "outputs": {
                "rectified_mesh": rectified_record,
                "plane_components_mesh": component_record,
                "polygon_info": {
                    "path": str(polygon_path),
                    "sha256": _sha256(polygon_path),
                    "size_bytes": polygon_path.stat().st_size,
                },
                "polygon_boundaries": {
                    "path": str(boundary_path),
                    "sha256": _sha256(boundary_path),
                    "line_count": len(boundary_lines),
                },
                "plane_candidates": {
                    "path": str(component_dir / "plane_candidates.jsonl"),
                    "sha256": _sha256(component_dir / "plane_candidates.jsonl"),
                    "count": len(candidate_records),
                },
                "rejected_components": {
                    "path": str(component_dir / "rejected_components.jsonl"),
                    "sha256": _sha256(component_dir / "rejected_components.jsonl"),
                    "count": len(rejected_components),
                },
                "arrays": {
                    name: {
                        "path": str(component_dir / name),
                        "sha256": _sha256(component_dir / name),
                    }
                    for name in array_names
                },
            },
            "statistics": {
                "structure_vertex_count": len(vertices),
                "structure_triangle_count": len(triangles),
                "candidate_plane_count": len(candidate_records),
                "polygon_count": len(polygon_info),
                "rejected_component_count": len(rejected_components),
                "assigned_polygon_vertex_count": int(accepted.sum()),
                "rejected_vertex_count": int((plane_assignments == -2).sum()),
                "residual_unassigned_vertex_count": int(unassigned.sum()),
                "contour_count": contour_count,
                "full_contour_vertex_count": full_contour_vertices,
                "simplified_contour_vertex_count": simplified_contour_vertices,
                "component_triangle_count": int(same_plane.sum()),
            },
            "validation": {
                "all_contour_indices_in_rectified_mesh": all(
                    0 <= vertex < len(vertices)
                    for polygon in polygon_info.values()
                    for contour in polygon["contours"]
                    for vertex in contour
                ),
                "all_contours_have_at_least_three_vertices": all(
                    len(contour) >= 3
                    for polygon in polygon_info.values()
                    for contour in polygon["contours"]
                ),
                "all_simplified_contours_have_at_least_three_vertices": all(
                    sum(mask) >= 3
                    for polygon in polygon_info.values()
                    for mask in polygon["contour_masks_rdp"]
                ),
                "plane_equations_are_unit_normalized": all(
                    np.isclose(np.linalg.norm(polygon["plane_eq"][:3]), 1.0)
                    for polygon in polygon_info.values()
                ),
                "plane_equations_are_finite": all(
                    np.isfinite(np.asarray(polygon["plane_eq"], dtype=np.float64)).all()
                    for polygon in polygon_info.values()
                ),
                "all_outer_contours_have_nonzero_area": all(
                    polygon["contour_areas_original"][0] > 0
                    for polygon in polygon_info.values()
                ),
                "unofficial_optimizer_required_keys_present": all(
                    {"vertices", "contours", "plane_eq", "color"} <= set(polygon)
                    for polygon in polygon_info.values()
                ),
                "structure_to_full_mapping_exact": True,
                "no_ground_truth_inputs_used": True,
            },
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "warnings": [
                "The Appendix specifies Algorithm 1 but does not disclose K, the RANSAC distance threshold, iteration count, or contour simplification tolerance; these are explicit reproducibility parameters in the YAML configuration.",
                "The supplied unofficial package consumes polygon_info.json but does not contain the code that creates it; this component independently implements the Appendix algorithm and its observed input schema.",
            ],
        }
        if not all(manifest["validation"].values()):
            raise PolygonInitError("polygon-init artifact validation failed")
        manifest_path = component_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(component_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(component_dir, "failed", str(error))
        if isinstance(error, PolygonInitError):
            raise
        raise PolygonInitError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize planar polygons from a completed Section 4.2 skeleton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--superpoint-level", type=int, default=3)
    parser.add_argument("--plane-distance-threshold-meters", type=float, default=0.04)
    parser.add_argument("--minimum-unassigned-vertices", type=int, default=100)
    parser.add_argument("--ransac-iterations", type=int, default=256)
    parser.add_argument("--rdp-epsilon-meters", type=float, default=0.03)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.superpoint_level, args.minimum_unassigned_vertices, args.ransac_iterations) <= 0:
        raise PolygonInitError("superpoint level, K, and RANSAC iterations must be positive")
    if args.plane_distance_threshold_meters <= 0 or args.rdp_epsilon_meters <= 0:
        raise PolygonInitError("distance thresholds must be positive")
    config = PolygonInitConfig(
        skeleton=args.skeleton,
        output=args.output,
        superpoint_level=args.superpoint_level,
        plane_distance_threshold_meters=args.plane_distance_threshold_meters,
        minimum_unassigned_vertices=args.minimum_unassigned_vertices,
        ransac_iterations=args.ransac_iterations,
        rdp_epsilon_meters=args.rdp_epsilon_meters,
        random_seed=args.random_seed,
    )
    manifest = run_polygon_init(config, command=sys.argv)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
