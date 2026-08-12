"""Extract the Section 4.2 semantic layout skeleton from a Poisson mesh."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.layout_skeleton"

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.rgb_to_mesh.dn_splatter import build_training_environment

from .labels import LAYOUT_LABELS, LAYOUT_PALETTE


class SkeletonError(RuntimeError):
    """Raised when semantic mesh voting or skeleton extraction fails."""


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class SuperpointConfig:
    repository: Path
    knn_neighbors: int = 15
    knn_radius_meters: float = 0.2
    adjacency_neighbors: int = 10
    adjacency_weight: float = 1.0
    regularization: tuple[float, ...] = (0.03, 0.06, 0.12)
    spatial_weight: tuple[float, ...] = (0.01, 0.02, 0.04)
    cutoff: tuple[int, ...] = (10, 20, 40)
    iterations: int = 10
    final_level: int = 3


@dataclass(frozen=True)
class SkeletonConfig:
    transforms: Path
    dn_splatter: Path
    mesh_manifest: Path
    oneformer: Path
    camera: CameraConfig
    ns_render: Path
    superpoint: SuperpointConfig
    output: Path
    random_seed: int = 0
    samples_per_frame: int = 5000
    preferred_gpu: int = 0
    torch_home: Path | None = None


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
    log_path = component_dir / "skeleton.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{datetime.now(timezone.utc).isoformat()} {state} {detail}\n")


def load_camera(path: Path) -> CameraConfig:
    payload = _read_json(path)
    intrinsics = payload.get("pinhole_intrinsics")
    resolution = payload.get("pinhole_resolution")
    if (
        not isinstance(intrinsics, list)
        or len(intrinsics) != 4
        or not isinstance(resolution, list)
        or len(resolution) != 2
    ):
        raise SkeletonError("camera file must contain pinhole_intrinsics and pinhole_resolution")
    camera = CameraConfig(
        width=int(resolution[0]),
        height=int(resolution[1]),
        fx=float(intrinsics[0]),
        fy=float(intrinsics[1]),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
    )
    if min(camera.width, camera.height, camera.fx, camera.fy) <= 0:
        raise SkeletonError("camera dimensions and focal lengths must be positive")
    return camera


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
        raise SkeletonError(f"{name} manifest is missing: {path}")
    return path


def build_depth_render_command(ns_render: Path, training_config: Path, output_dir: Path) -> list[str]:
    return [
        str(ns_render),
        "dataset",
        "--load-config",
        str(training_config),
        "--output-path",
        str(output_dir),
        "--rendered-output-names",
        "raw-depth",
        "--split",
        "train",
    ]


def backproject_samples(
    depth: np.ndarray,
    flat_indices: np.ndarray,
    c2w_opengl: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    height, width = depth.shape
    rows = flat_indices // width
    columns = flat_indices % width
    z = depth[rows, columns].astype(np.float32, copy=False)
    camera_points = np.column_stack(
        (
            (columns.astype(np.float32) + 0.5 - cx) * z / fx,
            (rows.astype(np.float32) + 0.5 - cy) * z / fy,
            z,
        )
    ).astype(np.float32, copy=False)
    c2w_opencv = np.asarray(c2w_opengl, dtype=np.float32).copy()
    c2w_opencv[:3, 1:3] *= -1
    return camera_points @ c2w_opencv[:3, :3].T + c2w_opencv[:3, 3]


def _load_inputs(config: SkeletonConfig) -> dict[str, Any]:
    transforms_path = config.transforms.expanduser().resolve()
    if not transforms_path.is_file():
        raise SkeletonError(f"transforms.json is missing: {transforms_path}")
    transforms = _read_json(transforms_path)
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise SkeletonError("transforms.json must contain non-empty frames")

    dn_manifest_path = _component_manifest(config.dn_splatter, "dn_splatter")
    dn_manifest = _read_json(dn_manifest_path)
    if dn_manifest.get("status") != "complete":
        raise SkeletonError("dn_splatter manifest is not complete")
    training_config = _resolve_component_artifact(
        dn_manifest_path, dn_manifest["outputs"]["training_config"], "dn_splatter"
    )
    checkpoint = _resolve_component_artifact(
        dn_manifest_path, dn_manifest["outputs"]["final_checkpoint"], "dn_splatter"
    )
    if not training_config.is_file() or _sha256(training_config) != dn_manifest["outputs"]["training_config_sha256"]:
        raise SkeletonError(f"dn_splatter training config hash mismatch: {training_config}")
    if not checkpoint.is_file() or _sha256(checkpoint) != dn_manifest["outputs"]["final_checkpoint_sha256"]:
        raise SkeletonError(f"dn_splatter checkpoint hash mismatch: {checkpoint}")

    mesh_manifest_path = _component_manifest(config.mesh_manifest, "mesh")
    mesh_manifest = _read_json(mesh_manifest_path)
    if mesh_manifest.get("status") != "complete":
        raise SkeletonError("mesh manifest is not complete")
    mesh_record = mesh_manifest["outputs"]["poisson_mesh"]
    mesh_path = _resolve_component_artifact(mesh_manifest_path, mesh_record["path"], "mesh")
    if not mesh_path.is_file() or _sha256(mesh_path) != mesh_record["sha256"]:
        raise SkeletonError(f"mesh Poisson hash mismatch: {mesh_path}")

    oneformer_manifest_path = _component_manifest(config.oneformer, "oneformer")
    oneformer_manifest = _read_json(oneformer_manifest_path)
    if oneformer_manifest.get("status") != "complete":
        raise SkeletonError("oneformer manifest is not complete")
    per_image = oneformer_manifest["outputs"]["per_image"]
    per_image_path = _resolve_component_artifact(oneformer_manifest_path, per_image["path"], "oneformer")
    if not per_image_path.is_file() or _sha256(per_image_path) != per_image["sha256"]:
        raise SkeletonError(f"oneformer per-image hash mismatch: {per_image_path}")
    semantic_records = [
        json.loads(line)
        for line in per_image_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(semantic_records) != oneformer_manifest["validation"]["frame_count"]:
        raise SkeletonError("oneformer per-image count is inconsistent")
    for record in semantic_records:
        layout_path = _resolve_component_artifact(oneformer_manifest_path, record["layout_path"], "oneformer")
        if not layout_path.is_file() or _sha256(layout_path) != record["layout_sha256"]:
            raise SkeletonError(f"oneformer layout hash mismatch: {layout_path}")
        record["layout_path"] = str(layout_path)

    frame_names = [Path(frame["file_path"]).name for frame in frames]
    semantic_names = [record["name"] for record in semantic_records]
    if frame_names != semantic_names:
        raise SkeletonError("transforms and OneFormer frame orders do not match")

    return {
        "transforms_path": transforms_path,
        "transforms": transforms,
        "dn_manifest_path": dn_manifest_path,
        "training_config": training_config,
        "checkpoint": checkpoint,
        "mesh_manifest_path": mesh_manifest_path,
        "mesh_path": mesh_path,
        "oneformer_manifest_path": oneformer_manifest_path,
        "semantic_records": semantic_records,
    }


def _load_rendered_depths(
    render_dir: Path,
    frames: list[dict[str, Any]],
    expected_shape: tuple[int, int],
) -> tuple[list[Path], list[dict[str, Any]]]:
    raw_dir = render_dir / "train" / "raw-depth"
    paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for frame in frames:
        stem = Path(frame["file_path"]).stem
        path = raw_dir / f"{stem}.npy.gz"
        if not path.is_file():
            matches = list(raw_dir.rglob(f"{stem}.npy.gz")) if raw_dir.is_dir() else []
            if len(matches) != 1:
                raise SkeletonError(f"rendered raw depth is missing: {path}")
            path = matches[0]
        with gzip.open(path, "rb") as handle:
            depth = np.load(handle)
        depth = np.asarray(depth).squeeze()
        if depth.shape != expected_shape:
            raise SkeletonError(f"rendered depth has shape {depth.shape}, expected {expected_shape}: {path}")
        finite = np.isfinite(depth)
        if not finite.all() or not (depth > 0).any():
            raise SkeletonError(f"rendered depth is invalid: {path}")
        paths.append(path)
        records.append(
            {
                "name": path.name,
                "path": str(path),
                "sha256": _sha256(path),
                "shape": list(depth.shape),
                "dtype": str(depth.dtype),
                "minimum_meters": float(depth.min()),
                "maximum_meters": float(depth.max()),
                "positive_fraction": float((depth > 0).mean()),
            }
        )
    extras = set(raw_dir.rglob("*.npy.gz")) - set(paths)
    if extras:
        raise SkeletonError(f"raw-depth rendering produced {len(extras)} unexpected files")
    return paths, records


def _sample_rays(
    config: SkeletonConfig,
    frames: list[dict[str, Any]],
    semantic_records: list[dict[str, Any]],
    depth_paths: list[Path],
    component_dir: Path,
) -> dict[str, Any]:
    from PIL import Image

    height = config.camera.height
    width = config.camera.width
    samples = config.samples_per_frame
    if samples > height * width:
        raise SkeletonError("samples_per_frame exceeds image pixel count")
    rng = np.random.default_rng(config.random_seed)
    origins: list[np.ndarray] = []
    destinations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    pixel_records: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    for index, (frame, semantic_record, depth_path) in enumerate(zip(frames, semantic_records, depth_paths)):
        with gzip.open(depth_path, "rb") as handle:
            depth = np.asarray(np.load(handle)).squeeze().astype(np.float32)
        with Image.open(semantic_record["layout_path"]) as image:
            semantic = np.asarray(image, dtype=np.uint8)
        if semantic.shape != (height, width):
            raise SkeletonError(f"OneFormer map shape changed: {semantic_record['layout_path']}")
        flat = rng.choice(height * width, size=samples, replace=False)
        valid = np.isfinite(depth.ravel()[flat]) & (depth.ravel()[flat] > 0)
        inpainted = depth.copy()
        inpainted[~np.isfinite(inpainted) | (inpainted <= 0)] = 0.5
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        points = backproject_samples(
            inpainted,
            flat,
            c2w,
            config.camera.fx,
            config.camera.fy,
            config.camera.cx,
            config.camera.cy,
        )
        origin = np.repeat(c2w[None, :3, 3], samples, axis=0)
        sampled_labels = semantic.ravel()[flat]
        if int(sampled_labels.max()) >= len(LAYOUT_LABELS):
            raise SkeletonError("sampled layout label is out of range")
        origins.append(origin.astype(np.float32))
        destinations.append(points.astype(np.float32))
        labels.append(sampled_labels.astype(np.uint8))
        validity.append(valid)
        rows = flat // width
        columns = flat % width
        pixel_records.append(
            np.column_stack(
                (
                    np.full(samples, index, dtype=np.int32),
                    rows.astype(np.int32),
                    columns.astype(np.int32),
                )
            )
        )
        counts = np.bincount(sampled_labels[valid], minlength=len(LAYOUT_LABELS))
        per_frame.append(
            {
                "index": index,
                "name": semantic_record["name"],
                "sample_count": samples,
                "valid_depth_count": int(valid.sum()),
                "layout_histogram": {str(class_id): int(value) for class_id, value in enumerate(counts)},
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(frames):
            _write_status(component_dir, "sampling_rays", f"{index + 1}/{len(frames)}")

    rays = {
        "origins": np.concatenate(origins),
        "destinations": np.concatenate(destinations),
        "labels": np.concatenate(labels),
        "valid": np.concatenate(validity),
        "pixels": np.concatenate(pixel_records),
        "per_frame": per_frame,
    }
    np.save(component_dir / "full_ray_origins.npy", rays["origins"])
    np.save(component_dir / "full_ray_dests.npy", rays["destinations"])
    np.save(component_dir / "ray_is_valid.npy", rays["valid"])
    np.save(component_dir / "hard_labels_simplified_segmentations.npy", rays["labels"])
    np.save(component_dir / "ray_frame_row_column.npy", rays["pixels"])
    with (component_dir / "per_frame_rays.jsonl").open("w", encoding="utf-8") as handle:
        for record in per_frame:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return rays


def _paper_vertex_votes(
    vertices: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(vertices)
    counts = np.zeros((len(vertices), len(LAYOUT_LABELS)), dtype=np.uint32)
    distances: list[np.ndarray] = []
    chunk_size = 250_000
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        distance, vertex_index = tree.query(points[start:stop], k=1, workers=-1)
        np.add.at(counts, (vertex_index, labels[start:stop]), 1)
        distances.append(distance.astype(np.float32))
    return counts, np.concatenate(distances)


def _source_knn_probabilities(
    vertices: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    probabilities = np.empty((len(vertices), len(LAYOUT_LABELS)), dtype=np.float32)
    chunk_size = 50_000
    for start in range(0, len(vertices), chunk_size):
        stop = min(start + chunk_size, len(vertices))
        _, indices = tree.query(vertices[start:stop], k=k, workers=-1)
        neighbor_labels = labels[indices]
        for class_id in range(len(LAYOUT_LABELS)):
            probabilities[start:stop, class_id] = np.mean(neighbor_labels == class_id, axis=1)
    return probabilities


def _superpoint_hierarchy(
    config: SkeletonConfig,
    vertices: np.ndarray,
    colors: np.ndarray,
    component_dir: Path,
) -> tuple[list[np.ndarray], list[int]]:
    repository = config.superpoint.repository.expanduser().resolve()
    if not repository.is_dir():
        raise SkeletonError(f"Superpoint Transformer repository is missing: {repository}")
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    try:
        import torch
        from src.data import Data
        from src.transforms.graph import AdjacencyGraph
        from src.transforms.neighbors import KNN
        from src.transforms.partition import CutPursuitPartition
        from src.transforms.point import GroundElevation, PointFeatures
    except ImportError as error:
        raise SkeletonError("Superpoint Transformer dependencies are unavailable") from error
    if not torch.cuda.is_available():
        raise SkeletonError("Superpoint preprocessing requires CUDA")

    _write_status(component_dir, "superpoints_knn", f"vertices={len(vertices)}")
    data = Data(
        pos=torch.from_numpy(vertices.astype(np.float32)).cuda(),
        rgb=torch.from_numpy(colors.astype(np.float32)).cuda(),
    )
    data = KNN(
        k=config.superpoint.knn_neighbors,
        r_max=config.superpoint.knn_radius_meters,
        verbose=False,
    )(data)
    data = data.cpu()
    _write_status(component_dir, "superpoints_features", "pgeof + ground elevation")
    data = PointFeatures(
        keys=["rgb", "linearity", "planarity", "scattering", "verticality"],
        k_min=1,
        k_step=-1,
        k_min_search=25,
        overwrite=False,
    )(data)
    data = GroundElevation(z_threshold=1.5, scale=4.0)(data)
    data = data.cuda()
    data = AdjacencyGraph(
        k=config.superpoint.adjacency_neighbors,
        w=config.superpoint.adjacency_weight,
    )(data)
    data = data.cpu()
    data.x = torch.cat(
        (
            data.rgb,
            data.linearity,
            data.planarity,
            data.scattering,
            data.verticality,
            data.elevation,
        ),
        dim=1,
    )
    _write_status(component_dir, "superpoints_cut_pursuit", "hierarchy")
    hierarchy = CutPursuitPartition(
        regularization=list(config.superpoint.regularization),
        spatial_weight=list(config.superpoint.spatial_weight),
        cutoff=list(config.superpoint.cutoff),
        parallel=True,
        iterations=config.superpoint.iterations,
        k_adjacency=config.superpoint.adjacency_neighbors,
        verbose=True,
    )(data)
    expected_levels = config.superpoint.final_level + 1
    if hierarchy.num_levels != expected_levels:
        raise SkeletonError(f"expected {expected_levels} SPT levels, got {hierarchy.num_levels}")
    spt_dir = component_dir / "spt"
    spt_dir.mkdir(exist_ok=False)
    segmentations: list[np.ndarray] = []
    segment_counts: list[int] = []
    for level in range(1, hierarchy.num_levels):
        segmentation = hierarchy.get_super_index(level, low=0).cpu().numpy()
        if len(segmentation) != len(vertices) or segmentation.min() != 0:
            raise SkeletonError("invalid SPT hierarchy index array")
        segment_count = int(segmentation.max()) + 1
        if set(np.unique(segmentation)) != set(range(segment_count)):
            raise SkeletonError("SPT hierarchy indices are not consecutive")
        np.save(spt_dir / f"level_{level}_segmentation.npy", segmentation.astype(np.int32))
        segmentations.append(segmentation)
        segment_counts.append(segment_count)
    del hierarchy, data
    torch.cuda.empty_cache()
    return segmentations, segment_counts


def _aggregate_superpoint_labels(
    component_dir: Path,
    segmentations: list[np.ndarray],
    vote_counts: np.ndarray,
    knn_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    level_records: list[dict[str, Any]] = []
    final_labels: np.ndarray | None = None
    final_probabilities: np.ndarray | None = None
    for level, segmentation in enumerate(segmentations, start=1):
        segment_count = int(segmentation.max()) + 1
        segment_votes = np.zeros((segment_count, len(LAYOUT_LABELS)), dtype=np.uint64)
        np.add.at(segment_votes, segmentation, vote_counts)
        segment_scores = segment_votes.astype(np.float64)
        zero_vote = segment_scores.sum(axis=1) == 0
        if zero_vote.any():
            fallback = np.zeros_like(segment_scores)
            np.add.at(fallback, segmentation, knn_probabilities)
            segment_scores[zero_vote] = fallback[zero_vote]
        totals = segment_scores.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise SkeletonError("a superpoint has no paper votes or KNN fallback")
        probabilities = (segment_scores / totals).astype(np.float32)
        hard = probabilities.argmax(axis=1).astype(np.uint8)
        spt_dir = component_dir / "spt"
        np.save(spt_dir / f"level_{level}_segment_vote_counts.npy", segment_votes)
        np.save(spt_dir / f"level_{level}_segment_probabilities_simplified.npy", probabilities.astype(np.float16))
        np.save(spt_dir / f"level_{level}_segment_hard_assignments_simplified.npy", hard)
        level_records.append(
            {
                "level": level,
                "segment_count": segment_count,
                "zero_paper_vote_segment_count": int(zero_vote.sum()),
                "hard_label_histogram": {
                    label: int((hard == class_id).sum())
                    for class_id, label in enumerate(LAYOUT_LABELS)
                },
            }
        )
        final_labels = hard[segmentation]
        final_probabilities = probabilities[segmentation]
    assert final_labels is not None and final_probabilities is not None
    return final_labels, final_probabilities, level_records


def _mesh_artifact(path: Path, mesh: Any) -> dict[str, Any]:
    import open3d as o3d

    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise SkeletonError(f"failed to write mesh: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
    }


def _write_semantic_meshes(
    component_dir: Path,
    mesh: Any,
    hard_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    import open3d as o3d

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        LAYOUT_PALETTE[hard_labels].astype(np.float64) / 255.0
    )
    records: dict[str, Any] = {
        "semantic_mesh": _mesh_artifact(component_dir / "semantic_mesh.ply", mesh)
    }
    _mesh_artifact(component_dir / "spt" / "mesh_class_colored.ply", mesh)
    groups = {
        "structure": np.isin(hard_labels, [0, 1, 2, 3]),
        "objects": hard_labels == LAYOUT_LABELS.index("object"),
        "stairs": hard_labels == LAYOUT_LABELS.index("stairs"),
        "inaccurate": np.isin(hard_labels, [4, 5, 6]),
    }
    filenames = {
        "structure": "ceiling_wall_floor_mesh.ply",
        "objects": "objects_mesh.ply",
        "stairs": "stair_mesh.ply",
        "inaccurate": "geometrically_inaccurate_mesh.ply",
    }
    for name, keep in groups.items():
        filtered = o3d.geometry.TriangleMesh(mesh)
        filtered.remove_vertices_by_mask(~keep)
        if len(filtered.vertices) == 0:
            records[name] = {"path": None, "vertex_count": 0, "triangle_count": 0}
            continue
        path = component_dir / filenames[name]
        record = _mesh_artifact(path, filtered)
        classes_path = component_dir / f"{Path(filenames[name]).stem}_classes.npy"
        np.save(classes_path, probabilities[keep].astype(np.float16))
        record["classes_path"] = str(classes_path)
        record["classes_sha256"] = _sha256(classes_path)
        records[name] = record
    return records


def _write_ray_visualizations(component_dir: Path, rays: dict[str, Any]) -> dict[str, Any]:
    import open3d as o3d

    valid = rays["valid"]
    points = rays["destinations"][valid]
    labels = rays["labels"][valid]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(LAYOUT_PALETTE[labels].astype(np.float64) / 255.0)
    cloud_path = component_dir / "sampled_semantic_points.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise SkeletonError("failed to write sampled semantic point cloud")
    count = min(20_000, int(valid.sum()))
    indices = np.flatnonzero(valid)[:count]
    origins = rays["origins"][indices]
    destinations = rays["destinations"][indices]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.concatenate((origins, destinations)).astype(np.float64)),
        lines=o3d.utility.Vector2iVector(np.column_stack((np.arange(count), np.arange(count) + count))),
    )
    line_set.colors = o3d.utility.Vector3dVector(LAYOUT_PALETTE[rays["labels"][indices]].astype(np.float64) / 255.0)
    line_path = component_dir / "rays_preview_20000.ply"
    if not o3d.io.write_line_set(str(line_path), line_set, write_ascii=False):
        raise SkeletonError("failed to write ray preview")
    return {
        "sampled_semantic_points": {
            "path": str(cloud_path),
            "sha256": _sha256(cloud_path),
            "point_count": len(points),
        },
        "ray_preview": {
            "path": str(line_path),
            "sha256": _sha256(line_path),
            "ray_count": count,
        },
    }


def run_skeleton(config: SkeletonConfig, command: list[str] | None = None) -> Path:
    """Render raw depths, vote semantics onto the mesh, and extract skeleton subsets."""

    output = config.output.expanduser()
    if output.exists():
        raise SkeletonError(f"skeleton output already exists: {output}")
    inputs = _load_inputs(config)
    output.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.preferred_gpu))

    render_dir = output / "rendered_depth"
    render_command = build_depth_render_command(config.ns_render.expanduser().resolve(), inputs["training_config"], render_dir)
    _write_json(output / "commands.json", {"render_depth": render_command})
    _write_status(output, "rendering_depth", "DN-Splatter raw-depth")
    if not config.ns_render.is_file() or not os.access(config.ns_render, os.X_OK):
        raise SkeletonError(f"ns-render is unavailable: {config.ns_render}")

    try:
        environment = build_training_environment(config.ns_render.expanduser().resolve())
        if config.torch_home is not None:
            environment["TORCH_HOME"] = str(config.torch_home.expanduser().resolve())
        render_cwd = Path("external/dn-splatter")
        if not render_cwd.is_dir():
            render_cwd = config.ns_render.expanduser().resolve().parent
        with (output / "render_depth.log").open("w", encoding="utf-8") as log:
            log.write("command: " + " ".join(render_command) + "\n\n")
            log.flush()
            result = subprocess.run(
                render_command,
                cwd=render_cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise SkeletonError(f"DN-Splatter depth rendering failed with code {result.returncode}; see {output / 'render_depth.log'}")

        frames = inputs["transforms"]["frames"]
        expected_shape = (config.camera.height, config.camera.width)
        _write_status(output, "validating_depth", f"frames={len(frames)}")
        depth_paths, depth_records = _load_rendered_depths(render_dir, frames, expected_shape)
        if len(depth_paths) != len(inputs["semantic_records"]):
            raise SkeletonError("rendered-depth and semantic frame counts differ")

        rays = _sample_rays(config, frames, inputs["semantic_records"], depth_paths, output)
        valid_points = rays["destinations"][rays["valid"]]
        valid_labels = rays["labels"][rays["valid"]]
        if len(valid_points) == 0:
            raise SkeletonError("no valid rays remain after depth validation")

        try:
            import open3d as o3d
        except ImportError as error:
            raise SkeletonError("Open3D is unavailable") from error
        mesh = o3d.io.read_triangle_mesh(str(inputs["mesh_path"]), enable_post_processing=False)
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        triangles = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors).astype(np.float32)
        if len(vertices) == 0 or len(triangles) == 0:
            raise SkeletonError("Poisson mesh is empty")
        if colors.shape != vertices.shape:
            colors = np.full_like(vertices, 0.5)
        _mesh_artifact(output / "mesh.ply", mesh)

        _write_status(output, "paper_vertex_voting", f"rays={len(valid_points)}")
        vertex_votes, projection_distances = _paper_vertex_votes(vertices, valid_points, valid_labels)
        np.save(output / "vertex_vote_counts.npy", vertex_votes)
        np.save(output / "ray_to_mesh_distance_meters.npy", projection_distances.astype(np.float32))
        _write_status(output, "source_knn_transfer", "k=5")
        knn_probabilities = _source_knn_probabilities(vertices, valid_points, valid_labels, k=5)
        np.save(output / "vertex_probabilities_knn5.npy", knn_probabilities.astype(np.float16))

        segmentations, segment_counts = _superpoint_hierarchy(config, vertices, colors, output)
        _write_status(output, "aggregating_superpoint_votes", f"levels={len(segmentations)}")
        hard_labels, vertex_probabilities, level_records = superpoint_aggregate_superpoint_labels(
            output, segmentations, vertex_votes, knn_probabilities
        )
        np.save(output / "vertex_probabilities.npy", vertex_probabilities.astype(np.float16))
        np.save(output / "vertex_hard_assignments.npy", hard_labels)
        np.save(output / "simplified_segmentation_labels.npy", np.asarray(LAYOUT_LABELS))

        _write_status(output, "writing_filtered_meshes", "structure/object/stair")
        mesh_records = _write_semantic_meshes(output, mesh, hard_labels, vertex_probabilities)
        ray_visualizations = _write_ray_visualizations(output, rays)

        array_names = (
            "full_ray_origins.npy",
            "full_ray_dests.npy",
            "ray_is_valid.npy",
            "hard_labels_simplified_segmentations.npy",
            "ray_frame_row_column.npy",
            "vertex_vote_counts.npy",
            "ray_to_mesh_distance_meters.npy",
            "vertex_probabilities_knn5.npy",
            "vertex_probabilities.npy",
            "vertex_hard_assignments.npy",
            "simplified_segmentation_labels.npy",
        )
        array_records = {
            name: {
                "path": str(output / name),
                "sha256": _sha256(output / name),
                "size_bytes": (output / name).stat().st_size,
            }
            for name in array_names
        }
        class_histogram = np.bincount(hard_labels, minlength=len(LAYOUT_LABELS))
        manifest = {
            "schema_version": 1,
            "component": "skeleton",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": config.random_seed,
            "inputs": {
                "transforms": {"path": str(inputs["transforms_path"]), "sha256": _sha256(inputs["transforms_path"])},
                "dn_splatter_manifest": {"path": str(inputs["dn_manifest_path"]), "sha256": _sha256(inputs["dn_manifest_path"])},
                "mesh_manifest": {"path": str(inputs["mesh_manifest_path"]), "sha256": _sha256(inputs["mesh_manifest_path"])},
                "oneformer_manifest": {"path": str(inputs["oneformer_manifest_path"]), "sha256": _sha256(inputs["oneformer_manifest_path"])},
            },
            "algorithm": {
                "depth": "DN-Splatter final-checkpoint rendered raw-depth in meters",
                "samples_per_frame": config.samples_per_frame,
                "sampling": "uniform without replacement over all image pixels",
                "pixel_center_offset": 0.5,
                "paper_projection": "each valid back-projected point votes at its nearest mesh vertex",
                "zero_vote_fallback": "K=5 nearest-ray probabilities, only for superpoints with zero paper votes",
                "semantic_classes": list(LAYOUT_LABELS),
                "superpoint_preprocessing": "Superpoint Transformer geometric features and Cut Pursuit hierarchy",
                "knn_neighbors": config.superpoint.knn_neighbors,
                "knn_radius_meters": config.superpoint.knn_radius_meters,
                "adjacency_neighbors": config.superpoint.adjacency_neighbors,
                "adjacency_weight": config.superpoint.adjacency_weight,
                "regularization": list(config.superpoint.regularization),
                "spatial_weight": list(config.superpoint.spatial_weight),
                "cutoff": list(config.superpoint.cutoff),
                "iterations": config.superpoint.iterations,
                "final_level": config.superpoint.final_level,
            },
            "outputs": {
                "rendered_depth_root": str(render_dir),
                "rendered_depth_records": depth_records,
                "arrays": array_records,
                "superpoint_dir": str(output / "spt"),
                "superpoint_levels": level_records,
                "meshes": mesh_records,
                "visualizations": ray_visualizations,
            },
            "statistics": {
                "frame_count": len(frames),
                "ray_count": len(rays["valid"]),
                "valid_ray_count": int(rays["valid"].sum()),
                "valid_ray_fraction": float(rays["valid"].mean()),
                "mesh_vertex_count": len(vertices),
                "mesh_triangle_count": len(triangles),
                "paper_voted_vertex_count": int((vertex_votes.sum(axis=1) > 0).sum()),
                "paper_unvoted_vertex_count": int((vertex_votes.sum(axis=1) == 0).sum()),
                "ray_to_mesh_distance_meters": {
                    "minimum": float(projection_distances.min()),
                    "median": float(np.median(projection_distances)),
                    "p95": float(np.quantile(projection_distances, 0.95)),
                    "maximum": float(projection_distances.max()),
                },
                "superpoint_segment_counts": segment_counts,
                "final_vertex_label_histogram": {
                    label: int(class_histogram[index])
                    for index, label in enumerate(LAYOUT_LABELS)
                },
            },
            "validation": {
                "render_return_code": result.returncode,
                "one_depth_per_frame": len(depth_paths) == len(frames),
                "one_semantic_map_per_frame": len(inputs["semantic_records"]) == len(frames),
                "exact_paper_sample_count": len(rays["valid"]) == len(frames) * config.samples_per_frame,
                "all_mesh_vertices_labeled": len(hard_labels) == len(vertices),
                "all_final_labels_in_range": int(hard_labels.max()) < len(LAYOUT_LABELS),
                "expected_superpoint_levels": len(segmentations) == config.superpoint.final_level,
                "no_ground_truth_inputs_used": True,
            },
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "warnings": [
                "K=5 transfer is retained only as a zero-vote-superpoint fallback.",
            ],
        }
        manifest_path = output / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(output, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(output, "failed", str(error))
        if isinstance(error, SkeletonError):
            raise
        raise SkeletonError(str(error)) from error


def _float_tuple(values: str) -> tuple[float, ...]:
    return tuple(float(value) for value in values.split(",") if value)


def _int_tuple(values: str) -> tuple[int, ...]:
    return tuple(int(value) for value in values.split(",") if value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a semantic layout skeleton from Section 4.1 artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--dn-splatter", type=Path, required=True)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--oneformer", type=Path, required=True)
    parser.add_argument("--camera", type=Path, default=Path("camera_param.json"))
    parser.add_argument("--ns-render", type=Path, required=True)
    parser.add_argument("--superpoint-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--samples-per-frame", type=int, default=5000)
    parser.add_argument("--preferred-gpu", type=int, default=0)
    parser.add_argument("--torch-home", type=Path)
    parser.add_argument("--spt-knn-neighbors", type=int, default=15)
    parser.add_argument("--spt-knn-radius-meters", type=float, default=0.2)
    parser.add_argument("--spt-adjacency-neighbors", type=int, default=10)
    parser.add_argument("--spt-adjacency-weight", type=float, default=1.0)
    parser.add_argument("--spt-regularization", default="0.03,0.06,0.12")
    parser.add_argument("--spt-spatial-weight", default="0.01,0.02,0.04")
    parser.add_argument("--spt-cutoff", default="10,20,40")
    parser.add_argument("--spt-iterations", type=int, default=10)
    parser.add_argument("--spt-final-level", type=int, default=3)
    args = parser.parse_args()
    superpoint = SuperpointConfig(
        repository=args.superpoint_repo,
        knn_neighbors=args.spt_knn_neighbors,
        knn_radius_meters=args.spt_knn_radius_meters,
        adjacency_neighbors=args.spt_adjacency_neighbors,
        adjacency_weight=args.spt_adjacency_weight,
        regularization=_float_tuple(args.spt_regularization),
        spatial_weight=_float_tuple(args.spt_spatial_weight),
        cutoff=_int_tuple(args.spt_cutoff),
        iterations=args.spt_iterations,
        final_level=args.spt_final_level,
    )
    if not (
        len(superpoint.regularization)
        == len(superpoint.spatial_weight)
        == len(superpoint.cutoff)
        == superpoint.final_level
    ):
        raise SkeletonError("SPT hierarchy parameters must all have final_level entries")
    config = SkeletonConfig(
        transforms=args.transforms,
        dn_splatter=args.dn_splatter,
        mesh_manifest=args.mesh_manifest,
        oneformer=args.oneformer,
        camera=load_camera(args.camera),
        ns_render=args.ns_render,
        superpoint=superpoint,
        output=args.output,
        random_seed=args.random_seed,
        samples_per_frame=args.samples_per_frame,
        preferred_gpu=args.preferred_gpu,
        torch_home=args.torch_home,
    )
    manifest = run_skeleton(config, command=sys.argv)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
