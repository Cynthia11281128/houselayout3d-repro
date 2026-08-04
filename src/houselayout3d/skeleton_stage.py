"""Paper-faithful semantic mesh voting and layout-skeleton extraction."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .dn_splatter_stage import build_training_environment
from .oneformer_stage import LAYOUT_LABELS, LAYOUT_PALETTE
from .stages import Stage


class SkeletonStageError(RuntimeError):
    """Raised when depth rendering, voting, or superpoint extraction fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_status(stage_dir: Path, state: str, detail: str = "") -> None:
    _write_json(
        stage_dir / "STATUS.json",
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    with (stage_dir / "skeleton.log").open("a", encoding="utf-8") as log:
        log.write(f"{datetime.now(timezone.utc).isoformat()} {state} {detail}\n")


def build_depth_render_command(
    config: PipelineConfig, training_config: Path, output_dir: Path
) -> list[str]:
    """Build the official Nerfstudio raw-depth rendering command."""

    return [
        str(config.skeleton.render_executable),
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
    """Back-project selected z-depth pixels to world coordinates."""

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


def _verify_prior_stages(
    config: PipelineConfig, run_id: str
) -> dict[str, Any]:
    run_dir = config.storage.outputs / config.scene / run_id
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for stage in (Stage.POSE, Stage.DN_SPLATTER, Stage.MESH, Stage.ONEFORMER):
        path = run_dir / stage.value / "manifest.json"
        if not path.is_file():
            raise SkeletonStageError(f"prior-stage manifest is missing: {path}")
        manifest = _read_json(path)
        if manifest.get("status") != "complete":
            raise SkeletonStageError(f"{stage.value} manifest is not complete")
        manifests[stage.value] = (path, manifest)

    pose_manifest = manifests[Stage.POSE.value][1]
    transforms_path = Path(pose_manifest["outputs"]["transforms_json"])
    if (
        not transforms_path.is_file()
        or _sha256(transforms_path)
        != pose_manifest["outputs"]["transforms_json_sha256"]
    ):
        raise SkeletonStageError("01_pose transforms hash no longer matches")

    dn_manifest = manifests[Stage.DN_SPLATTER.value][1]
    training_config = Path(dn_manifest["outputs"]["training_config"])
    checkpoint = Path(dn_manifest["outputs"]["final_checkpoint"])
    if (
        not training_config.is_file()
        or _sha256(training_config)
        != dn_manifest["outputs"]["training_config_sha256"]
    ):
        raise SkeletonStageError("03_dn_splatter training-config hash no longer matches")
    if (
        not checkpoint.is_file()
        or _sha256(checkpoint)
        != dn_manifest["outputs"]["final_checkpoint_sha256"]
    ):
        raise SkeletonStageError("03_dn_splatter checkpoint hash no longer matches")

    mesh_manifest = manifests[Stage.MESH.value][1]
    mesh_record = mesh_manifest["outputs"]["poisson_mesh"]
    mesh_path = Path(mesh_record["path"])
    if not mesh_path.is_file() or _sha256(mesh_path) != mesh_record["sha256"]:
        raise SkeletonStageError("04_mesh Poisson hash no longer matches")

    oneformer_manifest = manifests[Stage.ONEFORMER.value][1]
    per_image_path = Path(oneformer_manifest["outputs"]["per_image"]["path"])
    if (
        not per_image_path.is_file()
        or _sha256(per_image_path)
        != oneformer_manifest["outputs"]["per_image"]["sha256"]
    ):
        raise SkeletonStageError("05_oneformer per-image manifest hash no longer matches")
    semantic_records = [
        json.loads(line)
        for line in per_image_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(semantic_records) != oneformer_manifest["validation"]["frame_count"]:
        raise SkeletonStageError("05_oneformer per-image record count is inconsistent")
    for record in semantic_records:
        path = Path(record["layout_path"])
        if not path.is_file() or _sha256(path) != record["layout_sha256"]:
            raise SkeletonStageError(f"05_oneformer layout hash mismatch: {path}")

    transforms = _read_json(transforms_path)
    frames = transforms.get("frames", [])
    if [Path(frame["file_path"]).name for frame in frames] != [
        record["name"] for record in semantic_records
    ]:
        raise SkeletonStageError("pose and OneFormer frame orders do not match")
    return {
        "run_dir": run_dir,
        "manifest_records": manifests,
        "transforms_path": transforms_path,
        "transforms": transforms,
        "training_config": training_config,
        "checkpoint": checkpoint,
        "mesh_path": mesh_path,
        "mesh_record": mesh_record,
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
                raise SkeletonStageError(f"rendered raw depth is missing: {path}")
            path = matches[0]
        with gzip.open(path, "rb") as handle:
            depth = np.load(handle)
        depth = np.asarray(depth).squeeze()
        if depth.shape != expected_shape:
            raise SkeletonStageError(
                f"rendered depth has shape {depth.shape}, expected {expected_shape}: {path}"
            )
        finite = np.isfinite(depth)
        if not finite.all():
            raise SkeletonStageError(f"rendered depth contains non-finite values: {path}")
        valid = depth > 0
        if not valid.any():
            raise SkeletonStageError(f"rendered depth contains no positive values: {path}")
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
                "positive_fraction": float(valid.mean()),
            }
        )
    extras = set(raw_dir.rglob("*.npy.gz")) - set(paths)
    if extras:
        raise SkeletonStageError(
            f"raw-depth rendering produced {len(extras)} unexpected files"
        )
    return paths, records


def _sample_rays(
    config: PipelineConfig,
    frames: list[dict[str, Any]],
    semantic_records: list[dict[str, Any]],
    depth_paths: list[Path],
    stage_dir: Path,
) -> dict[str, Any]:
    from PIL import Image

    height = config.input.camera.height
    width = config.input.camera.width
    samples = config.skeleton.samples_per_frame
    if samples > height * width:
        raise SkeletonStageError("samples_per_frame exceeds the image pixel count")
    rng = np.random.default_rng(config.runtime.random_seed)
    origins: list[np.ndarray] = []
    destinations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    validity: list[np.ndarray] = []
    pixel_records: list[np.ndarray] = []
    per_frame: list[dict[str, Any]] = []
    for index, (frame, semantic_record, depth_path) in enumerate(
        zip(frames, semantic_records, depth_paths)
    ):
        with gzip.open(depth_path, "rb") as handle:
            depth = np.asarray(np.load(handle)).squeeze().astype(np.float32)
        with Image.open(semantic_record["layout_path"]) as image:
            semantic = np.asarray(image, dtype=np.uint8)
        if semantic.shape != (height, width):
            raise SkeletonStageError("OneFormer map shape changed during skeleton extraction")
        flat = rng.choice(height * width, size=samples, replace=False)
        valid = np.isfinite(depth.ravel()[flat]) & (depth.ravel()[flat] > 0)
        inpainted = depth.copy()
        inpainted[~np.isfinite(inpainted) | (inpainted <= 0)] = 0.5
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        points = backproject_samples(
            inpainted,
            flat,
            c2w,
            config.input.camera.fx,
            config.input.camera.fy,
            config.input.camera.cx,
            config.input.camera.cy,
        )
        origin = np.repeat(c2w[None, :3, 3], samples, axis=0)
        sampled_labels = semantic.ravel()[flat]
        if int(sampled_labels.max()) >= len(LAYOUT_LABELS):
            raise SkeletonStageError("sampled layout label is out of range")
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
                "layout_histogram": {
                    str(class_id): int(value)
                    for class_id, value in enumerate(counts)
                },
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(frames):
            _write_status(stage_dir, "sampling_rays", f"{index + 1}/{len(frames)}")

    ray_origins = np.concatenate(origins)
    ray_destinations = np.concatenate(destinations)
    ray_labels = np.concatenate(labels)
    ray_valid = np.concatenate(validity)
    ray_pixels = np.concatenate(pixel_records)
    np.save(stage_dir / "full_ray_origins.npy", ray_origins)
    np.save(stage_dir / "full_ray_dests.npy", ray_destinations)
    np.save(stage_dir / "ray_is_valid.npy", ray_valid)
    np.save(stage_dir / "hard_labels_simplified_segmentations.npy", ray_labels)
    np.save(stage_dir / "ray_frame_row_column.npy", ray_pixels)
    with (stage_dir / "per_frame_rays.jsonl").open("w", encoding="utf-8") as handle:
        for record in per_frame:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "origins": ray_origins,
        "destinations": ray_destinations,
        "labels": ray_labels,
        "valid": ray_valid,
        "pixels": ray_pixels,
        "per_frame": per_frame,
    }


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
    """Reproduce the unofficial code's K=5 mesh-vertex feature transfer."""

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    probabilities = np.empty((len(vertices), len(LAYOUT_LABELS)), dtype=np.float32)
    chunk_size = 50_000
    for start in range(0, len(vertices), chunk_size):
        stop = min(start + chunk_size, len(vertices))
        _, indices = tree.query(vertices[start:stop], k=k, workers=-1)
        neighbor_labels = labels[indices]
        for class_id in range(len(LAYOUT_LABELS)):
            probabilities[start:stop, class_id] = np.mean(
                neighbor_labels == class_id, axis=1
            )
    return probabilities


def _superpoint_hierarchy(
    config: PipelineConfig,
    vertices: np.ndarray,
    colors: np.ndarray,
    stage_dir: Path,
) -> tuple[list[np.ndarray], list[int]]:
    repository = config.skeleton.superpoint_repository
    if not repository.is_dir():
        raise SkeletonStageError(f"Superpoint Transformer is missing: {repository}")
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
        raise SkeletonStageError("Superpoint Transformer dependencies are unavailable") from error
    if not torch.cuda.is_available():
        raise SkeletonStageError("Superpoint preprocessing requires CUDA")

    _write_status(stage_dir, "superpoints_knn", f"vertices={len(vertices)}")
    data = Data(
        pos=torch.from_numpy(vertices.astype(np.float32)).cuda(),
        rgb=torch.from_numpy(colors.astype(np.float32)).cuda(),
    )
    data = KNN(
        k=config.skeleton.knn_neighbors,
        r_max=config.skeleton.knn_radius_meters,
        verbose=False,
    )(data)
    data = data.cpu()
    _write_status(stage_dir, "superpoints_features", "pgeof + ground elevation")
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
        k=config.skeleton.adjacency_neighbors,
        w=config.skeleton.adjacency_weight,
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
    _write_status(stage_dir, "superpoints_cut_pursuit", "three hierarchy levels")
    hierarchy = CutPursuitPartition(
        regularization=list(config.skeleton.regularization),
        spatial_weight=list(config.skeleton.spatial_weight),
        cutoff=list(config.skeleton.cutoff),
        parallel=True,
        iterations=config.skeleton.iterations,
        k_adjacency=config.skeleton.adjacency_neighbors,
        verbose=True,
    )(data)
    expected_levels = config.skeleton.final_level + 1
    if hierarchy.num_levels != expected_levels:
        raise SkeletonStageError(
            f"expected {expected_levels} SPT levels, got {hierarchy.num_levels}"
        )
    segmentations: list[np.ndarray] = []
    segment_counts: list[int] = []
    spt_dir = stage_dir / "spt"
    spt_dir.mkdir(exist_ok=False)
    for level in range(1, hierarchy.num_levels):
        segmentation = hierarchy.get_super_index(level, low=0).cpu().numpy()
        if len(segmentation) != len(vertices) or segmentation.min() != 0:
            raise SkeletonStageError("invalid SPT hierarchy index array")
        segment_count = int(segmentation.max()) + 1
        if set(np.unique(segmentation)) != set(range(segment_count)):
            raise SkeletonStageError("SPT hierarchy indices are not consecutive")
        np.save(spt_dir / f"level_{level}_segmentation.npy", segmentation.astype(np.int32))
        segmentations.append(segmentation)
        segment_counts.append(segment_count)
    del hierarchy, data
    torch.cuda.empty_cache()
    return segmentations, segment_counts


def _aggregate_superpoint_labels(
    stage_dir: Path,
    segmentations: list[np.ndarray],
    vote_counts: np.ndarray,
    knn_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    level_records: list[dict[str, Any]] = []
    final_labels: np.ndarray | None = None
    final_probabilities: np.ndarray | None = None
    for level, segmentation in enumerate(segmentations, start=1):
        segment_count = int(segmentation.max()) + 1
        segment_votes = np.zeros(
            (segment_count, len(LAYOUT_LABELS)), dtype=np.uint64
        )
        np.add.at(segment_votes, segmentation, vote_counts)
        segment_scores = segment_votes.astype(np.float64)
        zero_vote = segment_scores.sum(axis=1) == 0
        if zero_vote.any():
            fallback = np.zeros_like(segment_scores)
            np.add.at(fallback, segmentation, knn_probabilities)
            segment_scores[zero_vote] = fallback[zero_vote]
        totals = segment_scores.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise SkeletonStageError("a superpoint has no paper votes or KNN fallback")
        probabilities = (segment_scores / totals).astype(np.float32)
        hard = probabilities.argmax(axis=1).astype(np.uint8)
        spt_dir = stage_dir / "spt"
        np.save(
            spt_dir / f"level_{level}_segment_vote_counts.npy", segment_votes
        )
        np.save(
            spt_dir / f"level_{level}_segment_probabilities_simplified.npy",
            probabilities.astype(np.float16),
        )
        np.save(
            spt_dir / f"level_{level}_segment_hard_assignments_simplified.npy",
            hard,
        )
        level_records.append(
            {
                "level": level,
                "segment_count": segment_count,
                "zero_paper_vote_segment_count": int(zero_vote.sum()),
                "hard_label_histogram": {
                    LAYOUT_LABELS[class_id]: int((hard == class_id).sum())
                    for class_id in range(len(LAYOUT_LABELS))
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
        raise SkeletonStageError(f"failed to write mesh: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles),
    }


def _write_semantic_meshes(
    stage_dir: Path,
    mesh: Any,
    hard_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    import open3d as o3d

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        LAYOUT_PALETTE[hard_labels].astype(np.float64) / 255.0
    )
    semantic_record = _mesh_artifact(stage_dir / "semantic_mesh.ply", mesh)
    _mesh_artifact(stage_dir / "spt" / "mesh_class_colored.ply", mesh)
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
    records: dict[str, Any] = {"semantic_mesh": semantic_record}
    for name, keep in groups.items():
        filtered = o3d.geometry.TriangleMesh(mesh)
        filtered.remove_vertices_by_mask(~keep)
        if len(filtered.vertices) == 0:
            records[name] = {
                "path": None,
                "vertex_count": 0,
                "triangle_count": 0,
            }
            continue
        path = stage_dir / filenames[name]
        record = _mesh_artifact(path, filtered)
        classes_path = stage_dir / f"{Path(filenames[name]).stem}_classes.npy"
        np.save(classes_path, probabilities[keep].astype(np.float16))
        record["classes_path"] = str(classes_path)
        record["classes_sha256"] = _sha256(classes_path)
        records[name] = record
    return records


def _write_ray_visualizations(stage_dir: Path, rays: dict[str, Any]) -> dict[str, Any]:
    import open3d as o3d

    valid = rays["valid"]
    points = rays["destinations"][valid]
    labels = rays["labels"][valid]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(
        LAYOUT_PALETTE[labels].astype(np.float64) / 255.0
    )
    cloud_path = stage_dir / "sampled_semantic_points.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False):
        raise SkeletonStageError("failed to write sampled semantic point cloud")

    count = min(20_000, int(valid.sum()))
    indices = np.flatnonzero(valid)[:count]
    origins = rays["origins"][indices]
    destinations = rays["destinations"][indices]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(
            np.concatenate((origins, destinations)).astype(np.float64)
        ),
        lines=o3d.utility.Vector2iVector(
            np.column_stack((np.arange(count), np.arange(count) + count))
        ),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        LAYOUT_PALETTE[rays["labels"][indices]].astype(np.float64) / 255.0
    )
    line_path = stage_dir / "rays_preview_20000.ply"
    if not o3d.io.write_line_set(str(line_path), line_set, write_ascii=False):
        raise SkeletonStageError("failed to write ray preview")
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


def run_skeleton(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Render final-model depths, vote semantics, and extract the skeleton."""

    inputs = _verify_prior_stages(config, run_id)
    stage_dir = inputs["run_dir"] / Stage.SKELETON.value
    if stage_dir.exists():
        raise SkeletonStageError(
            f"skeleton stage already exists and will not be overwritten: {stage_dir}"
        )
    stage_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.runtime.preferred_gpu))
    render_dir = stage_dir / "rendered_depth"
    render_command = build_depth_render_command(
        config, inputs["training_config"], render_dir
    )
    _write_json(stage_dir / "commands.json", {"render_depth": render_command})
    _write_status(stage_dir, "rendering_depth", "DN-Splatter raw-depth")
    if (
        not config.skeleton.render_executable.is_file()
        or not os.access(config.skeleton.render_executable, os.X_OK)
    ):
        error = f"ns-render is unavailable: {config.skeleton.render_executable}"
        _write_status(stage_dir, "failed", error)
        raise SkeletonStageError(error)

    try:
        environment = build_training_environment(config.skeleton.render_executable)
        environment["TORCH_HOME"] = str(config.storage.weights / "torch")
        with (stage_dir / "render_depth.log").open("w", encoding="utf-8") as log:
            log.write("command: " + " ".join(render_command) + "\n\n")
            log.flush()
            result = subprocess.run(
                render_command,
                cwd=config.metric3d.repository.parent / "dn-splatter",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise SkeletonStageError(
                f"DN-Splatter depth rendering failed with code {result.returncode}; "
                f"see {stage_dir / 'render_depth.log'}"
            )

        frames = inputs["transforms"]["frames"]
        expected_shape = (config.input.camera.height, config.input.camera.width)
        _write_status(stage_dir, "validating_depth", f"frames={len(frames)}")
        depth_paths, depth_records = _load_rendered_depths(
            render_dir, frames, expected_shape
        )
        if len(depth_paths) != len(inputs["semantic_records"]):
            raise SkeletonStageError("rendered-depth and semantic frame counts differ")

        rays = _sample_rays(
            config,
            frames,
            inputs["semantic_records"],
            depth_paths,
            stage_dir,
        )
        valid_points = rays["destinations"][rays["valid"]]
        valid_labels = rays["labels"][rays["valid"]]
        if len(valid_points) == 0:
            raise SkeletonStageError("no valid rays remain after depth validation")

        try:
            import open3d as o3d
        except ImportError as error:
            raise SkeletonStageError("Open3D is unavailable") from error
        mesh = o3d.io.read_triangle_mesh(
            str(inputs["mesh_path"]), enable_post_processing=False
        )
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        triangles = np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors).astype(np.float32)
        if len(vertices) == 0 or len(triangles) == 0:
            raise SkeletonStageError("04_mesh Poisson artifact is empty")
        if colors.shape != vertices.shape:
            colors = np.full_like(vertices, 0.5)
        _mesh_artifact(stage_dir / "mesh.ply", mesh)

        _write_status(stage_dir, "paper_vertex_voting", f"rays={len(valid_points)}")
        vertex_votes, projection_distances = _paper_vertex_votes(
            vertices, valid_points, valid_labels
        )
        np.save(stage_dir / "vertex_vote_counts.npy", vertex_votes)
        np.save(
            stage_dir / "ray_to_mesh_distance_meters.npy",
            projection_distances.astype(np.float32),
        )
        _write_status(stage_dir, "source_knn_transfer", "k=5")
        knn_probabilities = _source_knn_probabilities(
            vertices, valid_points, valid_labels, k=5
        )
        np.save(
            stage_dir / "vertex_probabilities_knn5.npy",
            knn_probabilities.astype(np.float16),
        )

        segmentations, segment_counts = _superpoint_hierarchy(
            config, vertices, colors, stage_dir
        )
        _write_status(stage_dir, "aggregating_superpoint_votes", "levels=1..3")
        hard_labels, vertex_probabilities, level_records = (
            _aggregate_superpoint_labels(
                stage_dir, segmentations, vertex_votes, knn_probabilities
            )
        )
        np.save(stage_dir / "vertex_probabilities.npy", vertex_probabilities.astype(np.float16))
        np.save(stage_dir / "vertex_hard_assignments.npy", hard_labels)
        np.save(
            stage_dir / "simplified_segmentation_labels.npy",
            np.asarray(LAYOUT_LABELS),
        )

        _write_status(stage_dir, "writing_filtered_meshes", "structure/object/stair")
        mesh_records = _write_semantic_meshes(
            stage_dir, mesh, hard_labels, vertex_probabilities
        )
        ray_visualizations = _write_ray_visualizations(stage_dir, rays)

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
                "path": str(stage_dir / name),
                "sha256": _sha256(stage_dir / name),
                "size_bytes": (stage_dir / name).stat().st_size,
            }
            for name in array_names
        }
        finished_at = datetime.now(timezone.utc)
        class_histogram = np.bincount(
            hard_labels, minlength=len(LAYOUT_LABELS)
        )
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.SKELETON.value,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": config.runtime.random_seed,
            "inputs": {
                stage: {"path": str(path), "sha256": _sha256(path)}
                for stage, (path, _) in inputs["manifest_records"].items()
            },
            "algorithm": {
                "depth": "DN-Splatter final-checkpoint rendered raw-depth in meters",
                "samples_per_frame": config.skeleton.samples_per_frame,
                "sampling": "uniform without replacement over all image pixels",
                "pixel_center_offset": 0.5,
                "paper_projection": "each valid back-projected point votes at its nearest mesh vertex",
                "unofficial_source_transfer": "K=5 nearest-ray probabilities, used only when a whole superpoint has zero paper votes",
                "semantic_classes": list(LAYOUT_LABELS),
                "superpoint_preprocessing": "Superpoint Transformer ScanNet geometric feature and Cut Pursuit hierarchy",
                "knn_neighbors": config.skeleton.knn_neighbors,
                "knn_radius_meters": config.skeleton.knn_radius_meters,
                "adjacency_neighbors": config.skeleton.adjacency_neighbors,
                "adjacency_weight": config.skeleton.adjacency_weight,
                "regularization": list(config.skeleton.regularization),
                "spatial_weight": list(config.skeleton.spatial_weight),
                "cutoff": list(config.skeleton.cutoff),
                "iterations": config.skeleton.iterations,
                "final_level": config.skeleton.final_level,
            },
            "outputs": {
                "rendered_depth_root": str(render_dir),
                "rendered_depth_records": depth_records,
                "arrays": array_records,
                "superpoint_dir": str(stage_dir / "spt"),
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
                "exact_paper_sample_count": len(rays["valid"])
                == len(frames) * config.skeleton.samples_per_frame,
                "all_mesh_vertices_labeled": len(hard_labels) == len(vertices),
                "all_final_labels_in_range": int(hard_labels.max()) < len(LAYOUT_LABELS),
                "three_superpoint_levels": len(segmentations) == 3,
                "no_ground_truth_inputs_used": True,
            },
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "warnings": [
                "The unofficial source defaults to 3000 samples per frame; this implementation uses the paper's explicit M=5000.",
                "K=5 transfer from the partial source is retained only as a zero-vote-superpoint fallback; the primary labels use the paper's ray-to-nearest-vertex votes.",
            ],
        }
        manifest_path = stage_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(stage_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(stage_dir, "failed", str(error))
        if isinstance(error, SkeletonStageError):
            raise
        raise SkeletonStageError(str(error)) from error
