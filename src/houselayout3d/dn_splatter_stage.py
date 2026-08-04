"""Known-pose DN-Splatter dataset preparation and training stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .stages import Stage


class DNSplatterStageError(RuntimeError):
    """Raised when DN-Splatter preparation, training, or validation fails."""


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


def _verify_inputs(
    config: PipelineConfig, run_id: str
) -> tuple[Path, Path, Path, list[str], dict[str, Any]]:
    run_dir = config.storage.outputs / config.scene / run_id
    input_dir = run_dir / Stage.INPUT.value
    pose_dir = run_dir / Stage.POSE.value
    metric_dir = run_dir / Stage.METRIC3D.value
    image_list_path = input_dir / "images.txt"
    pose_transforms_path = pose_dir / "transforms.json"
    metric_manifest_path = metric_dir / "manifest.json"
    for required in (
        input_dir / "manifest.json",
        pose_dir / "manifest.json",
        metric_manifest_path,
        image_list_path,
        pose_transforms_path,
    ):
        if not required.is_file():
            raise DNSplatterStageError(f"required prior artifact is missing: {required}")

    input_manifest = _read_json(input_dir / "manifest.json")
    pose_manifest = _read_json(pose_dir / "manifest.json")
    metric_manifest = _read_json(metric_manifest_path)
    for label, manifest in (
        (Stage.INPUT.value, input_manifest),
        (Stage.POSE.value, pose_manifest),
        (Stage.METRIC3D.value, metric_manifest),
    ):
        if manifest.get("status") != "complete":
            raise DNSplatterStageError(f"{label} manifest is not complete")
    if _sha256(image_list_path) != input_manifest["outputs"]["image_list_sha256"]:
        raise DNSplatterStageError("00_input/images.txt hash no longer matches")
    if _sha256(pose_transforms_path) != pose_manifest["outputs"]["transforms_json_sha256"]:
        raise DNSplatterStageError("01_pose/transforms.json hash no longer matches")
    if _sha256(metric_dir / "per_image.jsonl") != metric_manifest["outputs"]["per_image_records_sha256"]:
        raise DNSplatterStageError("02_metric3d/per_image.jsonl hash no longer matches")

    image_names = image_list_path.read_text(encoding="utf-8").splitlines()
    expected_count = input_manifest["validation"]["image_count"]
    if len(image_names) != expected_count:
        raise DNSplatterStageError("approved image count changed")
    if metric_manifest["validation"]["depth_file_count"] != expected_count:
        raise DNSplatterStageError("Metric3D depth count does not match images")
    if pose_manifest["validation"]["pose_count"] != expected_count:
        raise DNSplatterStageError("known-pose count does not match images")
    return pose_transforms_path, metric_dir, metric_manifest_path, image_names, pose_manifest


def _create_seed_pointcloud(
    config: PipelineConfig,
    transforms: dict[str, Any],
    metric_dir: Path,
    image_names: list[str],
    output_path: Path,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        import open3d as o3d
    except ImportError as error:
        raise DNSplatterStageError(
            "seed point-cloud dependencies are unavailable; use the nerfstudio environment"
        ) from error

    frames = transforms.get("frames")
    if not isinstance(frames, list) or len(frames) != len(image_names):
        raise DNSplatterStageError("01_pose transforms frame count is invalid")
    frame_by_name = {Path(frame["file_path"]).name: frame for frame in frames}
    if sorted(frame_by_name) != sorted(image_names):
        raise DNSplatterStageError("01_pose transform names do not match approved images")

    camera = config.input.camera
    stride = config.dn_splatter.seed_stride
    pixel_y = np.arange(0, camera.height, stride, dtype=np.float32)
    pixel_x = np.arange(0, camera.width, stride, dtype=np.float32)
    uu, vv = np.meshgrid(pixel_x, pixel_y)
    point_chunks = []
    color_chunks = []
    valid_per_frame: list[int] = []
    saturated_rejected = 0

    for name in image_names:
        depth_path = metric_dir / "depth" / f"{Path(name).stem}.npy"
        if not depth_path.is_file():
            raise DNSplatterStageError(f"Metric3D depth is missing: {depth_path}")
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != (camera.height, camera.width) or depth.dtype != np.float32:
            raise DNSplatterStageError(f"invalid Metric3D depth artifact: {depth_path}")
        sampled_depth = depth[::stride, ::stride]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= config.dn_splatter.seed_minimum_depth_meters)
            & (sampled_depth <= config.dn_splatter.seed_maximum_depth_meters)
        )
        saturated_rejected += int((sampled_depth > config.dn_splatter.seed_maximum_depth_meters).sum())
        z = sampled_depth[valid]
        x = ((uu[valid] - camera.cx) / camera.fx) * z
        y = ((vv[valid] - camera.cy) / camera.fy) * z
        points_opengl = np.stack((x, -y, -z), axis=1).astype(np.float32)

        transform = np.asarray(
            frame_by_name[name]["transform_matrix"], dtype=np.float64
        )
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise DNSplatterStageError(f"invalid camera transform for {name}")
        world = (
            points_opengl.astype(np.float64) @ transform[:3, :3].T
            + transform[:3, 3]
        ).astype(np.float32)
        rgb_bgr = cv2.imread(str(config.input.images / name), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise DNSplatterStageError(f"cannot read image for seed colors: {name}")
        rgb = rgb_bgr[:, :, ::-1]
        colors = rgb[::stride, ::stride][valid]
        point_chunks.append(world)
        color_chunks.append(colors)
        valid_per_frame.append(int(valid.sum()))

    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    raw_count = int(points.shape[0])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    cloud = cloud.voxel_down_sample(config.dn_splatter.seed_voxel_size_meters)
    voxel_points = np.asarray(cloud.points)
    voxel_colors = np.asarray(cloud.colors)
    voxel_count = int(voxel_points.shape[0])
    if voxel_count > config.dn_splatter.maximum_seed_points:
        generator = np.random.default_rng(config.runtime.random_seed)
        selected = np.sort(
            generator.choice(
                voxel_count,
                config.dn_splatter.maximum_seed_points,
                replace=False,
            )
        )
        voxel_points = voxel_points[selected]
        voxel_colors = voxel_colors[selected]
        cloud.points = o3d.utility.Vector3dVector(voxel_points)
        cloud.colors = o3d.utility.Vector3dVector(voxel_colors)
    final_count = int(len(cloud.points))
    if final_count < 10_000:
        raise DNSplatterStageError(
            f"depth-unprojected seed cloud is unexpectedly small: {final_count} points"
        )
    if not o3d.io.write_point_cloud(
        str(output_path), cloud, write_ascii=False, compressed=False
    ):
        raise DNSplatterStageError(f"failed to write seed point cloud: {output_path}")
    final_points = np.asarray(cloud.points)
    return {
        "source": "Metric3D depth unprojected with known metric camera poses",
        "coordinate_system": "preserved known-pose world coordinates",
        "stride": stride,
        "depth_bounds_meters": [
            config.dn_splatter.seed_minimum_depth_meters,
            config.dn_splatter.seed_maximum_depth_meters,
        ],
        "voxel_size_meters": config.dn_splatter.seed_voxel_size_meters,
        "maximum_seed_points": config.dn_splatter.maximum_seed_points,
        "raw_point_count": raw_count,
        "voxel_point_count": voxel_count,
        "final_point_count": final_count,
        "valid_points_per_frame_min": min(valid_per_frame),
        "valid_points_per_frame_max": max(valid_per_frame),
        "sampled_far_points_rejected": saturated_rejected,
        "bounds_min_meters": final_points.min(axis=0).tolist(),
        "bounds_max_meters": final_points.max(axis=0).tolist(),
        "sha256": _sha256(output_path),
        "size_bytes": output_path.stat().st_size,
    }


def _validate_dataparser(
    dataset_dir: Path, expected_count: int, expected_seed_count: int
) -> dict[str, Any]:
    try:
        import numpy as np
        from dn_splatter.data.dn_dataset import GDataset
        from dn_splatter.data.normal_nerfstudio import NormalNerfstudioConfig
    except ImportError as error:
        raise DNSplatterStageError(
            "DN-Splatter is unavailable; use the nerfstudio environment"
        ) from error

    parser_config = NormalNerfstudioConfig(
        data=dataset_dir,
        downscale_factor=1,
        scene_scale=1.0,
        orientation_method="none",
        center_method="none",
        auto_scale_poses=False,
        eval_mode="all",
        depth_unit_scale_factor=1.0,
        load_3D_points=True,
        load_normals=False,
        load_depths=True,
        load_pcd_normals=False,
    )
    parser = parser_config.setup()
    outputs = parser.get_dataparser_outputs(split="train")
    if len(outputs.image_filenames) != expected_count:
        raise DNSplatterStageError("DN-Splatter dataparser image count is invalid")
    points = outputs.metadata.get("points3D_xyz")
    if points is None or int(points.shape[0]) != expected_seed_count:
        raise DNSplatterStageError("DN-Splatter dataparser seed point count is invalid")
    dataset = GDataset(outputs)
    first = dataset[0]
    if "sensor_depth" not in first:
        raise DNSplatterStageError("DN-Splatter dataparser did not load metric depth")
    depth = first["sensor_depth"].detach().cpu().numpy()
    if depth.shape != (600, 800, 1) or not np.isfinite(depth).all():
        raise DNSplatterStageError(
            f"DN-Splatter dataparser depth is invalid: {depth.shape}"
        )
    centers = outputs.cameras.camera_to_worlds[:, :3, 3].detach().cpu().numpy()
    return {
        "image_count": len(outputs.image_filenames),
        "camera_count": int(outputs.cameras.size),
        "seed_point_count": int(points.shape[0]),
        "first_depth_shape": list(depth.shape),
        "first_depth_min_meters": float(depth.min()),
        "first_depth_median_meters": float(np.median(depth)),
        "first_depth_max_meters": float(depth.max()),
        "dataparser_scale": float(outputs.dataparser_scale),
        "camera_translation_min": centers.min(axis=0).tolist(),
        "camera_translation_max": centers.max(axis=0).tolist(),
        "auto_scale_poses": False,
        "auto_orient_poses": False,
        "depth_unit_scale_factor": 1.0,
    }


def build_training_command(
    config: PipelineConfig, stage_dir: Path, scene_scale: float
) -> list[str]:
    """Build the exact headless DN-Splatter training command."""

    ns_train = Path(sys.executable).with_name("ns-train")
    dataset_dir = stage_dir / "dataset"
    return [
        str(ns_train),
        config.dn_splatter.method,
        "--output-dir",
        str(stage_dir / "training"),
        "--experiment-name",
        config.scene,
        "--timestamp",
        stage_dir.parent.name,
        "--vis",
        "tensorboard",
        "--machine.seed",
        str(config.runtime.random_seed),
        "--steps-per-save",
        str(config.dn_splatter.steps_per_save),
        "--max-num-iterations",
        str(config.dn_splatter.max_num_iterations),
        "--save-only-latest-checkpoint",
        "True",
        "--pipeline.datamanager.cache-images",
        "cpu",
        "--pipeline.datamanager.cache-images-type",
        "float32",
        "--pipeline.datamanager.train-cameras-sampling-seed",
        str(config.runtime.random_seed),
        "--pipeline.model.random-init",
        "False",
        "--pipeline.model.camera-optimizer.mode",
        "off",
        "--pipeline.model.use-depth-loss",
        "True",
        "--pipeline.model.depth-loss-type",
        config.dn_splatter.depth_loss_type,
        "--pipeline.model.depth-lambda",
        str(config.dn_splatter.depth_lambda),
        "--pipeline.model.predict-normals",
        "True",
        "--pipeline.model.use-normal-loss",
        "True",
        "--pipeline.model.use-normal-tv-loss",
        "True",
        "--pipeline.model.normal-supervision",
        config.dn_splatter.normal_supervision,
        "normal-nerfstudio",
        "--data",
        str(dataset_dir),
        "--downscale-factor",
        "1",
        "--scene-scale",
        str(scene_scale),
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
        "--eval-mode",
        "all",
        "--depth-unit-scale-factor",
        "1.0",
        "--load-3D-points",
        "True",
        "--load-normals",
        "False",
        "--load-depths",
        "True",
        "--load-pcd-normals",
        "True",
        "--load-depth-confidence-masks",
        "False",
    ]


def build_training_environment(
    ns_train: Path, base_environment: dict[str, str] | None = None
) -> dict[str, str]:
    """Expose the CUDA toolkit shipped in the active Nerfstudio environment."""

    environment = dict(os.environ if base_environment is None else base_environment)
    environment_bin = ns_train.parent.resolve()
    environment_root = environment_bin.parent
    existing_path = environment.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    if str(environment_bin) not in path_entries:
        environment["PATH"] = os.pathsep.join(
            [str(environment_bin), *path_entries]
        )
    nvcc = environment_bin / "nvcc"
    if nvcc.is_file():
        environment.setdefault("CUDA_HOME", str(environment_root))
    environment.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def prepare_dn_splatter(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Prepare and validate a known-pose DN-Splatter dataset."""

    pose_transforms_path, metric_dir, metric_manifest_path, image_names, pose_manifest = (
        _verify_inputs(config, run_id)
    )
    stage_dir = config.storage.outputs / config.scene / run_id / Stage.DN_SPLATTER.value
    if stage_dir.exists():
        raise DNSplatterStageError(
            f"DN-Splatter stage already exists and will not be overwritten: {stage_dir}"
        )
    dataset_dir = stage_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    _write_status(stage_dir, "preparing_seed_pointcloud")

    try:
        os.symlink(config.input.images, dataset_dir / "images")
        os.symlink(metric_dir / "depth", dataset_dir / "mono_depth")
        transforms = _read_json(pose_transforms_path)
        frames = transforms.get("frames", [])
        for frame in frames:
            stem = Path(frame["file_path"]).stem
            frame["depth_file_path"] = f"mono_depth/{stem}.npy"
        transforms["ply_file_path"] = "seed_pointcloud.ply"
        transforms["depth_unit"] = "meter"
        transforms["pose_scaling"] = "disabled"
        dataset_transforms_path = dataset_dir / "transforms.json"
        _write_json(dataset_transforms_path, transforms)

        seed_path = dataset_dir / "seed_pointcloud.ply"
        seed_stats = _create_seed_pointcloud(
            config,
            transforms,
            metric_dir,
            image_names,
            seed_path,
        )
        _write_json(dataset_dir / "seed_pointcloud.json", seed_stats)
        _write_status(stage_dir, "validating_dataparser")
        parser_validation = _validate_dataparser(
            dataset_dir,
            len(image_names),
            seed_stats["final_point_count"],
        )
        max_abs_bound = max(
            abs(value)
            for key in ("bounds_min_meters", "bounds_max_meters")
            for value in seed_stats[key]
        )
        scene_scale = float(max(1, math.ceil(max_abs_bound + 1.0)))
        training_command = build_training_command(config, stage_dir, scene_scale)
        _write_json(
            stage_dir / "commands.json", {"train": training_command}
        )

        finished_at = datetime.now(timezone.utc)
        prepared = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.DN_SPLATTER.value,
            "status": "prepared",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": config.runtime.random_seed,
            "inputs": {
                "pose_transforms": {
                    "path": str(pose_transforms_path),
                    "sha256": _sha256(pose_transforms_path),
                    "translation_scale": pose_manifest["conversion"]["translation_scale"],
                },
                "metric3d_manifest": {
                    "path": str(metric_manifest_path),
                    "sha256": _sha256(metric_manifest_path),
                },
            },
            "dataset": {
                "root": str(dataset_dir),
                "transforms": str(dataset_transforms_path),
                "transforms_sha256": _sha256(dataset_transforms_path),
                "images_symlink_target": str(config.input.images),
                "depth_symlink_target": str(metric_dir / "depth"),
                "seed_pointcloud": str(seed_path),
                "seed": seed_stats,
                "dataparser_validation": parser_validation,
                "scene_scale": scene_scale,
            },
            "training": {
                "method": config.dn_splatter.method,
                "max_num_iterations": config.dn_splatter.max_num_iterations,
                "steps_per_save": config.dn_splatter.steps_per_save,
                "depth_loss_type": config.dn_splatter.depth_loss_type,
                "depth_lambda": config.dn_splatter.depth_lambda,
                "normal_supervision": config.dn_splatter.normal_supervision,
                "camera_optimizer": "off",
                "pose_auto_scale": False,
                "pose_auto_orient": False,
                "depth_unit_scale_factor": 1.0,
                "command_path": str(stage_dir / "commands.json"),
            },
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "warnings": [
                "COLMAP sparse points were replaced by Metric3D depth unprojected through the approved known poses.",
                "Metric3D native normals are not used as monocular normal supervision; DN-Splatter derives normal supervision from rendered depth.",
            ],
        }
        prepared_path = stage_dir / "prepared.json"
        _write_json(prepared_path, prepared)
        _write_status(stage_dir, "prepared", str(prepared_path))
        return prepared_path
    except Exception as error:
        _write_status(stage_dir, "failed_preparation", str(error))
        if isinstance(error, DNSplatterStageError):
            raise
        raise DNSplatterStageError(str(error)) from error


def train_dn_splatter(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Train the already-prepared DN-Splatter stage and validate its checkpoint."""

    stage_dir = config.storage.outputs / config.scene / run_id / Stage.DN_SPLATTER.value
    prepared_path = stage_dir / "prepared.json"
    status_path = stage_dir / "STATUS.json"
    commands_path = stage_dir / "commands.json"
    if not prepared_path.is_file() or not status_path.is_file() or not commands_path.is_file():
        raise DNSplatterStageError(f"prepared DN-Splatter stage is missing: {stage_dir}")
    prepared = _read_json(prepared_path)
    status = _read_json(status_path)
    if prepared.get("status") != "prepared" or status.get("state") != "prepared":
        raise DNSplatterStageError(
            f"DN-Splatter stage is not in the prepared state: {status.get('state')}"
        )
    train_command = _read_json(commands_path)["train"]
    ns_train = Path(train_command[0])
    if not ns_train.is_file() or not os.access(ns_train, os.X_OK):
        raise DNSplatterStageError(
            f"ns-train is unavailable in the active environment: {ns_train}"
        )
    lpips_checkpoint = (
        config.storage.weights
        / "torch"
        / "hub"
        / "checkpoints"
        / "alexnet-owt-7be5be79.pth"
    )
    if not lpips_checkpoint.is_file():
        raise DNSplatterStageError(
            f"pinned LPIPS AlexNet checkpoint is missing: {lpips_checkpoint}"
        )

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    log_path = stage_dir / "training.log"
    _write_status(stage_dir, "training", str(log_path))
    environment = build_training_environment(ns_train)
    environment["TORCH_HOME"] = str(config.storage.weights / "torch")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(train_command) + "\n\n")
        log.write(
            "cuda_environment: "
            f"CUDA_HOME={environment.get('CUDA_HOME', '')} "
            f"TORCH_CUDA_ARCH_LIST={environment['TORCH_CUDA_ARCH_LIST']}\n\n"
        )
        log.flush()
        result = subprocess.run(
            train_command,
            cwd=config.metric3d.repository.parent / "dn-splatter",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        _write_status(
            stage_dir,
            "failed_training",
            f"exit code {result.returncode}; see {log_path}",
        )
        raise DNSplatterStageError(
            f"DN-Splatter training failed with exit code {result.returncode}; see {log_path}"
        )

    training_root = (
        stage_dir
        / "training"
        / config.scene
        / config.dn_splatter.method
        / run_id
    )
    config_path = training_root / "config.yml"
    checkpoints = sorted((training_root / "nerfstudio_models").glob("step-*.ckpt"))
    if not config_path.is_file() or not checkpoints:
        _write_status(stage_dir, "failed_validation", "training output is incomplete")
        raise DNSplatterStageError(
            f"DN-Splatter produced no complete checkpoint under {training_root}"
        )
    final_checkpoint = checkpoints[-1]
    match = re.fullmatch(r"step-([0-9]+)\.ckpt", final_checkpoint.name)
    if match is None:
        raise DNSplatterStageError(f"unexpected checkpoint name: {final_checkpoint.name}")
    final_step = int(match.group(1))
    minimum_expected_step = config.dn_splatter.max_num_iterations - 1
    if final_step < minimum_expected_step:
        raise DNSplatterStageError(
            f"final checkpoint step {final_step} is below expected {minimum_expected_step}"
        )

    finished_at = datetime.now(timezone.utc)
    manifest = {
        **prepared,
        "status": "complete",
        "training_started_at": started_at.isoformat(),
        "training_finished_at": finished_at.isoformat(),
        "training_elapsed_seconds": round(elapsed, 6),
        "training_command": command if command is not None else sys.argv,
        "training_dependencies": {
            "lpips_alexnet_checkpoint": str(lpips_checkpoint),
            "lpips_alexnet_sha256": _sha256(lpips_checkpoint),
        },
        "validation": {
            "return_code": result.returncode,
            "final_step": final_step,
            "minimum_expected_step": minimum_expected_step,
            "checkpoint_count": len(checkpoints),
            "training_config_present": True,
        },
        "outputs": {
            "training_root": str(training_root),
            "training_config": str(config_path),
            "training_config_sha256": _sha256(config_path),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": _sha256(final_checkpoint),
            "final_checkpoint_size_bytes": final_checkpoint.stat().st_size,
            "training_log": str(log_path),
        },
    }
    manifest_path = stage_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_status(stage_dir, "complete", str(manifest_path))
    return manifest_path
