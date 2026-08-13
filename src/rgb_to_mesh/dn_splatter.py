"""Prepare and train DN-Splatter from explicit RGB, pose, and depth inputs."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "houselayout3d.rgb_to_mesh"

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != SCRIPT_DIRECTORY
]

LPIPS_ALEXNET_CHECKPOINT = Path(
    "pretrained_weights/alexnet-owt-7be5be79.pth"
)
METRIC3D_REPOSITORY = Path(
    "external/Metric3D"
)
CAMERA_PARAMETERS = Path("camera_param.json")


class DNSplatterError(RuntimeError):
    """Raised when DN-Splatter preparation, training, or validation fails."""


@dataclass(frozen=True)
class CameraConfig:
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class InputConfig:
    images: Path
    camera: CameraConfig


@dataclass(frozen=True)
class StorageConfig:
    lpips_alexnet_checkpoint: Path
    output: Path


@dataclass(frozen=True)
class Metric3DConfig:
    repository: Path


@dataclass(frozen=True)
class DNSplatterSettings:
    method: str
    max_num_iterations: int
    steps_per_save: int
    depth_loss_type: str
    depth_lambda: float
    normal_supervision: str
    seed_stride: int
    seed_minimum_depth_meters: float
    seed_maximum_depth_meters: float
    seed_voxel_size_meters: float
    maximum_seed_points: int


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int


@dataclass(frozen=True)
class DNSplatterConfig:
    scene: str
    input: InputConfig
    storage: StorageConfig
    metric3d: Metric3DConfig
    dn_splatter: DNSplatterSettings
    runtime: RuntimeConfig


def _component_dir(config: DNSplatterConfig) -> Path:
    return config.storage.output.expanduser().resolve()


def load_camera_parameters(path: Path = CAMERA_PARAMETERS) -> CameraConfig:
    payload = _read_json(path)
    intrinsics = payload.get("pinhole_intrinsics")
    resolution = payload.get("pinhole_resolution")
    if (
        not isinstance(intrinsics, list)
        or len(intrinsics) != 4
        or not isinstance(resolution, list)
        or len(resolution) != 2
    ):
        raise DNSplatterError(
            f"camera parameters must contain pinhole_intrinsics[fx,fy,cx,cy] "
            f"and pinhole_resolution[width,height]: {path}"
        )
    camera = CameraConfig(
        model="PINHOLE",
        width=int(resolution[0]),
        height=int(resolution[1]),
        fx=float(intrinsics[0]),
        fy=float(intrinsics[1]),
        cx=float(intrinsics[2]),
        cy=float(intrinsics[3]),
    )
    if min(camera.width, camera.height) <= 0 or min(camera.fx, camera.fy) <= 0:
        raise DNSplatterError("camera dimensions and focal lengths must be positive")
    return camera


def build_config_from_args(args: argparse.Namespace) -> DNSplatterConfig:
    camera = load_camera_parameters()
    method = args.method
    if method != "dn-splatter":
        raise DNSplatterError("--method must be dn-splatter")
    if min(
        args.max_num_iterations,
        args.steps_per_save,
        args.seed_stride,
        args.maximum_seed_points,
    ) <= 0:
        raise DNSplatterError(
            "DN-Splatter iteration, save, stride, and point counts must be positive"
        )
    if not 0 < args.seed_minimum_depth_meters < args.seed_maximum_depth_meters:
        raise DNSplatterError("DN-Splatter seed depth bounds are invalid")
    if args.seed_voxel_size_meters <= 0 or args.depth_lambda <= 0:
        raise DNSplatterError("DN-Splatter voxel size and depth lambda must be positive")
    return DNSplatterConfig(
        scene=args.output.expanduser().parent.name,
        input=InputConfig(
            images=args.images.expanduser(),
            camera=camera,
        ),
        storage=StorageConfig(
            lpips_alexnet_checkpoint=LPIPS_ALEXNET_CHECKPOINT.expanduser().resolve(),
            output=args.output.expanduser().resolve(),
        ),
        metric3d=Metric3DConfig(
            repository=METRIC3D_REPOSITORY.expanduser().resolve(),
        ),
        dn_splatter=DNSplatterSettings(
            method=method,
            max_num_iterations=args.max_num_iterations,
            steps_per_save=args.steps_per_save,
            depth_loss_type=args.depth_loss_type,
            depth_lambda=args.depth_lambda,
            normal_supervision=args.normal_supervision,
            seed_stride=args.seed_stride,
            seed_minimum_depth_meters=args.seed_minimum_depth_meters,
            seed_maximum_depth_meters=args.seed_maximum_depth_meters,
            seed_voxel_size_meters=args.seed_voxel_size_meters,
            maximum_seed_points=args.maximum_seed_points,
        ),
        runtime=RuntimeConfig(random_seed=args.random_seed),
    )


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


def _write_status(component_dir: Path, state: str, detail: str = "") -> None:
    _write_json(
        component_dir / "STATUS.json",
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _remove_existing_output(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise DNSplatterError(f"cannot overwrite unsupported output path: {path}")


def build_training_environment(
    executable: Path, base_environment: dict[str, str] | None = None
) -> dict[str, str]:
    """Expose the CUDA toolkit shipped in the active Nerfstudio environment."""

    environment = dict(os.environ if base_environment is None else base_environment)
    environment_bin = executable.parent.resolve()
    environment_root = environment_bin.parent
    existing_path = environment.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    if str(environment_bin) not in path_entries:
        environment["PATH"] = os.pathsep.join([str(environment_bin), *path_entries])
    nvcc = environment_bin / "nvcc"
    if nvcc.is_file():
        environment.setdefault("CUDA_HOME", str(environment_root))
    environment.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _torch_home_for_checkpoint(path: Path) -> Path:
    if path.parent.name == "checkpoints" and path.parent.parent.name == "hub":
        return path.parent.parent.parent
    return path.parent


def _image_names_from_transforms(transforms: dict[str, Any]) -> list[str]:
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DNSplatterError("transforms.json must contain a non-empty frames list")
    image_names = [Path(str(frame["file_path"])).name for frame in frames]
    if len(set(image_names)) != len(image_names):
        raise DNSplatterError("transforms.json contains duplicate frame names")
    return image_names


def _create_seed_pointcloud(
    config: DNSplatterConfig,
    transforms: dict[str, Any],
    image_dir: Path,
    depth_dir: Path,
    image_names: list[str],
    output_path: Path,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        import open3d as o3d
    except ImportError as error:
        raise DNSplatterError(
            "seed point-cloud dependencies are unavailable; use the nerfstudio environment"
        ) from error

    frames = transforms.get("frames")
    if not isinstance(frames, list) or len(frames) != len(image_names):
        raise DNSplatterError("pose transforms frame count is invalid")
    frame_by_name = {Path(frame["file_path"]).name: frame for frame in frames}
    if sorted(frame_by_name) != sorted(image_names):
        raise DNSplatterError("pose transform names do not match approved images")

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
        depth_path = depth_dir / f"{Path(name).stem}.npy"
        if not depth_path.is_file():
            raise DNSplatterError(f"depth map is missing: {depth_path}")
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != (camera.height, camera.width) or depth.dtype != np.float32:
            raise DNSplatterError(f"invalid depth artifact: {depth_path}")
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
            raise DNSplatterError(f"invalid camera transform for {name}")
        world = (
            points_opengl.astype(np.float64) @ transform[:3, :3].T
            + transform[:3, 3]
        ).astype(np.float32)
        rgb_bgr = cv2.imread(str(image_dir / name), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise DNSplatterError(f"cannot read image for seed colors: {name}")
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
        raise DNSplatterError(
            f"depth-unprojected seed cloud is unexpectedly small: {final_count} points"
        )
    if not o3d.io.write_point_cloud(
        str(output_path), cloud, write_ascii=False, compressed=False
    ):
        raise DNSplatterError(f"failed to write seed point cloud: {output_path}")
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
        raise DNSplatterError(
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
        raise DNSplatterError("DN-Splatter dataparser image count is invalid")
    points = outputs.metadata.get("points3D_xyz")
    if points is None or int(points.shape[0]) != expected_seed_count:
        raise DNSplatterError("DN-Splatter dataparser seed point count is invalid")
    dataset = GDataset(outputs)
    first = dataset[0]
    if "sensor_depth" not in first:
        raise DNSplatterError("DN-Splatter dataparser did not load metric depth")
    depth = first["sensor_depth"].detach().cpu().numpy()
    if depth.shape != (600, 800, 1) or not np.isfinite(depth).all():
        raise DNSplatterError(
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
    config: DNSplatterConfig, component_dir: Path, scene_scale: float
) -> list[str]:
    """Build the exact headless DN-Splatter training command."""

    ns_train = Path(sys.executable).with_name("ns-train")
    dataset_dir = component_dir / "dataset"
    return [
        str(ns_train),
        config.dn_splatter.method,
        "--output-dir",
        str(component_dir / "training"),
        "--experiment-name",
        config.scene,
        "--timestamp",
        component_dir.name,
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


def prepare_dn_splatter(
    config: DNSplatterConfig,
    transforms_path: Path,
    image_dir: Path,
    depth_dir: Path,
    command: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare and validate a known-pose DN-Splatter dataset."""

    transforms_path = transforms_path.expanduser().resolve()
    image_dir = image_dir.expanduser().resolve()
    depth_dir = depth_dir.expanduser().resolve()
    if not transforms_path.is_file():
        raise DNSplatterError(f"transforms.json is missing: {transforms_path}")
    if not image_dir.is_dir():
        raise DNSplatterError(f"image folder is missing: {image_dir}")
    if not depth_dir.is_dir():
        raise DNSplatterError(f"depth folder is missing: {depth_dir}")
    transforms = _read_json(transforms_path)
    image_names = _image_names_from_transforms(transforms)
    component_dir = _component_dir(config)
    if component_dir.exists():
        if not overwrite:
            raise DNSplatterError(
                f"DN-Splatter component already exists and will not be overwritten: {component_dir}"
            )
        _remove_existing_output(component_dir)
    dataset_dir = component_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    _write_status(component_dir, "preparing_seed_pointcloud")

    try:
        os.symlink(image_dir, dataset_dir / "images")
        os.symlink(depth_dir, dataset_dir / "mono_depth")
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
            image_dir,
            depth_dir,
            image_names,
            seed_path,
        )
        _write_json(dataset_dir / "seed_pointcloud.json", seed_stats)
        _write_status(component_dir, "validating_dataparser")
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
        training_command = build_training_command(config, component_dir, scene_scale)
        _write_json(
            component_dir / "commands.json", {"train": training_command}
        )

        finished_at = datetime.now(timezone.utc)
        prepared = {
            "schema_version": 1,
            "scene": config.scene,
            "component": "dn_splatter",
            "status": "prepared",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": config.runtime.random_seed,
            "inputs": {
                "pose_transforms": {
                    "path": str(transforms_path),
                    "sha256": _sha256(transforms_path),
                },
                "images": str(image_dir),
                "depth": str(depth_dir),
            },
            "dataset": {
                "root": str(dataset_dir),
                "transforms": str(dataset_transforms_path),
                "transforms_sha256": _sha256(dataset_transforms_path),
                "images_symlink_target": str(image_dir),
                "depth_symlink_target": str(depth_dir),
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
                "command_path": str(component_dir / "commands.json"),
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
        prepared_path = component_dir / "prepared.json"
        _write_json(prepared_path, prepared)
        _write_status(component_dir, "prepared", str(prepared_path))
        return prepared_path
    except Exception as error:
        _write_status(component_dir, "failed_preparation", str(error))
        if isinstance(error, DNSplatterError):
            raise
        raise DNSplatterError(str(error)) from error


def train_dn_splatter(
    config: DNSplatterConfig,
    command: list[str] | None = None,
) -> Path:
    """Train the already-prepared DN-Splatter component and validate its checkpoint."""

    component_dir = _component_dir(config)
    prepared_path = component_dir / "prepared.json"
    status_path = component_dir / "STATUS.json"
    commands_path = component_dir / "commands.json"
    if (
        not prepared_path.is_file()
        or not status_path.is_file()
        or not commands_path.is_file()
    ):
        raise DNSplatterError(
            f"prepared DN-Splatter component is missing: {component_dir}"
        )
    prepared = _read_json(prepared_path)
    status = _read_json(status_path)
    if prepared.get("status") != "prepared" or status.get("state") != "prepared":
        raise DNSplatterError(
            f"DN-Splatter component is not in the prepared state: {status.get('state')}"
        )
    train_command = _read_json(commands_path)["train"]
    ns_train = Path(train_command[0])
    if not ns_train.is_file() or not os.access(ns_train, os.X_OK):
        raise DNSplatterError(
            f"ns-train is unavailable in the active environment: {ns_train}"
        )
    lpips_checkpoint = config.storage.lpips_alexnet_checkpoint
    if not lpips_checkpoint.is_file():
        raise DNSplatterError(
            f"pinned LPIPS AlexNet checkpoint is missing: {lpips_checkpoint}"
        )

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    log_path = component_dir / "training.log"
    _write_status(component_dir, "training", str(log_path))
    environment = build_training_environment(ns_train)
    environment["TORCH_HOME"] = str(_torch_home_for_checkpoint(lpips_checkpoint))
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
            component_dir,
            "failed_training",
            f"exit code {result.returncode}; see {log_path}",
        )
        raise DNSplatterError(
            f"DN-Splatter training failed with exit code {result.returncode}; see {log_path}"
        )

    training_root = (
        component_dir
        / "training"
        / config.scene
        / config.dn_splatter.method
        / component_dir.name
    )
    config_path = training_root / "config.yml"
    checkpoints = sorted((training_root / "nerfstudio_models").glob("step-*.ckpt"))
    if not config_path.is_file() or not checkpoints:
        _write_status(
            component_dir, "failed_validation", "training output is incomplete"
        )
        raise DNSplatterError(
            f"DN-Splatter produced no complete checkpoint under {training_root}"
        )
    final_checkpoint = checkpoints[-1]
    match = re.fullmatch(r"step-([0-9]+)\.ckpt", final_checkpoint.name)
    if match is None:
        raise DNSplatterError(f"unexpected checkpoint name: {final_checkpoint.name}")
    final_step = int(match.group(1))
    minimum_expected_step = config.dn_splatter.max_num_iterations - 1
    if final_step < minimum_expected_step:
        raise DNSplatterError(
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
    manifest_path = component_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_status(component_dir, "complete", str(manifest_path))
    return manifest_path


def run_dn_splatter(
    config: DNSplatterConfig,
    transforms_path: Path,
    image_dir: Path,
    depth_dir: Path,
    command: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare and train DN-Splatter in one direct run."""

    prepare_dn_splatter(
        config,
        transforms_path=transforms_path,
        image_dir=image_dir,
        depth_dir=depth_dir,
        command=command,
        overwrite=overwrite,
    )
    return train_dn_splatter(config, command=command)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and train the DN-Splatter component.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Exact DN-Splatter output folder, for example outputs/r04_front/dn_splatter.",
    )
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--method", default="dn-splatter")
    parser.add_argument("--max-num-iterations", type=int, default=30000)
    parser.add_argument("--steps-per-save", type=int, default=5000)
    parser.add_argument("--depth-loss-type", default="EdgeAwareLogL1")
    parser.add_argument("--depth-lambda", type=float, default=0.2)
    parser.add_argument("--normal-supervision", default="depth")
    parser.add_argument("--seed-stride", type=int, default=8)
    parser.add_argument("--seed-minimum-depth-meters", type=float, default=0.1)
    parser.add_argument("--seed-maximum-depth-meters", type=float, default=30.0)
    parser.add_argument("--seed-voxel-size-meters", type=float, default=0.03)
    parser.add_argument("--maximum-seed-points", type=int, default=500000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing DN-Splatter output path before running.",
    )
    args = parser.parse_args()
    path = run_dn_splatter(
        build_config_from_args(args),
        transforms_path=args.transforms,
        image_dir=args.images,
        depth_dir=args.depth,
        command=sys.argv,
        overwrite=args.overwrite,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
