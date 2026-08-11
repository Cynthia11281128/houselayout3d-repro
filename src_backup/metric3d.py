"""Metric3D v2 inference for the active ``metric3d`` component."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "houselayout3d"


import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PipelineConfig


class Metric3DError(RuntimeError):
    """Raised when Metric3D inference or validation fails."""


_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)
_ALLOWED_MISSING_CHECKPOINT_KEYS = {"depth_model.encoder.mask_token"}


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


def _verify_prior_components(
    config: PipelineConfig, run_id: str
) -> tuple[Path, Path, dict[str, Any], list[str]]:
    run_dir = config.storage.outputs / config.scene / run_id
    input_dir = run_dir / "input"
    pose_dir = run_dir / "pose"
    input_manifest_path = input_dir / "manifest.json"
    pose_manifest_path = pose_dir / "manifest.json"
    image_list_path = input_dir / "images.txt"
    transforms_path = pose_dir / "transforms.json"
    for required in (
        input_manifest_path,
        pose_manifest_path,
        image_list_path,
        transforms_path,
    ):
        if not required.is_file():
            raise Metric3DError(f"required prior artifact is missing: {required}")

    input_manifest = _read_json(input_manifest_path)
    pose_manifest = _read_json(pose_manifest_path)
    if input_manifest.get("status") != "complete":
        raise Metric3DError("input manifest is not complete")
    if pose_manifest.get("status") != "complete":
        raise Metric3DError("pose manifest is not complete")
    if _sha256(image_list_path) != input_manifest["outputs"]["image_list_sha256"]:
        raise Metric3DError("input/images.txt hash no longer matches")
    if _sha256(transforms_path) != pose_manifest["outputs"]["transforms_json_sha256"]:
        raise Metric3DError("pose/transforms.json hash no longer matches")

    image_names = image_list_path.read_text(encoding="utf-8").splitlines()
    expected_count = input_manifest["validation"]["image_count"]
    if len(image_names) != expected_count:
        raise Metric3DError("input image count no longer matches")
    if pose_manifest["validation"]["pose_count"] != expected_count:
        raise Metric3DError("pose count does not match input")
    return input_manifest_path, pose_manifest_path, pose_manifest, image_names


def _git_revision(repository: Path) -> str:
    if not (repository / ".git").exists():
        raise Metric3DError(f"Metric3D repository is unavailable: {repository}")
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_model(config: PipelineConfig):
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise Metric3DError(
            "Metric3D runtime dependencies are unavailable; use the layout environment"
        ) from error

    if not torch.cuda.is_available():
        raise Metric3DError("Metric3D requires a visible CUDA device")
    repository = config.metric3d.repository
    hubconf_path = repository / "hubconf.py"
    if not hubconf_path.is_file():
        raise Metric3DError(f"Metric3D hubconf.py is missing: {hubconf_path}")
    if not config.metric3d.weights.is_file():
        raise Metric3DError(
            f"Metric3D checkpoint is missing: {config.metric3d.weights}"
        )

    sys.path.insert(0, str(repository))
    spec = importlib.util.spec_from_file_location(
        "houselayout3d_metric3d_hubconf", hubconf_path
    )
    if spec is None or spec.loader is None:
        raise Metric3DError(f"cannot load Metric3D hubconf: {hubconf_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, config.metric3d.model, None)
    if factory is None:
        raise Metric3DError(
            f"Metric3D model factory is unavailable: {config.metric3d.model}"
        )
    model = factory(pretrain=False)
    checkpoint = torch.load(
        config.metric3d.weights, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise Metric3DError("unexpected Metric3D checkpoint structure")
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)
    if missing != _ALLOWED_MISSING_CHECKPOINT_KEYS or unexpected:
        raise Metric3DError(
            f"Metric3D checkpoint keys do not match: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.cuda().eval()
    return model, torch, functional, np, cv2, sorted(missing), sorted(unexpected)


def _crop_and_resize(tensor, padding: tuple[int, int, int, int], size, functional):
    top, bottom, left, right = padding
    height_end = tensor.shape[-2] - bottom if bottom else tensor.shape[-2]
    width_end = tensor.shape[-1] - right if right else tensor.shape[-1]
    tensor = tensor[..., top:height_end, left:width_end]
    return functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )


def _save_npy(path: Path, array, np) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _save_geometry(path: Path, normal, depth_confidence, normal_confidence, np) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            normal=normal.astype(np.float16),
            depth_confidence=depth_confidence.astype(np.float16),
            normal_confidence=normal_confidence.astype(np.float16),
        )
    os.replace(temporary, path)


def _write_previews(depth, normal, depth_path: Path, normal_path: Path, cv2, np) -> None:
    valid = depth[np.isfinite(depth) & (depth > 0)]
    low, high = np.percentile(valid, [2.0, 98.0])
    if not high > low:
        high = low + 1.0
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    depth_color = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    if not cv2.imwrite(str(depth_path), depth_color):
        raise Metric3DError(f"failed to write depth preview: {depth_path}")
    normal_rgb = np.clip((normal.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255)
    if not cv2.imwrite(str(normal_path), normal_rgb[..., ::-1].astype(np.uint8)):
        raise Metric3DError(f"failed to write normal preview: {normal_path}")


def _run_one(config, model, torch, functional, np, cv2, image_path: Path):
    rgb_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise Metric3DError(f"cannot read input image: {image_path}")
    rgb_origin = rgb_bgr[:, :, ::-1]
    height, width = rgb_origin.shape[:2]
    expected = (config.input.camera.height, config.input.camera.width)
    if (height, width) != expected:
        raise Metric3DError(
            f"input image dimensions changed: {image_path.name} is {(height, width)}"
        )

    input_height = config.metric3d.input_height
    input_width = config.metric3d.input_width
    scale = min(input_height / height, input_width / width)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    rgb = cv2.resize(
        rgb_origin, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_height = input_height - resized_height
    pad_width = input_width - resized_width
    padding = (
        pad_height // 2,
        pad_height - pad_height // 2,
        pad_width // 2,
        pad_width - pad_width // 2,
    )
    rgb = cv2.copyMakeBorder(
        rgb,
        padding[0],
        padding[1],
        padding[2],
        padding[3],
        cv2.BORDER_CONSTANT,
        value=_MEAN,
    )
    image = torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()
    mean = torch.tensor(_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(_STD, dtype=torch.float32)[:, None, None]
    image = ((image - mean) / std)[None].cuda(non_blocking=True)

    with torch.inference_mode():
        pred_depth, depth_confidence, output = model.inference({"input": image})
    if "prediction_normal" not in output:
        raise Metric3DError("Metric3D v2 output has no prediction_normal")
    prediction_normal = output["prediction_normal"]
    if pred_depth.ndim != 4 or prediction_normal.shape[1] < 4:
        raise Metric3DError(
            f"unexpected Metric3D output shapes: depth={tuple(pred_depth.shape)}, "
            f"normal={tuple(prediction_normal.shape)}"
        )

    depth = _crop_and_resize(
        pred_depth, padding, (height, width), functional
    ).squeeze(0).squeeze(0)
    canonical_to_real_scale = (
        config.input.camera.fx * scale / config.metric3d.canonical_focal_length
    )
    depth = torch.clamp(
        depth * canonical_to_real_scale,
        min=config.metric3d.minimum_depth_meters,
        max=config.metric3d.maximum_depth_meters,
    )
    depth_confidence = _crop_and_resize(
        depth_confidence, padding, (height, width), functional
    ).squeeze(0).squeeze(0)
    normal = _crop_and_resize(
        prediction_normal[:, :3], padding, (height, width), functional
    ).squeeze(0)
    normal = functional.normalize(normal, dim=0, eps=1.0e-6)
    normal_confidence = _crop_and_resize(
        prediction_normal[:, 3:4], padding, (height, width), functional
    ).squeeze(0).squeeze(0)

    depth_array = depth.float().cpu().numpy()
    normal_array = normal.float().cpu().numpy()
    depth_confidence_array = depth_confidence.float().cpu().numpy()
    normal_confidence_array = normal_confidence.float().cpu().numpy()
    arrays = (
        depth_array,
        normal_array,
        depth_confidence_array,
        normal_confidence_array,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise Metric3DError(f"Metric3D produced non-finite output: {image_path.name}")
    if depth_array.shape != (height, width) or normal_array.shape != (3, height, width):
        raise Metric3DError(f"Metric3D output dimensions are invalid: {image_path.name}")
    return arrays, scale, padding, canonical_to_real_scale


def run_metric3d(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Run Metric3D v2-L on every approved image and validate all outputs."""

    input_manifest_path, pose_manifest_path, pose_manifest, image_names = (
        _verify_prior_components(config, run_id)
    )
    weight_hash = _sha256(config.metric3d.weights)
    revision = _git_revision(config.metric3d.repository)

    run_dir = config.storage.outputs / config.scene / run_id
    component_dir = run_dir / "metric3d"
    if component_dir.exists():
        raise Metric3DError(
            f"Metric3D component already exists and will not be overwritten: {component_dir}"
        )
    depth_dir = component_dir / "depth"
    geometry_dir = component_dir / "geometry"
    preview_dir = component_dir / "previews"
    depth_dir.mkdir(parents=True, exist_ok=False)
    geometry_dir.mkdir()
    preview_dir.mkdir()

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    _write_status(component_dir, "loading_model", config.metric3d.model)
    records_path = component_dir / "per_image.jsonl"
    records: list[dict[str, Any]] = []

    try:
        load_start = time.perf_counter()
        model, torch, functional, np, cv2, missing, unexpected = _load_model(config)
        model_load_seconds = time.perf_counter() - load_start
        preview_indices = set(
            int(index)
            for index in np.linspace(0, len(image_names) - 1, num=12).round()
        )
        _write_status(component_dir, "running", f"0/{len(image_names)}")

        with records_path.open("w", encoding="utf-8") as record_file:
            for index, name in enumerate(image_names):
                image_start = time.perf_counter()
                stem = Path(name).stem
                arrays, resize_scale, padding, depth_scale = _run_one(
                    config,
                    model,
                    torch,
                    functional,
                    np,
                    cv2,
                    config.input.images / name,
                )
                depth, normal, depth_confidence, normal_confidence = arrays
                depth_path = depth_dir / f"{stem}.npy"
                geometry_path = geometry_dir / f"{stem}.npz"
                _save_npy(depth_path, depth.astype(np.float32), np)
                _save_geometry(
                    geometry_path,
                    normal,
                    depth_confidence,
                    normal_confidence,
                    np,
                )
                preview_paths = None
                if index in preview_indices:
                    depth_preview = preview_dir / f"{stem}_depth.png"
                    normal_preview = preview_dir / f"{stem}_normal.png"
                    _write_previews(
                        depth,
                        normal,
                        depth_preview,
                        normal_preview,
                        cv2,
                        np,
                    )
                    preview_paths = [str(depth_preview), str(normal_preview)]

                normal_norm = np.linalg.norm(normal, axis=0)
                record = {
                    "index": index,
                    "image": name,
                    "depth": str(depth_path),
                    "depth_sha256": _sha256(depth_path),
                    "geometry": str(geometry_path),
                    "geometry_sha256": _sha256(geometry_path),
                    "shape": [int(value) for value in depth.shape],
                    "resize_scale": resize_scale,
                    "padding": list(padding),
                    "canonical_to_real_depth_scale": depth_scale,
                    "depth_min_meters": float(depth.min()),
                    "depth_max_meters": float(depth.max()),
                    "depth_mean_meters": float(depth.mean()),
                    "depth_median_meters": float(np.median(depth)),
                    "depth_confidence_min": float(depth_confidence.min()),
                    "depth_confidence_max": float(depth_confidence.max()),
                    "normal_confidence_min": float(normal_confidence.min()),
                    "normal_confidence_max": float(normal_confidence.max()),
                    "normal_norm_mean": float(normal_norm.mean()),
                    "normal_norm_max_abs_error": float(
                        np.max(np.abs(normal_norm - 1.0))
                    ),
                    "preview_paths": preview_paths,
                    "elapsed_seconds": round(time.perf_counter() - image_start, 6),
                }
                records.append(record)
                record_file.write(json.dumps(record, sort_keys=True) + "\n")
                record_file.flush()
                _write_json(
                    component_dir / "progress.json",
                    {
                        "state": "running",
                        "completed": index + 1,
                        "total": len(image_names),
                        "last_image": name,
                        "elapsed_seconds": round(time.perf_counter() - start, 3),
                        "cuda_peak_memory_bytes": int(
                            torch.cuda.max_memory_allocated()
                        ),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

        finished_at = datetime.now(timezone.utc)
        elapsed = time.perf_counter() - start
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "component": "metric3d",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(elapsed, 6),
            "model_load_seconds": round(model_load_seconds, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": config.runtime.random_seed,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "inputs": {
                "input_manifest": {
                    "path": str(input_manifest_path),
                    "sha256": _sha256(input_manifest_path),
                },
                "pose_manifest": {
                    "path": str(pose_manifest_path),
                    "sha256": _sha256(pose_manifest_path),
                    "translation_scale": pose_manifest["conversion"][
                        "translation_scale"
                    ],
                },
            },
            "model": {
                "name": config.metric3d.model,
                "repository": str(config.metric3d.repository),
                "revision": revision,
                "weights": str(config.metric3d.weights),
                "weights_sha256": weight_hash,
                "checkpoint_missing_keys": missing,
                "checkpoint_unexpected_keys": unexpected,
                "input_size": [
                    config.metric3d.input_height,
                    config.metric3d.input_width,
                ],
                "canonical_focal_length": config.metric3d.canonical_focal_length,
            },
            "conversion": {
                "preprocessing": "official Metric3D aspect resize, mean padding, and normalization",
                "metric_depth_scale": "resized fx / canonical focal length",
                "depth_bounds_meters": [
                    config.metric3d.minimum_depth_meters,
                    config.metric3d.maximum_depth_meters,
                ],
                "normal_coordinates": "Metric3D native camera coordinates",
                "normal_resize": "bilinear followed by unit normalization",
                "additional_depth_alignment": None,
            },
            "validation": {
                "expected_image_count": len(image_names),
                "depth_file_count": len(list(depth_dir.glob("*.npy"))),
                "geometry_file_count": len(list(geometry_dir.glob("*.npz"))),
                "all_outputs_finite": True,
                "all_output_shapes": [
                    config.input.camera.height,
                    config.input.camera.width,
                ],
                "depth_global_min_meters": min(
                    record["depth_min_meters"] for record in records
                ),
                "depth_global_max_meters": max(
                    record["depth_max_meters"] for record in records
                ),
                "mean_per_image_depth_meters": sum(
                    record["depth_mean_meters"] for record in records
                )
                / len(records),
                "normal_norm_global_max_abs_error": max(
                    record["normal_norm_max_abs_error"] for record in records
                ),
                "metric_pose_scale_preserved": True,
            },
            "outputs": {
                "depth_directory": str(depth_dir),
                "geometry_directory": str(geometry_dir),
                "preview_directory": str(preview_dir),
                "per_image_records": str(records_path),
                "per_image_records_sha256": _sha256(records_path),
                "progress": str(component_dir / "progress.json"),
            },
            "warnings": [
                "The paper names Metric3D but does not specify its checkpoint variant; this run uses the pinned v2 ViT-L checkpoint.",
                "Metric3D normals are preserved as auxiliary output and are not yet assumed to be DN-Splatter supervision.",
            ],
        }
        manifest_path = component_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_json(
            component_dir / "progress.json",
            {
                "state": "complete",
                "completed": len(image_names),
                "total": len(image_names),
                "elapsed_seconds": round(elapsed, 3),
                "updated_at": finished_at.isoformat(),
            },
        )
        _write_status(component_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(component_dir, "failed", str(error))
        _write_json(
            component_dir / "failure.json",
            {
                "schema_version": 1,
                "scene": config.scene,
                "run_id": run_id,
                "component": "metric3d",
                "status": "failed",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "completed_image_count": len(records),
                "error": str(error),
            },
        )
        if isinstance(error, Metric3DError):
            raise
        raise Metric3DError(str(error)) from error


def main() -> int:
    from .direct import run_component

    return run_component(run_metric3d, "Run Metric3D depth and normal inference.")


if __name__ == "__main__":
    raise SystemExit(main())
