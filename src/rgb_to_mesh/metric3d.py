"""Run Metric3D on an image folder and write metric depth artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.camera import load_pinhole_camera_from_transforms


METRIC3D_REPOSITORY = Path("external/Metric3D")
METRIC3D_CHECKPOINT = Path("pretrained_weights/metric_depth_vit_large_800k.pth")
METRIC3D_MODEL = "metric3d_vit_large"
INPUT_HEIGHT = 616
INPUT_WIDTH = 1064
CANONICAL_FOCAL_LENGTH = 1000.0
MINIMUM_DEPTH_METERS = 0.0
MAXIMUM_DEPTH_METERS = 300.0
PREVIEW_COUNT = 12

_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)
_ALLOWED_MISSING_CHECKPOINT_KEYS = {"depth_model.encoder.mask_token"}


class Metric3DError(RuntimeError):
    """Raised when Metric3D inference or validation fails."""


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_status(output_dir: Path, state: str, detail: str = "") -> None:
    _write_json(
        output_dir / "STATUS.json",
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
        raise Metric3DError(f"cannot overwrite unsupported output path: {path}")


def load_camera_from_transforms(path: Path) -> CameraConfig:
    try:
        camera = load_pinhole_camera_from_transforms(path)
    except Exception as error:
        raise Metric3DError(str(error)) from error
    return CameraConfig(
        width=camera.width,
        height=camera.height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
    )


def _git_revision(repository: Path) -> str:
    if not (repository / ".git").exists():
        return "unknown"
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_model():
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise Metric3DError(
            "Metric3D runtime dependencies are unavailable; use the Metric3D environment"
        ) from error

    if not torch.cuda.is_available():
        raise Metric3DError("Metric3D requires a visible CUDA device")
    hubconf_path = METRIC3D_REPOSITORY / "hubconf.py"
    if not hubconf_path.is_file():
        raise Metric3DError(f"Metric3D hubconf.py is missing: {hubconf_path}")
    if not METRIC3D_CHECKPOINT.is_file():
        raise Metric3DError(f"Metric3D checkpoint is missing: {METRIC3D_CHECKPOINT}")

    sys.path.insert(0, str(METRIC3D_REPOSITORY))
    spec = importlib.util.spec_from_file_location("metric3d_hubconf", hubconf_path)
    if spec is None or spec.loader is None:
        raise Metric3DError(f"cannot load Metric3D hubconf: {hubconf_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, METRIC3D_MODEL, None)
    if factory is None:
        raise Metric3DError(f"Metric3D model factory is unavailable: {METRIC3D_MODEL}")
    model = factory(pretrain=False)
    checkpoint = torch.load(METRIC3D_CHECKPOINT, map_location="cpu", weights_only=False)
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


def _run_one(model, torch, functional, np, cv2, camera: CameraConfig, image_path: Path):
    rgb_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise Metric3DError(f"cannot read input image: {image_path}")
    rgb_origin = rgb_bgr[:, :, ::-1]
    height, width = rgb_origin.shape[:2]
    if (height, width) != (camera.height, camera.width):
        raise Metric3DError(
            f"input image dimensions changed: {image_path.name} is {(height, width)}"
        )

    scale = min(INPUT_HEIGHT / height, INPUT_WIDTH / width)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    rgb = cv2.resize(
        rgb_origin, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_height = INPUT_HEIGHT - resized_height
    pad_width = INPUT_WIDTH - resized_width
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
    canonical_to_real_scale = camera.fx * scale / CANONICAL_FOCAL_LENGTH
    depth = torch.clamp(
        depth * canonical_to_real_scale,
        min=MINIMUM_DEPTH_METERS,
        max=MAXIMUM_DEPTH_METERS,
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

    arrays = (
        depth.float().cpu().numpy(),
        normal.float().cpu().numpy(),
        depth_confidence.float().cpu().numpy(),
        normal_confidence.float().cpu().numpy(),
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise Metric3DError(f"Metric3D produced non-finite output: {image_path.name}")
    return arrays, scale, padding, canonical_to_real_scale


def run_metric3d(
    image_dir: Path,
    transforms_path: Path,
    output_dir: Path,
    image_glob: str,
    command: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    image_dir = image_dir.expanduser().resolve()
    transforms_path = transforms_path.expanduser().resolve()
    output_dir = output_dir.expanduser()
    if not image_dir.is_dir():
        raise Metric3DError(f"image folder is missing: {image_dir}")
    image_paths = sorted(path for path in image_dir.glob(image_glob) if path.is_file())
    if not image_paths:
        raise Metric3DError(f"no images match {image_glob}: {image_dir}")
    if output_dir.exists():
        if not overwrite:
            raise Metric3DError(
                f"Metric3D output already exists and will not be overwritten: {output_dir}"
            )
        _remove_existing_output(output_dir)
    depth_dir = output_dir / "depth"
    geometry_dir = output_dir / "geometry"
    preview_dir = output_dir / "previews"
    depth_dir.mkdir(parents=True, exist_ok=False)
    geometry_dir.mkdir()
    preview_dir.mkdir()

    camera = load_camera_from_transforms(transforms_path)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    _write_status(output_dir, "loading_model", METRIC3D_MODEL)
    records_path = output_dir / "per_image.jsonl"
    records: list[dict[str, Any]] = []

    try:
        revision = _git_revision(METRIC3D_REPOSITORY)
        load_start = time.perf_counter()
        model, torch, functional, np, cv2, missing, unexpected = _load_model()
        model_load_seconds = time.perf_counter() - load_start
        preview_count = min(PREVIEW_COUNT, len(image_paths))
        preview_indices = set(
            int(index)
            for index in np.linspace(0, len(image_paths) - 1, num=preview_count).round()
        )
        _write_status(output_dir, "running", f"0/{len(image_paths)}")

        progress = tqdm(
            image_paths,
            desc="Metric3D",
            unit="image",
            dynamic_ncols=True,
        )
        with records_path.open("w", encoding="utf-8") as record_file:
            for index, image_path in enumerate(progress):
                progress.set_postfix_str(image_path.name, refresh=False)
                image_start = time.perf_counter()
                arrays, resize_scale, padding, depth_scale = _run_one(
                    model, torch, functional, np, cv2, camera, image_path
                )
                depth, normal, depth_confidence, normal_confidence = arrays
                depth_path = depth_dir / f"{image_path.stem}.npy"
                geometry_path = geometry_dir / f"{image_path.stem}.npz"
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
                    depth_preview = preview_dir / f"{image_path.stem}_depth.png"
                    normal_preview = preview_dir / f"{image_path.stem}_normal.png"
                    _write_previews(depth, normal, depth_preview, normal_preview, cv2, np)
                    preview_paths = [str(depth_preview), str(normal_preview)]

                normal_norm = np.linalg.norm(normal, axis=0)
                record = {
                    "index": index,
                    "image": image_path.name,
                    "depth": str(depth_path),
                    "geometry": str(geometry_path),
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
                    output_dir / "progress.json",
                    {
                        "state": "running",
                        "completed": index + 1,
                        "total": len(image_paths),
                        "last_image": image_path.name,
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
            "component": "metric3d",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(elapsed, 6),
            "model_load_seconds": round(model_load_seconds, 6),
            "command": command if command is not None else sys.argv,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "inputs": {
                "images": str(image_dir),
                "image_glob": image_glob,
                "transforms": {
                    "path": str(transforms_path),
                },
            },
            "model": {
                "name": METRIC3D_MODEL,
                "repository": str(METRIC3D_REPOSITORY),
                "revision": revision,
                "weights": str(METRIC3D_CHECKPOINT),
                "checkpoint_missing_keys": missing,
                "checkpoint_unexpected_keys": unexpected,
                "input_size": [INPUT_HEIGHT, INPUT_WIDTH],
                "canonical_focal_length": CANONICAL_FOCAL_LENGTH,
            },
            "validation": {
                "expected_image_count": len(image_paths),
                "depth_file_count": len(list(depth_dir.glob("*.npy"))),
                "geometry_file_count": len(list(geometry_dir.glob("*.npz"))),
                "all_outputs_finite": True,
                "all_output_shapes": [camera.height, camera.width],
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
            },
            "outputs": {
                "depth_directory": str(depth_dir),
                "geometry_directory": str(geometry_dir),
                "preview_directory": str(preview_dir),
                "per_image_records": str(records_path),
                "progress": str(output_dir / "progress.json"),
            },
        }
        manifest_path = output_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_json(
            output_dir / "progress.json",
            {
                "state": "complete",
                "completed": len(image_paths),
                "total": len(image_paths),
                "elapsed_seconds": round(elapsed, 3),
                "updated_at": finished_at.isoformat(),
            },
        )
        _write_status(output_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(output_dir, "failed", str(error))
        if isinstance(error, Metric3DError):
            raise
        raise Metric3DError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Metric3D depth and normal inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-glob", default="*.png")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output path before running.",
    )
    args = parser.parse_args()
    path = run_metric3d(
        image_dir=args.images,
        transforms_path=args.transforms,
        output_dir=args.output,
        image_glob=args.image_glob,
        command=sys.argv,
        overwrite=args.overwrite,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
