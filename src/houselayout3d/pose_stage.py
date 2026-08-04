"""Known-pose adapter for the active ``01_pose`` stage."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .stages import Stage


class PoseStageError(RuntimeError):
    """Raised when known poses cannot be matched or converted safely."""


_EXPECTED_HEADER = [
    "# counter",
    "sec",
    "nsec",
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
]


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


def _verify_input_stage(
    config: PipelineConfig, run_id: str
) -> tuple[Path, dict[str, Any], list[str]]:
    input_dir = config.storage.outputs / config.scene / run_id / Stage.INPUT.value
    manifest_path = input_dir / "manifest.json"
    image_list_path = input_dir / "images.txt"
    if not manifest_path.is_file() or not image_list_path.is_file():
        raise PoseStageError(f"complete 00_input stage is missing: {input_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise PoseStageError("00_input manifest is not complete")
    if _sha256(image_list_path) != manifest["outputs"]["image_list_sha256"]:
        raise PoseStageError("00_input/images.txt hash no longer matches its manifest")
    names = image_list_path.read_text(encoding="utf-8").splitlines()
    if len(names) != manifest["validation"]["image_count"]:
        raise PoseStageError("00_input image count no longer matches images.txt")
    return input_dir, manifest, names


def _quaternion_to_matrix(values: tuple[float, float, float, float]) -> list[list[float]]:
    qx, qy, qz, qw = values
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise PoseStageError("pose contains a zero-norm quaternion")
    x, y, z, w = (value / norm for value in (qx, qy, qz, qw))
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _determinant(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _orthogonality_error(matrix: list[list[float]]) -> float:
    error = 0.0
    for row in range(3):
        for column in range(3):
            value = sum(matrix[k][row] * matrix[k][column] for k in range(3))
            expected = 1.0 if row == column else 0.0
            error = max(error, abs(value - expected))
    return error


def _read_pose_rows(path: Path, expected_names: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise PoseStageError(f"configured pose CSV does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise PoseStageError(f"pose CSV is empty: {path}") from error
        if header != _EXPECTED_HEADER:
            raise PoseStageError(
                f"unexpected pose CSV header: expected {_EXPECTED_HEADER}, got {header}"
            )

        for line_number, raw in enumerate(reader, start=2):
            if len(raw) != len(_EXPECTED_HEADER):
                raise PoseStageError(
                    f"pose CSV row {line_number} has {len(raw)} columns, expected 10"
                )
            try:
                counter, sec, nsec = (int(raw[index]) for index in range(3))
                numeric = tuple(float(value) for value in raw[3:])
            except ValueError as error:
                raise PoseStageError(
                    f"pose CSV row {line_number} contains an invalid number"
                ) from error
            if counter != len(rows):
                raise PoseStageError(
                    f"pose counter is not contiguous at row {line_number}: {counter}"
                )
            if not 0 <= nsec < 1_000_000_000:
                raise PoseStageError(
                    f"pose CSV row {line_number} has invalid nanoseconds: {nsec}"
                )
            if not all(math.isfinite(value) for value in numeric):
                raise PoseStageError(f"pose CSV row {line_number} is not finite")

            x, y, z, qx, qy, qz, qw = numeric
            name = f"{sec:010d}_{nsec:09d}.png"
            quaternion = (qx, qy, qz, qw)
            quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
            if abs(quaternion_norm - 1.0) > 1.0e-5:
                raise PoseStageError(
                    f"pose CSV row {line_number} quaternion norm is {quaternion_norm}"
                )
            rotation_opencv = _quaternion_to_matrix(quaternion)
            # Nerfstudio transforms use OpenGL camera axes. The source is OpenCV
            # c2w, so flip its camera-space Y and Z columns and preserve world XYZ.
            rotation_opengl = [
                [row[0], -row[1], -row[2]] for row in rotation_opencv
            ]
            transform = [
                [*rotation_opengl[0], x],
                [*rotation_opengl[1], y],
                [*rotation_opengl[2], z],
                [0.0, 0.0, 0.0, 1.0],
            ]
            rows.append(
                {
                    "counter": counter,
                    "name": name,
                    "timestamp": {"sec": sec, "nsec": nsec},
                    "translation": [x, y, z],
                    "quaternion_norm": quaternion_norm,
                    "rotation_determinant": _determinant(rotation_opengl),
                    "rotation_orthogonality_error": _orthogonality_error(rotation_opengl),
                    "transform_matrix": transform,
                }
            )

    actual_names = [row["name"] for row in rows]
    if actual_names != expected_names:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_names, expected_names)
                )
                if actual != expected
            ),
            min(len(actual_names), len(expected_names)),
        )
        actual = actual_names[mismatch] if mismatch < len(actual_names) else "<missing>"
        expected = expected_names[mismatch] if mismatch < len(expected_names) else "<none>"
        raise PoseStageError(
            f"pose/image timestamp mismatch at index {mismatch}: pose={actual}, image={expected}"
        )

    translations = [row["translation"] for row in rows]
    steps = [
        math.dist(previous, current)
        for previous, current in zip(translations, translations[1:])
    ]
    stats = {
        "pose_count": len(rows),
        "quaternion_norm_min": min(row["quaternion_norm"] for row in rows),
        "quaternion_norm_max": max(row["quaternion_norm"] for row in rows),
        "rotation_determinant_min": min(row["rotation_determinant"] for row in rows),
        "rotation_determinant_max": max(row["rotation_determinant"] for row in rows),
        "rotation_orthogonality_max_abs_error": max(
            row["rotation_orthogonality_error"] for row in rows
        ),
        "translation_min_meters": [min(values) for values in zip(*translations)],
        "translation_max_meters": [max(values) for values in zip(*translations)],
        "trajectory_path_length_meters": sum(steps),
        "translation_step_max_meters": max(steps, default=0.0),
    }
    return rows, stats


def prepare_poses(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Validate configured poses and emit a metric Nerfstudio dataset adapter."""

    if config.input.poses_csv is None or config.input.pose_convention is None:
        raise PoseStageError("known poses are not configured")
    if config.input.pose_convention != "opencv_c2w_xyzw_meters":
        raise PoseStageError(f"unsupported pose convention: {config.input.pose_convention}")

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    input_dir, input_manifest, image_names = _verify_input_stage(config, run_id)
    rows, pose_stats = _read_pose_rows(config.input.poses_csv, image_names)

    stage_dir = config.storage.outputs / config.scene / run_id / Stage.POSE.value
    if stage_dir.exists():
        raise PoseStageError(
            f"pose stage already exists and will not be overwritten: {stage_dir}"
        )
    stage_dir.mkdir(parents=True, exist_ok=False)
    images_link = stage_dir / "images"
    os.symlink(config.input.images, images_link)

    camera = config.input.camera
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": camera.fx,
        "fl_y": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "w": camera.width,
        "h": camera.height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "orientation_override": "none",
        "source_pose_convention": config.input.pose_convention,
        "world_translation_unit": "meter",
        "frames": [
            {
                "file_path": f"images/{row['name']}",
                "transform_matrix": row["transform_matrix"],
            }
            for row in rows
        ],
    }
    transforms_path = stage_dir / "transforms.json"
    _write_json(transforms_path, transforms)

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "scene": config.scene,
        "run_id": run_id,
        "stage": Stage.POSE.value,
        "status": "complete",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start, 6),
        "command": command if command is not None else sys.argv,
        "random_seed": config.runtime.random_seed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "inputs": {
            "input_manifest": {
                "path": str(input_dir / "manifest.json"),
                "sha256": _sha256(input_dir / "manifest.json"),
            },
            "poses_csv": {
                "path": str(config.input.poses_csv),
                "sha256": _sha256(config.input.poses_csv),
                "convention": config.input.pose_convention,
            },
        },
        "conversion": {
            "source": "OpenCV camera-to-world, XYZW quaternion, meters",
            "output": "Nerfstudio/OpenGL camera-to-world, meters",
            "camera_axis_transform": "diag(1,-1,-1) applied on the right",
            "world_transform": "identity",
            "translation_scale": 1.0,
            "auto_orient_poses": False,
            "auto_scale_poses": False,
        },
        "validation": {
            **pose_stats,
            "image_count": input_manifest["validation"]["image_count"],
            "pose_image_count_match": True,
            "timestamps_match_exactly_in_order": True,
            "all_values_finite": True,
            "world_coordinates_preserved": True,
            "translation_unit_preserved": True,
        },
        "outputs": {
            "dataset_root": str(stage_dir),
            "transforms_json": str(transforms_path),
            "transforms_json_sha256": _sha256(transforms_path),
            "images": str(images_link),
            "images_symlink_target": str(config.input.images),
        },
        "warnings": [
            "Downstream Nerfstudio commands must set auto_scale_poses=False."
        ],
    }
    manifest_path = stage_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        stage_dir / "STATUS.json",
        {
            "state": "complete",
            "detail": str(transforms_path),
            "updated_at": finished_at.isoformat(),
        },
    )
    return manifest_path
