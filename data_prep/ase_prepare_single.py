#!/usr/bin/env python3
"""Prepare one raw Aria ASE scene for the HouseLayout3D known-pose path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SOURCE = Path(
    "/home/xinyuan/houselayout3d-repro/raw_data/"
    "aria_ase_train_random10_chunks_seed42/raw/4090"
)
DEFAULT_DEST_ROOT = Path("/home/xinyuan/houselayout3d-repro/data/aria_ase/4090")
EXPECTED_POSE_HEADER = [
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
TRAJECTORY_COLUMNS = [
    "tracking_timestamp_us",
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
    "qx_world_device",
    "qy_world_device",
    "qz_world_device",
    "qw_world_device",
]

FRAME_RE = re.compile(r"^vignette(\d{7})\.(jpg|jpeg|png)$", re.IGNORECASE)
ASE_RGB_BASE_SIZE = 704
ASE_RGB_VALID_RADIUS = 1415.0 / 4.0
ASE_RGB_PROJECTION_PARAMS = [
    297.6375381033778,
    357.6599197217746,
    349.1922497127481,
    0.3650890375644368,
    -0.1738082418112771,
    -0.7534945484033189,
    2.434788882752295,
    -2.57786220300886,
    0.8788483538598834,
    0.0008005198595407136,
    -0.000294237814554143,
    0.0,
    0.0,
    0.0,
    0.0,
]
ASE_PINHOLE_RESOLUTION = [800, 600]
ASE_PINHOLE_INTRINSICS = [463.99945, 463.25045, 400.0, 300.0]
ASE_PINHOLE_SOURCE_ROTATION = "cw90"
ASE_T_DEVICE_CAMERA_TRANSLATION = [
    -0.007530096566173914,
    -0.010908549841580260,
    -0.003598063315542823,
]
ASE_T_DEVICE_CAMERA_QUAT_XYZW = [
    0.326409343828490850,
    0.029274992008313648,
    0.033361059956531547,
    0.9441858687689326,
]


class PrepareError(RuntimeError):
    """Raised when a raw ASE scene cannot be converted safely."""


@dataclass(frozen=True)
class FrameRecord:
    counter: int
    source_frame_id: int
    source_image: Path
    target_name: str
    sec: int
    nsec: int
    pose_values: tuple[float, float, float, float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one raw Aria ASE scene into HouseLayout3D known-pose input: "
            "front/*.png, front/poses.csv, intrinsics.json, and transfer_manifest.json. "
            "Raw 704x704 ASE fisheye frames are rectified to an 800x600 front pinhole view."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Raw ASE scene root. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=f"HouseLayout3D scene output root. Default: {DEFAULT_DEST_ROOT}",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth raw RGB frame. Default: 1",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Output pinhole fx. Defaults to the ASE front pinhole calibration.",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Output pinhole fy. Defaults to the ASE front pinhole calibration.",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Output pinhole cx. Defaults to the ASE front pinhole calibration.",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Output pinhole cy. Defaults to the ASE front pinhole calibration.",
    )
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON with pinhole_intrinsics/intrinsics and "
            "pinhole_resolution/resolution. CLI fx/fy/cx/cy override it."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the manifest without writing output files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace conflicting files managed by this script.",
    )
    return parser.parse_args()


def require_pillow():
    try:
        from PIL import Image
    except ImportError as error:
        raise PrepareError(
            "Pillow is required to decode and rotate raw ASE frames"
        ) from error
    return Image


def require_cv2_numpy() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise PrepareError(
            "OpenCV and NumPy are required to rectify ASE Fisheye624 frames to pinhole PNGs"
        ) from error
    return cv2, np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_scene_id(source: Path) -> str:
    scene_id = source.name
    if not scene_id:
        raise PrepareError(f"cannot infer scene id from source path: {source}")
    return scene_id


def finite_float(row: Mapping[str, str], key: str, line_number: int) -> float:
    value = row.get(key)
    if value is None:
        raise PrepareError(f"trajectory.csv is missing required column: {key}")
    try:
        result = float(value)
    except ValueError as error:
        raise PrepareError(
            f"trajectory row {line_number} column {key} is not a float: {value!r}"
        ) from error
    if not math.isfinite(result):
        raise PrepareError(f"trajectory row {line_number} column {key} is not finite")
    return result


def timestamp_us_to_sec_nsec(value: str, line_number: int) -> tuple[int, int]:
    try:
        timestamp_us = int(value)
    except ValueError as error:
        raise PrepareError(
            f"trajectory row {line_number} has invalid tracking_timestamp_us"
        ) from error
    if timestamp_us < 0:
        raise PrepareError(
            f"trajectory row {line_number} has negative tracking_timestamp_us"
        )
    timestamp_ns = timestamp_us * 1000
    return timestamp_ns // 1_000_000_000, timestamp_ns % 1_000_000_000


def normalize_quat_xyzw(
    quat: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(quat) != 4:
        raise PrepareError(f"quaternion must have 4 values, got {len(quat)}")
    qx, qy, qz, qw = (float(value) for value in quat)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1.0e-12:
        raise PrepareError("pose contains a zero-norm quaternion")
    return qx / norm, qy / norm, qz / norm, qw / norm


def quat_to_matrix(quat_xyzw: Sequence[float]) -> list[list[float]]:
    qx, qy, qz, qw = normalize_quat_xyzw(quat_xyzw)
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


def matrix_to_quat_xyzw(
    rotation: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    r = [[float(rotation[row][col]) for col in range(3)] for row in range(3)]
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r[2][1] - r[1][2]) / scale
        qy = (r[0][2] - r[2][0]) / scale
        qz = (r[1][0] - r[0][1]) / scale
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        scale = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / scale
        qx = 0.25 * scale
        qy = (r[0][1] + r[1][0]) / scale
        qz = (r[0][2] + r[2][0]) / scale
    elif r[1][1] > r[2][2]:
        scale = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / scale
        qx = (r[0][1] + r[1][0]) / scale
        qy = 0.25 * scale
        qz = (r[1][2] + r[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / scale
        qx = (r[0][2] + r[2][0]) / scale
        qy = (r[1][2] + r[2][1]) / scale
        qz = 0.25 * scale
    return normalize_quat_xyzw((qx, qy, qz, qw))


def matmul3(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> list[list[float]]:
    return [
        [
            sum(float(a[row][idx]) * float(b[idx][col]) for idx in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def matvec3(
    a: Sequence[Sequence[float]],
    v: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(
        sum(float(a[row][col]) * float(v[col]) for col in range(3))
        for row in range(3)
    )


def source_rotation_to_raw_camera_rotation(
    source_rotation: str,
) -> list[list[float]]:
    if source_rotation == "none":
        angle = 0.0
    elif source_rotation == "cw90":
        angle = -math.pi / 2.0
    elif source_rotation == "ccw90":
        angle = math.pi / 2.0
    elif source_rotation == "180":
        angle = math.pi
    else:
        raise PrepareError(f"unsupported source rotation: {source_rotation}")
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0],
    ]


def world_camera_pose_from_ase(
    row: Mapping[str, str],
    line_number: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    t_world_device = (
        finite_float(row, "tx_world_device", line_number),
        finite_float(row, "ty_world_device", line_number),
        finite_float(row, "tz_world_device", line_number),
    )
    q_world_device = (
        finite_float(row, "qx_world_device", line_number),
        finite_float(row, "qy_world_device", line_number),
        finite_float(row, "qz_world_device", line_number),
        finite_float(row, "qw_world_device", line_number),
    )
    r_world_device = quat_to_matrix(q_world_device)
    r_device_camera = quat_to_matrix(ASE_T_DEVICE_CAMERA_QUAT_XYZW)
    rotated_camera_t = matvec3(r_world_device, ASE_T_DEVICE_CAMERA_TRANSLATION)
    t_world_camera = tuple(t_world_device[idx] + rotated_camera_t[idx] for idx in range(3))
    r_world_camera = matmul3(r_world_device, r_device_camera)
    q_world_camera = matrix_to_quat_xyzw(r_world_camera)
    return t_world_camera, q_world_camera


def front_pinhole_pose_from_ase(
    row: Mapping[str, str],
    line_number: int,
) -> tuple[float, float, float, float, float, float, float]:
    position, q_world_camera = world_camera_pose_from_ase(row, line_number)
    r_world_camera = quat_to_matrix(q_world_camera)
    r_view_to_camera = source_rotation_to_raw_camera_rotation(ASE_PINHOLE_SOURCE_ROTATION)
    r_world_view = matmul3(r_world_camera, r_view_to_camera)
    q_world_view = matrix_to_quat_xyzw(r_world_view)
    if sum(a * b for a, b in zip(q_world_camera, q_world_view)) < 0.0:
        q_world_view = tuple(-value for value in q_world_view)
    return (*position, *q_world_view)


def read_raw_trajectory(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in TRAJECTORY_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise PrepareError(
                "trajectory.csv is missing required columns: " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise PrepareError(f"trajectory.csv contains no frame rows: {path}")
    return rows


def collect_rgb_frames(rgb_dir: Path, frame_stride: int) -> list[tuple[int, Path]]:
    if frame_stride <= 0:
        raise PrepareError(f"--frame-stride must be positive, got {frame_stride}")
    if not rgb_dir.is_dir():
        raise PrepareError(f"raw RGB directory does not exist: {rgb_dir}")

    frames: list[tuple[int, Path]] = []
    for path in sorted(rgb_dir.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_RE.fullmatch(path.name)
        if match is None:
            continue
        frames.append((int(match.group(1)), path))
    if not frames:
        raise PrepareError(f"no RGB frames named vignette####### found in {rgb_dir}")
    return frames[::frame_stride]


def read_scene(source: Path, frame_stride: int) -> tuple[list[FrameRecord], int]:
    rgb_dir = source / "rgb"
    trajectory = source / "trajectory.csv"
    if not source.is_dir():
        raise PrepareError(f"raw scene root does not exist: {source}")
    if not trajectory.is_file():
        raise PrepareError(f"trajectory.csv is missing: {trajectory}")

    trajectory_rows = read_raw_trajectory(trajectory)
    rgb_frames = collect_rgb_frames(rgb_dir, frame_stride)
    records: list[FrameRecord] = []
    previous_timestamp: tuple[int, int] | None = None
    seen_timestamps: set[tuple[int, int]] = set()

    for output_counter, (frame_id, source_image) in enumerate(rgb_frames):
        if frame_id >= len(trajectory_rows):
            raise PrepareError(
                f"RGB frame {frame_id} has no matching trajectory row; "
                f"trajectory has {len(trajectory_rows)} rows"
            )
        row = trajectory_rows[frame_id]
        line_number = frame_id + 2
        sec, nsec = timestamp_us_to_sec_nsec(
            row["tracking_timestamp_us"],
            line_number,
        )
        timestamp = (sec, nsec)
        if previous_timestamp is not None and previous_timestamp >= timestamp:
            raise PrepareError(
                f"selected trajectory timestamps are not strictly increasing at row {line_number}"
            )
        if timestamp in seen_timestamps:
            raise PrepareError(f"duplicate output timestamp: {sec:010d}_{nsec:09d}")
        previous_timestamp = timestamp
        seen_timestamps.add(timestamp)
        records.append(
            FrameRecord(
                counter=output_counter,
                source_frame_id=frame_id,
                source_image=source_image,
                target_name=f"{sec:010d}_{nsec:09d}.png",
                sec=sec,
                nsec=nsec,
                pose_values=front_pinhole_pose_from_ase(row, line_number),
            )
        )
    return records, len(trajectory_rows)


def image_info(path: Path) -> tuple[int, int, str, str]:
    Image = require_pillow()
    try:
        with Image.open(path) as image:
            return image.size[0], image.size[1], str(image.format), str(image.mode)
    except Exception as error:
        raise PrepareError(f"cannot read RGB image {path}: {error}") from error


def validate_images(records: list[FrameRecord]) -> dict[str, Any]:
    dimensions: set[tuple[int, int]] = set()
    formats: set[str] = set()
    modes: set[str] = set()
    resolved: set[Path] = set()
    names: set[str] = set()
    for record in records:
        resolved_path = record.source_image.resolve()
        if resolved_path in resolved:
            raise PrepareError(f"duplicate resolved RGB frame: {resolved_path}")
        resolved.add(resolved_path)
        if record.target_name in names:
            raise PrepareError(f"duplicate target frame name: {record.target_name}")
        names.add(record.target_name)
        width, height, image_format, mode = image_info(record.source_image)
        dimensions.add((width, height))
        formats.add(image_format)
        modes.add(mode)
    if len(dimensions) != 1:
        raise PrepareError(f"raw RGB frames have inconsistent dimensions: {sorted(dimensions)}")
    width, height = next(iter(dimensions))
    if (width, height) != (ASE_RGB_BASE_SIZE, ASE_RGB_BASE_SIZE):
        raise PrepareError(
            "ASE Fisheye624 calibration expects raw RGB frames to be "
            f"{ASE_RGB_BASE_SIZE}x{ASE_RGB_BASE_SIZE}, got {width}x{height}"
        )
    return {
        "width": width,
        "height": height,
        "source_formats": sorted(formats),
        "source_modes": sorted(modes),
        "image_count": len(records),
    }


def read_intrinsics_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PrepareError(f"invalid intrinsics JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PrepareError(f"intrinsics JSON must contain an object: {path}")
    intrinsics = payload.get("pinhole_intrinsics", payload.get("intrinsics"))
    resolution = payload.get("pinhole_resolution", payload.get("resolution"))
    if (
        not isinstance(intrinsics, list)
        or len(intrinsics) != 4
        or not all(isinstance(value, (int, float)) for value in intrinsics)
    ):
        raise PrepareError("intrinsics JSON must provide four numeric intrinsics")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(isinstance(value, int) for value in resolution)
    ):
        raise PrepareError("intrinsics JSON must provide integer width and height")
    return {
        "fx": float(intrinsics[0]),
        "fy": float(intrinsics[1]),
        "cx": float(intrinsics[2]),
        "cy": float(intrinsics[3]),
        "width": int(resolution[0]),
        "height": int(resolution[1]),
        "source": str(path),
    }


def camera_from_args(args: argparse.Namespace) -> dict[str, Any]:
    source = "ase_default_aria_fisheye624_front_pinhole"
    values: dict[str, Any] = {
        "width": ASE_PINHOLE_RESOLUTION[0],
        "height": ASE_PINHOLE_RESOLUTION[1],
        "fx": ASE_PINHOLE_INTRINSICS[0],
        "fy": ASE_PINHOLE_INTRINSICS[1],
        "cx": ASE_PINHOLE_INTRINSICS[2],
        "cy": ASE_PINHOLE_INTRINSICS[3],
    }
    if args.intrinsics_json is not None:
        values = read_intrinsics_json(args.intrinsics_json.expanduser())
        source = str(args.intrinsics_json.expanduser())
    for name in ("fx", "fy", "cx", "cy"):
        override = getattr(args, name)
        if override is not None:
            values[name] = float(override)
            source = "command_line_override"
    for name in ("fx", "fy", "cx", "cy"):
        if not math.isfinite(float(values[name])) or float(values[name]) <= 0.0:
            raise PrepareError(f"camera {name} must be finite and positive")
    for name in ("width", "height"):
        if int(values[name]) <= 0:
            raise PrepareError(f"camera {name} must be positive")
    return {
        "model": "PINHOLE",
        "width": int(values["width"]),
        "height": int(values["height"]),
        "fx": float(values["fx"]),
        "fy": float(values["fy"]),
        "cx": float(values["cx"]),
        "cy": float(values["cy"]),
        "intrinsics_source": source,
    }


def build_aria_fisheye624_remap(camera: dict[str, Any], np: Any) -> tuple[Any, Any, Any]:
    output_width, output_height = int(camera["width"]), int(camera["height"])
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])

    uu, vv = np.meshgrid(
        np.arange(output_width, dtype=np.float64),
        np.arange(output_height, dtype=np.float64),
    )

    source_ab_x = (uu - cx) / fx
    source_ab_y = (vv - cy) / fy
    source_z = np.ones_like(source_ab_x)
    front_valid = source_z > 1.0e-12

    radius = np.sqrt(source_ab_x * source_ab_x + source_ab_y * source_ab_y)
    theta = np.arctan(radius)
    theta_sq = theta * theta

    params = np.asarray(ASE_RGB_PROJECTION_PARAMS, dtype=np.float64)
    theta_radial = np.ones_like(theta)
    theta_power = theta_sq.copy()
    for coefficient in params[3:9]:
        theta_radial += coefficient * theta_power
        theta_power *= theta_sq

    theta_div_radius = np.ones_like(theta)
    nonzero = radius > 1.0e-12
    theta_div_radius[nonzero] = theta[nonzero] / radius[nonzero]

    xr = theta_radial * theta_div_radius * source_ab_x
    yr = theta_radial * theta_div_radius * source_ab_y
    r2 = xr * xr + yr * yr
    r4 = r2 * r2

    p0, p1 = params[9], params[10]
    s0, s1, s2, s3 = params[11:15]
    tangential = 2.0 * (xr * p0 + yr * p1)

    u_distorted = xr + tangential * xr + r2 * p0 + s0 * r2 + s1 * r4
    v_distorted = yr + tangential * yr + r2 * p1 + s2 * r2 + s3 * r4

    map_x = params[0] * u_distorted + params[1]
    map_y = params[0] * v_distorted + params[2]
    source_width = ASE_RGB_BASE_SIZE
    source_height = ASE_RGB_BASE_SIZE
    valid = (
        front_valid
        & (map_x >= 0.0)
        & (map_x < source_width)
        & (map_y >= 0.0)
        & (map_y < source_height)
        & (
            (map_x - params[1]) * (map_x - params[1])
            + (map_y - params[2]) * (map_y - params[2])
            <= ASE_RGB_VALID_RADIUS * ASE_RGB_VALID_RADIUS
        )
    )

    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x.astype("float32"), map_y.astype("float32"), valid


def same_bytes(path: Path, payload: bytes) -> bool:
    return path.is_file() and path.read_bytes() == payload


def write_bytes(path: Path, payload: bytes, *, force: bool) -> str:
    if path.exists() or path.is_symlink():
        if path.is_dir():
            raise PrepareError(f"refusing to replace directory: {path}")
        if same_bytes(path, payload):
            return "reused"
        if not force:
            raise PrepareError(f"target exists with different content: {path}")
        path.unlink()
    path.write_bytes(payload)
    return "written"


def pinhole_png_payload(
    source: Path,
    expected_raw_size: tuple[int, int],
    map_x: Any,
    map_y: Any,
    *,
    cv2: Any,
    np: Any,
    Image: Any,
) -> bytes:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if rgb.size != expected_raw_size:
            raise PrepareError(
                f"RGB frame changed size while writing: {source} is {rgb.size}, "
                f"expected {expected_raw_size}"
            )
        try:
            rotated = rgb.transpose(Image.Transpose.ROTATE_270)
        except AttributeError:
            rotated = rgb.transpose(Image.ROTATE_270)
        rotated_bgr = cv2.cvtColor(np.asarray(rotated), cv2.COLOR_RGB2BGR)

    rectified = cv2.remap(
        rotated_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    success, encoded = cv2.imencode(".png", rectified)
    if not success:
        raise PrepareError(f"failed to encode pinhole PNG for {source}")
    return bytes(encoded)


def poses_csv_payload(records: list[FrameRecord]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(EXPECTED_POSE_HEADER)
    for record in records:
        writer.writerow(
            [
                record.counter,
                record.sec,
                record.nsec,
                *(f"{value:.9f}" for value in record.pose_values),
            ]
        )
    return output.getvalue().encode("utf-8")


def intrinsics_payload(camera: dict[str, Any]) -> bytes:
    payload = {
        "camera_model": camera["model"],
        "pinhole_intrinsics": [
            camera["fx"],
            camera["fy"],
            camera["cx"],
            camera["cy"],
        ],
        "pinhole_resolution": [camera["width"], camera["height"]],
        "intrinsics_source": camera["intrinsics_source"],
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def output_root_for(dest_root: Path) -> Path:
    return dest_root.expanduser().resolve(strict=False)


def count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.is_dir() else 0


def transfer(
    source: Path,
    dest_root: Path,
    *,
    camera: dict[str, Any],
    records: list[FrameRecord],
    trajectory_row_count: int,
    image_summary: dict[str, Any],
    dry_run: bool,
    force: bool,
    frame_stride: int,
) -> dict[str, Any]:
    requested_dest = dest_root.expanduser()
    output_root = output_root_for(requested_dest)
    front_dir = output_root / "front"
    image_status = {"written": 0, "reused": 0}
    file_status: dict[str, str] = {}
    remap_status: dict[str, Any] = {
        "algorithm": "aria_fisheye624_to_front_pinhole",
        "raw_image_rotation": ASE_PINHOLE_SOURCE_ROTATION,
        "pose_rotation_rule": (
            "T_world_view = T_world_device * T_device_camera, then "
            "R_world_view = R_world_camera @ R_source_rotation_cw90"
        ),
        "valid_pixel_fraction": None,
    }

    if not dry_run:
        cv2, np = require_cv2_numpy()
        Image = require_pillow()
        map_x, map_y, valid = build_aria_fisheye624_remap(camera, np)
        remap_status["valid_pixel_fraction"] = float(np.mean(valid))

        if output_root.exists() and not output_root.is_dir():
            raise PrepareError(f"destination root is not a directory: {output_root}")
        front_dir.mkdir(parents=True, exist_ok=True)
        expected_raw_size = (int(image_summary["width"]), int(image_summary["height"]))
        for record in records:
            status = write_bytes(
                front_dir / record.target_name,
                pinhole_png_payload(
                    record.source_image,
                    expected_raw_size,
                    map_x,
                    map_y,
                    cv2=cv2,
                    np=np,
                    Image=Image,
                ),
                force=force,
            )
            image_status[status] = image_status.get(status, 0) + 1
        file_status["poses.csv"] = write_bytes(
            front_dir / "poses.csv",
            poses_csv_payload(records),
            force=force,
        )
        file_status["intrinsics.json"] = write_bytes(
            output_root / "intrinsics.json",
            intrinsics_payload(camera),
            force=force,
        )

    depth_dir = source / "depth"
    instance_dir = source / "instances"
    first = records[0]
    last = records[-1]
    manifest = {
        "schema_version": 2,
        "task": "ase_prepare_single",
        "dry_run": dry_run,
        "scene_id": raw_scene_id(source),
        "source": {
            "scene_root": str(source),
            "rgb_dir": str(source / "rgb"),
            "trajectory_csv": str(source / "trajectory.csv"),
            "trajectory_row_count": trajectory_row_count,
            "depth_dir": str(depth_dir) if depth_dir.is_dir() else None,
            "instance_dir": str(instance_dir) if instance_dir.is_dir() else None,
            "depth_file_count": count_pngs(depth_dir),
            "instance_file_count": count_pngs(instance_dir),
        },
        "destination": {
            "requested_root": str(requested_dest),
            "resolved_root": str(output_root),
            "front_dir": str(front_dir),
            "poses_csv": str(front_dir / "poses.csv"),
            "intrinsics_json": str(output_root / "intrinsics.json"),
            "manifest": str(output_root / "transfer_manifest.json"),
        },
        "camera": camera,
        "source_camera": {
            "camera_model": "aria_fisheye624",
            "fisheye_model": "Fisheye624",
            "fisheye_resolution": [ASE_RGB_BASE_SIZE, ASE_RGB_BASE_SIZE],
            "fisheye_projection_params": ASE_RGB_PROJECTION_PARAMS,
            "fisheye_valid_radius_px": ASE_RGB_VALID_RADIUS,
            "T_device_camera": {
                "translation": ASE_T_DEVICE_CAMERA_TRANSLATION,
                "quaternion_xyzw": ASE_T_DEVICE_CAMERA_QUAT_XYZW,
            },
        },
        "image_summary": {
            "raw": image_summary,
            "output": {
                "width": camera["width"],
                "height": camera["height"],
                "format": "PNG",
                "mode": "RGB",
            },
        },
        "frame_count": len(records),
        "frame_stride": frame_stride,
        "filename_rule": {
            "source": "rgb/vignette<frame_id:07d>.<jpg|jpeg|png>",
            "target": "<tracking_timestamp_us as sec/nsec>.png",
        },
        "pose_convention": "opencv_c2w_xyzw_meters",
        "pose_source": (
            "ASE world_device pose converted with T_device_camera, then rotated "
            "to match the clockwise-rotated front pinhole image"
        ),
        "remap": remap_status,
        "first_frame": {
            "source_frame_id": first.source_frame_id,
            "source": str(first.source_image),
            "target": first.target_name,
            "timestamp": {"sec": first.sec, "nsec": first.nsec},
        },
        "last_frame": {
            "source_frame_id": last.source_frame_id,
            "source": str(last.source_image),
            "target": last.target_name,
            "timestamp": {"sec": last.sec, "nsec": last.nsec},
        },
        "write_status": {
            "images": image_status,
            "files": file_status,
        },
        "validation": {
            "selected_rgb_frames": len(records),
            "timestamps_strictly_increasing": True,
            "raw_rgb_size_matches_aria_fisheye624": True,
            "all_rgb_frames_readable": True,
            "all_source_image_dimensions_match": True,
            "unique_source_images": True,
            "unique_target_names": True,
            "output_images_are_png_rgb": True,
            "pose_header_matches_prepare_poses": True,
            "pose_counter_is_contiguous": True,
            "no_depth_or_instance_ground_truth_copied": True,
        },
        "warnings": [
            "Raw ASE depth and instance masks are recorded for provenance but are not copied into the HouseLayout3D input tree.",
            "Only the front pinhole view is exported for HouseLayout3D.",
        ],
    }

    if not dry_run:
        manifest_path = output_root / "transfer_manifest.json"
        file_status["transfer_manifest.json"] = "written"
        manifest["write_status"]["files"] = file_status
        manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        manifest["destination"]["manifest_sha256"] = sha256(manifest_path)
    return manifest


def main() -> int:
    args = parse_args()
    try:
        source = args.source.expanduser().resolve()
        records, trajectory_row_count = read_scene(source, int(args.frame_stride))
        image_summary = validate_images(records)
        camera = camera_from_args(args)
        manifest = transfer(
            source,
            args.dest_root,
            camera=camera,
            records=records,
            trajectory_row_count=trajectory_row_count,
            image_summary=image_summary,
            dry_run=bool(args.dry_run),
            force=bool(args.force),
            frame_stride=int(args.frame_stride),
        )
    except PrepareError as error:
        raise SystemExit(f"ase_prepare_single: error: {error}") from error
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
