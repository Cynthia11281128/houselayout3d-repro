#!/usr/bin/env python3
"""Prepare r04 Insta360 keyframes as a combined Nerfstudio pinhole dataset.

The script reads the r04 root folder, rectifies selected fisheye keyframes into
virtual pinhole views, and writes one combined ``images`` folder plus one
``transforms.json`` file for the RGB-to-mesh pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError as error:  # pragma: no cover - checked at runtime.
    raise ImportError(
        "insta360_conver.py requires cv2, numpy, and tqdm. "
        "Run it with an environment such as: "
        "conda run -n houselayout3d-layout python data_preparation/insta360_conver.py"
    ) from error


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "insta360" / "r04"
DEFAULT_CAMERA_PARAMS = REPO_ROOT / "camera_param.json"
DEFAULT_DIRECTIONS = ("up", "front", "down")
SUPPORTED_DIRECTIONS = ("up", "front", "down")
IMAGE_SUFFIX = ".png"

# Source fisheye calibration from the reference feed_forward/parameters/insta360.json.
# Intrinsics are EUCM [alpha, beta, fx, fy, cx, cy].
INSTA360_EUCM_CAMERAS = {
    "cam0": {
        "intrinsics": [
            0.6957674149719358,
            0.8672913754267156,
            463.9942294132061,
            463.2452444287406,
            735.024509773121,
            719.708087825285,
        ],
        "resolution": [1472, 1440],
    },
    "cam1": {
        "intrinsics": [
            0.6963497976033663,
            0.8636463534883493,
            465.5457429529284,
            464.403997986113,
            735.812188071991,
            718.3267258840588,
        ],
        "resolution": [1472, 1440],
    },
}


class Insta360PreparationError(RuntimeError):
    """Raised when r04 data cannot be converted safely."""


@dataclass(frozen=True)
class SourceCamera:
    camera: str
    alpha: float
    beta: float
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class Keyframe:
    index: int
    sec: int
    nsec: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    source_image: Path

    @property
    def timestamp(self) -> str:
        return f"{self.sec:010d}_{self.nsec:09d}"


@dataclass(frozen=True)
class DirectionView:
    name: str
    yaw_deg: float
    pitch_deg: float
    view_to_camera: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray


def _finite_float(value: str, name: str, row_number: int) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise Insta360PreparationError(
            f"row {row_number}: {name} is not a valid float: {value!r}"
        ) from error
    if not math.isfinite(number):
        raise Insta360PreparationError(
            f"row {row_number}: {name} is not finite: {value!r}"
        )
    return number


def rotation_x(angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_a, -sin_a],
            [0.0, sin_a, cos_a],
        ],
        dtype=np.float64,
    )


def rotation_y(angle_rad: float) -> np.ndarray:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.asarray(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float64,
    )


def direction_angles(direction: str, view_angle_deg: float) -> tuple[float, float]:
    if direction == "front":
        return 0.0, 0.0
    if direction == "up":
        return 0.0, -view_angle_deg
    if direction == "down":
        return 0.0, view_angle_deg
    raise Insta360PreparationError(f"unsupported direction: {direction}")


def view_to_camera_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    camera_to_view = rotation_x(math.radians(pitch_deg)) @ rotation_y(
        math.radians(yaw_deg)
    )
    return camera_to_view.T


def quaternion_xyzw_to_matrix(values: Sequence[float]) -> np.ndarray:
    quat = np.asarray(values, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise Insta360PreparationError(f"zero-norm quaternion: {values}")
    x, y, z, w = (quat / norm).tolist()
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def opencv_c2w_to_nerfstudio_transform(
    rotation_opencv: np.ndarray, translation: Sequence[float]
) -> list[list[float]]:
    rotation_opengl = rotation_opencv @ np.diag([1.0, -1.0, -1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_opengl
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform.tolist()


def load_source_camera(camera: str) -> SourceCamera:
    raw = INSTA360_EUCM_CAMERAS.get(camera)
    if raw is None:
        available = ", ".join(sorted(INSTA360_EUCM_CAMERAS))
        raise Insta360PreparationError(
            f"unsupported camera {camera!r}; available: {available}"
        )
    alpha, beta, fx, fy, cx, cy = (float(value) for value in raw["intrinsics"])
    width, height = (int(value) for value in raw["resolution"])
    if not 0.0 <= alpha <= 1.0:
        raise Insta360PreparationError(f"EUCM alpha must be in [0,1], got {alpha}")
    if min(beta, fx, fy, width, height) <= 0:
        raise Insta360PreparationError(f"invalid EUCM calibration for {camera}")
    return SourceCamera(
        camera=camera,
        alpha=alpha,
        beta=beta,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )


def load_pinhole_camera(path: Path) -> PinholeCamera:
    if not path.is_file():
        raise FileNotFoundError(f"camera parameter JSON is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = payload.get("pinhole_intrinsics")
    resolution = payload.get("pinhole_resolution")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        raise Insta360PreparationError(
            f"{path} must contain pinhole_intrinsics=[fx,fy,cx,cy]"
        )
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise Insta360PreparationError(
            f"{path} must contain pinhole_resolution=[width,height]"
        )
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    width, height = (int(value) for value in resolution)
    if min(fx, fy, width, height) <= 0:
        raise Insta360PreparationError(f"invalid pinhole camera parameters in {path}")
    return PinholeCamera(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def build_eucm_remap(
    source: SourceCamera,
    pinhole: PinholeCamera,
    view_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uu, vv = np.meshgrid(
        np.arange(pinhole.width, dtype=np.float64),
        np.arange(pinhole.height, dtype=np.float64),
    )
    rays_view = np.stack(
        (
            (uu - pinhole.cx) / pinhole.fx,
            (vv - pinhole.cy) / pinhole.fy,
            np.ones_like(uu),
        ),
        axis=-1,
    )

    rays_camera = rays_view @ view_to_camera.T
    x = rays_camera[..., 0]
    y = rays_camera[..., 1]
    z = rays_camera[..., 2]

    distance = np.sqrt(source.beta * (x * x + y * y) + z * z)
    denominator = source.alpha * distance + (1.0 - source.alpha) * z
    projectable = denominator > 1.0e-12

    map_x = np.full_like(x, -1.0, dtype=np.float64)
    map_y = np.full_like(y, -1.0, dtype=np.float64)
    map_x[projectable] = source.fx * x[projectable] / denominator[projectable] + source.cx
    map_y[projectable] = source.fy * y[projectable] / denominator[projectable] + source.cy

    valid = (
        projectable
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x < source.width)
        & (map_y >= 0.0)
        & (map_y < source.height)
    )
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def build_direction_views(
    directions: Sequence[str],
    view_angle_deg: float,
    source: SourceCamera,
    pinhole: PinholeCamera,
) -> list[DirectionView]:
    views: list[DirectionView] = []
    for direction in directions:
        yaw_deg, pitch_deg = direction_angles(direction, view_angle_deg)
        view_to_camera = view_to_camera_rotation(yaw_deg, pitch_deg)
        map_x, map_y, valid = build_eucm_remap(source, pinhole, view_to_camera)
        views.append(
            DirectionView(
                name=direction,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                view_to_camera=view_to_camera,
                map_x=map_x,
                map_y=map_y,
                valid=valid,
            )
        )
    return views


def keyframe_image_path(root: Path, camera: str, sec: int, nsec: int) -> Path:
    return root / "fisheye_images" / camera / f"image_{sec:010d}_{nsec:09d}{IMAGE_SUFFIX}"


def load_keyframes(root: Path, camera: str, keyframes_csv: Path) -> list[Keyframe]:
    if not keyframes_csv.is_file():
        raise FileNotFoundError(f"keyframes CSV is missing: {keyframes_csv}")

    required = {"sec", "nsec", "x", "y", "z", "qx", "qy", "qz", "qw"}
    keyframes: list[Keyframe] = []
    seen: set[tuple[int, int]] = set()
    with keyframes_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise Insta360PreparationError(f"keyframes CSV has no header: {keyframes_csv}")
        missing_columns = sorted(required - set(reader.fieldnames))
        if missing_columns:
            raise Insta360PreparationError(
                f"{keyframes_csv} is missing required columns: {missing_columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            sec = int(_finite_float(row["sec"], "sec", row_number))
            nsec = int(_finite_float(row["nsec"], "nsec", row_number))
            if not 0 <= nsec < 1_000_000_000:
                raise Insta360PreparationError(
                    f"row {row_number}: nsec must be in [0, 1000000000), got {nsec}"
                )
            timestamp = (sec, nsec)
            if timestamp in seen:
                raise Insta360PreparationError(
                    f"duplicate keyframe timestamp {sec:010d}_{nsec:09d}"
                )
            seen.add(timestamp)

            values = {
                name: _finite_float(row[name], name, row_number)
                for name in ("x", "y", "z", "qx", "qy", "qz", "qw")
            }
            quaternion_xyzw_to_matrix(
                (values["qx"], values["qy"], values["qz"], values["qw"])
            )
            source_image = keyframe_image_path(root, camera, sec, nsec)
            if not source_image.is_file():
                candidate = Path(row.get("source_image_path") or "").expanduser()
                if candidate.is_file():
                    source_image = candidate
                else:
                    raise FileNotFoundError(
                        f"row {row_number}: source image is missing: {source_image}"
                    )
            keyframes.append(
                Keyframe(
                    index=len(keyframes),
                    sec=sec,
                    nsec=nsec,
                    source_image=source_image,
                    **values,
                )
            )

    if not keyframes:
        raise Insta360PreparationError(f"no keyframes loaded from {keyframes_csv}")
    return keyframes


def output_image_name(direction: str, keyframe: Keyframe) -> str:
    return f"image_{direction}_{keyframe.timestamp}{IMAGE_SUFFIX}"


def write_image(
    source_image: Path,
    output_path: Path,
    source: SourceCamera,
    view: DirectionView,
    overwrite: bool,
) -> str:
    if output_path.exists() and not overwrite:
        existing = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if existing is None:
            raise Insta360PreparationError(f"existing output image is unreadable: {output_path}")
        return "existing"

    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read source image: {source_image}")
    height, width = image.shape[:2]
    if (width, height) != (source.width, source.height):
        raise Insta360PreparationError(
            f"{source_image} has resolution {width}x{height}; "
            f"expected {source.width}x{source.height} for {source.camera}"
        )

    rectified = cv2.remap(
        image,
        view.map_x,
        view.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rectified):
        raise RuntimeError(f"failed to write output image: {output_path}")
    return "written"


def build_transforms(
    keyframes: Sequence[Keyframe],
    views: Sequence[DirectionView],
    output_images: Path,
    transforms_path: Path,
    pinhole: PinholeCamera,
) -> dict[str, object]:
    dataset_root = transforms_path.parent.resolve()
    frames: list[dict[str, object]] = []
    for keyframe in keyframes:
        base_rotation = quaternion_xyzw_to_matrix(
            (keyframe.qx, keyframe.qy, keyframe.qz, keyframe.qw)
        )
        translation = (keyframe.x, keyframe.y, keyframe.z)
        for view in views:
            output_path = output_images / output_image_name(view.name, keyframe)
            try:
                file_path = output_path.resolve().relative_to(dataset_root)
            except ValueError:
                file_path = output_path.resolve()
            rotated_opencv = base_rotation @ view.view_to_camera
            frames.append(
                {
                    "file_path": str(file_path),
                    "transform_matrix": opencv_c2w_to_nerfstudio_transform(
                        rotated_opencv, translation
                    ),
                }
            )

    return {
        "camera_model": "OPENCV",
        "fl_x": pinhole.fx,
        "fl_y": pinhole.fy,
        "cx": pinhole.cx,
        "cy": pinhole.cy,
        "w": pinhole.width,
        "h": pinhole.height,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "orientation_override": "none",
        "source_pose_convention": "opencv_c2w_xyzw_meters",
        "world_translation_unit": "meter",
        "frames": frames,
    }


def write_json_if_allowed(path: Path, payload: dict[str, object], overwrite: bool) -> str:
    content = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if path.exists() and not overwrite:
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return "existing"
        raise FileExistsError(f"{path} already exists and differs; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "written"


def write_manifest(
    path: Path,
    *,
    root: Path,
    keyframes_csv: Path,
    camera_params: Path,
    output_images: Path,
    transforms_path: Path,
    source: SourceCamera,
    pinhole: PinholeCamera,
    views: Sequence[DirectionView],
    keyframe_count: int,
    written_images: int,
    existing_images: int,
    overwrite: bool,
) -> str:
    manifest = {
        "task": "insta360_keyframes_to_combined_pinhole_nerfstudio",
        "root": str(root),
        "keyframes_csv": str(keyframes_csv),
        "camera_param_json": str(camera_params),
        "source_camera": {
            "model": "eucm",
            "camera": source.camera,
            "intrinsics": [source.alpha, source.beta, source.fx, source.fy, source.cx, source.cy],
            "resolution": [source.width, source.height],
        },
        "pinhole_camera": {
            "intrinsics": [pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy],
            "resolution": [pinhole.width, pinhole.height],
        },
        "directions": [
            {
                "name": view.name,
                "yaw_deg": view.yaw_deg,
                "pitch_deg": view.pitch_deg,
                "valid_pixel_fraction": float(np.mean(view.valid)),
            }
            for view in views
        ],
        "keyframes": keyframe_count,
        "frame_count": keyframe_count * len(views),
        "written_images": written_images,
        "existing_images": existing_images,
        "output_images": str(output_images),
        "transforms_json": str(transforms_path),
        "filename_rule": "images/image_<direction>_<sec:010d>_<nsec:09d>.png",
        "pose_rotation_rule": "R_world_view = R_world_camera @ R_view_to_camera; translation unchanged",
    }
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        ignored = {"written_images", "existing_images"}
        stable_existing = {
            key: value for key, value in existing.items() if key not in ignored
        }
        stable_manifest = {
            key: value for key, value in manifest.items() if key not in ignored
        }
        if stable_existing == stable_manifest:
            return "existing"
        raise FileExistsError(f"{path} already exists and differs; pass --overwrite")
    return write_json_if_allowed(path, manifest, overwrite)


def prepare_dataset(args: argparse.Namespace) -> dict[str, object]:
    root = args.root.expanduser().resolve()
    camera_params = args.camera_params.expanduser().resolve()
    keyframes_csv = args.keyframes_csv.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_images = (
        args.output_images.expanduser().resolve()
        if args.output_images is not None
        else output_root / "images"
    )
    transforms_path = (
        args.output_transforms.expanduser().resolve()
        if args.output_transforms is not None
        else output_root / "transforms.json"
    )
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else output_root / "preparation_manifest.json"

    if len(set(args.directions)) != len(args.directions):
        raise Insta360PreparationError("--directions contains duplicate values")

    source = load_source_camera(args.camera)
    pinhole = load_pinhole_camera(camera_params)
    views = build_direction_views(args.directions, args.view_angle_deg, source, pinhole)
    keyframes = load_keyframes(root, args.camera, keyframes_csv)
    if args.limit is not None:
        keyframes = keyframes[: args.limit]
    if not keyframes:
        raise Insta360PreparationError("no keyframes selected")

    output_images.mkdir(parents=True, exist_ok=True)
    written_images = 0
    existing_images = 0
    for keyframe in tqdm(keyframes, desc="fisheye -> pinhole", unit="keyframe"):
        for view in views:
            output_path = output_images / output_image_name(view.name, keyframe)
            status = write_image(
                keyframe.source_image,
                output_path,
                source,
                view,
                args.overwrite,
            )
            if status == "written":
                written_images += 1
            else:
                existing_images += 1

    transforms = build_transforms(
        keyframes,
        views,
        output_images,
        transforms_path,
        pinhole,
    )
    transform_status = write_json_if_allowed(
        transforms_path,
        transforms,
        args.overwrite,
    )
    manifest_status = write_manifest(
        manifest_path,
        root=root,
        keyframes_csv=keyframes_csv,
        camera_params=camera_params,
        output_images=output_images,
        transforms_path=transforms_path,
        source=source,
        pinhole=pinhole,
        views=views,
        keyframe_count=len(keyframes),
        written_images=written_images,
        existing_images=existing_images,
        overwrite=args.overwrite,
    )
    return {
        "root": str(root),
        "images": str(output_images),
        "transforms": str(transforms_path),
        "manifest": str(manifest_path),
        "keyframes": len(keyframes),
        "directions": list(args.directions),
        "frames": len(keyframes) * len(views),
        "written_images": written_images,
        "existing_images": existing_images,
        "transforms_status": transform_status,
        "manifest_status": manifest_status,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert r04 Insta360 keyframes into a combined up/front/down "
            "Nerfstudio pinhole dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--camera", choices=tuple(INSTA360_EUCM_CAMERAS), default="cam0")
    parser.add_argument("--camera-params", type=Path, default=DEFAULT_CAMERA_PARAMS)
    parser.add_argument(
        "--keyframes-csv",
        type=Path,
        default=DEFAULT_ROOT / "keyframes.csv",
        help="CSV with selected keyframes and x,y,z,qx,qy,qz,qw pose columns.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=SUPPORTED_DIRECTIONS,
        default=list(DEFAULT_DIRECTIONS),
    )
    parser.add_argument(
        "--view-angle-deg",
        type=float,
        default=57.0,
        help="Pitch angle used for up/down virtual views.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root containing images/ and transforms.json unless explicit output paths are set.",
    )
    parser.add_argument("--output-images", type=Path)
    parser.add_argument("--output-transforms", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, help="Only convert the first N keyframes.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if not 0.0 <= args.view_angle_deg < 90.0:
        parser.error("--view-angle-deg must be in [0, 90)")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = prepare_dataset(parse_args(argv))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
