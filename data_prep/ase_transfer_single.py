#!/usr/bin/env python3
"""Transfer one processed ASE scene into the HouseLayout3D input layout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("/home/xinyuan/layout_reconstruction/data/aria_ase/14240")
DEFAULT_DEST_ROOT = Path("/home/xinyuan/houselayout3d-repro/data/aria_ase/14240")
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
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_WIDTH = 800
EXPECTED_HEIGHT = 600


class TransferError(RuntimeError):
    """Raised when the source scene cannot be transferred safely."""


@dataclass(frozen=True)
class FrameRecord:
    counter: int
    sec: int
    nsec: int
    pose_values: tuple[str, ...]
    source_image: Path
    target_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a HouseLayout3D-ready front-view input directory from one "
            "processed layout_reconstruction ASE scene."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Processed ASE scene root. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help=(
            "Final HouseLayout3D scene output directory. The script writes "
            "front/, intrinsics.json, and transfer_manifest.json directly here. "
            f"Default: {DEFAULT_DEST_ROOT}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the transfer plan without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace individual conflicting files or symlinks managed by this script.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TransferError(f"missing intrinsics JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise TransferError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TransferError(f"intrinsics JSON must contain an object: {path}")
    return payload


def camera_from_intrinsics(payload: dict[str, Any]) -> dict[str, Any]:
    intrinsics = payload.get("pinhole_intrinsics", payload.get("intrinsics"))
    resolution = payload.get("pinhole_resolution", payload.get("resolution"))
    if (
        not isinstance(intrinsics, list)
        or len(intrinsics) != 4
        or not all(isinstance(value, (int, float)) for value in intrinsics)
    ):
        raise TransferError("intrinsics JSON must provide four numeric pinhole intrinsics")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(isinstance(value, int) for value in resolution)
    ):
        raise TransferError("intrinsics JSON must provide two integer resolution values")
    width, height = int(resolution[0]), int(resolution[1])
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise TransferError(
            f"expected pinhole resolution {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, "
            f"got {width}x{height}"
        )
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy, cx, cy)):
        raise TransferError("pinhole intrinsics must be finite positive numbers")
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def png_info(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as handle:
        header = handle.read(33)
    if len(header) < 33 or not header.startswith(PNG_SIGNATURE):
        raise TransferError(f"not a PNG file: {path}")
    ihdr_length = int.from_bytes(header[8:12], "big")
    ihdr_type = header[12:16]
    if ihdr_length != 13 or ihdr_type != b"IHDR":
        raise TransferError(f"invalid PNG IHDR chunk: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    bit_depth = header[24]
    color_type = header[25]
    return width, height, bit_depth, color_type


def find_source_image(front_dir: Path, sec: int, nsec: int) -> Path:
    candidates = [
        front_dir / f"image_{sec}_{nsec:09d}.png",
        front_dir / f"image_{sec:010d}_{nsec:09d}.png",
        front_dir / f"{sec:010d}_{nsec:09d}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TransferError(
        "missing source image for timestamp "
        f"{sec}.{nsec:09d}; tried: {', '.join(str(path) for path in candidates)}"
    )


def read_pose_records(source_front: Path, pose_csv: Path) -> list[FrameRecord]:
    if not pose_csv.is_file():
        raise TransferError(f"missing pose CSV: {pose_csv}")
    records: list[FrameRecord] = []
    previous_timestamp: tuple[int, int] | None = None
    with pose_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise TransferError(f"pose CSV is empty: {pose_csv}") from error
        if header != EXPECTED_POSE_HEADER:
            raise TransferError(
                f"unexpected pose CSV header: expected {EXPECTED_POSE_HEADER}, got {header}"
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_POSE_HEADER):
                raise TransferError(
                    f"pose CSV row {line_number} has {len(row)} columns, expected 10"
                )
            try:
                counter = int(row[0])
                sec = int(row[1])
                nsec = int(row[2])
                numeric = tuple(float(value) for value in row[3:])
            except ValueError as error:
                raise TransferError(
                    f"pose CSV row {line_number} contains an invalid number"
                ) from error
            if counter != len(records):
                raise TransferError(
                    f"pose counter is not contiguous at row {line_number}: {counter}"
                )
            if not 0 <= nsec < 1_000_000_000:
                raise TransferError(
                    f"pose CSV row {line_number} has invalid nanoseconds: {nsec}"
                )
            timestamp = (sec, nsec)
            if previous_timestamp is not None and previous_timestamp >= timestamp:
                raise TransferError(
                    f"pose timestamps are not strictly increasing at row {line_number}"
                )
            previous_timestamp = timestamp
            if not all(math.isfinite(value) for value in numeric):
                raise TransferError(f"pose CSV row {line_number} is not finite")
            qx, qy, qz, qw = numeric[3:]
            quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if abs(quaternion_norm - 1.0) > 1.0e-5:
                raise TransferError(
                    f"pose CSV row {line_number} quaternion norm is {quaternion_norm}"
                )
            source_image = find_source_image(source_front, sec, nsec)
            target_name = f"{sec:010d}_{nsec:09d}.png"
            records.append(
                FrameRecord(
                    counter=counter,
                    sec=sec,
                    nsec=nsec,
                    pose_values=tuple(row[3:]),
                    source_image=source_image,
                    target_name=target_name,
                )
            )
    if not records:
        raise TransferError("pose CSV contains no frame rows")
    return records


def validate_images(records: list[FrameRecord], camera: dict[str, Any]) -> None:
    expected = (int(camera["width"]), int(camera["height"]))
    resolved: set[Path] = set()
    target_names: set[str] = set()
    for record in records:
        resolved_image = record.source_image.resolve()
        if resolved_image in resolved:
            raise TransferError(f"duplicate resolved source image: {resolved_image}")
        resolved.add(resolved_image)
        if record.target_name in target_names:
            raise TransferError(f"duplicate target image name: {record.target_name}")
        target_names.add(record.target_name)
        width, height, bit_depth, color_type = png_info(record.source_image)
        if (width, height) != expected:
            raise TransferError(
                f"image size mismatch for {record.source_image}: "
                f"expected {expected[0]}x{expected[1]}, got {width}x{height}"
            )
        if bit_depth != 8 or color_type != 2:
            raise TransferError(
                f"expected 8-bit RGB PNG, got bit_depth={bit_depth}, "
                f"color_type={color_type}: {record.source_image}"
            )


def same_bytes(path: Path, payload: bytes) -> bool:
    return path.is_file() and path.read_bytes() == payload


def write_file(path: Path, payload: bytes, *, force: bool) -> str:
    if path.exists() or path.is_symlink():
        if path.is_dir():
            raise TransferError(f"refusing to replace directory: {path}")
        if same_bytes(path, payload):
            return "reused"
        if not force:
            raise TransferError(f"target exists with different content: {path}")
        path.unlink()
    path.write_bytes(payload)
    return "written"


def link_image(source: Path, target: Path, *, force: bool) -> str:
    if target.exists() or target.is_symlink():
        if target.is_dir():
            raise TransferError(f"refusing to replace directory: {target}")
        if target.is_symlink() and target.resolve() == source.resolve():
            return "reused"
        if not force:
            raise TransferError(f"target exists with different destination: {target}")
        target.unlink()
    target.symlink_to(source.resolve())
    return "linked"


def pose_csv_payload(records: list[FrameRecord]) -> bytes:
    rows = [
        [
            str(record.counter),
            str(record.sec),
            str(record.nsec),
            *record.pose_values,
        ]
        for record in records
    ]
    parts = [",".join(EXPECTED_POSE_HEADER)]
    parts.extend(",".join(row) for row in rows)
    return ("\n".join(parts) + "\n").encode("utf-8")


def manifest_payload(
    *,
    source: Path,
    source_front: Path,
    pose_csv: Path,
    intrinsics_json: Path,
    dest_root: Path,
    dest_front: Path,
    records: list[FrameRecord],
    camera: dict[str, Any],
    image_status: dict[str, int],
    file_status: dict[str, str],
    dry_run: bool,
) -> dict[str, Any]:
    first = records[0]
    last = records[-1]
    return {
        "schema_version": 1,
        "task": "ase_transfer_single",
        "dry_run": dry_run,
        "source": {
            "scene_root": str(source),
            "front_dir": str(source_front),
            "poses_csv": str(pose_csv),
            "intrinsics_json": str(intrinsics_json),
        },
        "destination": {
            "root": str(dest_root),
            "front_dir": str(dest_front),
            "poses_csv": str(dest_front / "poses.csv"),
            "intrinsics_json": str(dest_root / "intrinsics.json"),
            "manifest": str(dest_root / "transfer_manifest.json"),
        },
        "camera": camera,
        "image_count": len(records),
        "link_mode": "absolute_symlink",
        "filename_rule": {
            "source_candidates": [
                "image_<sec>_<nsec:09d>.png",
                "image_<sec:010d>_<nsec:09d>.png",
                "<sec:010d>_<nsec:09d>.png",
            ],
            "target": "<sec:010d>_<nsec:09d>.png",
        },
        "first_frame": {
            "source": str(first.source_image),
            "target": first.target_name,
            "timestamp": {"sec": first.sec, "nsec": first.nsec},
        },
        "last_frame": {
            "source": str(last.source_image),
            "target": last.target_name,
            "timestamp": {"sec": last.sec, "nsec": last.nsec},
        },
        "write_status": {
            "images": image_status,
            "files": file_status,
        },
        "validation": {
            "pose_header_matches": True,
            "pose_counter_contiguous": True,
            "timestamps_strictly_increasing": True,
            "quaternion_norm_tolerance": 1.0e-5,
            "all_images_png_rgb_8bit": True,
            "all_dimensions_match_camera": True,
            "unique_source_images": True,
            "unique_target_names": True,
        },
    }


def transfer(source: Path, dest_root: Path, *, dry_run: bool, force: bool) -> dict[str, Any]:
    source = source.expanduser().resolve()
    dest_root = dest_root.expanduser()
    source_front = source / "feed_forward" / "keyframes_all3" / "front"
    pose_csv = source_front / "poses.csv"
    intrinsics_json = source / "intrinsics.json"
    if not source.is_dir():
        raise TransferError(f"source scene root does not exist: {source}")
    if not source_front.is_dir():
        raise TransferError(f"source front directory does not exist: {source_front}")

    intrinsics_payload = read_json(intrinsics_json)
    camera = camera_from_intrinsics(intrinsics_payload)
    records = read_pose_records(source_front, pose_csv)
    validate_images(records, camera)

    dest_front = dest_root / "front"
    image_status = {"linked": 0, "reused": 0}
    file_status: dict[str, str] = {}
    if not dry_run:
        dest_front.mkdir(parents=True, exist_ok=True)
        for record in records:
            status = link_image(
                record.source_image,
                dest_front / record.target_name,
                force=force,
            )
            image_status[status] = image_status.get(status, 0) + 1
        file_status["poses.csv"] = write_file(
            dest_front / "poses.csv",
            pose_csv_payload(records),
            force=force,
        )
        file_status["intrinsics.json"] = write_file(
            dest_root / "intrinsics.json",
            json.dumps(intrinsics_payload, indent=2, sort_keys=True).encode("utf-8")
            + b"\n",
            force=force,
        )

    manifest = manifest_payload(
        source=source,
        source_front=source_front,
        pose_csv=pose_csv,
        intrinsics_json=intrinsics_json,
        dest_root=dest_root,
        dest_front=dest_front,
        records=records,
        camera=camera,
        image_status=image_status,
        file_status=file_status,
        dry_run=dry_run,
    )
    if not dry_run:
        manifest_path = dest_root / "transfer_manifest.json"
        if manifest_path.is_dir():
            raise TransferError(f"refusing to replace directory: {manifest_path}")
        file_status["transfer_manifest.json"] = "written"
        manifest["write_status"]["files"] = file_status
        manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
    return manifest


def main() -> int:
    args = parse_args()
    try:
        manifest = transfer(
            args.source,
            args.dest_root,
            dry_run=bool(args.dry_run),
            force=bool(args.force),
        )
    except TransferError as error:
        raise SystemExit(f"ase_transfer_single: error: {error}") from error
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
