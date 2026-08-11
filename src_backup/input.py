"""Immutable image input audit for component ``input``."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "houselayout3d"


import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .config import PipelineConfig


class InputAuditError(RuntimeError):
    """Raised when input files do not satisfy the configured contract."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_digest(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _colmap_version(executable: Path) -> str:
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise InputAuditError(f"COLMAP executable is unavailable: {executable}")
    result = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    if not output:
        raise InputAuditError("COLMAP --version returned no output")
    return output.splitlines()[0]


def _timestamp(match: re.Match[str], name: str) -> tuple[int, int]:
    groups = match.groupdict()
    if "sec" not in groups or "nsec" not in groups:
        raise InputAuditError(
            "strict timestamp order requires named regex groups 'sec' and 'nsec'"
        )
    sec, nsec = int(groups["sec"]), int(groups["nsec"])
    if not 0 <= nsec < 1_000_000_000:
        raise InputAuditError(f"invalid nanosecond field in image name: {name}")
    return sec, nsec


def _scan_images(config: PipelineConfig) -> tuple[list[dict[str, Any]], list[str]]:
    source = config.input.images
    if not source.is_dir():
        raise InputAuditError(f"input image directory does not exist: {source}")

    paths = sorted(source.glob(config.input.image_glob), key=lambda item: item.name)
    if not paths:
        raise InputAuditError(
            f"no images match {config.input.image_glob!r} under {source}"
        )
    if len({path.name for path in paths}) != len(paths):
        raise InputAuditError("input image names are not unique")

    pattern = (
        re.compile(config.input.filename_regex)
        if config.input.filename_regex is not None
        else None
    )
    timestamps: list[tuple[int, int]] = []
    records: list[dict[str, Any]] = []
    resolved_paths: set[Path] = set()
    expected_size = (config.input.camera.width, config.input.camera.height)

    for path in paths:
        if path.is_symlink() and not path.exists():
            raise InputAuditError(f"broken image symlink: {path}")
        if not path.is_file():
            raise InputAuditError(f"input is not a regular file: {path}")

        resolved = path.resolve()
        if resolved in resolved_paths:
            raise InputAuditError(f"duplicate resolved image target: {resolved}")
        resolved_paths.add(resolved)

        match = pattern.fullmatch(path.name) if pattern is not None else None
        if pattern is not None and match is None:
            raise InputAuditError(
                f"image name does not match input.filename_regex: {path.name}"
            )
        if match is not None and config.input.require_strict_timestamp_order:
            timestamps.append(_timestamp(match, path.name))

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_format = image.format
                image_mode = image.mode
                image_size = image.size
        except Exception as error:
            raise InputAuditError(f"invalid image {path}: {error}") from error

        if image_format != "PNG":
            raise InputAuditError(f"expected PNG image, got {image_format}: {path}")
        if image_mode != "RGB":
            raise InputAuditError(f"expected RGB image, got {image_mode}: {path}")
        if image_size != expected_size:
            raise InputAuditError(
                f"image size mismatch for {path.name}: expected {expected_size}, "
                f"got {image_size}"
            )

        stat = path.stat()
        record: dict[str, Any] = {
            "name": path.name,
            "path": str(path),
            "resolved_path": str(resolved),
            "size_bytes": stat.st_size,
            "sha256": _sha256(path),
            "format": image_format,
            "mode": image_mode,
            "width": image_size[0],
            "height": image_size[1],
            "is_symlink": path.is_symlink(),
        }
        if match is not None and config.input.require_strict_timestamp_order:
            sec, nsec = timestamps[-1]
            record["timestamp"] = {"sec": sec, "nsec": nsec}
        records.append(record)

    if config.input.require_strict_timestamp_order and any(
        previous >= current
        for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise InputAuditError("image timestamps are not strictly increasing")

    selected = {path.name for path in paths}
    ignored = sorted(entry.name for entry in source.iterdir() if entry.name not in selected)
    return records, ignored


def prepare_input(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Audit configured images and write a new immutable ``input`` component."""

    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise InputAuditError("run_id must be one non-empty path component")

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    records, ignored = _scan_images(config)
    colmap_version = (
        None
        if config.input.poses_csv is not None
        else _colmap_version(config.runtime.colmap_executable)
    )

    run_dir = config.storage.outputs / config.scene / run_id
    component_dir = run_dir / "input"
    if component_dir.exists():
        raise InputAuditError(f"input component already exists and will not be overwritten: {component_dir}")
    component_dir.mkdir(parents=True, exist_ok=False)

    image_list_path = component_dir / "images.txt"
    image_list_path.write_text(
        "".join(f"{record['name']}\n" for record in records), encoding="utf-8"
    )

    revisions_path = PROJECT_ROOT / "references" / "external_revisions.json"
    finished_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "scene": config.scene,
        "run_id": run_id,
        "component": "input",
        "status": "complete",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(time.perf_counter() - start, 6),
        "command": command if command is not None else sys.argv,
        "random_seed": config.runtime.random_seed,
        "source_revisions": {
            "path": str(revisions_path),
            "sha256": _sha256(revisions_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "input": {
            "images_root": str(config.input.images),
            "image_glob": config.input.image_glob,
            "filename_regex": config.input.filename_regex,
            "require_strict_timestamp_order": (
                config.input.require_strict_timestamp_order
            ),
            "camera": {
                "model": config.input.camera.model,
                "width": config.input.camera.width,
                "height": config.input.camera.height,
                "fx": config.input.camera.fx,
                "fy": config.input.camera.fy,
                "cx": config.input.camera.cx,
                "cy": config.input.camera.cy,
            },
            "configured_pose_source": (
                str(config.input.poses_csv)
                if config.input.poses_csv is not None
                else None
            ),
            "pose_convention": config.input.pose_convention,
            "ignored_non_image_entries": ignored,
        },
        "validation": {
            "image_count": len(records),
            "total_size_bytes": sum(record["size_bytes"] for record in records),
            "combined_sha256": _combined_digest(records),
            "unique_names": True,
            "unique_resolved_targets": True,
            "all_images_readable": True,
            "all_images_png_rgb": True,
            "all_dimensions_match_camera": True,
            "timestamps_strictly_increasing": (
                True if config.input.require_strict_timestamp_order else None
            ),
            "symlink_count": sum(record["is_symlink"] for record in records),
            "broken_symlink_count": 0,
            "pose_or_ground_truth_inputs_used": False,
            "reconstruction_source": (
                "known_pose" if config.input.poses_csv is not None else "colmap"
            ),
            "colmap": (
                None
                if colmap_version is None
                else {
                    "executable": str(config.runtime.colmap_executable),
                    "version": colmap_version,
                    "matcher": config.runtime.colmap_matcher,
                }
            ),
        },
        "outputs": {
            "image_list": str(image_list_path),
            "image_list_sha256": _sha256(image_list_path),
        },
        "images": records,
        "warnings": [],
    }
    manifest_path = component_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> int:
    from .direct import run_component

    return run_component(prepare_input, "Audit and freeze the input component.")


if __name__ == "__main__":
    raise SystemExit(main())
