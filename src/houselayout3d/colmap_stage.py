"""COLMAP sparse reconstruction stage with durable logs and validation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .stages import Stage


class ColmapStageError(RuntimeError):
    """Raised when reconstruction or its acceptance checks fail."""


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


def _verify_input_stage(config: PipelineConfig, run_id: str) -> tuple[Path, dict[str, Any]]:
    input_dir = config.storage.outputs / config.scene / run_id / Stage.INPUT.value
    manifest_path = input_dir / "manifest.json"
    image_list = input_dir / "images.txt"
    if not manifest_path.is_file() or not image_list.is_file():
        raise ColmapStageError(f"complete 00_input stage is missing: {input_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ColmapStageError("00_input manifest is not complete")
    expected_list_hash = manifest["outputs"]["image_list_sha256"]
    if _sha256(image_list) != expected_list_hash:
        raise ColmapStageError("00_input/images.txt hash no longer matches its manifest")
    names = image_list.read_text(encoding="utf-8").splitlines()
    if len(names) != manifest["validation"]["image_count"]:
        raise ColmapStageError("00_input image count no longer matches images.txt")

    records = {record["name"]: record for record in manifest["images"]}
    for name in names:
        record = records.get(name)
        path = config.input.images / name
        if record is None or not path.is_file():
            raise ColmapStageError(f"audited input image is missing: {name}")
        if path.stat().st_size != record["size_bytes"] or _sha256(path) != record["sha256"]:
            raise ColmapStageError(f"audited input image changed: {name}")
    return input_dir, manifest


def build_commands(
    config: PipelineConfig,
    stage_dir: Path,
    image_list: Path,
) -> list[tuple[str, list[str]]]:
    """Return the exact feature, matching, and mapping commands."""

    executable = str(config.runtime.colmap_executable)
    database = str(stage_dir / "database.db")
    images = str(config.input.images)
    sparse = str(stage_dir / "sparse")
    camera = config.input.camera
    camera_params = f"{camera.fx},{camera.fy},{camera.cx},{camera.cy}"
    common = [
        executable,
        "--default_random_seed",
        str(config.runtime.random_seed),
        "--log_target",
        "stdout",
        "--log_color",
        "0",
    ]
    feature = [
        executable,
        "feature_extractor",
        *common[1:],
        "--database_path",
        database,
        "--image_path",
        images,
        "--image_list_path",
        str(image_list),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera.model,
        "--ImageReader.camera_params",
        camera_params,
        "--FeatureExtraction.use_gpu",
        "0",
    ]
    matcher = [
        executable,
        f"{config.runtime.colmap_matcher}_matcher",
        *common[1:],
        "--database_path",
        database,
        "--FeatureMatching.use_gpu",
        "0",
    ]
    if config.runtime.colmap_matcher == "sequential":
        matcher.extend(
            [
                "--SequentialMatching.overlap",
                str(config.runtime.colmap_sequential_overlap),
            ]
        )
    mapper = [
        executable,
        "mapper",
        *common[1:],
        "--database_path",
        database,
        "--image_path",
        images,
        "--output_path",
        sparse,
        "--Mapper.image_list_path",
        str(image_list),
        "--Mapper.random_seed",
        str(config.runtime.random_seed),
    ]
    return [
        ("feature_extractor", feature),
        (f"{config.runtime.colmap_matcher}_matcher", matcher),
        ("mapper", mapper),
    ]


def _run_command(stage_dir: Path, name: str, command: list[str]) -> float:
    log_path = stage_dir / "logs" / f"{name}.log"
    _write_status(stage_dir, name, str(log_path))
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise ColmapStageError(
            f"{name} failed with exit code {result.returncode}; see {log_path}"
        )
    return elapsed


def _first_uint64(path: Path) -> int:
    with path.open("rb") as handle:
        data = handle.read(8)
    if len(data) != 8:
        raise ColmapStageError(f"invalid COLMAP binary: {path}")
    return int(struct.unpack("<Q", data)[0])


def _database_stats(database_path: Path) -> dict[str, Any]:
    with sqlite3.connect(database_path) as connection:
        camera_row = connection.execute(
            "SELECT model, width, height, params FROM cameras"
        ).fetchone()
        if camera_row is None:
            raise ColmapStageError("COLMAP database has no camera")
        params_blob = camera_row[3]
        camera_params = list(struct.unpack(f"<{len(params_blob) // 8}d", params_blob))
        return {
            "camera_count": connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0],
            "image_count": connection.execute("SELECT COUNT(*) FROM images").fetchone()[0],
            "keypoint_image_count": connection.execute("SELECT COUNT(*) FROM keypoints").fetchone()[0],
            "keypoint_count": connection.execute("SELECT COALESCE(SUM(rows), 0) FROM keypoints").fetchone()[0],
            "descriptor_image_count": connection.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0],
            "match_pair_count": connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
            "verified_pair_count": connection.execute(
                "SELECT COUNT(*) FROM two_view_geometries"
            ).fetchone()[0],
            "camera": {
                "model_id": camera_row[0],
                "width": camera_row[1],
                "height": camera_row[2],
                "params": camera_params,
            },
        }


def _model_stats(sparse_dir: Path) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_dir in sorted(
        (path for path in sparse_dir.iterdir() if path.is_dir()),
        key=lambda path: int(path.name) if path.name.isdigit() else path.name,
    ):
        required = [model_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
        if not all(path.is_file() for path in required):
            continue
        models.append(
            {
                "name": model_dir.name,
                "path": str(model_dir),
                "camera_count": _first_uint64(model_dir / "cameras.bin"),
                "registered_image_count": _first_uint64(model_dir / "images.bin"),
                "point3D_count": _first_uint64(model_dir / "points3D.bin"),
            }
        )
    if not models:
        raise ColmapStageError("mapper produced no valid sparse models")
    return models


def _write_failure_manifest(
    stage_dir: Path,
    config: PipelineConfig,
    run_id: str,
    started_at: datetime,
    command_times: dict[str, float],
    error: Exception,
) -> None:
    _write_status(stage_dir, "failed", str(error))
    _write_json(
        stage_dir / "manifest.json",
        {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.COLMAP.value,
            "status": "failed",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "command_elapsed_seconds": command_times,
            "error": str(error),
        },
    )


def run_colmap(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Execute and validate sparse reconstruction for one audited run."""

    input_dir, input_manifest = _verify_input_stage(config, run_id)
    stage_dir = config.storage.outputs / config.scene / run_id / Stage.COLMAP.value
    if stage_dir.exists():
        raise ColmapStageError(f"COLMAP stage already exists and will not be overwritten: {stage_dir}")
    (stage_dir / "logs").mkdir(parents=True, exist_ok=False)
    (stage_dir / "sparse").mkdir()

    started_at = datetime.now(timezone.utc)
    command_times: dict[str, float] = {}
    commands = build_commands(config, stage_dir, input_dir / "images.txt")
    _write_json(
        stage_dir / "commands.json",
        {name: command_line for name, command_line in commands},
    )

    try:
        for name, command_line in commands:
            command_times[name] = round(
                _run_command(stage_dir, name, command_line), 6
            )

        database_stats = _database_stats(stage_dir / "database.db")
        expected_count = input_manifest["validation"]["image_count"]
        if database_stats["image_count"] != expected_count:
            raise ColmapStageError(
                f"database contains {database_stats['image_count']} images, expected {expected_count}"
            )
        expected_camera = config.input.camera
        actual_camera = database_stats["camera"]
        expected_params = [
            expected_camera.fx,
            expected_camera.fy,
            expected_camera.cx,
            expected_camera.cy,
        ]
        if (
            database_stats["camera_count"] != 1
            or actual_camera["width"] != expected_camera.width
            or actual_camera["height"] != expected_camera.height
            or any(abs(a - b) > 1e-8 for a, b in zip(actual_camera["params"], expected_params))
        ):
            raise ColmapStageError(f"database camera does not match configured PINHOLE camera: {actual_camera}")

        models = _model_stats(stage_dir / "sparse")
        main_model = max(
            models,
            key=lambda item: (item["registered_image_count"], item["point3D_count"]),
        )
        main_link = stage_dir / "sparse" / "main"
        os.symlink(main_model["name"], main_link)
        registration_ratio = main_model["registered_image_count"] / expected_count

        analyzer_log = stage_dir / "logs" / "model_analyzer.log"
        analyzer_command = [
            str(config.runtime.colmap_executable),
            "model_analyzer",
            "--path",
            main_model["path"],
            "--log_target",
            "stdout",
            "--log_color",
            "0",
        ]
        with analyzer_log.open("w", encoding="utf-8") as log:
            subprocess.run(
                analyzer_command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        accepted = registration_ratio >= config.runtime.minimum_registered_image_ratio
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.COLMAP.value,
            "status": "complete" if accepted else "failed",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "command": command if command is not None else sys.argv,
            "command_elapsed_seconds": command_times,
            "input_manifest": {
                "path": str(input_dir / "manifest.json"),
                "sha256": _sha256(input_dir / "manifest.json"),
            },
            "commands_path": str(stage_dir / "commands.json"),
            "database": database_stats,
            "models": models,
            "main_model": main_model,
            "registration": {
                "expected_image_count": expected_count,
                "registered_image_count": main_model["registered_image_count"],
                "ratio": registration_ratio,
                "minimum_required_ratio": config.runtime.minimum_registered_image_ratio,
                "accepted": accepted,
            },
            "outputs": {
                "database": str(stage_dir / "database.db"),
                "sparse": str(stage_dir / "sparse"),
                "main_model": str(main_link),
                "model_analyzer_log": str(analyzer_log),
            },
            "warnings": [] if accepted else [
                "largest sparse component did not meet the configured registration ratio"
            ],
        }
        manifest_path = stage_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        if not accepted:
            error = ColmapStageError(
                f"largest model registered {main_model['registered_image_count']}/{expected_count} "
                f"images ({registration_ratio:.1%}), below required "
                f"{config.runtime.minimum_registered_image_ratio:.1%}"
            )
            _write_status(stage_dir, "failed_acceptance", str(error))
            raise error
        _write_status(stage_dir, "complete", str(main_link))
        return manifest_path
    except Exception as error:
        if not (stage_dir / "manifest.json").exists():
            _write_failure_manifest(
                stage_dir, config, run_id, started_at, command_times, error
            )
        raise
