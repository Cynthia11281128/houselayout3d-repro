"""Run COLMAP feature extraction, matching, and mapping."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "houselayout3d.rgb_to_mesh"


import argparse
import ast
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class ColmapError(RuntimeError):
    """Raised when a COLMAP command fails."""


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
    outputs: Path


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int
    colmap_executable: Path
    colmap_matcher: str
    colmap_sequential_overlap: int


@dataclass(frozen=True)
class ColmapConfig:
    scene: str
    input: InputConfig
    storage: StorageConfig
    runtime: RuntimeConfig


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ColmapError(f"{name} must be a mapping")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ColmapError(f"{name} must be a non-empty path string")
    return Path(value).expanduser()


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_yaml_subset(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ColmapError(f"unsupported config line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_colmap_config(path: Path) -> ColmapConfig:
    raw = _mapping(_read_yaml_subset(path), "config")
    input_raw = _mapping(raw.get("input"), "input")
    camera_raw = _mapping(input_raw.get("camera"), "input.camera")
    storage_raw = _mapping(raw.get("storage"), "storage")
    runtime_raw = _mapping(raw.get("runtime"), "runtime")
    camera = CameraConfig(
        model=str(camera_raw.get("model", "")),
        width=int(camera_raw.get("width", 0)),
        height=int(camera_raw.get("height", 0)),
        fx=float(camera_raw.get("fx", 0)),
        fy=float(camera_raw.get("fy", 0)),
        cx=float(camera_raw.get("cx", 0)),
        cy=float(camera_raw.get("cy", 0)),
    )
    if camera.model != "PINHOLE":
        raise ColmapError("input.camera.model must be PINHOLE")
    if min(camera.width, camera.height) <= 0 or min(camera.fx, camera.fy) <= 0:
        raise ColmapError("camera dimensions and focal lengths must be positive")
    matcher = str(runtime_raw.get("colmap_matcher", "sequential"))
    if matcher not in {"sequential", "exhaustive", "spatial", "vocab_tree"}:
        raise ColmapError(f"unsupported colmap matcher: {matcher}")
    overlap = int(runtime_raw.get("colmap_sequential_overlap", 10))
    if overlap <= 0:
        raise ColmapError("runtime.colmap_sequential_overlap must be positive")
    return ColmapConfig(
        scene=str(raw.get("scene", "")),
        input=InputConfig(
            images=_path(input_raw.get("images"), "input.images"),
            camera=camera,
        ),
        storage=StorageConfig(
            outputs=_path(storage_raw.get("outputs"), "storage.outputs"),
        ),
        runtime=RuntimeConfig(
            random_seed=int(runtime_raw.get("random_seed", 0)),
            colmap_executable=_path(
                runtime_raw.get("colmap_executable"),
                "runtime.colmap_executable",
            ),
            colmap_matcher=matcher,
            colmap_sequential_overlap=overlap,
        ),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
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


def build_commands(
    config: ColmapConfig,
    component_dir: Path,
    image_list: Path | None,
) -> list[tuple[str, list[str]]]:
    executable = str(config.runtime.colmap_executable)
    database = str(component_dir / "database.db")
    images = str(config.input.images)
    sparse = str(component_dir / "sparse")
    camera = config.input.camera
    camera_params = f"{camera.fx},{camera.fy},{camera.cx},{camera.cy}"
    common = [
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
        *common,
        "--database_path",
        database,
        "--image_path",
        images,
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera.model,
        "--ImageReader.camera_params",
        camera_params,
        "--FeatureExtraction.use_gpu",
        "0",
    ]
    if image_list is not None:
        feature.extend(["--image_list_path", str(image_list)])

    matcher = [
        executable,
        f"{config.runtime.colmap_matcher}_matcher",
        *common,
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
        *common,
        "--database_path",
        database,
        "--image_path",
        images,
        "--output_path",
        sparse,
        "--Mapper.random_seed",
        str(config.runtime.random_seed),
    ]
    if image_list is not None:
        mapper.extend(["--Mapper.image_list_path", str(image_list)])

    return [
        ("feature_extractor", feature),
        (f"{config.runtime.colmap_matcher}_matcher", matcher),
        ("mapper", mapper),
    ]


def _run_command(component_dir: Path, name: str, command: list[str]) -> float:
    log_path = component_dir / "logs" / f"{name}.log"
    _write_status(component_dir, name, str(log_path))
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise ColmapError(
            f"{name} failed with exit code {result.returncode}; see {log_path}"
        )
    return elapsed


def run_colmap(
    config: ColmapConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Run COLMAP commands and return the manifest path."""

    run_dir = config.storage.outputs / config.scene / run_id
    input_image_list = run_dir / "input" / "images.txt"
    image_list = input_image_list if input_image_list.is_file() else None
    component_dir = run_dir / "colmap"
    if component_dir.exists():
        raise ColmapError(
            f"COLMAP component already exists and will not be overwritten: {component_dir}"
        )
    (component_dir / "logs").mkdir(parents=True, exist_ok=False)
    (component_dir / "sparse").mkdir()

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    command_times: dict[str, float] = {}
    commands = build_commands(config, component_dir, image_list)
    _write_json(
        component_dir / "commands.json",
        {name: command_line for name, command_line in commands},
    )

    try:
        for name, command_line in commands:
            command_times[name] = round(_run_command(component_dir, name, command_line), 6)

        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "component": "colmap",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "command": command if command is not None else sys.argv,
            "command_elapsed_seconds": command_times,
            "commands_path": str(component_dir / "commands.json"),
            "inputs": {
                "images": str(config.input.images),
                "image_list": str(image_list) if image_list is not None else None,
            },
            "outputs": {
                "database": str(component_dir / "database.db"),
                "sparse": str(component_dir / "sparse"),
                "logs": str(component_dir / "logs"),
            },
        }
        manifest_path = component_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(component_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(component_dir, "failed", str(error))
        _write_json(
            component_dir / "manifest.json",
            {
                "schema_version": 1,
                "scene": config.scene,
                "run_id": run_id,
                "component": "colmap",
                "status": "failed",
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "command": command if command is not None else sys.argv,
                "command_elapsed_seconds": command_times,
                "commands_path": str(component_dir / "commands.json"),
                "error": str(error),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run COLMAP feature extraction, matching, and mapping."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    manifest = run_colmap(load_colmap_config(args.config), args.run_id, sys.argv)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
