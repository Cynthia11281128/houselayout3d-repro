"""Prepare and run the Section 4.3 prototype fitting wrapper."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.layout_prototype"

import argparse
import hashlib
import importlib.util
import json
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

import numpy as np

from src.layout_skeleton.labels import LAYOUT_LABELS


class PrototypeError(RuntimeError):
    """Raised when prototype preparation or fitting fails."""


SOURCE_FILE_SHA256 = {
    "fit_prototype.py": "1b4cdeeb3148fd6f145b29f12018a8308601b7da3f2d47608b01d8ab7757cb90",
    "mesh_fitting_3D/cgal_triangulations.py": "0ce34746691bb227c011f0a63d9e7425d47a4c7d98109e1638f092b3e34ba3f7",
    "mesh_fitting_3D/chamfer_distance.py": "110e98550340ee34f53620540c5dd412660a6438c62dfa6cb1964b9d9ba9f1fc",
    "mesh_fitting_3D/differentiable_3D_polygon_stuctures.py": "d976df8d3cd81e7bee9566a663558e1661eb399815ac6608ffee653c58efb7c0",
    "mesh_fitting_3D/differentiable_mesh_sampling.py": "c531414fdbfdf47a194b021b807b5dce1b06ca1b95252347abf44e9836c915df",
    "mesh_fitting_3D/geometry_utils.py": "19194eda3c71f45704cda0ed745b0fe8c36bb1cbde423e7c149b1fd5fce9516c",
    "mesh_fitting_3D/merge_split_util.py": "54b1169b3801770e850ac49d29c0aa95ba7b5b27ff2ed2041e90576b5503ec42",
    "mesh_fitting_3D/point_triangle_distance_vectorized.py": "3a30af9998d103be8f42c2f9fcaaad989214b85e336db753550e107abbb5b007",
    "mesh_fitting_3D/polygon_fitting_config.py": "7fe51903979c2ef10e15c30fcd4d15417d6d9fc9fe1ec8904c2fb733fdc601d6",
}


@dataclass(frozen=True)
class PrepareConfig:
    skeleton: Path
    polygon_init: Path
    source_repo: Path
    output: Path
    python: Path
    random_seed: int = 0
    scene_type: str = "matterport"
    iterations: int = 4000
    checkpoint_interval: int = 100
    object_target_triangles: int = 200_000
    strict_source_hashes: bool = False
    skip_runtime_probe: bool = False


@dataclass(frozen=True)
class FitConfig:
    prepared: Path
    source_repo: Path | None
    python: Path | None
    preferred_gpu: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_status(path: Path, state: str, detail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _component_manifest(path_or_dir: Path, name: str) -> Path:
    path = path_or_dir.expanduser().resolve()
    if path.is_dir():
        candidates = [path / "prepare_manifest.json", path / "manifest.json"]
        path = next((candidate for candidate in candidates if candidate.is_file()), path / "manifest.json")
    if not path.is_file():
        raise PrototypeError(f"{name} manifest is missing: {path}")
    return path


def _resolve_component_artifact(manifest_path: Path, path_text: str, component: str) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    parts = path.parts
    if component in parts:
        suffix = Path(*parts[parts.index(component) + 1 :])
        candidate = manifest_path.parent / suffix
        if candidate.is_file():
            return candidate
    if not path.is_absolute():
        candidate = manifest_path.parent / path
        if candidate.is_file():
            return candidate
    return path


def _verify_manifest_record(
    manifest_path: Path,
    record: dict[str, Any],
    name: str,
    component: str,
) -> Path:
    path = _resolve_component_artifact(manifest_path, record["path"], component)
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise PrototypeError(f"artifact hash mismatch for {name}: {path}")
    return path


def _copy_record(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _record(destination)


def _verify_source(source_repo: Path, strict_hashes: bool) -> dict[str, dict[str, Any]]:
    root = source_repo.expanduser().resolve()
    records: dict[str, dict[str, Any]] = {}
    for relative, expected in SOURCE_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise PrototypeError(f"unofficial source file is missing: {path}")
        actual = _sha256(path)
        if strict_hashes and actual != expected:
            raise PrototypeError(f"unofficial source hash changed for {relative}: {actual}")
        records[relative] = {
            "path": str(path),
            "sha256": actual,
            "expected_sha256": expected,
            "matches_expected": actual == expected,
            "size_bytes": path.stat().st_size,
        }
    return records


def _source_config(source_repo: Path, iterations: int, checkpoint_interval: int) -> dict[str, Any]:
    path = source_repo.expanduser().resolve() / "mesh_fitting_3D" / "polygon_fitting_config.py"
    spec = importlib.util.spec_from_file_location("multifloor3d_polygon_config", path)
    if spec is None or spec.loader is None:
        raise PrototypeError("cannot load the unofficial MatterportConfig")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.MatterportConfig()
    values = {
        "iterations": int(source.iterations),
        "save_interval": int(source.save_interval),
        "up_vector": str(source.up_vector),
        "multi_floor": bool(source.multi_floor),
        "project_objects_to_floor_at_step": int(source.project_objects_to_floor_at_step),
        "simplification_steps": list(source.simplification_steps),
        "plane_merge_steps": list(source.plane_merge_steps),
        "ray_tracing_distance": bool(source.ray_tracing_distance),
        "chamfer_distance": bool(source.chamfer_distance),
        "point_triangle_distance": bool(source.point_triangle_distance),
        "vertex_merge_thresh": float(source.vertex_merge_thresh),
        "max_dist": float(source.max_dist),
        "regularization_strength": float(source.regularization_strength),
        "max_plane_merge_dist": float(source.max_plane_merge_dist),
        "max_plane_merge_angle": float(source.max_plane_merge_angle),
        "max_ray_intersects_per_m2": int(source.max_ray_intersects_per_m2),
    }
    if values["iterations"] != iterations:
        raise PrototypeError("configured iterations differ from MatterportConfig")
    if values["save_interval"] != checkpoint_interval:
        raise PrototypeError("configured checkpoint interval differs from source")
    if values["up_vector"] != "Z" or not values["multi_floor"]:
        raise PrototypeError("MatterportConfig must remain Z-up and multi-floor")
    return values


def _verify_inputs(config: PrepareConfig) -> dict[str, Path]:
    skeleton_manifest_path = _component_manifest(config.skeleton, "skeleton")
    polygon_manifest_path = _component_manifest(config.polygon_init, "polygon_init")
    skeleton = _read_json(skeleton_manifest_path)
    polygon = _read_json(polygon_manifest_path)
    if skeleton.get("status") != "complete" or polygon.get("status") != "complete":
        raise PrototypeError("skeleton and polygon_init must both be complete")

    skeleton_outputs = skeleton["outputs"]
    structure_record = skeleton_outputs["meshes"]["structure"]
    if not structure_record.get("classes_path"):
        raise PrototypeError("structure class-probability artifact is missing")
    structure_classes_record = {
        "path": structure_record["classes_path"],
        "sha256": structure_record["classes_sha256"],
    }
    object_record = skeleton_outputs["meshes"]["objects"]
    object_mesh = None
    if object_record.get("path"):
        object_mesh = _verify_manifest_record(skeleton_manifest_path, object_record, "object mesh", "skeleton")
    rectified_record = polygon["outputs"].get("rectified_mesh") or polygon["outputs"]["clean_edge_mesh"]
    return {
        "skeleton_manifest": skeleton_manifest_path,
        "polygon_manifest": polygon_manifest_path,
        "target_mesh": _verify_manifest_record(skeleton_manifest_path, structure_record, "structure mesh", "skeleton"),
        "target_classes": _verify_manifest_record(
            skeleton_manifest_path, structure_classes_record, "structure classes", "skeleton"
        ),
        "object_mesh": object_mesh,
        "ray_origins": _verify_manifest_record(
            skeleton_manifest_path, skeleton_outputs["arrays"]["full_ray_origins.npy"], "ray origins", "skeleton"
        ),
        "ray_destinations": _verify_manifest_record(
            skeleton_manifest_path, skeleton_outputs["arrays"]["full_ray_dests.npy"], "ray destinations", "skeleton"
        ),
        "ray_validity": _verify_manifest_record(
            skeleton_manifest_path, skeleton_outputs["arrays"]["ray_is_valid.npy"], "ray validity", "skeleton"
        ),
        "ray_classes": _verify_manifest_record(
            skeleton_manifest_path,
            skeleton_outputs["arrays"]["hard_labels_simplified_segmentations.npy"],
            "ray classes",
            "skeleton",
        ),
        "class_names": _verify_manifest_record(
            skeleton_manifest_path,
            skeleton_outputs["arrays"]["simplified_segmentation_labels.npy"],
            "class names",
            "skeleton",
        ),
        "rectified_mesh": _verify_manifest_record(polygon_manifest_path, rectified_record, "rectified mesh", "polygon_init"),
        "polygon_info": _verify_manifest_record(
            polygon_manifest_path, polygon["outputs"]["polygon_info"], "polygon info", "polygon_init"
        ),
    }


def _verify_runtime(executable: Path) -> dict[str, Any]:
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PrototypeError(f"prototype Python is unavailable: {executable}")
    probe = (
        "import json,torch,pytorch3d,open3d,shapely,sklearn,rdp,CGAL; "
        "from CGAL.CGAL_Kernel import Point_2,Polygon_2; "
        "from CGAL.CGAL_Triangulation_2 import Constrained_triangulation_2; "
        "print(json.dumps({'torch':torch.__version__,'pytorch3d':pytorch3d.__version__,"
        "'open3d':open3d.__version__,'shapely':shapely.__version__,"
        "'sklearn':sklearn.__version__,'cgal':CGAL.__version__,"
        "'cuda':torch.cuda.is_available()}))"
    )
    result = subprocess.run([str(executable), "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        raise PrototypeError("prototype dependency probe failed: " + result.stderr.strip())
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    if not versions["cuda"]:
        raise PrototypeError("prototype runtime cannot access CUDA")
    return versions


def _prepare_semantic_classes(
    input_classes: Path,
    input_names: Path,
    output_classes: Path,
    output_names: Path,
) -> dict[str, Any]:
    probabilities = np.load(input_classes)
    names = tuple(str(value) for value in np.load(input_names))
    if names != LAYOUT_LABELS:
        raise PrototypeError("skeleton class order changed")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(names):
        raise PrototypeError("structure class-probability shape is invalid")
    if "door" in names:
        raise PrototypeError("unexpected upstream door class")
    prepared_names = np.asarray((*names, "door"))
    prepared_probabilities = np.column_stack(
        (probabilities, np.zeros(len(probabilities), dtype=probabilities.dtype))
    )
    np.save(output_classes, prepared_probabilities)
    np.save(output_names, prepared_names)
    return {
        "source_shape": list(probabilities.shape),
        "prepared_shape": list(prepared_probabilities.shape),
        "source_names": list(names),
        "prepared_names": prepared_names.tolist(),
        "door_probability_is_zero": bool(np.all(prepared_probabilities[:, -1] == 0)),
    }


def _write_empty_object_mesh(output_path: Path) -> dict[str, Any]:
    output_path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 0\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n",
        encoding="ascii",
    )
    return {
        "source_vertex_count": 0,
        "source_triangle_count": 0,
        "target_triangle_count": 0,
        "output_vertex_count": 0,
        "output_triangle_count": 0,
        "empty_object_mesh_placeholder": True,
    }


def _simplify_object_mesh(input_path: Path | None, output_path: Path, target_triangles: int) -> dict[str, Any]:
    if input_path is None:
        return _write_empty_object_mesh(output_path)
    try:
        import open3d as o3d
    except ImportError as error:
        raise PrototypeError("Open3D is required to prepare the object mesh") from error
    mesh = o3d.io.read_triangle_mesh(str(input_path), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return _write_empty_object_mesh(output_path)
    source_vertices = len(mesh.vertices)
    source_triangles = len(mesh.triangles)
    source_area = float(mesh.get_surface_area())
    if source_triangles > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return _write_empty_object_mesh(output_path)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False):
        raise PrototypeError("failed to write simplified object mesh")
    return {
        "source_vertex_count": source_vertices,
        "source_triangle_count": source_triangles,
        "source_surface_area_square_meters": source_area,
        "target_triangle_count": target_triangles,
        "output_vertex_count": len(mesh.vertices),
        "output_triangle_count": len(mesh.triangles),
        "output_surface_area_square_meters": float(mesh.get_surface_area()),
        "empty_object_mesh_placeholder": False,
    }


def build_prototype_command(
    python: Path,
    source_repo: Path,
    prepared: dict[str, Path],
    output_dir: Path,
    random_seed: int,
    scene_type: str,
) -> list[str]:
    source_script = source_repo.expanduser().resolve() / "fit_prototype.py"
    return [
        str(python),
        "-m",
        "src.layout_prototype.prototype_entry",
        "--source-script",
        str(source_script),
        "--random-seed",
        str(random_seed),
        "--scene-type",
        scene_type,
        "--rectified-ply-path",
        str(prepared["rectified_mesh"]),
        "--target-pcd-path",
        str(prepared["target_mesh"]),
        "--target-vertex-classes",
        str(prepared["target_classes"]),
        "--target-vertex-class-names",
        str(prepared["class_names"]),
        "--polygon-info-path",
        str(prepared["polygon_info"]),
        "--target-pcd-ray-origins-path",
        str(prepared["ray_origins"]),
        "--target-pcd-ray-dests-path",
        str(prepared["ray_destinations"]),
        "--object-mesh",
        str(prepared["object_mesh"]),
        "--ray-classes",
        str(prepared["ray_classes"]),
        "--output-dir",
        str(output_dir),
        "--device",
        "cuda",
    ]


def prepare_prototype(config: PrepareConfig) -> Path:
    inputs = _verify_inputs(config)
    component_dir = config.output.expanduser()
    if component_dir.exists():
        raise PrototypeError(f"prototype component already exists: {component_dir}")
    frozen_dir = component_dir / "frozen_inputs"
    frozen_dir.mkdir(parents=True, exist_ok=False)
    _write_status(component_dir / "STATUS.json", "preparing", "freezing prototype inputs")
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    try:
        source_records = _verify_source(config.source_repo, config.strict_source_hashes)
        source_config = _source_config(config.source_repo, config.iterations, config.checkpoint_interval)
        runtime_versions = (
            {"probe_skipped": True}
            if config.skip_runtime_probe
            else _verify_runtime(config.python.expanduser().resolve())
        )
        semantic = _prepare_semantic_classes(
            inputs["target_classes"],
            inputs["class_names"],
            frozen_dir / "target_vertex_classes.npy",
            frozen_dir / "target_vertex_class_names.npy",
        )
        object_stats = _simplify_object_mesh(
            inputs["object_mesh"],
            frozen_dir / "objects_mesh_simplified.ply",
            config.object_target_triangles,
        )
        frozen_records = {
            "rectified_mesh": _copy_record(inputs["rectified_mesh"], frozen_dir / "rectified_mesh.ply"),
            "target_mesh": _copy_record(inputs["target_mesh"], frozen_dir / "target_mesh.ply"),
            "target_classes": _record(frozen_dir / "target_vertex_classes.npy"),
            "class_names": _record(frozen_dir / "target_vertex_class_names.npy"),
            "polygon_info": _copy_record(inputs["polygon_info"], frozen_dir / "polygon_info.json"),
            "ray_origins": _copy_record(inputs["ray_origins"], frozen_dir / "full_ray_origins.npy"),
            "ray_destinations": _copy_record(inputs["ray_destinations"], frozen_dir / "full_ray_dests.npy"),
            "ray_validity": _copy_record(inputs["ray_validity"], frozen_dir / "ray_is_valid.npy"),
            "ray_classes": _copy_record(inputs["ray_classes"], frozen_dir / "hard_labels_simplified_segmentations.npy"),
            "object_mesh": _record(frozen_dir / "objects_mesh_simplified.ply"),
        }
        frozen_paths = {name: Path(record["path"]) for name, record in frozen_records.items()}
        command_template = build_prototype_command(
            config.python.expanduser().resolve(),
            config.source_repo,
            frozen_paths,
            Path("<ATTEMPT_OUTPUT_DIR>"),
            config.random_seed,
            config.scene_type,
        )
        _write_json(component_dir / "commands.json", {"fit_prototype_template": command_template})
        prepared_manifest = {
            "schema_version": 1,
            "component": "prototype",
            "status": "prepared",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "random_seed": config.random_seed,
            "inputs": {
                "skeleton_manifest": _record(inputs["skeleton_manifest"]),
                "polygon_manifest": _record(inputs["polygon_manifest"]),
            },
            "source": {
                "repository": str(config.source_repo.expanduser().resolve()),
                "files": source_records,
                "matterport_config": source_config,
                "source_files_modified": any(
                    not record["matches_expected"] for record in source_records.values()
                ),
                "strict_source_hashes": config.strict_source_hashes,
            },
            "prepared_inputs": frozen_records,
            "adaptations": {
                "semantic_classes": semantic,
                "object_mesh_simplification": object_stats,
                "seeded_launcher": "Seeds Python, NumPy, torch, and all CUDA devices before runpy executes the unmodified entrypoint.",
                "all_rays_including_invalid_depth_fallbacks": True,
            },
            "environment": {
                "python_executable": str(config.python.expanduser().resolve()),
                "versions": runtime_versions,
                "platform": platform.platform(),
            },
            "warnings": [
                "The unofficial optimizer requires a door class; a zero-probability door channel is appended only for source compatibility.",
                "Object mesh simplification target is explicit because the supplied source does not specify one.",
                "The supplied source fits with every ray, including invalid-depth fallback destinations; this behavior is preserved.",
            ],
        }
        manifest_path = component_dir / "prepare_manifest.json"
        _write_json(manifest_path, prepared_manifest)
        _write_status(component_dir / "STATUS.json", "prepared", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(component_dir / "STATUS.json", "failed", str(error))
        if isinstance(error, PrototypeError):
            raise
        raise PrototypeError(str(error)) from error


def _load_prepared(config: FitConfig) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    component_dir = config.prepared.expanduser().resolve()
    manifest_path = component_dir / "prepare_manifest.json"
    if not manifest_path.is_file():
        raise PrototypeError(f"prepare_manifest.json is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "prepared":
        raise PrototypeError("prepared prototype manifest is invalid")
    if (component_dir / "manifest.json").is_file() and _read_json(component_dir / "manifest.json").get("status") == "complete":
        raise PrototypeError("prototype fitting is already complete")
    source_repo = config.source_repo or Path(manifest["source"]["repository"])
    _verify_source(source_repo, bool(manifest["source"].get("strict_source_hashes", False)))
    frozen: dict[str, Path] = {}
    for name, record in manifest["prepared_inputs"].items():
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise PrototypeError(f"prepared input hash mismatch: {name}")
        frozen[name] = path
    return component_dir, manifest, frozen


def _next_attempt(component_dir: Path) -> Path:
    attempts = component_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    indices = []
    for path in attempts.glob("attempt_*_*"):
        match = re.match(r"attempt_(\d+)_", path.name)
        if match:
            indices.append(int(match.group(1)))
    index = max(indices, default=-1) + 1
    return attempts / f"attempt_{index:03d}_running"


def _rename_attempt(attempt: Path, suffix: str) -> Path:
    destination = attempt.with_name(attempt.name.rsplit("_", 1)[0] + f"_{suffix}")
    if destination.exists():
        raise PrototypeError(f"attempt destination already exists: {destination}")
    attempt.rename(destination)
    return destination


def _output_artifact(path: Path) -> dict[str, Any]:
    try:
        import open3d as o3d
    except ImportError as error:
        raise PrototypeError("Open3D is required to validate fitted meshes") from error
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise PrototypeError(f"fitted mesh is empty: {path}")
    if not np.isfinite(vertices).all():
        raise PrototypeError(f"fitted mesh has non-finite vertices: {path}")
    record = _record(path)
    record.update(
        {
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
            "surface_area_square_meters": float(mesh.get_surface_area()),
            "axis_aligned_minimum": vertices.min(axis=0).tolist(),
            "axis_aligned_maximum": vertices.max(axis=0).tolist(),
        }
    )
    return record


def fit_prototype(config: FitConfig) -> Path:
    component_dir, prepared_manifest, frozen = _load_prepared(config)
    source_repo = (config.source_repo or Path(prepared_manifest["source"]["repository"])).expanduser().resolve()
    python = (config.python or Path(prepared_manifest["environment"]["python_executable"])).expanduser().resolve()
    attempt = _next_attempt(component_dir)
    attempt.mkdir(parents=False, exist_ok=False)
    command = build_prototype_command(
        python,
        source_repo,
        frozen,
        attempt,
        int(prepared_manifest["random_seed"]),
        "matterport",
    )
    _write_json(attempt / "command.json", {"command": command})
    _write_status(attempt / "STATUS.json", "running", "unofficial MatterportConfig")
    _write_status(component_dir / "STATUS.json", "running", str(attempt))
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    environment = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    python_paths = [str(project_root), str(source_repo), str(source_repo / "mesh_fitting_3D")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["CUDA_VISIBLE_DEVICES"] = str(config.preferred_gpu)
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONUNBUFFERED"] = "1"
    log_path = attempt / "fit.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, cwd=attempt, env=environment, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        failed_attempt = _rename_attempt(attempt, "failed")
        _write_status(
            failed_attempt / "STATUS.json",
            "failed",
            f"return_code={result.returncode}; log={failed_attempt / log_path.name}",
        )
        _write_status(component_dir / "STATUS.json", "failed", str(failed_attempt))
        raise PrototypeError(f"prototype optimizer failed with code {result.returncode}; see {failed_attempt / log_path.name}")

    completed_attempt = _rename_attempt(attempt, "complete")
    final_mesh_path = completed_attempt / "fitted_mesh.ply"
    checkpoint_path = completed_attempt / "polygon_set_3d.pt"
    if not final_mesh_path.is_file() or not checkpoint_path.is_file():
        raise PrototypeError("optimizer returned zero but final outputs are missing")
    final_mesh = _output_artifact(final_mesh_path)
    initial_mesh = _output_artifact(completed_attempt / "fitted_mesh_00.ply")
    snapshots = []
    for path in sorted(completed_attempt.glob("fitted_mesh_[0-9]*.ply")):
        match = re.fullmatch(r"fitted_mesh_(\d+)\.ply", path.name)
        if match and path.name != "fitted_mesh_00.ply":
            snapshots.append({"step": int(match.group(1)), **_record(path)})
    state_snapshots = [
        {"step": int(match.group(1)), **_record(path)}
        for path in sorted(completed_attempt.glob("polygon_set_3d_[0-9]*.pt"))
        if (match := re.fullmatch(r"polygon_set_3d_(\d+)\.pt", path.name))
    ]
    snapshots.sort(key=lambda record: record["step"])
    state_snapshots.sort(key=lambda record: record["step"])
    manifest = {
        "schema_version": 1,
        "component": "prototype",
        "status": "complete",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 6),
        "random_seed": int(prepared_manifest["random_seed"]),
        "command": command,
        "return_code": result.returncode,
        "inputs": {
            "prepared_manifest": _record(component_dir / "prepare_manifest.json"),
            **prepared_manifest["prepared_inputs"],
        },
        "source": prepared_manifest["source"],
        "algorithm": {
            "paper_objectives": ["Lgeo=Lprox+Lempty", "Lconnect", "Lsimple"],
            "source_config": prepared_manifest["source"]["matterport_config"],
            "semantic_preparation": prepared_manifest["adaptations"]["semantic_classes"],
            "object_mesh_simplification": prepared_manifest["adaptations"]["object_mesh_simplification"],
        },
        "outputs": {
            "attempt_dir": str(completed_attempt),
            "log": _record(completed_attempt / "fit.log"),
            "initial_mesh": initial_mesh,
            "final_mesh": final_mesh,
            "final_model_state": _record(checkpoint_path),
            "mesh_checkpoints": snapshots,
            "model_state_checkpoints": state_snapshots,
        },
        "validation": {
            "source_files_recorded": True,
            "optimizer_exit_code_zero": result.returncode == 0,
            "final_mesh_nonempty": final_mesh["vertex_count"] > 0 and final_mesh["triangle_count"] > 0,
            "final_mesh_finite": True,
            "no_ground_truth_inputs_used": True,
        },
        "environment": {
            **prepared_manifest["environment"],
            "cuda_visible_devices": str(config.preferred_gpu),
        },
        "warnings": prepared_manifest["warnings"],
    }
    if not all(manifest["validation"].values()):
        raise PrototypeError("prototype validation failed")
    manifest_path = component_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_status(completed_attempt / "STATUS.json", "complete", str(final_mesh_path))
    _write_status(component_dir / "STATUS.json", "complete", str(manifest_path))
    return manifest_path


def _default_root() -> Path:
    return Path(os.environ.get("ROOT", "data/insta360/r04"))


def _default_python() -> Path:
    return Path(os.environ.get("PROTOTYPE_PYTHON", sys.executable))


def _add_prepare_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    root = _default_root()
    parser.add_argument("--skeleton", type=Path, default=root / "skeleton", required=required)
    parser.add_argument("--polygon-init", type=Path, default=root / "polygon_init", required=required)
    parser.add_argument("--source-repo", type=Path, default=Path("MultiFloor3D-unofficial"), required=required)
    parser.add_argument("--output", type=Path, default=root / "prototype", required=required)
    parser.add_argument("--python", type=Path, default=_default_python())
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--scene-type", default="matterport")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--object-target-triangles", type=int, default=200_000)
    parser.add_argument("--strict-source-hashes", action="store_true")
    parser.add_argument("--skip-runtime-probe", action="store_true")


def _prepare_config_from_args(args: argparse.Namespace) -> PrepareConfig:
    if args.scene_type != "matterport":
        raise PrototypeError("--scene-type must be matterport for the Z-up source configuration")
    if min(args.iterations, args.checkpoint_interval, args.object_target_triangles) <= 0:
        raise PrototypeError("iteration, checkpoint, and object triangle counts must be positive")
    return PrepareConfig(
        skeleton=args.skeleton,
        polygon_init=args.polygon_init,
        source_repo=args.source_repo,
        output=args.output,
        python=args.python,
        random_seed=args.random_seed,
        scene_type=args.scene_type,
        iterations=args.iterations,
        checkpoint_interval=args.checkpoint_interval,
        object_target_triangles=args.object_target_triangles,
        strict_source_hashes=args.strict_source_hashes,
        skip_runtime_probe=args.skip_runtime_probe,
    )


def _run_direct(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and fit the Section 4.3 prototype component.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_prepare_arguments(parser, required=False)
    parser.add_argument("--preferred-gpu", type=int, default=0)
    args = parser.parse_args(argv)
    component_dir = args.output.expanduser().resolve()
    prepared_manifest = component_dir / "prepare_manifest.json"
    complete_manifest = component_dir / "manifest.json"
    if complete_manifest.is_file() and _read_json(complete_manifest).get("status") == "complete":
        raise PrototypeError("prototype fitting is already complete")
    if not prepared_manifest.is_file():
        print(prepare_prototype(_prepare_config_from_args(args)))
    print(
        fit_prototype(
            FitConfig(
                prepared=args.output,
                source_repo=args.source_repo,
                python=args.python,
                preferred_gpu=args.preferred_gpu,
            )
        )
    )
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"prepare", "fit"}:
        return _run_direct(argv)

    parser = argparse.ArgumentParser(
        description=(
            "Prepare or fit the Section 4.3 prototype component. Omit the subcommand "
            "to run prepare followed by fit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    _add_prepare_arguments(prepare, required=True)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--prepared", type=Path, required=True)
    fit.add_argument("--source-repo", type=Path)
    fit.add_argument("--python", type=Path)
    fit.add_argument("--preferred-gpu", type=int, default=0)
    fit.add_argument("--output", type=Path, help="Accepted for CLI symmetry; must match --prepared when provided.")

    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_prototype(_prepare_config_from_args(args))
    else:
        if args.output is not None and args.output.expanduser().resolve() != args.prepared.expanduser().resolve():
            raise PrototypeError("--output must match --prepared for fit")
        path = fit_prototype(
            FitConfig(
                prepared=args.prepared,
                source_repo=args.source_repo,
                python=args.python,
                preferred_gpu=args.preferred_gpu,
            )
        )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
