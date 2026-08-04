"""Preparation, execution, and auditing for the unofficial prototype fitter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .oneformer_stage import LAYOUT_LABELS
from .stages import Stage


class PrototypeStageError(RuntimeError):
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_status(path: Path, state: str, detail: str = "") -> None:
    _write_json(
        path,
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_source(config: PipelineConfig) -> dict[str, dict[str, Any]]:
    root = config.prototype.source_repository
    records: dict[str, dict[str, Any]] = {}
    for relative, expected in SOURCE_FILE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise PrototypeStageError(f"unofficial source file is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise PrototypeStageError(
                f"unofficial source hash changed for {relative}: {actual}"
            )
        records[relative] = {
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return records


def _source_config(config: PipelineConfig) -> dict[str, Any]:
    path = config.prototype.source_repository / "mesh_fitting_3D" / "polygon_fitting_config.py"
    spec = importlib.util.spec_from_file_location("multifloor3d_polygon_config", path)
    if spec is None or spec.loader is None:
        raise PrototypeStageError("cannot load the unofficial MatterportConfig")
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
    if values["iterations"] != config.prototype.iterations:
        raise PrototypeStageError("configured iterations differ from MatterportConfig")
    if values["save_interval"] != config.prototype.checkpoint_interval:
        raise PrototypeStageError("configured checkpoint interval differs from source")
    if values["up_vector"] != "Z" or not values["multi_floor"]:
        raise PrototypeStageError("MatterportConfig must remain Z-up and multi-floor")
    return values


def _verify_manifest_record(record: dict[str, Any], name: str) -> Path:
    path = Path(record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise PrototypeStageError(f"prior artifact hash mismatch: {name}")
    return path


def _verify_inputs(config: PipelineConfig, run_id: str) -> dict[str, Any]:
    run_dir = config.storage.outputs / config.scene / run_id
    skeleton_manifest_path = run_dir / Stage.SKELETON.value / "manifest.json"
    polygon_manifest_path = run_dir / Stage.POLYGON_INIT.value / "manifest.json"
    if not skeleton_manifest_path.is_file() or not polygon_manifest_path.is_file():
        raise PrototypeStageError("06_skeleton and 07_polygon_init manifests are required")
    skeleton = _read_json(skeleton_manifest_path)
    polygon = _read_json(polygon_manifest_path)
    if skeleton.get("status") != "complete" or polygon.get("status") != "complete":
        raise PrototypeStageError("prior stage is not complete")

    skeleton_outputs = skeleton["outputs"]
    structure_record = skeleton_outputs["meshes"]["structure"]
    object_record = skeleton_outputs["meshes"]["objects"]
    if not structure_record.get("classes_path"):
        raise PrototypeStageError("structure class-probability artifact is missing")
    records = {
        "target_mesh": _verify_manifest_record(structure_record, "structure mesh"),
        "target_classes": _verify_manifest_record(
            {
                "path": structure_record["classes_path"],
                "sha256": structure_record["classes_sha256"],
            },
            "structure classes",
        ),
        "object_mesh": _verify_manifest_record(object_record, "object mesh"),
        "ray_origins": _verify_manifest_record(
            skeleton_outputs["arrays"]["full_ray_origins.npy"], "ray origins"
        ),
        "ray_destinations": _verify_manifest_record(
            skeleton_outputs["arrays"]["full_ray_dests.npy"], "ray destinations"
        ),
        "ray_validity": _verify_manifest_record(
            skeleton_outputs["arrays"]["ray_is_valid.npy"], "ray validity"
        ),
        "ray_classes": _verify_manifest_record(
            skeleton_outputs["arrays"]["hard_labels_simplified_segmentations.npy"],
            "ray classes",
        ),
        "class_names": _verify_manifest_record(
            skeleton_outputs["arrays"]["simplified_segmentation_labels.npy"],
            "class names",
        ),
        "rectified_mesh": _verify_manifest_record(
            polygon["outputs"]["rectified_mesh"], "rectified mesh"
        ),
        "polygon_info": _verify_manifest_record(
            polygon["outputs"]["polygon_info"], "polygon info"
        ),
    }
    return {
        "run_dir": run_dir,
        "skeleton_manifest": skeleton_manifest_path,
        "polygon_manifest": polygon_manifest_path,
        **records,
    }


def _verify_runtime(config: PipelineConfig) -> dict[str, Any]:
    executable = config.prototype.python_executable
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PrototypeStageError(f"prototype Python is unavailable: {executable}")
    probe = (
        "import json,torch,pytorch3d,open3d,shapely,sklearn,rdp,CGAL; "
        "from CGAL.CGAL_Kernel import Point_2,Polygon_2; "
        "from CGAL.CGAL_Triangulation_2 import Constrained_triangulation_2; "
        "print(json.dumps({'torch':torch.__version__,'pytorch3d':pytorch3d.__version__,"
        "'open3d':open3d.__version__,'shapely':shapely.__version__,"
        "'sklearn':sklearn.__version__,'cgal':CGAL.__version__,"
        "'cuda':torch.cuda.is_available()}))"
    )
    result = subprocess.run(
        [str(executable), "-c", probe],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PrototypeStageError(
            "prototype dependency probe failed: " + result.stderr.strip()
        )
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    if not versions["cuda"]:
        raise PrototypeStageError("prototype runtime cannot access CUDA")
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
        raise PrototypeStageError("06_skeleton class order changed")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(names):
        raise PrototypeStageError("structure class-probability shape is invalid")
    if "door" in names:
        raise PrototypeStageError("unexpected upstream door class")
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
        "door_probability_is_zero": bool(
            np.all(prepared_probabilities[:, -1] == 0)
        ),
    }


def _simplify_object_mesh(
    input_path: Path,
    output_path: Path,
    target_triangles: int,
) -> dict[str, Any]:
    try:
        import open3d as o3d
    except ImportError as error:
        raise PrototypeStageError("Open3D is required to prepare the object mesh") from error
    mesh = o3d.io.read_triangle_mesh(str(input_path), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise PrototypeStageError("06_skeleton object mesh is empty")
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
        raise PrototypeStageError("object simplification produced an empty mesh")
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False):
        raise PrototypeStageError("failed to write simplified object mesh")
    return {
        "source_vertex_count": source_vertices,
        "source_triangle_count": source_triangles,
        "source_surface_area_square_meters": source_area,
        "target_triangle_count": target_triangles,
        "output_vertex_count": len(mesh.vertices),
        "output_triangle_count": len(mesh.triangles),
        "output_surface_area_square_meters": float(mesh.get_surface_area()),
    }


def build_prototype_command(
    config: PipelineConfig,
    prepared: dict[str, Path],
    output_dir: Path,
) -> list[str]:
    """Build the seeded command around the unmodified source entrypoint."""

    source_script = config.prototype.source_repository / "fit_prototype.py"
    return [
        str(config.prototype.python_executable),
        "-m",
        "houselayout3d.prototype_entry",
        "--source-script",
        str(source_script),
        "--random-seed",
        str(config.runtime.random_seed),
        "--scene-type",
        config.prototype.scene_type,
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


def prepare_prototype(config: PipelineConfig, run_id: str) -> Path:
    """Freeze all Stage-08 inputs and source/runtime audits before optimization."""

    inputs = _verify_inputs(config, run_id)
    source_records = _verify_source(config)
    source_config = _source_config(config)
    runtime_versions = _verify_runtime(config)
    stage_dir = inputs["run_dir"] / Stage.PROTOTYPE.value
    if stage_dir.exists():
        raise PrototypeStageError(
            f"prototype stage already exists and will not be overwritten: {stage_dir}"
        )
    prepared_dir = stage_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=False)
    _write_status(stage_dir / "STATUS.json", "preparing", "freezing Stage-08 inputs")
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    try:
        semantic = _prepare_semantic_classes(
            inputs["target_classes"],
            inputs["class_names"],
            prepared_dir / "target_vertex_classes.npy",
            prepared_dir / "target_vertex_class_names.npy",
        )
        object_stats = _simplify_object_mesh(
            inputs["object_mesh"],
            prepared_dir / "objects_mesh_simplified.ply",
            config.prototype.object_target_triangles,
        )
        frozen = {
            "rectified_mesh": inputs["rectified_mesh"],
            "target_mesh": inputs["target_mesh"],
            "target_classes": prepared_dir / "target_vertex_classes.npy",
            "class_names": prepared_dir / "target_vertex_class_names.npy",
            "polygon_info": inputs["polygon_info"],
            "ray_origins": inputs["ray_origins"],
            "ray_destinations": inputs["ray_destinations"],
            "ray_validity": inputs["ray_validity"],
            "ray_classes": inputs["ray_classes"],
            "object_mesh": prepared_dir / "objects_mesh_simplified.ply",
        }
        frozen_records = {name: _record(path) for name, path in frozen.items()}
        command_template = build_prototype_command(
            config, frozen, Path("<ATTEMPT_OUTPUT_DIR>")
        )
        _write_json(
            stage_dir / "commands.json",
            {"fit_prototype_template": command_template},
        )
        prepared_manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.PROTOTYPE.value,
            "status": "prepared",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "random_seed": config.runtime.random_seed,
            "inputs": {
                "skeleton_manifest": _record(inputs["skeleton_manifest"]),
                "polygon_manifest": _record(inputs["polygon_manifest"]),
            },
            "source": {
                "repository": str(config.prototype.source_repository),
                "files": source_records,
                "matterport_config": source_config,
                "source_files_modified": False,
            },
            "prepared_inputs": frozen_records,
            "adaptations": {
                "semantic_classes": semantic,
                "object_mesh_simplification": object_stats,
                "seeded_launcher": "Seeds Python, NumPy, torch, and all CUDA devices before runpy executes the unmodified entrypoint.",
                "all_rays_including_invalid_depth_fallbacks": True,
            },
            "environment": {
                "python_executable": str(config.prototype.python_executable),
                "versions": runtime_versions,
                "platform": platform.platform(),
            },
            "warnings": [
                "The unofficial optimizer requires a door class even though Appendix Table 5 does not retain a distinct door class during prototype fitting. A zero-probability door channel is appended only for source compatibility.",
                "The unofficial README requests objects_mesh_simplified.ply but extract_skeleton.py never creates it and discloses no simplification target. This reproduction records an explicit quadric-decimation target in YAML.",
                "The supplied source saves ray_is_valid.npy but fits with every ray, including the 0.5 m fallback destinations for invalid depths; this behavior is preserved.",
            ],
        }
        manifest_path = stage_dir / "prepared_manifest.json"
        _write_json(manifest_path, prepared_manifest)
        _write_status(stage_dir / "STATUS.json", "prepared", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(stage_dir / "STATUS.json", "failed", str(error))
        if isinstance(error, PrototypeStageError):
            raise
        raise PrototypeStageError(str(error)) from error


def _load_prepared(config: PipelineConfig, run_id: str) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    stage_dir = config.storage.outputs / config.scene / run_id / Stage.PROTOTYPE.value
    manifest_path = stage_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise PrototypeStageError("prepare-prototype must complete first")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "prepared":
        raise PrototypeStageError("prepared prototype manifest is invalid")
    if (stage_dir / "manifest.json").is_file():
        complete = _read_json(stage_dir / "manifest.json")
        if complete.get("status") == "complete":
            raise PrototypeStageError("prototype fitting is already complete")
    _verify_source(config)
    frozen: dict[str, Path] = {}
    for name, record in manifest["prepared_inputs"].items():
        frozen[name] = _verify_manifest_record(record, f"prepared {name}")
    return stage_dir, manifest, frozen


def _next_attempt(stage_dir: Path) -> Path:
    indices = []
    for path in stage_dir.glob("attempt_*_*"):
        match = re.match(r"attempt_(\d+)_", path.name)
        if match:
            indices.append(int(match.group(1)))
    index = max(indices, default=0) + 1
    return stage_dir / f"attempt_{index:03d}_running"


def _rename_attempt(attempt: Path, suffix: str) -> Path:
    destination = attempt.with_name(attempt.name.rsplit("_", 1)[0] + f"_{suffix}")
    if destination.exists():
        raise PrototypeStageError(f"attempt destination already exists: {destination}")
    attempt.rename(destination)
    return destination


def _output_artifact(path: Path) -> dict[str, Any]:
    try:
        import open3d as o3d
    except ImportError as error:
        raise PrototypeStageError("Open3D is required to validate fitted meshes") from error
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise PrototypeStageError(f"fitted mesh is empty: {path}")
    if not np.isfinite(vertices).all():
        raise PrototypeStageError(f"fitted mesh has non-finite vertices: {path}")
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


def fit_prototype(config: PipelineConfig, run_id: str) -> Path:
    """Run the full unmodified 4,000-step Matterport prototype optimizer."""

    stage_dir, prepared_manifest, frozen = _load_prepared(config, run_id)
    attempt = _next_attempt(stage_dir)
    attempt.mkdir(parents=False, exist_ok=False)
    command = build_prototype_command(config, frozen, attempt)
    _write_json(attempt / "command.json", {"command": command})
    _write_status(attempt / "STATUS.json", "running", "unofficial MatterportConfig")
    _write_status(stage_dir / "STATUS.json", "running", str(attempt))
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    environment = os.environ.copy()
    source_root = config.prototype.source_repository
    python_paths = [str(source_root), str(source_root / "mesh_fitting_3D")]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["CUDA_VISIBLE_DEVICES"] = str(config.runtime.preferred_gpu)
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONUNBUFFERED"] = "1"
    log_path = attempt / "fit_prototype.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=attempt,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        failed_attempt = _rename_attempt(attempt, "failed")
        _write_status(
            failed_attempt / "STATUS.json",
            "failed",
            f"return_code={result.returncode}; log={failed_attempt / log_path.name}",
        )
        _write_status(stage_dir / "STATUS.json", "failed", str(failed_attempt))
        raise PrototypeStageError(
            f"prototype optimizer failed with code {result.returncode}; "
            f"see {failed_attempt / log_path.name}"
        )

    completed_attempt = _rename_attempt(attempt, "complete")
    final_mesh_path = completed_attempt / "fitted_mesh.ply"
    checkpoint_path = completed_attempt / "polygon_set_3d.pt"
    if not final_mesh_path.is_file() or not checkpoint_path.is_file():
        raise PrototypeStageError("optimizer returned zero but final outputs are missing")
    final_mesh = _output_artifact(final_mesh_path)
    initial_mesh = _output_artifact(completed_attempt / "fitted_mesh_00.ply")
    checkpoint_record = _record(checkpoint_path)
    snapshots: list[dict[str, Any]] = []
    for path in sorted(completed_attempt.glob("fitted_mesh_[0-9]*.ply")):
        if path.name == "fitted_mesh_00.ply":
            continue
        match = re.fullmatch(r"fitted_mesh_(\d+)\.ply", path.name)
        if match:
            record = _record(path)
            record["step"] = int(match.group(1))
            snapshots.append(record)
    state_snapshots = [
        {"step": int(match.group(1)), **_record(path)}
        for path in sorted(completed_attempt.glob("polygon_set_3d_[0-9]*.pt"))
        if (match := re.fullmatch(r"polygon_set_3d_(\d+)\.pt", path.name))
    ]
    snapshots.sort(key=lambda record: record["step"])
    state_snapshots.sort(key=lambda record: record["step"])
    expected_steps = list(
        range(0, config.prototype.iterations, config.prototype.checkpoint_interval)
    )
    actual_steps = [record["step"] for record in snapshots]
    if actual_steps != expected_steps:
        raise PrototypeStageError(
            f"mesh checkpoint steps differ: expected {expected_steps}, got {actual_steps}"
        )
    if [record["step"] for record in state_snapshots] != expected_steps:
        raise PrototypeStageError("model-state checkpoint steps are incomplete")

    manifest = {
        "schema_version": 1,
        "scene": config.scene,
        "run_id": run_id,
        "stage": Stage.PROTOTYPE.value,
        "status": "complete",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 6),
        "random_seed": config.runtime.random_seed,
        "command": command,
        "return_code": result.returncode,
        "inputs": {
            "prepared_manifest": _record(stage_dir / "prepared_manifest.json"),
            **prepared_manifest["prepared_inputs"],
        },
        "source": prepared_manifest["source"],
        "algorithm": {
            "paper_objectives": ["Lgeo=Lprox+Lempty", "Lconnect", "Lsimple"],
            "source_config": prepared_manifest["source"]["matterport_config"],
            "iterations": config.prototype.iterations,
            "checkpoint_interval": config.prototype.checkpoint_interval,
            "semantic_preparation": prepared_manifest["adaptations"]["semantic_classes"],
            "object_mesh_simplification": prepared_manifest["adaptations"]["object_mesh_simplification"],
        },
        "outputs": {
            "attempt_dir": str(completed_attempt),
            "log": _record(completed_attempt / "fit_prototype.log"),
            "initial_mesh": initial_mesh,
            "final_mesh": final_mesh,
            "final_model_state": checkpoint_record,
            "mesh_checkpoints": snapshots,
            "model_state_checkpoints": state_snapshots,
        },
        "validation": {
            "source_files_byte_identical": True,
            "full_4000_iterations_requested": config.prototype.iterations == 4000,
            "all_expected_mesh_checkpoints_present": actual_steps == expected_steps,
            "all_expected_state_checkpoints_present": [
                record["step"] for record in state_snapshots
            ]
            == expected_steps,
            "final_mesh_nonempty": final_mesh["vertex_count"] > 0
            and final_mesh["triangle_count"] > 0,
            "final_mesh_finite": True,
            "no_ground_truth_inputs_used": True,
        },
        "environment": {
            **prepared_manifest["environment"],
            "cuda_visible_devices": str(config.runtime.preferred_gpu),
        },
        "warnings": prepared_manifest["warnings"],
    }
    if not all(manifest["validation"].values()):
        raise PrototypeStageError("prototype validation failed")
    manifest_path = stage_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_status(completed_attempt / "STATUS.json", "complete", str(final_mesh_path))
    _write_status(stage_dir / "STATUS.json", "complete", str(manifest_path))
    return manifest_path
