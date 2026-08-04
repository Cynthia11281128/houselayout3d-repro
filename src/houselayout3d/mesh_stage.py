"""Paper-path depth-and-normal Poisson mesh export for ``04_mesh``."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .dn_splatter_stage import build_training_environment
from .stages import Stage


class MeshStageError(RuntimeError):
    """Raised when the formal mesh export or its validation fails."""


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


def _verify_dn_splatter(
    config: PipelineConfig, run_id: str
) -> tuple[Path, Path, dict[str, Any]]:
    dn_dir = config.storage.outputs / config.scene / run_id / Stage.DN_SPLATTER.value
    manifest_path = dn_dir / "manifest.json"
    if not manifest_path.is_file():
        raise MeshStageError(f"completed DN-Splatter manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise MeshStageError("03_dn_splatter manifest is not complete")
    config_path = Path(manifest["outputs"]["training_config"])
    checkpoint_path = Path(manifest["outputs"]["final_checkpoint"])
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise MeshStageError("DN-Splatter config or checkpoint is missing")
    if _sha256(config_path) != manifest["outputs"]["training_config_sha256"]:
        raise MeshStageError("DN-Splatter training config hash no longer matches")
    if _sha256(checkpoint_path) != manifest["outputs"]["final_checkpoint_sha256"]:
        raise MeshStageError("DN-Splatter checkpoint hash no longer matches")
    return config_path, checkpoint_path, manifest


def build_mesh_command(
    config: PipelineConfig, training_config: Path, export_dir: Path
) -> list[str]:
    """Build the exact official DN-Splatter Poisson export command."""

    gs_mesh = Path(sys.executable).with_name("gs-mesh")
    return [
        str(gs_mesh),
        config.mesh.exporter,
        "--load-config",
        str(training_config),
        "--output-dir",
        str(export_dir),
        "--total-points",
        str(config.mesh.total_points),
        "--normal-method",
        config.mesh.normal_method,
        "--use-masks",
        str(config.mesh.use_masks),
        "--filter-edges-from-depth-maps",
        str(config.mesh.filter_edges_from_depth_maps),
        "--poisson-depth",
        str(config.mesh.poisson_depth),
    ]


def _pointcloud_stats(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as error:
        raise MeshStageError("Open3D is unavailable in the mesh environment") from error

    pointcloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pointcloud.points)
    normals = np.asarray(pointcloud.normals)
    colors = np.asarray(pointcloud.colors)
    if len(points) < 100_000 or points.shape != normals.shape:
        raise MeshStageError("Poisson oriented point cloud is incomplete")
    if colors.shape != points.shape:
        raise MeshStageError("Poisson point cloud has no RGB vertex colors")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise MeshStageError("Poisson point cloud contains non-finite values")
    lengths = np.linalg.norm(normals, axis=1)
    valid_normals = lengths > 1e-8
    invalid_normal_count = int((~valid_normals).sum())
    invalid_normal_fraction = float((~valid_normals).mean())
    if invalid_normal_fraction > 0.01:
        raise MeshStageError(
            "more than one percent of Poisson point-cloud normals are invalid"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "point_count": int(len(points)),
        "has_normals": True,
        "has_colors": True,
        "valid_normal_count": int(valid_normals.sum()),
        "invalid_normal_count": invalid_normal_count,
        "invalid_normal_fraction": invalid_normal_fraction,
        "valid_normal_length_max_error": float(
            np.max(np.abs(lengths[valid_normals] - 1.0))
        ),
        "bounds_min_meters": points.min(axis=0).tolist(),
        "bounds_max_meters": points.max(axis=0).tolist(),
    }


def _mesh_stats(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as error:
        raise MeshStageError("Open3D is unavailable in the mesh environment") from error

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if len(vertices) < 100_000 or len(triangles) < 100_000:
        raise MeshStageError("Poisson mesh is unexpectedly small")
    if not np.isfinite(vertices).all():
        raise MeshStageError("Poisson mesh contains non-finite vertices")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise MeshStageError("Poisson mesh contains invalid triangle indices")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "has_vertex_colors": colors.shape == vertices.shape,
        "bounds_min_meters": vertices.min(axis=0).tolist(),
        "bounds_max_meters": vertices.max(axis=0).tolist(),
    }


def run_mesh(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Run and validate the official depth-and-normal Poisson exporter."""

    training_config, checkpoint_path, dn_manifest = _verify_dn_splatter(
        config, run_id
    )
    stage_dir = config.storage.outputs / config.scene / run_id / Stage.MESH.value
    if stage_dir.exists():
        raise MeshStageError(
            f"mesh stage already exists and will not be overwritten: {stage_dir}"
        )
    export_dir = stage_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=False)
    mesh_command = build_mesh_command(config, training_config, export_dir)
    gs_mesh = Path(mesh_command[0])
    if not gs_mesh.is_file() or not os.access(gs_mesh, os.X_OK):
        raise MeshStageError(f"gs-mesh is unavailable: {gs_mesh}")
    _write_json(stage_dir / "commands.json", {"export": mesh_command})
    log_path = stage_dir / "export.log"
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    _write_status(stage_dir, "exporting", str(log_path))
    environment = build_training_environment(gs_mesh)
    environment["TORCH_HOME"] = str(config.storage.weights / "torch")

    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("command: " + " ".join(mesh_command) + "\n\n")
            log.flush()
            result = subprocess.run(
                mesh_command,
                cwd=config.metric3d.repository.parent / "dn-splatter",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise MeshStageError(
                f"gs-mesh failed with exit code {result.returncode}; see {log_path}"
            )
        pointcloud_path = export_dir / "DepthAndNormalMapsPoisson_pcd.ply"
        mesh_path = export_dir / "DepthAndNormalMapsPoisson_poisson_mesh.ply"
        if not pointcloud_path.is_file() or not mesh_path.is_file():
            raise MeshStageError("gs-mesh did not produce both expected Poisson outputs")
        _write_status(stage_dir, "validating", str(mesh_path))
        pointcloud = _pointcloud_stats(pointcloud_path)
        mesh = _mesh_stats(mesh_path)
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.MESH.value,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "inputs": {
                "dn_splatter_manifest": {
                    "path": str(
                        config.storage.outputs
                        / config.scene
                        / run_id
                        / Stage.DN_SPLATTER.value
                        / "manifest.json"
                    ),
                    "sha256": _sha256(
                        config.storage.outputs
                        / config.scene
                        / run_id
                        / Stage.DN_SPLATTER.value
                        / "manifest.json"
                    ),
                },
                "training_config": str(training_config),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": dn_manifest["outputs"][
                    "final_checkpoint_sha256"
                ],
            },
            "algorithm": {
                "implementation": "DN-Splatter gs-mesh dn",
                "total_points": config.mesh.total_points,
                "normal_method": config.mesh.normal_method,
                "use_masks": config.mesh.use_masks,
                "filter_edges_from_depth_maps": (
                    config.mesh.filter_edges_from_depth_maps
                ),
                "poisson_depth": config.mesh.poisson_depth,
            },
            "outputs": {
                "oriented_pointcloud": pointcloud,
                "poisson_mesh": mesh,
                "export_log": str(log_path),
            },
            "validation": {
                "return_code": result.returncode,
                "pointcloud_present": True,
                "mesh_present": True,
                "finite_geometry": True,
            },
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "warnings": [
                "The earlier Open3D TSDF mesh remains an auxiliary visualization artifact under 03_dn_splatter; this stage is the paper-path depth-and-normal Poisson export."
            ],
        }
        manifest_path = stage_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(stage_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(stage_dir, "failed", str(error))
        if isinstance(error, MeshStageError):
            raise
        raise MeshStageError(str(error)) from error
