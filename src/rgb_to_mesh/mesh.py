"""Export a Poisson mesh from a completed DN-Splatter output folder."""

from __future__ import annotations

import argparse
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


DN_SPLATTER_REPOSITORY = Path("external/dn-splatter")
LPIPS_ALEXNET_CHECKPOINT = Path("pretrained_weights/alexnet-owt-7be5be79.pth")
EXPORTER = "dn"
TOTAL_POINTS = 2_000_000
NORMAL_METHOD = "normal_maps"
USE_MASKS = True
FILTER_EDGES_FROM_DEPTH_MAPS = False
POISSON_DEPTH = 9


class MeshError(RuntimeError):
    """Raised when the mesh export or validation fails."""


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


def _write_status(output_dir: Path, state: str, detail: str = "") -> None:
    _write_json(
        output_dir / "STATUS.json",
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def build_training_environment(
    executable: Path, base_environment: dict[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    environment_bin = executable.parent.resolve()
    environment_root = environment_bin.parent
    existing_path = environment.get("PATH", "")
    path_entries = existing_path.split(os.pathsep) if existing_path else []
    if str(environment_bin) not in path_entries:
        environment["PATH"] = os.pathsep.join([str(environment_bin), *path_entries])
    nvcc = environment_bin / "nvcc"
    if nvcc.is_file():
        environment.setdefault("CUDA_HOME", str(environment_root))
    environment.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _torch_home_for_checkpoint(path: Path) -> Path:
    if path.parent.name == "checkpoints" and path.parent.parent.name == "hub":
        return path.parent.parent.parent
    return path.parent


def _verify_dn_splatter(dn_splatter_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = dn_splatter_dir / "manifest.json"
    if not manifest_path.is_file():
        raise MeshError(f"completed DN-Splatter manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise MeshError("dn_splatter manifest is not complete")
    config_path = Path(manifest["outputs"]["training_config"])
    checkpoint_path = Path(manifest["outputs"]["final_checkpoint"])
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise MeshError("DN-Splatter config or checkpoint is missing")
    if _sha256(config_path) != manifest["outputs"]["training_config_sha256"]:
        raise MeshError("DN-Splatter training config hash no longer matches")
    if _sha256(checkpoint_path) != manifest["outputs"]["final_checkpoint_sha256"]:
        raise MeshError("DN-Splatter checkpoint hash no longer matches")
    return config_path, checkpoint_path, manifest


def build_mesh_command(training_config: Path, export_dir: Path) -> list[str]:
    gs_mesh = Path(sys.executable).with_name("gs-mesh")
    return [
        str(gs_mesh),
        EXPORTER,
        "--load-config",
        str(training_config),
        "--output-dir",
        str(export_dir),
        "--total-points",
        str(TOTAL_POINTS),
        "--normal-method",
        NORMAL_METHOD,
        "--use-masks",
        str(USE_MASKS),
        "--filter-edges-from-depth-maps",
        str(FILTER_EDGES_FROM_DEPTH_MAPS),
        "--poisson-depth",
        str(POISSON_DEPTH),
    ]


def _pointcloud_stats(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as error:
        raise MeshError("Open3D is unavailable in the mesh environment") from error

    pointcloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pointcloud.points)
    normals = np.asarray(pointcloud.normals)
    colors = np.asarray(pointcloud.colors)
    if len(points) < 100_000 or points.shape != normals.shape:
        raise MeshError("Poisson oriented point cloud is incomplete")
    if colors.shape != points.shape:
        raise MeshError("Poisson point cloud has no RGB vertex colors")
    if not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise MeshError("Poisson point cloud contains non-finite values")
    lengths = np.linalg.norm(normals, axis=1)
    valid_normals = lengths > 1e-8
    invalid_fraction = float((~valid_normals).mean())
    if invalid_fraction > 0.01:
        raise MeshError("more than one percent of point-cloud normals are invalid")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "point_count": int(len(points)),
        "has_normals": True,
        "has_colors": True,
        "invalid_normal_fraction": invalid_fraction,
        "bounds_min_meters": points.min(axis=0).tolist(),
        "bounds_max_meters": points.max(axis=0).tolist(),
    }


def _mesh_stats(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as error:
        raise MeshError("Open3D is unavailable in the mesh environment") from error

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if len(vertices) < 100_000 or len(triangles) < 100_000:
        raise MeshError("Poisson mesh is unexpectedly small")
    if not np.isfinite(vertices).all():
        raise MeshError("Poisson mesh contains non-finite vertices")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise MeshError("Poisson mesh contains invalid triangle indices")
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
    dn_splatter_dir: Path,
    output_dir: Path,
    command: list[str] | None = None,
) -> Path:
    dn_splatter_dir = dn_splatter_dir.expanduser().resolve()
    output_dir = output_dir.expanduser()
    if output_dir.exists():
        raise MeshError(f"mesh output already exists and will not be overwritten: {output_dir}")
    training_config, checkpoint_path, dn_manifest = _verify_dn_splatter(dn_splatter_dir)
    if not LPIPS_ALEXNET_CHECKPOINT.is_file():
        raise MeshError(f"LPIPS checkpoint is missing: {LPIPS_ALEXNET_CHECKPOINT}")
    if not DN_SPLATTER_REPOSITORY.is_dir():
        raise MeshError(f"DN-Splatter repository is missing: {DN_SPLATTER_REPOSITORY}")

    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=False)
    mesh_command = build_mesh_command(training_config, export_dir)
    gs_mesh = Path(mesh_command[0])
    if not gs_mesh.is_file() or not os.access(gs_mesh, os.X_OK):
        raise MeshError(f"gs-mesh is unavailable: {gs_mesh}")
    _write_json(output_dir / "commands.json", {"export": mesh_command})
    log_path = output_dir / "export.log"
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    _write_status(output_dir, "exporting", str(log_path))
    environment = build_training_environment(gs_mesh)
    environment["TORCH_HOME"] = str(_torch_home_for_checkpoint(LPIPS_ALEXNET_CHECKPOINT))

    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("command: " + " ".join(mesh_command) + "\n\n")
            log.flush()
            result = subprocess.run(
                mesh_command,
                cwd=DN_SPLATTER_REPOSITORY,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise MeshError(
                f"gs-mesh failed with exit code {result.returncode}; see {log_path}"
            )
        pointcloud_path = export_dir / "DepthAndNormalMapsPoisson_pcd.ply"
        mesh_path = export_dir / "DepthAndNormalMapsPoisson_poisson_mesh.ply"
        if not pointcloud_path.is_file() or not mesh_path.is_file():
            raise MeshError("gs-mesh did not produce both expected Poisson outputs")
        _write_status(output_dir, "validating", str(mesh_path))
        pointcloud = _pointcloud_stats(pointcloud_path)
        mesh = _mesh_stats(mesh_path)
        manifest = {
            "schema_version": 1,
            "component": "mesh",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "inputs": {
                "dn_splatter": str(dn_splatter_dir),
                "dn_splatter_manifest_sha256": _sha256(dn_splatter_dir / "manifest.json"),
                "training_config": str(training_config),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": dn_manifest["outputs"]["final_checkpoint_sha256"],
            },
            "algorithm": {
                "implementation": "DN-Splatter gs-mesh dn",
                "total_points": TOTAL_POINTS,
                "normal_method": NORMAL_METHOD,
                "use_masks": USE_MASKS,
                "filter_edges_from_depth_maps": FILTER_EDGES_FROM_DEPTH_MAPS,
                "poisson_depth": POISSON_DEPTH,
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
        }
        manifest_path = output_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(output_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(output_dir, "failed", str(error))
        if isinstance(error, MeshError):
            raise
        raise MeshError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a Poisson mesh from DN-Splatter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dn-splatter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = run_mesh(
        dn_splatter_dir=args.dn_splatter,
        output_dir=args.output,
        command=sys.argv,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
