"""Strict configuration contract shared by all future pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a pipeline configuration violates the public contract."""


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
    poses_csv: Path | None
    pose_convention: str | None
    image_glob: str
    filename_regex: str | None
    require_strict_timestamp_order: bool
    camera: CameraConfig


@dataclass(frozen=True)
class StorageConfig:
    data: Path
    weights: Path
    outputs: Path
    cache: Path


@dataclass(frozen=True)
class Metric3DConfig:
    repository: Path
    weights: Path
    model: str
    input_height: int
    input_width: int
    canonical_focal_length: float
    minimum_depth_meters: float
    maximum_depth_meters: float


@dataclass(frozen=True)
class DNSplatterConfig:
    method: str
    max_num_iterations: int
    steps_per_save: int
    depth_loss_type: str
    depth_lambda: float
    normal_supervision: str
    seed_stride: int
    seed_minimum_depth_meters: float
    seed_maximum_depth_meters: float
    seed_voxel_size_meters: float
    maximum_seed_points: int


@dataclass(frozen=True)
class MeshConfig:
    exporter: str
    total_points: int
    normal_method: str
    use_masks: bool
    filter_edges_from_depth_maps: bool
    poisson_depth: int


@dataclass(frozen=True)
class OneFormerConfig:
    model_dir: Path
    task: str
    preview_count: int


@dataclass(frozen=True)
class SkeletonConfig:
    render_executable: Path
    superpoint_repository: Path
    samples_per_frame: int
    knn_neighbors: int
    knn_radius_meters: float
    adjacency_neighbors: int
    adjacency_weight: float
    regularization: tuple[float, ...]
    spatial_weight: tuple[float, ...]
    cutoff: tuple[int, ...]
    iterations: int
    final_level: int


@dataclass(frozen=True)
class PolygonInitConfig:
    superpoint_level: int
    minimum_unassigned_vertices: int
    plane_distance_threshold_meters: float
    ransac_iterations: int
    rdp_epsilon_meters: float


@dataclass(frozen=True)
class PrototypeConfig:
    python_executable: Path
    source_repository: Path
    scene_type: str
    iterations: int
    checkpoint_interval: int
    object_target_triangles: int


@dataclass(frozen=True)
class SceneGraphConfig:
    openseg_python: Path
    openseg_model: Path
    clip_weights: Path
    grid_resolution_meters: float
    floor_merge_height_meters: float
    ceiling_minimum_clearance_meters: float
    wall_interval_height_meters: float
    wall_line_width_meters: float
    bottleneck_widths_meters: tuple[float, float]
    door_maximum_width_meters: float
    minimum_seed_area_square_meters: float
    minimum_room_area_square_meters: float
    visibility_tolerance_meters: float
    image_boundary_pixels: int
    frame_stride: int
    stair_room_maximum_distance_meters: float
    stair_minimum_triangles: int


@dataclass(frozen=True)
class LayoutConfig:
    door_height_meters: float
    maximum_ceilings_per_room: int
    window_minimum_cluster_points: int
    window_minimum_size_meters: float
    window_dbscan_epsilon_meters: float
    window_dbscan_minimum_samples: int
    window_outlier_neighbors: int
    window_voxel_size_meters: float
    window_frame_stride: int
    window_pixel_stride: int
    window_maximum_ray_distance_meters: float
    stair_step_height_meters: float


@dataclass(frozen=True)
class RuntimeConfig:
    random_seed: int
    preferred_gpu: int
    colmap_executable: Path
    colmap_matcher: str
    colmap_sequential_overlap: int
    minimum_registered_image_ratio: float
    canonical_up_axis: str


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    scene: str
    input: InputConfig
    storage: StorageConfig
    metric3d: Metric3DConfig
    dn_splatter: DNSplatterConfig
    mesh: MeshConfig
    oneformer: OneFormerConfig
    skeleton: SkeletonConfig
    polygon_init: PolygonInitConfig
    prototype: PrototypeConfig
    scene_graph: SceneGraphConfig
    layout: LayoutConfig
    runtime: RuntimeConfig


_FORBIDDEN_INFERENCE_KEYS = {
    "gt",
    "gt_mesh",
    "ground_truth",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty path string")
    return Path(value).expanduser()


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a version-1 pipeline YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "config")

    if raw.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    input_raw = _mapping(raw.get("input"), "input")
    forbidden = sorted(_FORBIDDEN_INFERENCE_KEYS.intersection(input_raw))
    if forbidden:
        raise ConfigError(
            "inference input must not contain ground-truth keys: "
            + ", ".join(forbidden)
        )

    poses_csv_raw = input_raw.get("poses_csv")
    poses_csv = (
        None if poses_csv_raw is None else _path(poses_csv_raw, "input.poses_csv")
    )
    pose_convention_raw = input_raw.get("pose_convention")
    pose_convention = (
        None if pose_convention_raw is None else str(pose_convention_raw)
    )
    if (poses_csv is None) != (pose_convention is None):
        raise ConfigError(
            "input.poses_csv and input.pose_convention must be configured together"
        )
    if pose_convention not in {None, "opencv_c2w_xyzw_meters"}:
        raise ConfigError(
            "input.pose_convention must be opencv_c2w_xyzw_meters"
        )

    camera_raw = _mapping(input_raw.get("camera"), "input.camera")
    filename_regex_raw = input_raw.get("filename_regex")
    filename_regex = (
        None if filename_regex_raw is None else str(filename_regex_raw)
    )
    if filename_regex is not None:
        try:
            re.compile(filename_regex)
        except re.error as error:
            raise ConfigError(f"input.filename_regex is invalid: {error}") from error
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
        raise ConfigError("milestone-1 camera contract requires PINHOLE")
    if min(camera.width, camera.height) <= 0 or min(camera.fx, camera.fy) <= 0:
        raise ConfigError("camera dimensions and focal lengths must be positive")

    storage_raw = _mapping(raw.get("storage"), "storage")
    metric3d_raw = _mapping(raw.get("metric3d"), "metric3d")
    metric3d_input_height = int(metric3d_raw.get("input_height", 0))
    metric3d_input_width = int(metric3d_raw.get("input_width", 0))
    metric3d_canonical_focal = float(
        metric3d_raw.get("canonical_focal_length", 0)
    )
    metric3d_minimum_depth = float(metric3d_raw.get("minimum_depth_meters", 0))
    metric3d_maximum_depth = float(metric3d_raw.get("maximum_depth_meters", 0))
    if min(metric3d_input_height, metric3d_input_width) <= 0:
        raise ConfigError("metric3d input dimensions must be positive")
    if metric3d_canonical_focal <= 0:
        raise ConfigError("metric3d.canonical_focal_length must be positive")
    if not 0 <= metric3d_minimum_depth < metric3d_maximum_depth:
        raise ConfigError(
            "metric3d depth bounds must satisfy 0 <= minimum < maximum"
        )
    metric3d_model = str(metric3d_raw.get("model", ""))
    if metric3d_model != "metric3d_vit_large":
        raise ConfigError("metric3d.model must be metric3d_vit_large")
    dn_splatter_raw = _mapping(raw.get("dn_splatter"), "dn_splatter")
    dn_method = str(dn_splatter_raw.get("method", ""))
    if dn_method != "dn-splatter":
        raise ConfigError("dn_splatter.method must be dn-splatter")
    dn_iterations = int(dn_splatter_raw.get("max_num_iterations", 0))
    dn_steps_per_save = int(dn_splatter_raw.get("steps_per_save", 0))
    dn_depth_lambda = float(dn_splatter_raw.get("depth_lambda", 0))
    dn_seed_stride = int(dn_splatter_raw.get("seed_stride", 0))
    dn_seed_minimum = float(
        dn_splatter_raw.get("seed_minimum_depth_meters", 0)
    )
    dn_seed_maximum = float(
        dn_splatter_raw.get("seed_maximum_depth_meters", 0)
    )
    dn_seed_voxel = float(dn_splatter_raw.get("seed_voxel_size_meters", 0))
    dn_maximum_seed_points = int(dn_splatter_raw.get("maximum_seed_points", 0))
    if min(dn_iterations, dn_steps_per_save, dn_seed_stride, dn_maximum_seed_points) <= 0:
        raise ConfigError("DN-Splatter iteration, save, stride, and point counts must be positive")
    if dn_depth_lambda <= 0:
        raise ConfigError("dn_splatter.depth_lambda must be positive")
    if not 0 < dn_seed_minimum < dn_seed_maximum:
        raise ConfigError("DN-Splatter seed depth bounds are invalid")
    if dn_seed_voxel <= 0:
        raise ConfigError("dn_splatter.seed_voxel_size_meters must be positive")
    dn_depth_loss_type = str(dn_splatter_raw.get("depth_loss_type", ""))
    if dn_depth_loss_type != "EdgeAwareLogL1":
        raise ConfigError("dn_splatter.depth_loss_type must be EdgeAwareLogL1")
    dn_normal_supervision = str(dn_splatter_raw.get("normal_supervision", ""))
    if dn_normal_supervision != "depth":
        raise ConfigError("dn_splatter.normal_supervision must be depth")
    mesh_raw = _mapping(raw.get("mesh"), "mesh")
    mesh_exporter = str(mesh_raw.get("exporter", ""))
    mesh_total_points = int(mesh_raw.get("total_points", 0))
    mesh_normal_method = str(mesh_raw.get("normal_method", ""))
    mesh_poisson_depth = int(mesh_raw.get("poisson_depth", 0))
    if mesh_exporter != "dn":
        raise ConfigError("mesh.exporter must be dn")
    if mesh_normal_method != "normal_maps":
        raise ConfigError("mesh.normal_method must be normal_maps")
    if mesh_total_points <= 0 or mesh_poisson_depth <= 0:
        raise ConfigError("mesh point count and Poisson depth must be positive")
    oneformer_raw = _mapping(raw.get("oneformer"), "oneformer")
    oneformer_task = str(oneformer_raw.get("task", ""))
    oneformer_preview_count = int(oneformer_raw.get("preview_count", 0))
    if oneformer_task != "semantic":
        raise ConfigError("oneformer.task must be semantic")
    if oneformer_preview_count < 0:
        raise ConfigError("oneformer.preview_count must be non-negative")
    skeleton_raw = _mapping(raw.get("skeleton"), "skeleton")
    samples_per_frame = int(skeleton_raw.get("samples_per_frame", 0))
    knn_neighbors = int(skeleton_raw.get("knn_neighbors", 0))
    knn_radius = float(skeleton_raw.get("knn_radius_meters", 0))
    adjacency_neighbors = int(skeleton_raw.get("adjacency_neighbors", 0))
    adjacency_weight = float(skeleton_raw.get("adjacency_weight", 0))
    iterations = int(skeleton_raw.get("iterations", 0))
    final_level = int(skeleton_raw.get("final_level", 0))
    regularization = tuple(
        float(value) for value in skeleton_raw.get("regularization", [])
    )
    spatial_weight = tuple(
        float(value) for value in skeleton_raw.get("spatial_weight", [])
    )
    cutoff = tuple(int(value) for value in skeleton_raw.get("cutoff", []))
    if samples_per_frame != 5000:
        raise ConfigError("skeleton.samples_per_frame must be the paper value 5000")
    if min(knn_neighbors, adjacency_neighbors, iterations, final_level) <= 0:
        raise ConfigError("skeleton neighbor, iteration, and level values must be positive")
    if knn_radius <= 0 or adjacency_weight <= 0:
        raise ConfigError("skeleton KNN radius and adjacency weight must be positive")
    if not regularization or not (
        len(regularization) == len(spatial_weight) == len(cutoff)
    ):
        raise ConfigError("skeleton superpoint hierarchy lists must have equal length")
    if final_level != len(regularization):
        raise ConfigError("skeleton.final_level must select the final configured hierarchy")
    if min(regularization) <= 0 or min(spatial_weight) <= 0 or min(cutoff) <= 0:
        raise ConfigError("skeleton superpoint hierarchy parameters must be positive")
    polygon_init_raw = _mapping(raw.get("polygon_init"), "polygon_init")
    polygon_superpoint_level = int(polygon_init_raw.get("superpoint_level", 0))
    minimum_unassigned_vertices = int(
        polygon_init_raw.get("minimum_unassigned_vertices", 0)
    )
    plane_distance_threshold = float(
        polygon_init_raw.get("plane_distance_threshold_meters", 0)
    )
    polygon_ransac_iterations = int(polygon_init_raw.get("ransac_iterations", 0))
    rdp_epsilon = float(polygon_init_raw.get("rdp_epsilon_meters", 0))
    if polygon_superpoint_level != final_level:
        raise ConfigError(
            "polygon_init.superpoint_level must select skeleton.final_level"
        )
    if min(minimum_unassigned_vertices, polygon_ransac_iterations) <= 0:
        raise ConfigError("polygon-init K and RANSAC iteration count must be positive")
    if plane_distance_threshold <= 0 or rdp_epsilon <= 0:
        raise ConfigError("polygon-init distance thresholds must be positive")
    prototype_raw = _mapping(raw.get("prototype"), "prototype")
    prototype_scene_type = str(prototype_raw.get("scene_type", ""))
    prototype_iterations = int(prototype_raw.get("iterations", 0))
    prototype_checkpoint_interval = int(
        prototype_raw.get("checkpoint_interval", 0)
    )
    object_target_triangles = int(
        prototype_raw.get("object_target_triangles", 0)
    )
    if prototype_scene_type != "matterport":
        raise ConfigError("prototype.scene_type must be matterport for the Z-up run")
    if prototype_iterations != 4000:
        raise ConfigError("prototype.iterations must match MatterportConfig value 4000")
    if prototype_checkpoint_interval != 100:
        raise ConfigError(
            "prototype.checkpoint_interval must match source value 100"
        )
    if object_target_triangles <= 0:
        raise ConfigError("prototype.object_target_triangles must be positive")
    scene_graph_raw = _mapping(raw.get("scene_graph"), "scene_graph")
    grid_resolution = float(scene_graph_raw.get("grid_resolution_meters", 0))
    floor_merge_height = float(
        scene_graph_raw.get("floor_merge_height_meters", 0)
    )
    ceiling_clearance = float(
        scene_graph_raw.get("ceiling_minimum_clearance_meters", 0)
    )
    wall_interval_height = float(
        scene_graph_raw.get("wall_interval_height_meters", 0)
    )
    wall_line_width = float(scene_graph_raw.get("wall_line_width_meters", 0))
    bottleneck_widths = tuple(
        float(value)
        for value in scene_graph_raw.get("bottleneck_widths_meters", ())
    )
    door_maximum_width = float(
        scene_graph_raw.get("door_maximum_width_meters", 0)
    )
    minimum_seed_area = float(
        scene_graph_raw.get("minimum_seed_area_square_meters", 0)
    )
    minimum_room_area = float(
        scene_graph_raw.get("minimum_room_area_square_meters", 0)
    )
    visibility_tolerance = float(
        scene_graph_raw.get("visibility_tolerance_meters", 0)
    )
    image_boundary_pixels = int(scene_graph_raw.get("image_boundary_pixels", 0))
    frame_stride = int(scene_graph_raw.get("frame_stride", 0))
    stair_room_distance = float(
        scene_graph_raw.get("stair_room_maximum_distance_meters", 0)
    )
    stair_minimum_triangles = int(
        scene_graph_raw.get("stair_minimum_triangles", 0)
    )
    paper_constants = {
        "floor_merge_height_meters": (floor_merge_height, 0.5),
        "ceiling_minimum_clearance_meters": (ceiling_clearance, 1.0),
        "wall_interval_height_meters": (wall_interval_height, 2.5),
        "door_maximum_width_meters": (door_maximum_width, 1.5),
        "stair_room_maximum_distance_meters": (stair_room_distance, 0.5),
    }
    for name, (actual, expected) in paper_constants.items():
        if actual != expected:
            raise ConfigError(f"scene_graph.{name} must match Appendix D value {expected}")
    if bottleneck_widths != (2.5, 1.5):
        raise ConfigError(
            "scene_graph.bottleneck_widths_meters must match Appendix D values [2.5, 1.5]"
        )
    if min(
        grid_resolution,
        wall_line_width,
        minimum_seed_area,
        minimum_room_area,
        visibility_tolerance,
    ) <= 0:
        raise ConfigError("scene-graph measured geometry defaults must be positive")
    if min(image_boundary_pixels, frame_stride, stair_minimum_triangles) <= 0:
        raise ConfigError("scene-graph integer defaults must be positive")
    layout_raw = _mapping(raw.get("layout"), "layout")
    door_height = float(layout_raw.get("door_height_meters", 0))
    maximum_ceilings = int(layout_raw.get("maximum_ceilings_per_room", 0))
    window_minimum_points = int(
        layout_raw.get("window_minimum_cluster_points", 0)
    )
    window_minimum_size = float(
        layout_raw.get("window_minimum_size_meters", 0)
    )
    window_dbscan_epsilon = float(
        layout_raw.get("window_dbscan_epsilon_meters", 0)
    )
    window_dbscan_minimum_samples = int(
        layout_raw.get("window_dbscan_minimum_samples", 0)
    )
    window_outlier_neighbors = int(
        layout_raw.get("window_outlier_neighbors", 0)
    )
    window_voxel_size = float(layout_raw.get("window_voxel_size_meters", 0))
    window_frame_stride = int(layout_raw.get("window_frame_stride", 0))
    window_pixel_stride = int(layout_raw.get("window_pixel_stride", 0))
    window_maximum_ray_distance = float(
        layout_raw.get("window_maximum_ray_distance_meters", 0)
    )
    stair_step_height = float(layout_raw.get("stair_step_height_meters", 0))
    if door_height != 2.1:
        raise ConfigError("layout.door_height_meters must match Appendix D.6 value 2.1")
    if maximum_ceilings != 30:
        raise ConfigError(
            "layout.maximum_ceilings_per_room must match Sec. 4.4 value 30"
        )
    if window_minimum_points != 10:
        raise ConfigError(
            "layout.window_minimum_cluster_points must match Sec. 4.4 value 10"
        )
    if window_minimum_size != 0.3:
        raise ConfigError(
            "layout.window_minimum_size_meters must match Sec. 4.4 value 0.3"
        )
    if min(
        window_dbscan_epsilon,
        window_voxel_size,
        window_maximum_ray_distance,
        stair_step_height,
    ) <= 0:
        raise ConfigError("layout measured geometry defaults must be positive")
    if min(
        window_dbscan_minimum_samples,
        window_outlier_neighbors,
        window_frame_stride,
        window_pixel_stride,
    ) <= 0:
        raise ConfigError("layout integer defaults must be positive")
    runtime_raw = _mapping(raw.get("runtime"), "runtime")
    registration_ratio = float(
        runtime_raw.get("minimum_registered_image_ratio", 0)
    )
    if not 0 < registration_ratio <= 1:
        raise ConfigError("minimum_registered_image_ratio must be in (0, 1]")
    sequential_overlap = int(runtime_raw.get("colmap_sequential_overlap", 10))
    if sequential_overlap <= 0:
        raise ConfigError("colmap_sequential_overlap must be positive")

    return PipelineConfig(
        schema_version=1,
        scene=str(raw.get("scene", "")),
        input=InputConfig(
            images=_path(input_raw.get("images"), "input.images"),
            poses_csv=poses_csv,
            pose_convention=pose_convention,
            image_glob=str(input_raw.get("image_glob", "*.png")),
            filename_regex=filename_regex,
            require_strict_timestamp_order=bool(
                input_raw.get("require_strict_timestamp_order", False)
            ),
            camera=camera,
        ),
        storage=StorageConfig(
            data=_path(storage_raw.get("data"), "storage.data"),
            weights=_path(storage_raw.get("weights"), "storage.weights"),
            outputs=_path(storage_raw.get("outputs"), "storage.outputs"),
            cache=_path(storage_raw.get("cache"), "storage.cache"),
        ),
        metric3d=Metric3DConfig(
            repository=_path(metric3d_raw.get("repository"), "metric3d.repository"),
            weights=_path(metric3d_raw.get("weights"), "metric3d.weights"),
            model=metric3d_model,
            input_height=metric3d_input_height,
            input_width=metric3d_input_width,
            canonical_focal_length=metric3d_canonical_focal,
            minimum_depth_meters=metric3d_minimum_depth,
            maximum_depth_meters=metric3d_maximum_depth,
        ),
        dn_splatter=DNSplatterConfig(
            method=dn_method,
            max_num_iterations=dn_iterations,
            steps_per_save=dn_steps_per_save,
            depth_loss_type=dn_depth_loss_type,
            depth_lambda=dn_depth_lambda,
            normal_supervision=dn_normal_supervision,
            seed_stride=dn_seed_stride,
            seed_minimum_depth_meters=dn_seed_minimum,
            seed_maximum_depth_meters=dn_seed_maximum,
            seed_voxel_size_meters=dn_seed_voxel,
            maximum_seed_points=dn_maximum_seed_points,
        ),
        mesh=MeshConfig(
            exporter=mesh_exporter,
            total_points=mesh_total_points,
            normal_method=mesh_normal_method,
            use_masks=bool(mesh_raw.get("use_masks", True)),
            filter_edges_from_depth_maps=bool(
                mesh_raw.get("filter_edges_from_depth_maps", False)
            ),
            poisson_depth=mesh_poisson_depth,
        ),
        oneformer=OneFormerConfig(
            model_dir=_path(oneformer_raw.get("model_dir"), "oneformer.model_dir"),
            task=oneformer_task,
            preview_count=oneformer_preview_count,
        ),
        skeleton=SkeletonConfig(
            render_executable=_path(
                skeleton_raw.get("render_executable"),
                "skeleton.render_executable",
            ),
            superpoint_repository=_path(
                skeleton_raw.get("superpoint_repository"),
                "skeleton.superpoint_repository",
            ),
            samples_per_frame=samples_per_frame,
            knn_neighbors=knn_neighbors,
            knn_radius_meters=knn_radius,
            adjacency_neighbors=adjacency_neighbors,
            adjacency_weight=adjacency_weight,
            regularization=regularization,
            spatial_weight=spatial_weight,
            cutoff=cutoff,
            iterations=iterations,
            final_level=final_level,
        ),
        polygon_init=PolygonInitConfig(
            superpoint_level=polygon_superpoint_level,
            minimum_unassigned_vertices=minimum_unassigned_vertices,
            plane_distance_threshold_meters=plane_distance_threshold,
            ransac_iterations=polygon_ransac_iterations,
            rdp_epsilon_meters=rdp_epsilon,
        ),
        prototype=PrototypeConfig(
            python_executable=_path(
                prototype_raw.get("python_executable"),
                "prototype.python_executable",
            ),
            source_repository=_path(
                prototype_raw.get("source_repository"),
                "prototype.source_repository",
            ),
            scene_type=prototype_scene_type,
            iterations=prototype_iterations,
            checkpoint_interval=prototype_checkpoint_interval,
            object_target_triangles=object_target_triangles,
        ),
        scene_graph=SceneGraphConfig(
            openseg_python=_path(
                scene_graph_raw.get("openseg_python"),
                "scene_graph.openseg_python",
            ),
            openseg_model=_path(
                scene_graph_raw.get("openseg_model"),
                "scene_graph.openseg_model",
            ),
            clip_weights=_path(
                scene_graph_raw.get("clip_weights"),
                "scene_graph.clip_weights",
            ),
            grid_resolution_meters=grid_resolution,
            floor_merge_height_meters=floor_merge_height,
            ceiling_minimum_clearance_meters=ceiling_clearance,
            wall_interval_height_meters=wall_interval_height,
            wall_line_width_meters=wall_line_width,
            bottleneck_widths_meters=bottleneck_widths,
            door_maximum_width_meters=door_maximum_width,
            minimum_seed_area_square_meters=minimum_seed_area,
            minimum_room_area_square_meters=minimum_room_area,
            visibility_tolerance_meters=visibility_tolerance,
            image_boundary_pixels=image_boundary_pixels,
            frame_stride=frame_stride,
            stair_room_maximum_distance_meters=stair_room_distance,
            stair_minimum_triangles=stair_minimum_triangles,
        ),
        layout=LayoutConfig(
            door_height_meters=door_height,
            maximum_ceilings_per_room=maximum_ceilings,
            window_minimum_cluster_points=window_minimum_points,
            window_minimum_size_meters=window_minimum_size,
            window_dbscan_epsilon_meters=window_dbscan_epsilon,
            window_dbscan_minimum_samples=window_dbscan_minimum_samples,
            window_outlier_neighbors=window_outlier_neighbors,
            window_voxel_size_meters=window_voxel_size,
            window_frame_stride=window_frame_stride,
            window_pixel_stride=window_pixel_stride,
            window_maximum_ray_distance_meters=window_maximum_ray_distance,
            stair_step_height_meters=stair_step_height,
        ),
        runtime=RuntimeConfig(
            random_seed=int(runtime_raw.get("random_seed", 0)),
            preferred_gpu=int(runtime_raw.get("preferred_gpu", 0)),
            colmap_executable=_path(
                runtime_raw.get("colmap_executable"),
                "runtime.colmap_executable",
            ),
            colmap_matcher=str(runtime_raw.get("colmap_matcher", "")),
            colmap_sequential_overlap=sequential_overlap,
            minimum_registered_image_ratio=registration_ratio,
            canonical_up_axis=str(runtime_raw.get("canonical_up_axis", "")),
        ),
    )
