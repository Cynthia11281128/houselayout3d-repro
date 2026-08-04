"""Small bootstrap CLI; execution commands are added in later milestones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .colmap_stage import ColmapStageError, run_colmap
from .config import load_config
from .dn_splatter_stage import (
    DNSplatterStageError,
    prepare_dn_splatter,
    train_dn_splatter,
)
from .input_stage import InputAuditError, prepare_input
from .layout_stage import LayoutStageError, run_layout
from .metric3d_stage import Metric3DStageError, run_metric3d
from .mesh_stage import MeshStageError, run_mesh
from .oneformer_stage import OneFormerStageError, run_oneformer
from .pose_stage import PoseStageError, prepare_poses
from .polygon_init_stage import PolygonInitStageError, run_polygon_init
from .prototype_stage import PrototypeStageError, fit_prototype, prepare_prototype
from .scene_graph_stage import SceneGraphStageError, run_scene_graph
from .skeleton_stage import SkeletonStageError, run_skeleton
from .stages import STAGE_ORDER
from .validation_stage import ValidationStageError, run_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="houselayout3d")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-config", help="validate and summarize a pipeline config"
    )
    inspect_parser.add_argument("config", type=Path)
    prepare_parser = subparsers.add_parser(
        "prepare-input", help="audit and freeze the pose-free 00_input stage"
    )
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("--run-id", required=True)
    colmap_parser = subparsers.add_parser(
        "run-colmap", help="run and validate the 01_colmap stage"
    )
    colmap_parser.add_argument("config", type=Path)
    colmap_parser.add_argument("--run-id", required=True)
    pose_parser = subparsers.add_parser(
        "prepare-poses", help="validate known poses and write the 01_pose stage"
    )
    pose_parser.add_argument("config", type=Path)
    pose_parser.add_argument("--run-id", required=True)
    metric3d_parser = subparsers.add_parser(
        "run-metric3d", help="run metric depth and normal inference for 02_metric3d"
    )
    metric3d_parser.add_argument("config", type=Path)
    metric3d_parser.add_argument("--run-id", required=True)
    dn_prepare_parser = subparsers.add_parser(
        "prepare-dn-splatter",
        help="prepare known-pose depth inputs and a seed cloud for 03_dn_splatter",
    )
    dn_prepare_parser.add_argument("config", type=Path)
    dn_prepare_parser.add_argument("--run-id", required=True)
    dn_train_parser = subparsers.add_parser(
        "train-dn-splatter", help="train the prepared 03_dn_splatter stage"
    )
    dn_train_parser.add_argument("config", type=Path)
    dn_train_parser.add_argument("--run-id", required=True)
    mesh_parser = subparsers.add_parser(
        "run-mesh", help="export and validate the formal 04_mesh Poisson mesh"
    )
    mesh_parser.add_argument("config", type=Path)
    mesh_parser.add_argument("--run-id", required=True)
    oneformer_parser = subparsers.add_parser(
        "run-oneformer",
        help="run COCO OneFormer inference and Appendix-A remapping for 05_oneformer",
    )
    oneformer_parser.add_argument("config", type=Path)
    oneformer_parser.add_argument("--run-id", required=True)
    skeleton_parser = subparsers.add_parser(
        "run-skeleton",
        help="render DN depths and extract the semantic 06_skeleton mesh",
    )
    skeleton_parser.add_argument("config", type=Path)
    skeleton_parser.add_argument("--run-id", required=True)
    polygon_parser = subparsers.add_parser(
        "run-polygon-init",
        help="fit Appendix Algorithm-1 planes and initialize 07_polygon_init",
    )
    polygon_parser.add_argument("config", type=Path)
    polygon_parser.add_argument("--run-id", required=True)
    prototype_prepare_parser = subparsers.add_parser(
        "prepare-prototype",
        help="freeze and audit inputs for the 08_prototype optimizer",
    )
    prototype_prepare_parser.add_argument("config", type=Path)
    prototype_prepare_parser.add_argument("--run-id", required=True)
    prototype_fit_parser = subparsers.add_parser(
        "fit-prototype",
        help="run the full 4,000-step unofficial Matterport optimizer",
    )
    prototype_fit_parser.add_argument("config", type=Path)
    prototype_fit_parser.add_argument("--run-id", required=True)
    scene_graph_parser = subparsers.add_parser(
        "run-scene-graph",
        help="construct Appendix-D levels, rooms, openings, stairs, and semantics",
    )
    scene_graph_parser.add_argument("config", type=Path)
    scene_graph_parser.add_argument("--run-id", required=True)
    layout_parser = subparsers.add_parser(
        "run-layout",
        help="extrude rooms and generate final walls, floors, ceilings, openings, stairs, doors, and windows",
    )
    layout_parser.add_argument("config", type=Path)
    layout_parser.add_argument("--run-id", required=True)
    validation_parser = subparsers.add_parser(
        "validate-layout",
        help="independently validate Stage10 hashes, topology, entities, and graph references",
    )
    validation_parser.add_argument("config", type=Path)
    validation_parser.add_argument("--run-id", required=True)
    subparsers.add_parser("stages", help="print the stable pipeline stage order")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "stages":
        print("\n".join(stage.value for stage in STAGE_ORDER))
        return 0

    config = load_config(args.config)
    if args.command == "prepare-input":
        try:
            manifest = prepare_input(config, args.run_id)
        except InputAuditError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-colmap":
        try:
            manifest = run_colmap(config, args.run_id)
        except ColmapStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "prepare-poses":
        try:
            manifest = prepare_poses(config, args.run_id)
        except PoseStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-metric3d":
        try:
            manifest = run_metric3d(config, args.run_id)
        except Metric3DStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "prepare-dn-splatter":
        try:
            prepared = prepare_dn_splatter(config, args.run_id)
        except DNSplatterStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(prepared)
        return 0
    if args.command == "train-dn-splatter":
        try:
            manifest = train_dn_splatter(config, args.run_id)
        except DNSplatterStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-mesh":
        try:
            manifest = run_mesh(config, args.run_id)
        except MeshStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-oneformer":
        try:
            manifest = run_oneformer(config, args.run_id)
        except OneFormerStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-skeleton":
        try:
            manifest = run_skeleton(config, args.run_id)
        except SkeletonStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-polygon-init":
        try:
            manifest = run_polygon_init(config, args.run_id)
        except PolygonInitStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "prepare-prototype":
        try:
            manifest = prepare_prototype(config, args.run_id)
        except PrototypeStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "fit-prototype":
        try:
            manifest = fit_prototype(config, args.run_id)
        except PrototypeStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-scene-graph":
        try:
            manifest = run_scene_graph(config, args.run_id)
        except SceneGraphStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "run-layout":
        try:
            manifest = run_layout(config, args.run_id)
        except LayoutStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0
    if args.command == "validate-layout":
        try:
            manifest = run_validation(config, args.run_id)
        except ValidationStageError as error:
            parser = _parser()
            parser.error(str(error))
        print(manifest)
        return 0

    summary = {
        "schema_version": config.schema_version,
        "scene": config.scene,
        "images": str(config.input.images),
        "poses_csv": (
            str(config.input.poses_csv)
            if config.input.poses_csv is not None
            else None
        ),
        "pose_convention": config.input.pose_convention,
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
        "outputs": str(config.storage.outputs),
        "metric3d": {
            "repository": str(config.metric3d.repository),
            "weights": str(config.metric3d.weights),
            "model": config.metric3d.model,
            "input_height": config.metric3d.input_height,
            "input_width": config.metric3d.input_width,
            "canonical_focal_length": config.metric3d.canonical_focal_length,
            "minimum_depth_meters": config.metric3d.minimum_depth_meters,
            "maximum_depth_meters": config.metric3d.maximum_depth_meters,
        },
        "dn_splatter": {
            "method": config.dn_splatter.method,
            "max_num_iterations": config.dn_splatter.max_num_iterations,
            "steps_per_save": config.dn_splatter.steps_per_save,
            "depth_loss_type": config.dn_splatter.depth_loss_type,
            "depth_lambda": config.dn_splatter.depth_lambda,
            "normal_supervision": config.dn_splatter.normal_supervision,
            "seed_stride": config.dn_splatter.seed_stride,
            "seed_minimum_depth_meters": (
                config.dn_splatter.seed_minimum_depth_meters
            ),
            "seed_maximum_depth_meters": (
                config.dn_splatter.seed_maximum_depth_meters
            ),
            "seed_voxel_size_meters": (
                config.dn_splatter.seed_voxel_size_meters
            ),
            "maximum_seed_points": config.dn_splatter.maximum_seed_points,
        },
        "mesh": {
            "exporter": config.mesh.exporter,
            "total_points": config.mesh.total_points,
            "normal_method": config.mesh.normal_method,
            "use_masks": config.mesh.use_masks,
            "filter_edges_from_depth_maps": (
                config.mesh.filter_edges_from_depth_maps
            ),
            "poisson_depth": config.mesh.poisson_depth,
        },
        "oneformer": {
            "model_dir": str(config.oneformer.model_dir),
            "task": config.oneformer.task,
            "preview_count": config.oneformer.preview_count,
        },
        "skeleton": {
            "render_executable": str(config.skeleton.render_executable),
            "superpoint_repository": str(config.skeleton.superpoint_repository),
            "samples_per_frame": config.skeleton.samples_per_frame,
            "knn_neighbors": config.skeleton.knn_neighbors,
            "knn_radius_meters": config.skeleton.knn_radius_meters,
            "regularization": list(config.skeleton.regularization),
            "final_level": config.skeleton.final_level,
        },
        "polygon_init": {
            "superpoint_level": config.polygon_init.superpoint_level,
            "minimum_unassigned_vertices": (
                config.polygon_init.minimum_unassigned_vertices
            ),
            "plane_distance_threshold_meters": (
                config.polygon_init.plane_distance_threshold_meters
            ),
            "ransac_iterations": config.polygon_init.ransac_iterations,
            "rdp_epsilon_meters": config.polygon_init.rdp_epsilon_meters,
        },
        "prototype": {
            "python_executable": str(config.prototype.python_executable),
            "source_repository": str(config.prototype.source_repository),
            "scene_type": config.prototype.scene_type,
            "iterations": config.prototype.iterations,
            "checkpoint_interval": config.prototype.checkpoint_interval,
            "object_target_triangles": config.prototype.object_target_triangles,
        },
        "scene_graph": {
            "openseg_python": str(config.scene_graph.openseg_python),
            "openseg_model": str(config.scene_graph.openseg_model),
            "clip_weights": str(config.scene_graph.clip_weights),
            "grid_resolution_meters": config.scene_graph.grid_resolution_meters,
            "floor_merge_height_meters": config.scene_graph.floor_merge_height_meters,
            "ceiling_minimum_clearance_meters": config.scene_graph.ceiling_minimum_clearance_meters,
            "wall_interval_height_meters": config.scene_graph.wall_interval_height_meters,
            "bottleneck_widths_meters": list(config.scene_graph.bottleneck_widths_meters),
            "door_maximum_width_meters": config.scene_graph.door_maximum_width_meters,
            "stair_room_maximum_distance_meters": config.scene_graph.stair_room_maximum_distance_meters,
        },
        "layout": {
            "door_height_meters": config.layout.door_height_meters,
            "maximum_ceilings_per_room": config.layout.maximum_ceilings_per_room,
            "window_minimum_cluster_points": config.layout.window_minimum_cluster_points,
            "window_minimum_size_meters": config.layout.window_minimum_size_meters,
            "window_dbscan_epsilon_meters": config.layout.window_dbscan_epsilon_meters,
            "window_dbscan_minimum_samples": config.layout.window_dbscan_minimum_samples,
            "window_outlier_neighbors": config.layout.window_outlier_neighbors,
            "window_voxel_size_meters": config.layout.window_voxel_size_meters,
            "window_frame_stride": config.layout.window_frame_stride,
            "window_pixel_stride": config.layout.window_pixel_stride,
            "window_maximum_ray_distance_meters": config.layout.window_maximum_ray_distance_meters,
            "stair_step_height_meters": config.layout.stair_step_height_meters,
        },
        "colmap_matcher": config.runtime.colmap_matcher,
        "colmap_sequential_overlap": config.runtime.colmap_sequential_overlap,
        "minimum_registered_image_ratio": (
            config.runtime.minimum_registered_image_ratio
        ),
        "canonical_up_axis": config.runtime.canonical_up_axis,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
