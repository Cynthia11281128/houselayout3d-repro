from typing import List

class PolygonFittingConfig:
    project_objects_to_floor_at_step: int = 600
    simplification_steps: List[int] = [50, 100, 150, 225, 300, 375, 450, 525, 600, 675, 750, 825, 900, 975, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    plane_merge_steps: List[int] = [90, 200, 275, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950, 2500, 3000, 3500, 4000, 4500, 5000]
    # extend_walls_to_floor_steps: List[int] = 
    recompute_interval: int = 10
    save_interval: int = 100
    iterations: int = 4000
    
    max_dist: float = 0.2
    magnetism_threshold: float = 0.1
    max_angle: float = 80
    vertex_merge_thresh: float = 0.02
    ray_intersection_margin: float = 0.1

    warmup_steps: int = 50
    window: bool = True
    chamfer_distance: bool = False
    point_triangle_distance: bool = False
    ray_tracing_distance: bool = True
    include_normals: bool = False

    merge_planes: bool = True
    simplify: bool = True
    do_extend_walls_to_floor: bool = True
    do_extend_walls_to_ceiling: bool = True
    regularize: bool = True
    regularize_wall_reach_ceiling: bool = False
    regularize_non_watertight: bool = True
    regularize_edge_magnetism: bool = True
    regularize_vertex_magnetism: bool = True
    regularize_face_magnetism: bool = True
    regularize_area: bool = False
    regularize_straightness: bool = False
    regularize_rectangularity: bool = False
    regularize_manhattan: bool = False

    regularization_strength = 5 * 1e-6

    delete_polygons_not_aligned_with_semantics = True
    
    up_vector: str = "Y"

    multi_floor: bool = False
    max_ray_intersects_per_m2 : int = 1000

    # plane merging
    max_plane_merge_dist = 0.1
    projection_baseline_error = 0.08 # roughly 10% of what we expect
    normal_baseline_error = 0.5 # roughly 10% of what we expect
    max_plane_merge_angle = 45

    recompute_normals: bool = False
    plane_lr_downscale: float = 10

    max_wall_angle_for_extension: float = 10

    recompute_polygon_areas_every: int = 10

    lr_start = 1e-3
    lr_normal = 5 * 1e-4


class ScannetppConfig(PolygonFittingConfig):
    up_vector: str = "Y"

class MatterportConfig(PolygonFittingConfig):
    up_vector: str = "Z"

    regularization_strength = 1e-6
    delete_polygons_not_aligned_with_semantics = False
    multi_floor: bool = True
    project_objects_to_floor_at_step: int = 600

    max_ray_intersects_per_m2 : int = 50000

    max_plane_merge_dist = 0.2
    max_plane_merge_angle = 30
    projection_baseline_error = 0.1
    normal_baseline_error = 0.3
    
    ray_intersection_margin: float = 0.1

    recompute_normals: bool = True

    plane_lr_downscale: float = 50

    regularize_edge_magnetism: bool = True

    max_wall_angle_for_extension: float = 5

    # plane_merge_steps = [50, 100, 150, 225, 300, 375, 450, 525, 600, 675, 750, 825, 900, 975, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    
    iterations: int = 4000
    #NO LEARN
    # project_objects_to_floor_at_step: int = 20
    # plane_merge_steps = [5,10,15,20,25,30,45,50]
    # simplification_steps = []
    # do_extend_walls_to_floor: bool = False
    # regularize = False
    # save_interval = 5
    # iterations: int = 100
    # lr_start = 0
    # lr_normal = 0
    # warmup_steps = 0
