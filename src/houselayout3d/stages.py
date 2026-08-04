"""Stable names and ordering for pipeline artifact directories."""

from enum import Enum


class Stage(str, Enum):
    INPUT = "00_input"
    POSE = "01_pose"
    # Preserved for the two rejected SfM attempts; not part of the active order.
    COLMAP = "01_colmap"
    METRIC3D = "02_metric3d"
    DN_SPLATTER = "03_dn_splatter"
    MESH = "04_mesh"
    ONEFORMER = "05_oneformer"
    SKELETON = "06_skeleton"
    POLYGON_INIT = "07_polygon_init"
    PROTOTYPE = "08_prototype"
    SCENE_GRAPH = "09_scene_graph"
    LAYOUT = "10_layout"
    VALIDATION = "11_validation"


STAGE_ORDER = (
    Stage.INPUT,
    Stage.POSE,
    Stage.METRIC3D,
    Stage.DN_SPLATTER,
    Stage.MESH,
    Stage.ONEFORMER,
    Stage.SKELETON,
    Stage.POLYGON_INIT,
    Stage.PROTOTYPE,
    Stage.SCENE_GRAPH,
    Stage.LAYOUT,
    Stage.VALIDATION,
)
