from __future__ import annotations

import unittest

import numpy as np
from shapely.geometry import Polygon

from houselayout3d.config import LayoutConfig
from houselayout3d.layout_stage import (
    CeilingPlane,
    assign_triangle_ceilings,
    constrained_room_triangles,
    extrude_room,
)


def horizontal_ceiling(polygon_id: int, geometry: Polygon, height: float) -> CeilingPlane:
    return CeilingPlane(
        polygon_id=polygon_id,
        coefficients=np.asarray([0.0, 0.0, 1.0, -height]),
        geometry_xy=geometry,
        area_square_meters=geometry.area,
    )


class LayoutStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = LayoutConfig(
            door_height_meters=2.1,
            maximum_ceilings_per_room=30,
            window_minimum_cluster_points=10,
            window_minimum_size_meters=0.3,
            window_dbscan_epsilon_meters=0.15,
            window_dbscan_minimum_samples=5,
            window_outlier_neighbors=20,
            window_voxel_size_meters=0.03,
            window_frame_stride=1,
            window_pixel_stride=1,
            window_maximum_ray_distance_meters=30.0,
            stair_step_height_meters=0.18,
        )

    def test_constrained_triangles_cover_room_with_hole(self) -> None:
        room = Polygon(
            [(0, 0), (4, 0), (4, 4), (0, 4)],
            [[(1, 1), (2, 1), (2, 2), (1, 2)]],
        )
        ceilings = [horizontal_ceiling(1, Polygon([(0, 0), (2, 0), (2, 4), (0, 4)]), 3.0)]
        triangles = constrained_room_triangles(room, ceilings)
        area = sum(abs(np.cross(value[1] - value[0], value[2] - value[0])) / 2 for value in triangles)
        self.assertAlmostEqual(area, room.area, places=6)

    def test_ray_assignment_chooses_lowest_covering_ceiling(self) -> None:
        triangle = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        footprint = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])
        values = assign_triangle_ceilings(
            [triangle],
            [
                horizontal_ceiling(1, footprint, 3.2),
                horizontal_ceiling(2, footprint, 2.8),
            ],
            4.0,
        )
        self.assertEqual(values[0].polygon_id, 2)

    def test_square_room_closed_shell_is_watertight_after_weld(self) -> None:
        room = Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
        level = {
            "level_id": "level_00",
            "elevation_meters": 0.0,
            "ceiling_elevation_meters": 2.8,
            "wall_polygon_ids": [],
        }
        closed, final, _, _, _, diagnostics = extrude_room(
            "room_0",
            room,
            level,
            [horizontal_ceiling(1, room, 2.8)],
            [],
            {},
            self.layout,
            0,
        )
        mesh = closed.as_open3d()
        mesh.merge_close_vertices(1.0e-6)
        self.assertTrue(mesh.is_watertight())
        self.assertEqual(diagnostics["opening_segment_count"], 0)
        self.assertGreater(len(final.triangles), 0)

    def test_opening_removes_shared_wall_segments(self) -> None:
        room = Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
        level = {
            "level_id": "level_00",
            "elevation_meters": 0.0,
            "ceiling_elevation_meters": 2.8,
            "wall_polygon_ids": [],
        }
        edge = {
            "edge_id": "opening_0",
            "kind": "opening",
            "room_ids": ["room_0", "room_1"],
            "line_xy": [[0.0, 0.0], [3.0, 0.0]],
            "width_meters": 3.0,
            "pruned": False,
        }
        _, _, _, _, _, diagnostics = extrude_room(
            "room_0",
            room,
            level,
            [horizontal_ceiling(1, room, 2.8)],
            [edge],
            {},
            self.layout,
            0,
        )
        self.assertGreater(diagnostics["opening_segment_count"], 0)


if __name__ == "__main__":
    unittest.main()
