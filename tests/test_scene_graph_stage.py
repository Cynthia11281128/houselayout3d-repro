from __future__ import annotations

import unittest

import numpy as np

from houselayout3d.scene_graph_stage import (
    Grid2D,
    assign_ceilings_to_levels,
    group_floor_polygons,
    mask_geometry,
    opening_edges,
    two_stage_room_watershed,
)


class SceneGraphStageTests(unittest.TestCase):
    def test_floor_height_graph_uses_connected_components(self) -> None:
        groups = group_floor_polygons(
            [4, 8, 12, 16],
            {4: 0.0, 8: 0.4, 12: 0.8, 16: 3.1},
            0.5,
        )
        self.assertEqual(groups, [[4, 8, 12], [16]])

    def test_ceilings_choose_closest_next_lower_level(self) -> None:
        assignments = assign_ceilings_to_levels(
            [2, 6, 10],
            {2: 2.8, 6: 5.7, 10: 0.7},
            [0.0, 3.0],
            1.0,
        )
        self.assertEqual(assignments, {0: [2], 1: [6]})

    def test_two_stage_bottleneck_splits_two_chambers(self) -> None:
        resolution = 0.1
        free = np.zeros((60, 120), dtype=bool)
        free[10:50, 5:45] = True
        free[10:50, 75:115] = True
        free[25:35, 45:75] = True
        labels, distance, markers, stats = two_stage_room_watershed(
            free,
            resolution,
            (2.5, 1.5),
            0.02,
            0.5,
        )
        self.assertEqual(stats["room_count"], 2)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertGreater(float(distance.max()), 1.5)
        self.assertEqual(len(set(np.unique(markers))) - 1, 2)

    def test_opening_width_controls_edge_kind(self) -> None:
        grid = Grid2D(np.asarray([0.0, 0.0]), 0.1, (30, 20))
        labels = np.zeros(grid.shape, dtype=np.int32)
        labels[5:15, 2:10] = 1
        labels[5:15, 10:18] = 2
        edges = opening_edges(labels, grid, 1.5, {1: "left", 2: "right"})
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["kind"], "door")
        self.assertEqual(edges[0]["room_ids"], ["left", "right"])
        self.assertLess(edges[0]["width_meters"], 1.5)

    def test_mask_geometry_is_world_aligned(self) -> None:
        grid = Grid2D(np.asarray([2.0, -3.0]), 0.25, (8, 8))
        mask = np.zeros(grid.shape, dtype=np.uint8)
        mask[2:6, 1:5] = 1
        geometry = mask_geometry(mask, grid)
        self.assertFalse(geometry.is_empty)
        self.assertGreater(geometry.area, 0.5)
        self.assertGreaterEqual(geometry.bounds[0], 2.0)
        self.assertGreaterEqual(geometry.bounds[1], -3.0)


if __name__ == "__main__":
    unittest.main()
