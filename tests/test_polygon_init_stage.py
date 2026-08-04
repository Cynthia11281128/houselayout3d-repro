from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.cli import _parser  # noqa: E402
from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.polygon_init_stage import (  # noqa: E402
    closed_rdp_mask,
    extract_boundary_loops,
    fit_plane_ransac,
    group_boundary_contours,
    polygon_area,
)


class PolygonInitStageTest(unittest.TestCase):
    def test_ransac_recovers_plane_with_outliers(self) -> None:
        rng = np.random.default_rng(7)
        xy = rng.uniform(-2.0, 2.0, size=(200, 2))
        plane_points = np.column_stack((xy, np.full(len(xy), 1.25)))
        plane_points[:, 2] += rng.normal(scale=0.002, size=len(xy))
        outliers = rng.uniform(-2.0, 2.0, size=(20, 3))
        points = np.concatenate((plane_points, outliers))
        plane, inliers = fit_plane_ransac(
            points,
            threshold=0.01,
            iterations=128,
            rng=np.random.default_rng(0),
            preferred_normal=np.asarray([0.0, 0.0, 1.0]),
        )
        self.assertGreaterEqual(int(inliers.sum()), 198)
        np.testing.assert_allclose(plane[:3], [0.0, 0.0, 1.0], atol=0.002)
        self.assertAlmostEqual(float(plane[3]), -1.25, places=2)

    def test_triangle_boundary_is_one_closed_square(self) -> None:
        triangles = np.asarray([[0, 1, 2], [0, 2, 3]])
        loops, edges, branches = extract_boundary_loops(triangles)
        self.assertEqual(branches, 0)
        self.assertEqual(len(edges), 4)
        self.assertEqual(len(loops), 1)
        self.assertEqual(set(loops[0].tolist()), {0, 1, 2, 3})

    def test_closed_rdp_keeps_polygon_and_reduces_collinear_samples(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
                [0.5, 1.0],
                [0.0, 1.0],
                [0.0, 0.5],
            ]
        )
        mask = closed_rdp_mask(points, epsilon=0.01)
        self.assertEqual(int(mask.sum()), 4)
        self.assertAlmostEqual(abs(polygon_area(points[mask])), 1.0)

    def test_touching_triangle_components_recover_two_boundary_cycles(self) -> None:
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        triangles = np.asarray([[0, 1, 2], [0, 3, 4]])
        plane = np.asarray([0.0, 0.0, 1.0, 0.0])
        loops, _, branches = extract_boundary_loops(
            triangles, vertices=vertices, plane=plane
        )
        self.assertEqual(branches, 1)
        self.assertEqual(len(loops), 2)
        groups = group_boundary_contours(loops, vertices, plane)
        self.assertEqual(len(groups), 2)

    def test_config_and_cli_expose_polygon_init(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml")
        self.assertEqual(config.polygon_init.superpoint_level, 3)
        self.assertEqual(config.polygon_init.minimum_unassigned_vertices, 100)
        self.assertEqual(config.polygon_init.plane_distance_threshold_meters, 0.05)
        args = _parser().parse_args(
            ["run-polygon-init", "config.yaml", "--run-id", "run"]
        )
        self.assertEqual(args.command, "run-polygon-init")


if __name__ == "__main__":
    unittest.main()
