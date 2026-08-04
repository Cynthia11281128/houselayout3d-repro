from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.cli import _parser  # noqa: E402
from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.skeleton_stage import (  # noqa: E402
    backproject_samples,
    build_depth_render_command,
)


class SkeletonStageImplementationTest(unittest.TestCase):
    def test_backprojection_converts_opengl_pose_to_opencv(self) -> None:
        depth = np.asarray([[2.0]], dtype=np.float32)
        c2w_opengl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        point = backproject_samples(
            depth,
            np.asarray([0]),
            c2w_opengl,
            fx=1.0,
            fy=1.0,
            cx=0.5,
            cy=0.5,
        )
        np.testing.assert_allclose(point, [[0.0, 0.0, 2.0]])

    def test_render_command_requests_metric_raw_depth_for_train_cameras(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml")
        command = build_depth_render_command(
            config, Path("/tmp/config.yml"), Path("/tmp/rendered")
        )
        self.assertIn("raw-depth", command)
        self.assertEqual(command[-2:], ["--split", "train"])
        self.assertEqual(config.skeleton.samples_per_frame, 5000)
        self.assertEqual(config.skeleton.regularization, (0.01, 0.1, 0.5))
        self.assertEqual(config.skeleton.final_level, 3)

    def test_cli_exposes_skeleton(self) -> None:
        args = _parser().parse_args(
            ["run-skeleton", "config.yaml", "--run-id", "run"]
        )
        self.assertEqual(args.command, "run-skeleton")


if __name__ == "__main__":
    unittest.main()
