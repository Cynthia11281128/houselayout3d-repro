from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.mesh_stage import build_mesh_command  # noqa: E402


class MeshStageTest(unittest.TestCase):
    def test_command_uses_paper_path_depth_normal_poisson(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = build_mesh_command(config, root / "config.yml", root / "mesh")

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(command[1], "dn")
        self.assertEqual(value("--total-points"), "2000000")
        self.assertEqual(value("--normal-method"), "normal_maps")
        self.assertEqual(value("--use-masks"), "True")
        self.assertEqual(value("--filter-edges-from-depth-maps"), "False")
        self.assertEqual(value("--poisson-depth"), "9")


if __name__ == "__main__":
    unittest.main()
