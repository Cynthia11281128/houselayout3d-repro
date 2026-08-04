from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.cli import _parser  # noqa: E402
from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.prototype_stage import (  # noqa: E402
    _prepare_semantic_classes,
    _source_config,
    _verify_source,
    build_prototype_command,
)


class PrototypeStageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(
            PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml"
        )

    def test_unofficial_source_is_byte_preserved_and_matterport_config_matches(self) -> None:
        records = _verify_source(self.config)
        self.assertIn("fit_prototype.py", records)
        source = _source_config(self.config)
        self.assertEqual(source["iterations"], 4000)
        self.assertEqual(source["save_interval"], 100)
        self.assertEqual(source["up_vector"], "Z")
        self.assertTrue(source["multi_floor"])
        self.assertTrue(source["ray_tracing_distance"])

    def test_compatibility_door_channel_is_explicitly_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classes = np.asarray([[0.2] * 5 + [0.0] * 4], dtype=np.float16)
            classes /= classes.sum(axis=1, keepdims=True)
            names = np.asarray(
                [
                    "wall",
                    "ceiling",
                    "floor",
                    "surface",
                    "inaccurate_window",
                    "inaccurate_mirror",
                    "inaccurate_outdoor",
                    "stairs",
                    "object",
                ]
            )
            np.save(root / "classes.npy", classes)
            np.save(root / "names.npy", names)
            record = _prepare_semantic_classes(
                root / "classes.npy",
                root / "names.npy",
                root / "prepared_classes.npy",
                root / "prepared_names.npy",
            )
            prepared = np.load(root / "prepared_classes.npy")
            prepared_names = np.load(root / "prepared_names.npy")
        self.assertEqual(prepared.shape, (1, 10))
        self.assertEqual(prepared_names[-1], "door")
        self.assertTrue(np.all(prepared[:, -1] == 0))
        self.assertTrue(record["door_probability_is_zero"])

    def test_command_uses_real_unofficial_argument_names(self) -> None:
        prepared = {
            name: Path(f"/tmp/{name}")
            for name in (
                "rectified_mesh",
                "target_mesh",
                "target_classes",
                "class_names",
                "polygon_info",
                "ray_origins",
                "ray_destinations",
                "object_mesh",
                "ray_classes",
            )
        }
        command = build_prototype_command(
            self.config, prepared, Path("/tmp/output")
        )
        self.assertIn("--target-pcd-ray-origins-path", command)
        self.assertIn("--target-pcd-ray-dests-path", command)
        self.assertIn("--scene-type", command)
        self.assertEqual(command[command.index("--scene-type") + 1], "matterport")
        self.assertEqual(command[-1], "cuda")

    def test_cli_exposes_prepare_and_fit(self) -> None:
        prepare = _parser().parse_args(
            ["prepare-prototype", "config.yaml", "--run-id", "run"]
        )
        fit = _parser().parse_args(
            ["fit-prototype", "config.yaml", "--run-id", "run"]
        )
        self.assertEqual(prepare.command, "prepare-prototype")
        self.assertEqual(fit.command, "fit-prototype")


if __name__ == "__main__":
    unittest.main()
