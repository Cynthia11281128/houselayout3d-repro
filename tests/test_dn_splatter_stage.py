from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.dn_splatter_stage import (  # noqa: E402
    build_training_command,
    build_training_environment,
)


class DNSplatterStageTest(unittest.TestCase):
    def test_training_command_preserves_metric_known_poses(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            command = build_training_command(config, Path(temporary), 48.0)

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(value("--max-num-iterations"), "30000")
        self.assertEqual(value("--pipeline.model.depth-loss-type"), "EdgeAwareLogL1")
        self.assertEqual(value("--pipeline.model.depth-lambda"), "0.2")
        self.assertEqual(value("--pipeline.model.normal-supervision"), "depth")
        self.assertEqual(value("--pipeline.model.camera-optimizer.mode"), "off")
        self.assertEqual(value("--pipeline.model.random-init"), "False")
        self.assertEqual(value("--pipeline.datamanager.cache-images-type"), "float32")
        self.assertEqual(value("--auto-scale-poses"), "False")
        self.assertEqual(value("--orientation-method"), "none")
        self.assertEqual(value("--center-method"), "none")
        self.assertEqual(value("--depth-unit-scale-factor"), "1.0")
        self.assertEqual(value("--scene-scale"), "48.0")
        self.assertEqual(value("--load-3D-points"), "True")
        self.assertEqual(value("--load-normals"), "False")

    def test_training_environment_exposes_conda_cuda_toolkit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment_root = Path(temporary) / "nerfstudio"
            environment_bin = environment_root / "bin"
            environment_bin.mkdir(parents=True)
            ns_train = environment_bin / "ns-train"
            ns_train.touch()
            (environment_bin / "nvcc").touch()
            environment = build_training_environment(
                ns_train,
                {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "1"},
            )

        self.assertEqual(
            environment["PATH"].split(":"),
            [str(environment_bin.resolve()), "/usr/bin"],
        )
        self.assertEqual(environment["CUDA_HOME"], str(environment_root.resolve()))
        self.assertEqual(environment["TORCH_CUDA_ARCH_LIST"], "8.9")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")


if __name__ == "__main__":
    unittest.main()
