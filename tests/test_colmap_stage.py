from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.colmap_stage import build_commands  # noqa: E402
from houselayout3d.config import load_config  # noqa: E402


class ColmapStageTest(unittest.TestCase):
    def test_commands_use_only_audited_images_and_fixed_camera(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "01_colmap"
            image_list = Path(temporary) / "00_input" / "images.txt"
            commands = dict(build_commands(config, stage, image_list))

        feature = commands["feature_extractor"]
        self.assertIn("--image_list_path", feature)
        self.assertEqual(feature[feature.index("--image_list_path") + 1], str(image_list))
        self.assertEqual(
            feature[feature.index("--ImageReader.camera_model") + 1], "PINHOLE"
        )
        self.assertEqual(
            feature[feature.index("--ImageReader.camera_params") + 1],
            "463.99945,463.25045,400.0,300.0",
        )
        self.assertEqual(feature[feature.index("--FeatureExtraction.use_gpu") + 1], "0")

        matcher = commands["sequential_matcher"]
        self.assertEqual(matcher[matcher.index("--FeatureMatching.use_gpu") + 1], "0")
        self.assertEqual(
            matcher[matcher.index("--SequentialMatching.overlap") + 1], "10"
        )
        mapper = commands["mapper"]
        self.assertEqual(
            mapper[mapper.index("--Mapper.image_list_path") + 1], str(image_list)
        )

    def test_overlap30_retry_is_explicit(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front_overlap30.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            commands = dict(
                build_commands(
                    config,
                    Path(temporary) / "01_colmap",
                    Path(temporary) / "00_input" / "images.txt",
                )
            )
        matcher = commands["sequential_matcher"]
        self.assertEqual(
            matcher[matcher.index("--SequentialMatching.overlap") + 1], "30"
        )


if __name__ == "__main__":
    unittest.main()
