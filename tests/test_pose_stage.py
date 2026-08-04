from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.input_stage import prepare_input  # noqa: E402
from houselayout3d.pose_stage import PoseStageError, prepare_poses  # noqa: E402


class PoseStageTest(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        images = root / "images"
        outputs = root / "outputs"
        images.mkdir()
        names = ["1772886535_000000001.png", "1772886536_000000002.png"]
        for name in names:
            Image.new("RGB", (800, 600), (10, 20, 30)).save(images / name)
        poses = images / "poses.csv"
        poses.write_text(
            "# counter,sec,nsec,x,y,z,qx,qy,qz,qw\n"
            "0,1772886535,1,1,2,3,0,0,0,1\n"
            "1,1772886536,2,2,2,3,0,0,0,1\n",
            encoding="utf-8",
        )
        text = (PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml").read_text()
        text = text.replace(
            "/home/xinyuan/GRIP-Layout/data/r04/feed_forward/keyframes_all3/front/poses.csv",
            str(poses),
        )
        text = text.replace(
            "/home/xinyuan/GRIP-Layout/data/r04/feed_forward/keyframes_all3/front",
            str(images),
        )
        text = text.replace(
            "/tmp/tmp_data/GRIP-Layout/baselines/HouseLayout3D/outputs",
            str(outputs),
        )
        path = root / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_known_pose_conversion_preserves_world_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(self._config(root))
            prepare_input(config, "known-pose", ["unit-test-input"])
            manifest_path = prepare_poses(config, "known-pose", ["unit-test-pose"])
            manifest = json.loads(manifest_path.read_text())
            transforms = json.loads(
                (manifest_path.parent / "transforms.json").read_text()
            )

            self.assertEqual(manifest["stage"], "01_pose")
            self.assertEqual(manifest["validation"]["pose_count"], 2)
            self.assertTrue(manifest["validation"]["pose_image_count_match"])
            matrix = transforms["frames"][0]["transform_matrix"]
            self.assertEqual([row[3] for row in matrix[:3]], [1.0, 2.0, 3.0])
            self.assertEqual(
                [row[:3] for row in matrix[:3]],
                [[1.0, -0.0, -0.0], [0.0, -1.0, -0.0], [0.0, -0.0, -1.0]],
            )
            self.assertEqual(transforms["orientation_override"], "none")
            self.assertTrue((manifest_path.parent / "images").is_symlink())

    def test_pose_timestamp_mismatch_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(self._config(root))
            prepare_input(config, "mismatch")
            pose_path = config.input.poses_csv
            assert pose_path is not None
            pose_path.write_text(
                pose_path.read_text().replace("1772886536,2", "1772886537,2"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PoseStageError, "timestamp mismatch"):
                prepare_poses(config, "mismatch")
            self.assertFalse(
                root.joinpath("outputs", "r04_front", "mismatch", "01_pose").exists()
            )


if __name__ == "__main__":
    unittest.main()
