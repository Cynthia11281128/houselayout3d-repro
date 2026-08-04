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
from houselayout3d.input_stage import InputAuditError, prepare_input  # noqa: E402


class InputStageTest(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        images = root / "images"
        outputs = root / "outputs"
        images.mkdir()
        Image.new("RGB", (800, 600), (10, 20, 30)).save(
            images / "1772886535_000000001.png"
        )
        Image.new("RGB", (800, 600), (40, 50, 60)).save(
            images / "1772886536_000000002.png"
        )
        (images / "poses.csv").write_text("must not be consumed\n")

        text = (PROJECT_ROOT / "configs" / "r04_front.yaml").read_text()
        text = text.replace(
            "/home/xinyuan/GRIP-Layout/data/r04/feed_forward/keyframes_all3/front",
            str(images),
        )
        text = text.replace(
            "/tmp/tmp_data/GRIP-Layout/baselines/HouseLayout3D/outputs",
            str(outputs),
        )
        path = root / "config.yaml"
        path.write_text(text)
        return path

    def test_prepare_input_writes_pose_free_immutable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = load_config(self._config(root))
            manifest_path = prepare_input(config, "test-run", ["unit-test"])
            manifest = json.loads(manifest_path.read_text())

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["validation"]["image_count"], 2)
            self.assertEqual(
                manifest["input"]["ignored_non_image_entries"], ["poses.csv"]
            )
            self.assertFalse(
                manifest["validation"]["pose_or_ground_truth_inputs_used"]
            )
            self.assertEqual(
                (manifest_path.parent / "images.txt").read_text().splitlines(),
                ["1772886535_000000001.png", "1772886536_000000002.png"],
            )
            with self.assertRaises(InputAuditError):
                prepare_input(config, "test-run")

    def test_dimension_mismatch_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            Image.new("RGB", (10, 10)).save(
                root / "images" / "1772886537_000000003.png"
            )
            config = load_config(config_path)
            with self.assertRaisesRegex(InputAuditError, "image size mismatch"):
                prepare_input(config, "bad-run")
            self.assertFalse(root.joinpath("outputs", "r04_front", "bad-run").exists())

    def test_bad_timestamp_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = self._config(root)
            Image.new("RGB", (800, 600)).save(root / "images" / "bad.png")
            with self.assertRaisesRegex(InputAuditError, "filename_regex"):
                prepare_input(load_config(config_path), "bad-name")


if __name__ == "__main__":
    unittest.main()
