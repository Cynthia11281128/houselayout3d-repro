from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.metric3d_stage import (  # noqa: E402
    Metric3DStageError,
    _verify_prior_stages,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Metric3DStageTest(unittest.TestCase):
    def _config_and_run(self, root: Path):
        text = (PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml").read_text()
        text = text.replace(
            "/tmp/tmp_data/GRIP-Layout/baselines/HouseLayout3D/outputs",
            str(root / "outputs"),
        )
        config_path = root / "config.yaml"
        config_path.write_text(text, encoding="utf-8")
        config = load_config(config_path)
        run = config.storage.outputs / config.scene / "test-run"
        input_dir = run / "00_input"
        pose_dir = run / "01_pose"
        input_dir.mkdir(parents=True)
        pose_dir.mkdir()
        image_list = input_dir / "images.txt"
        image_list.write_text("1772886535_000000001.png\n", encoding="utf-8")
        transforms = pose_dir / "transforms.json"
        transforms.write_text('{"frames": []}\n', encoding="utf-8")
        (input_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "validation": {"image_count": 1},
                    "outputs": {"image_list_sha256": _sha256(image_list)},
                }
            ),
            encoding="utf-8",
        )
        (pose_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "validation": {"pose_count": 1},
                    "outputs": {"transforms_json_sha256": _sha256(transforms)},
                }
            ),
            encoding="utf-8",
        )
        return config, image_list

    def test_prior_artifact_hashes_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, image_list = self._config_and_run(Path(temporary))
            _, _, _, names = _verify_prior_stages(config, "test-run")
            self.assertEqual(names, ["1772886535_000000001.png"])
            image_list.write_text("changed.png\n", encoding="utf-8")
            with self.assertRaisesRegex(Metric3DStageError, "hash no longer matches"):
                _verify_prior_stages(config, "test-run")


if __name__ == "__main__":
    unittest.main()
