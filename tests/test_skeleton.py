from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.config import ConfigError, load_config  # noqa: E402
from houselayout3d.stages import STAGE_ORDER  # noqa: E402


class SkeletonContractTest(unittest.TestCase):
    def test_r04_config_is_unposed_pinhole(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front.yaml")
        self.assertEqual(config.scene, "r04_front")
        self.assertEqual(config.input.camera.model, "PINHOLE")
        self.assertEqual((config.input.camera.width, config.input.camera.height), (800, 600))
        self.assertEqual(config.runtime.colmap_matcher, "sequential")
        self.assertEqual(config.runtime.colmap_sequential_overlap, 10)
        self.assertEqual(config.runtime.minimum_registered_image_ratio, 0.8)
        self.assertEqual(config.metric3d.model, "metric3d_vit_large")
        self.assertEqual(
            (config.metric3d.input_height, config.metric3d.input_width),
            (616, 1064),
        )
        self.assertEqual(config.metric3d.canonical_focal_length, 1000.0)
        self.assertEqual(config.dn_splatter.method, "dn-splatter")
        self.assertEqual(config.dn_splatter.max_num_iterations, 30000)
        self.assertEqual(config.dn_splatter.depth_loss_type, "EdgeAwareLogL1")
        self.assertEqual(config.dn_splatter.depth_lambda, 0.2)
        self.assertEqual(config.dn_splatter.normal_supervision, "depth")

    def test_ground_truth_inputs_are_rejected(self) -> None:
        original = (PROJECT_ROOT / "configs" / "r04_front.yaml").read_text()
        modified = original.replace(
            "  image_glob: \"*.png\"",
            "  image_glob: \"*.png\"\n  gt_mesh: /forbidden/mesh.ply",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(modified)
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_stage_order_is_stable(self) -> None:
        self.assertEqual(STAGE_ORDER[0].value, "00_input")
        self.assertEqual(STAGE_ORDER[1].value, "01_pose")
        self.assertEqual(STAGE_ORDER[-1].value, "11_validation")
        self.assertEqual(len(STAGE_ORDER), 12)

    def test_heavy_storage_entries_are_symlinks(self) -> None:
        for name in ("data", "weights", "outputs", "cache", "reference_files"):
            path = PROJECT_ROOT / name
            self.assertTrue(path.is_symlink(), name)
            self.assertTrue(path.resolve().is_dir(), name)

    def test_preserved_source_hashes(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "references" / "source_manifest.json").read_text()
        )
        for source in manifest["sources"]:
            path = Path(source["preserved_path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(path.stat().st_size, source["size_bytes"])
            self.assertEqual(digest, source["sha256"])


if __name__ == "__main__":
    unittest.main()
