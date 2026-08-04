from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from houselayout3d.cli import _parser  # noqa: E402
from houselayout3d.config import load_config  # noqa: E402
from houselayout3d.oneformer_stage import (  # noqa: E402
    APPENDIX_COCO_IDS,
    LAYOUT_LABELS,
    appendix_layout_lut,
)


class OneFormerStageTest(unittest.TestCase):
    def test_appendix_mapping_is_disjoint_and_total(self) -> None:
        lut = appendix_layout_lut()
        self.assertEqual(lut.shape, (133,))
        self.assertEqual(lut.dtype, np.uint8)
        self.assertEqual(set(lut.tolist()), set(range(len(LAYOUT_LABELS))))

        explicit = set()
        for ids in APPENDIX_COCO_IDS.values():
            self.assertFalse(explicit.intersection(ids))
            explicit.update(ids)
        object_id = LAYOUT_LABELS.index("object")
        self.assertEqual(int((lut == object_id).sum()), 133 - len(explicit))

    def test_table_7_representative_ids(self) -> None:
        lut = appendix_layout_lut()
        expected = {
            109: "wall",
            118: "ceiling",
            87: "floor",
            120: "surface",
            115: "inaccurate_window",
            93: "inaccurate_mirror",
            119: "inaccurate_outdoor",
            106: "stairs",
            39: "object",
        }
        for coco_id, label in expected.items():
            self.assertEqual(LAYOUT_LABELS[int(lut[coco_id])], label)

    def test_config_and_cli_expose_oneformer(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "r04_front_known_pose.yaml")
        self.assertEqual(config.oneformer.task, "semantic")
        self.assertEqual(config.oneformer.preview_count, 24)
        args = _parser().parse_args(
            ["run-oneformer", "config.yaml", "--run-id", "run"]
        )
        self.assertEqual(args.command, "run-oneformer")


if __name__ == "__main__":
    unittest.main()
