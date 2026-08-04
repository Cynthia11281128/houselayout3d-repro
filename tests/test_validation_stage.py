from __future__ import annotations

import unittest

from houselayout3d.validation_stage import triangle_xy_area, validate_window


class ValidationStageTests(unittest.TestCase):
    def test_triangle_xy_area_ignores_height(self) -> None:
        self.assertAlmostEqual(
            triangle_xy_area([[0, 0, 4], [2, 0, 4], [0, 3, 4]]),
            3.0,
        )

    def test_window_rectangle_contract(self) -> None:
        result = validate_window(
            {
                "entity_id": "window_0",
                "vertices": [[0, 0, 1], [2, 0, 1], [2, 0, 2], [0, 0, 2]],
                "point_count": 10,
                "width_meters": 2.0,
                "height_meters": 1.0,
                "room_ids": ["room_0"],
            },
            10,
            0.3,
        )
        self.assertEqual(result["entity_id"], "window_0")


if __name__ == "__main__":
    unittest.main()
