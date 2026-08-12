"""HouseLayout3D Appendix-A semantic label mapping."""

from __future__ import annotations

import numpy as np


LAYOUT_LABELS = (
    "wall",
    "ceiling",
    "floor",
    "surface",
    "inaccurate_window",
    "inaccurate_mirror",
    "inaccurate_outdoor",
    "stairs",
    "object",
)

# Appendix A / Table 5 mapping. Everything not listed explicitly is an object.
APPENDIX_COCO_IDS: dict[str, frozenset[int]] = {
    "wall": frozenset({109, 110, 111, 112, 131}),
    "ceiling": frozenset({118}),
    "floor": frozenset({87, 122, 132}),
    "surface": frozenset({85, 86, 114, 120}),
    "inaccurate_window": frozenset({115}),
    "inaccurate_mirror": frozenset({93}),
    "inaccurate_outdoor": frozenset({90, 116, 119, 123, 125, 126}),
    "stairs": frozenset({106}),
}

LAYOUT_PALETTE = np.asarray(
    [
        [210, 68, 68],
        [90, 150, 230],
        [80, 185, 105],
        [238, 180, 70],
        [75, 205, 215],
        [205, 105, 220],
        [130, 130, 130],
        [245, 105, 30],
        [150, 105, 70],
    ],
    dtype=np.uint8,
)


def appendix_layout_lut() -> np.ndarray:
    """Return the COCO-133 to HouseLayout3D intermediate layout-label LUT."""

    lut = np.full(133, LAYOUT_LABELS.index("object"), dtype=np.uint8)
    assigned: set[int] = set()
    for layout_id, label in enumerate(LAYOUT_LABELS[:-1]):
        ids = APPENDIX_COCO_IDS[label]
        overlap = assigned.intersection(ids)
        if overlap:
            raise AssertionError(
                f"Appendix-A COCO mapping overlaps at IDs: {sorted(overlap)}"
            )
        lut[list(ids)] = layout_id
        assigned.update(ids)
    return lut


def label_contract(id2label: dict[int, str]) -> dict[str, object]:
    """Validate OneFormer's COCO label contract and describe the remapping."""

    if set(id2label) != set(range(133)):
        raise ValueError("OneFormer checkpoint must expose COCO's 133 IDs")
    expected = {
        85: "curtain",
        86: "door-stuff",
        87: "floor-wood",
        90: "gravel",
        93: "mirror-stuff",
        106: "stairs",
        109: "wall-brick",
        110: "wall-stone",
        111: "wall-tile",
        112: "wall-wood",
        114: "window-blind",
        115: "window-other",
        116: "tree-merged",
        118: "ceiling-merged",
        119: "sky-other-merged",
        120: "cabinet-merged",
        122: "floor-other-merged",
        123: "pavement-merged",
        125: "grass-merged",
        126: "dirt-merged",
        131: "wall-other-merged",
        132: "rug-merged",
    }
    mismatches = {
        str(class_id): {"expected": name, "actual": id2label[class_id]}
        for class_id, name in expected.items()
        if id2label[class_id] != name
    }
    if mismatches:
        raise ValueError(f"COCO label contract mismatch: {mismatches}")

    lut = appendix_layout_lut()
    coco_labels = []
    for class_id in range(133):
        layout_id = int(lut[class_id])
        coco_labels.append(
            {
                "id": class_id,
                "name": id2label[class_id],
                "layout_id": layout_id,
                "layout_name": LAYOUT_LABELS[layout_id],
            }
        )
    return {
        "source": "HouseLayout3D Appendix A, Table 5",
        "coco_label_count": 133,
        "layout_labels": [
            {
                "id": layout_id,
                "name": name,
                "color_rgb": LAYOUT_PALETTE[layout_id].tolist(),
                "coco_ids": [
                    entry["id"]
                    for entry in coco_labels
                    if entry["layout_id"] == layout_id
                ],
            }
            for layout_id, name in enumerate(LAYOUT_LABELS)
        ],
        "coco_labels": coco_labels,
    }

