#!/usr/bin/env python3
"""Export every inspectable Stage09 room-segmentation intermediate as PNG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


ROOM_COLORS = np.asarray(
    [
        [35, 35, 35],
        [76, 175, 255],
        [114, 221, 114],
        [255, 170, 80],
        [203, 120, 255],
        [255, 102, 153],
        [80, 215, 215],
        [180, 160, 75],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage09_attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=3)
    return parser.parse_args()


def title_image(image: np.ndarray, title: str, subtitle: str, scale: int) -> np.ndarray:
    display = np.flipud(image)
    display = cv2.resize(
        display,
        (display.shape[1] * scale, display.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    banner = np.full((78, display.shape[1], 3), 22, dtype=np.uint8)
    cv2.putText(banner, title, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(banner, subtitle, (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 195, 205), 1, cv2.LINE_AA)
    return np.vstack((banner, display))


def write_png(output: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write {output}")


def colored_labels(labels: np.ndarray, support: np.ndarray, walls: np.ndarray) -> np.ndarray:
    image = np.full((*labels.shape, 3), 16, dtype=np.uint8)
    image[support] = (48, 48, 48)
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        image[labels == label] = ROOM_COLORS[label % len(ROOM_COLORS)]
    image[walls] = (5, 5, 5)
    return image


def grid_point(xy: list[float], origin: np.ndarray, resolution: float) -> tuple[int, int]:
    column = int(round((float(xy[0]) - float(origin[0])) / resolution))
    row = int(round((float(xy[1]) - float(origin[1])) / resolution))
    return column, row


def main() -> int:
    args = parse_args()
    attempt = args.stage09_attempt.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    grid_path = attempt / "level_00_grid.npz"
    graph_path = attempt / "scene_graph.json"
    preview_path = attempt / "level_00_rooms.png"
    if not grid_path.is_file() or not graph_path.is_file() or not preview_path.is_file():
        raise SystemExit(f"incomplete Stage09 attempt: {attempt}")

    payload = np.load(grid_path, allow_pickle=False)
    labels = np.asarray(payload["labels"], dtype=np.int32)
    distance = np.asarray(payload["distance_meters"], dtype=np.float32)
    markers = np.asarray(payload["markers"], dtype=np.int32)
    walls = np.asarray(payload["walls"], dtype=bool)
    support = np.asarray(payload["support"], dtype=bool)
    origin = np.asarray(payload["origin_xy"], dtype=np.float64)
    resolution = float(payload["resolution_meters"])
    free = support & ~walls
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    images: list[tuple[str, np.ndarray, str, str]] = []
    support_image = np.zeros((*support.shape, 3), dtype=np.uint8)
    support_image[support] = (215, 215, 215)
    images.append((
        "00_support_footprint.png",
        support_image,
        "00 - Floor / ceiling footprint",
        f"union of projected floor and ceiling polygons | support cells={int(support.sum())}",
    ))

    wall_image = np.full((*support.shape, 3), 12, dtype=np.uint8)
    wall_image[support] = (55, 55, 55)
    wall_image[walls] = (40, 40, 245)
    images.append((
        "01_projected_wall_mask.png",
        wall_image,
        "01 - Projected wall mask",
        f"selected 3D walls rasterized in BEV | wall cells={int(walls.sum())}",
    ))

    free_image = np.zeros((*support.shape, 3), dtype=np.uint8)
    free_image[free] = (225, 225, 225)
    free_image[walls] = (40, 40, 245)
    images.append((
        "02_free_space.png",
        free_image,
        "02 - Free space",
        f"support minus walls | free cells={int(free.sum())}",
    ))

    normalized = np.clip(distance / max(float(distance.max()), 1.0e-12), 0.0, 1.0)
    distance_image = cv2.applyColorMap(np.uint8(np.round(normalized * 255)), cv2.COLORMAP_TURBO)
    distance_image[~free] = 0
    images.append((
        "03_distance_transform.png",
        distance_image,
        "03 - Distance to nearest wall",
        f"Euclidean distance transform | maximum={float(distance.max()):.3f} m",
    ))

    seed_25 = free & (distance >= 2.5 / 2.0)
    seed_25_image = distance_image // 3
    seed_25_image[seed_25] = (70, 245, 70)
    images.append((
        "04_seed_mask_2p5m.png",
        seed_25_image,
        "04 - First seed mask (2.5 m bottleneck)",
        f"distance >= 1.25 m | raw seed cells={int(seed_25.sum())}",
    ))

    seed_15 = free & (distance >= 1.5 / 2.0)
    seed_15_image = distance_image // 3
    seed_15_image[seed_15] = (30, 225, 245)
    seed_15_image[seed_25] = (70, 245, 70)
    images.append((
        "05_seed_mask_1p5m.png",
        seed_15_image,
        "05 - Second seed mask (1.5 m bottleneck)",
        f"yellow: distance >= 0.75 m | green: first-scale subset | cells={int(seed_15.sum())}",
    ))

    marker_image = np.full((*support.shape, 3), 18, dtype=np.uint8)
    marker_image[free] = (52, 52, 52)
    for marker in sorted(int(value) for value in np.unique(markers) if value > 0):
        marker_image[markers == marker] = ROOM_COLORS[marker % len(ROOM_COLORS)]
    marker_image[walls] = (5, 5, 5)
    images.append((
        "06_final_markers.png",
        marker_image,
        "06 - Accepted watershed markers",
        f"stable markers from both scales | marker count={int(markers.max())}",
    ))

    label_image = colored_labels(labels, support, walls)
    images.append((
        "07_watershed_labels.png",
        label_image,
        "07 - Watershed room labels",
        f"watershed(-distance, markers, mask=free_space) | rooms={int(labels.max())}",
    ))

    boundary_image = label_image.copy()
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        contours, _ = cv2.findContours(
            np.uint8(labels == label), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(boundary_image, contours, -1, (255, 255, 255), 1, cv2.LINE_8)
    active_edges = [edge for edge in graph["edges"] if not edge.get("pruned", False)]
    for edge in active_edges:
        if "line_xy" not in edge:
            continue
        first = grid_point(edge["line_xy"][0], origin, resolution)
        second = grid_point(edge["line_xy"][1], origin, resolution)
        cv2.line(boundary_image, first, second, (255, 0, 255), 3, cv2.LINE_AA)
        center = ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)
        cv2.circle(boundary_image, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
    images.append((
        "08_room_boundaries_and_openings.png",
        boundary_image,
        "08 - Vectorized room boundaries and graph openings",
        f"white: room contours | magenta: door/opening edges | edges={len(active_edges)}",
    ))

    rendered: list[tuple[str, np.ndarray]] = []
    for name, image, title, subtitle in images:
        titled = title_image(image, title, subtitle, args.scale)
        write_png(output / name, titled)
        rendered.append((name, titled))

    saved_preview_name = "09_original_stage09_room_preview.png"
    shutil.copy2(preview_path, output / saved_preview_name)

    tile_width = max(image.shape[1] for _, image in rendered)
    tile_height = max(image.shape[0] for _, image in rendered)
    tiles = []
    for _, image in rendered:
        canvas = np.full((tile_height, tile_width, 3), 18, dtype=np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        tiles.append(canvas)
    while len(tiles) < 9:
        tiles.append(np.full((tile_height, tile_width, 3), 18, dtype=np.uint8))
    rows = [np.hstack(tiles[index : index + 3]) for index in range(0, 9, 3)]
    montage = np.vstack(rows)
    write_png(output / "10_pipeline_montage.png", montage)

    file_names = [name for name, _ in rendered] + [saved_preview_name, "10_pipeline_montage.png"]
    manifest = {
        "source_stage09_attempt": str(attempt),
        "source_grid": str(grid_path),
        "coordinate_display": "world Y points upward; source arrays are flipped vertically for display",
        "grid_shape": list(labels.shape),
        "resolution_meters": resolution,
        "support_cells": int(support.sum()),
        "wall_cells": int(walls.sum()),
        "free_space_cells": int(free.sum()),
        "room_count": int(labels.max()),
        "marker_count": int(markers.max()),
        "graph_edge_count": len(active_edges),
        "images": file_names,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cards = "\n".join(
        f'<figure><a href="{html.escape(name)}"><img src="{html.escape(name)}" loading="lazy"></a>'
        f'<figcaption>{html.escape(name)}</figcaption></figure>'
        for name in file_names
    )
    (output / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Stage09 intermediate images</title>"
        "<style>body{margin:0;background:#111827;color:#e5e7eb;font-family:system-ui;padding:24px}"
        "h1{margin-top:0}.meta{color:#9ca3af;margin-bottom:22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}"
        "figure{margin:0;background:#1f2937;border:1px solid #374151;border-radius:10px;padding:10px}"
        "img{width:100%;height:auto;display:block;background:#000;border-radius:6px}figcaption{padding:8px 2px 2px;font-family:monospace}</style>"
        "</head><body><h1>HouseLayout3D Stage09 room segmentation</h1>"
        f"<div class='meta'>grid={labels.shape[1]}x{labels.shape[0]}, resolution={resolution:.3f} m, rooms={int(labels.max())}</div>"
        f"<div class='grid'>{cards}</div></body></html>\n",
        encoding="utf-8",
    )
    print(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
