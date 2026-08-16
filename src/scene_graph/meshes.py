"""Debug mesh writers for scene graph artifacts."""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any, Sequence


def _candidate_color(polygon_id: int) -> tuple[int, int, int]:
    hue = (0.6180339887498949 * (polygon_id + 1)) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.95)
    return int(round(red * 255)), int(round(green * 255)), int(round(blue * 255))


def write_ceiling_candidate_ply(
    path: Path,
    vertices: Any,
    triangles: Any,
    triangle_polygons: Any,
    polygon_ids: Sequence[int],
) -> dict[str, int] | None:
    """Write selected prototype ceiling polygons as one colored triangle PLY."""

    selected = {int(value) for value in polygon_ids}
    if not selected:
        return None

    rows = [(triangle, int(polygon_id)) for triangle, polygon_id in zip(triangles, triangle_polygons) if int(polygon_id) in selected]
    if not rows:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_count = len(rows) * 3
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"comment ceiling candidate polygons: {len(selected)}\n")
        handle.write(f"element vertex {vertex_count}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(rows)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for triangle, polygon_id in rows:
            red, green, blue = _candidate_color(polygon_id)
            for vertex_index in triangle:
                x, y, z = vertices[int(vertex_index)]
                handle.write(f"{float(x)} {float(y)} {float(z)} {red} {green} {blue}\n")
        for index in range(len(rows)):
            base = index * 3
            handle.write(f"3 {base} {base + 1} {base + 2}\n")

    return {"vertices": vertex_count, "triangles": len(rows)}
