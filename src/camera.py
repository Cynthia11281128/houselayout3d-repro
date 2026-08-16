"""Camera parameter helpers for Nerfstudio-style transforms files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def _number(payload: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, int | float):
            return float(value)
    return None


def _payloads(transforms: dict[str, Any]) -> list[dict[str, Any]]:
    payloads = [transforms]
    frames = transforms.get("frames")
    if isinstance(frames, list):
        payloads.extend(frame for frame in frames if isinstance(frame, dict))
    return payloads


def _first_number(transforms: dict[str, Any], *names: str) -> float | None:
    for payload in _payloads(transforms):
        value = _number(payload, *names)
        if value is not None:
            return value
    return None


def load_pinhole_camera_from_transforms(path: Path) -> PinholeCamera:
    transforms = json.loads(path.read_text(encoding="utf-8"))
    width = _first_number(transforms, "w", "width")
    height = _first_number(transforms, "h", "height")
    fx = _first_number(transforms, "fl_x", "fx")
    fy = _first_number(transforms, "fl_y", "fy")
    cx = _first_number(transforms, "cx")
    cy = _first_number(transforms, "cy")
    if width is None or height is None:
        raise ValueError(f"transforms file must contain w/h camera resolution: {path}")
    if fx is None:
        camera_angle_x = _first_number(transforms, "camera_angle_x")
        if camera_angle_x is not None:
            fx = 0.5 * width / math.tan(0.5 * camera_angle_x)
    if fy is None:
        camera_angle_y = _first_number(transforms, "camera_angle_y")
        if camera_angle_y is not None:
            fy = 0.5 * height / math.tan(0.5 * camera_angle_y)
        elif fx is not None:
            fy = fx
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    if fx is None or fy is None:
        raise ValueError(
            f"transforms file must contain fl_x/fl_y or camera_angle_x/y: {path}"
        )
    camera = PinholeCamera(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )
    if min(camera.width, camera.height, camera.fx, camera.fy) <= 0:
        raise ValueError("camera dimensions and focal lengths must be positive")
    return camera
