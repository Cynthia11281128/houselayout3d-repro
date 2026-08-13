from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.layout_prototype.polygon_init import (
    PolygonInitError,
    extract_boundary_loops,
    fit_plane_ransac,
    polygon_area,
    validate_full_mesh_arrays,
)


def test_fit_plane_ransac_recovers_horizontal_plane() -> None:
    rng = np.random.default_rng(7)
    xy = rng.uniform(-2.0, 2.0, size=(200, 2))
    z = 1.25 + rng.normal(0.0, 0.002, size=(200, 1))
    points = np.column_stack((xy, z))

    plane, inliers = fit_plane_ransac(points, threshold=0.02, iterations=128, rng=rng)

    assert inliers.sum() == len(points)
    assert np.isclose(np.linalg.norm(plane[:3]), 1.0)
    assert abs(abs(plane[2]) - 1.0) < 1e-3
    assert np.abs(points @ plane[:3] + plane[3]).max() < 0.02


def test_extract_boundary_loops_for_square_mesh() -> None:
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    loops, boundary, branch_count = extract_boundary_loops(triangles)

    assert branch_count == 0
    assert len(boundary) == 4
    assert len(loops) == 1
    assert set(loops[0].tolist()) == {0, 1, 2, 3}


def test_polygon_area_signed_square() -> None:
    square = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

    assert polygon_area(square) == pytest.approx(4.0)
    assert polygon_area(square[::-1]) == pytest.approx(-4.0)


def test_validate_full_mesh_arrays_rejects_length_mismatch() -> None:
    with pytest.raises(PolygonInitError, match="full-mesh arrays"):
        validate_full_mesh_arrays(
            4,
            hard_labels=np.zeros(4, dtype=np.uint8),
            segmentation=np.zeros(3, dtype=np.int32),
        )
