from __future__ import annotations

import numpy as np

from stereo_spatial_calib.coordinate_transform import (
    compose_transform,
    invert_transform,
    transform_point,
    transform_points,
)


def test_identity_transform():
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(transform_points(np.eye(4), pts), pts)


def test_translation_transform():
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [1.0, -2.0, 0.5]
    out = transform_point(T, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out, [2.0, 0.0, 3.5], atol=1e-6)


def test_inverse_transform():
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [1.0, 2.0, 3.0]
    inv = invert_transform(T)
    p = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    np.testing.assert_allclose(transform_point(inv, transform_point(T, p)), p, atol=1e-6)


def test_compose_transform():
    T_ab = np.eye(4, dtype=np.float32)
    T_bc = np.eye(4, dtype=np.float32)
    T_ab[:3, 3] = [1.0, 0.0, 0.0]
    T_bc[:3, 3] = [0.0, 2.0, 0.0]
    T_ac = compose_transform(T_ab, T_bc)
    np.testing.assert_allclose(transform_point(T_ac, [0.0, 0.0, 0.0]), [1.0, 2.0, 0.0], atol=1e-6)

