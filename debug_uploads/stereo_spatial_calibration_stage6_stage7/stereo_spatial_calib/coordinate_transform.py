"""Homogeneous coordinate transform helpers.

Naming convention:
``T_world_cam`` maps points from camera frame to world frame:
``P_world = T_world_cam @ P_cam_homo``.
"""

from __future__ import annotations

import numpy as np


def _as_transform(T):
    arr = np.asarray(T, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"expected 4x4 transform, got {arr.shape}")
    return arr


def transform_points(T_target_source, points_source):
    T = _as_transform(T_target_source)
    pts = np.asarray(points_source, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return pts.astype(np.float32)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ homo.T).T[:, :3]
    return out.astype(np.float32)


def transform_point(T_target_source, p_source):
    return transform_points(T_target_source, np.asarray(p_source, dtype=np.float64).reshape(1, 3))[0]


def invert_transform(T):
    T_arr = _as_transform(T)
    R = T_arr[:3, :3]
    t = T_arr[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv.astype(np.float32)


def compose_transform(T_ab, T_bc):
    """Compose transforms where T_ab maps b->a and T_bc maps c->b.

    Returns T_ac mapping c->a.
    """

    return (_as_transform(T_ab) @ _as_transform(T_bc)).astype(np.float32)


def rotation_matrix_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to a scalar-first quaternion."""

    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.astype(np.float32)


def quat_wxyz_to_rotation_matrix(q):
    q_arr = np.asarray(q, dtype=np.float64).reshape(4)
    q_arr = q_arr / np.linalg.norm(q_arr)
    w, x, y, z = q_arr
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_to_rotation_matrix(axis, angle_rad):
    axis_arr = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis_arr)
    if norm < 1e-12 or abs(float(angle_rad)) < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis_arr / norm
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def perturb_transform(T_world_cam, trans_noise_std_m=0.0, rot_noise_deg=0.0, seed=0):
    """Perturb a transform reproducibly with translation noise and SO(3) rotation noise."""

    T = _as_transform(T_world_cam).copy()
    if float(trans_noise_std_m) == 0.0 and float(rot_noise_deg) == 0.0:
        return T.astype(np.float32)
    rng = np.random.default_rng(seed)
    if float(trans_noise_std_m) != 0.0:
        T[:3, 3] += rng.normal(0.0, float(trans_noise_std_m), size=3)
    if float(rot_noise_deg) != 0.0:
        axis = rng.normal(0.0, 1.0, size=3)
        angle = np.deg2rad(float(rot_noise_deg))
        R_noise = axis_angle_to_rotation_matrix(axis, angle)
        T[:3, :3] = R_noise @ T[:3, :3]
    return T.astype(np.float32)
