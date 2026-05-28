"""Shared utilities for Stage 1.1 constrained reach."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

# Legacy constant; Stage 1.1B uses per-episode sampled position_tolerance instead.
STAGE1_SUCCESS_THRESHOLD = 0.08

STAGE1_POS_TOL_MIN = 0.05
STAGE1_POS_TOL_MAX = 0.10

STAGE1_ORI_TOL_MIN = 0.30
STAGE1_ORI_TOL_MAX = 0.70


def get_stage1_position_tolerance(env) -> torch.Tensor:
    """Return per-env position tolerance command with shape [num_envs, 3].

    Each episode samples one scalar tolerance in [STAGE1_POS_TOL_MIN, STAGE1_POS_TOL_MAX],
    then repeats it to xyz: [tol, tol, tol].
    """
    if (
        not hasattr(env, "stage1_position_tolerance")
        or env.stage1_position_tolerance is None
        or env.stage1_position_tolerance.shape != (env.num_envs, 3)
    ):
        env.stage1_position_tolerance = torch.full(
            (env.num_envs, 3),
            STAGE1_POS_TOL_MAX,
            device=env.device,
        )
    if not hasattr(env, "stage1_tolerance_ready"):
        env.stage1_tolerance_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reset_mask = env.episode_length_buf == 0
    need_sample = reset_mask & (~env.stage1_tolerance_ready)
    if torch.any(need_sample):
        n = int(need_sample.sum().item())
        tol_scalar = torch.empty(n, 1, device=env.device).uniform_(STAGE1_POS_TOL_MIN, STAGE1_POS_TOL_MAX)
        env.stage1_position_tolerance[need_sample] = tol_scalar.repeat(1, 3)
        env.stage1_tolerance_ready[need_sample] = True

    # Arm for next episode: after the first post-reset step, allow resampling again.
    env.stage1_tolerance_ready[env.episode_length_buf == 1] = False

    return env.stage1_position_tolerance


def get_stage1_orientation_tolerance(env) -> torch.Tensor:
    """Return per-env orientation tolerance command with shape [num_envs, 3]."""
    if (
        not hasattr(env, "stage1_orientation_tolerance")
        or env.stage1_orientation_tolerance is None
        or env.stage1_orientation_tolerance.shape != (env.num_envs, 3)
    ):
        env.stage1_orientation_tolerance = torch.full(
            (env.num_envs, 3),
            STAGE1_ORI_TOL_MAX,
            device=env.device,
        )
    if not hasattr(env, "stage1_orientation_tolerance_ready"):
        env.stage1_orientation_tolerance_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reset_mask = env.episode_length_buf == 0
    need_sample = reset_mask & (~env.stage1_orientation_tolerance_ready)
    if torch.any(need_sample):
        n = int(need_sample.sum().item())
        tol_scalar = torch.empty(n, 1, device=env.device).uniform_(STAGE1_ORI_TOL_MIN, STAGE1_ORI_TOL_MAX)
        env.stage1_orientation_tolerance[need_sample] = tol_scalar.repeat(1, 3)
        env.stage1_orientation_tolerance_ready[need_sample] = True

    # Arm for next episode: after the first post-reset step, allow resampling again.
    env.stage1_orientation_tolerance_ready[env.episode_length_buf == 1] = False

    return env.stage1_orientation_tolerance


def get_ee_and_target_position_env(env, asset_cfg: SceneEntityCfg, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return EE and target positions in env-local frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    hand_id = robot.find_bodies("panda_hand")[0][0]
    ee_position_w = robot.data.body_pos_w[:, hand_id]
    target_position_w = env.command_manager.get_term(command_name).pose_command_w[:, :3]
    env_origins = env.scene.env_origins
    ee_position = ee_position_w - env_origins
    target_position = target_position_w - env_origins
    return ee_position, target_position


def get_ee_and_target_orientation(env, asset_cfg: SceneEntityCfg, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return EE and target quaternions in world frame (wxyz)."""
    robot: Articulation = env.scene[asset_cfg.name]
    hand_id = robot.find_bodies("panda_hand")[0][0]
    ee_quat_w = robot.data.body_quat_w[:, hand_id]
    target_quat_w = env.command_manager.get_term(command_name).pose_command_w[:, 3:7]
    return ee_quat_w, target_quat_w


def quat_to_axis_angle_error(current_quat: torch.Tensor, target_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute shortest-arc axis-angle orientation error from current -> target.

    Args:
        current_quat: Tensor [N, 4] in wxyz.
        target_quat: Tensor [N, 4] in wxyz.

    Returns:
        axis_angle_error: Tensor [N, 3].
        angle_error: Tensor [N], radians.
    """
    eps = 1e-8
    current = torch.nn.functional.normalize(current_quat, dim=1)
    target = torch.nn.functional.normalize(target_quat, dim=1)

    # q_err = q_target * conjugate(q_current)
    w1, x1, y1, z1 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    w2, x2, y2, z2 = current[:, 0], -current[:, 1], -current[:, 2], -current[:, 3]
    q_err = torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=1,
    )
    q_err = torch.nn.functional.normalize(q_err, dim=1)

    # Enforce shortest rotation.
    sign = torch.where(q_err[:, :1] < 0.0, -1.0, 1.0)
    q_err = q_err * sign

    w = torch.clamp(q_err[:, 0], -1.0 + eps, 1.0 - eps)
    xyz = q_err[:, 1:4]
    sin_half = torch.norm(xyz, dim=1)
    angle = 2.0 * torch.atan2(sin_half, w)
    angle = torch.clamp(angle, min=0.0, max=torch.pi)

    axis = xyz / (sin_half.unsqueeze(1) + eps)
    axis_angle = axis * angle.unsqueeze(1)
    small_mask = sin_half < 1e-6
    if torch.any(small_mask):
        axis_angle[small_mask] = 0.0
        angle = torch.where(small_mask, torch.zeros_like(angle), angle)

    return axis_angle, angle


def get_joint_limit_terms(
    robot: Articulation, joint_ids: slice | list[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft joint lower/upper limits and safe range."""
    joint_lower = robot.data.soft_joint_pos_limits[:, joint_ids, 0]
    joint_upper = robot.data.soft_joint_pos_limits[:, joint_ids, 1]
    joint_range = torch.clamp(joint_upper - joint_lower, min=1e-6)
    return joint_lower, joint_upper, joint_range

