"""Reward terms for Stage 1.1 constrained position reach."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .utils import STAGE1_SUCCESS_THRESHOLD, get_ee_and_target_position_env, get_joint_limit_terms



def stage1_position_reach_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Exact reward formula requested for Stage 1.1."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    ee_position, target_position = get_ee_and_target_position_env(env, asset_cfg, command_name)
    pos_error_vec = target_position - ee_position
    pos_error = torch.norm(pos_error_vec, dim=1)

    # Stage 1.1A: shape reward for coarse position convergence.
    r_pos = torch.exp(-5.0 * pos_error)
    r_dist = -2.0 * pos_error
    r_success = torch.where(
        pos_error < STAGE1_SUCCESS_THRESHOLD,
        torch.full_like(pos_error, 10.0),
        torch.zeros_like(pos_error),
    )

    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    dq = robot.data.joint_vel[:, joint_ids]

    q = robot.data.joint_pos[:, joint_ids]
    joint_lower, joint_upper, joint_range = get_joint_limit_terms(robot, joint_ids)
    margin_to_lower = q - joint_lower
    margin_to_upper = joint_upper - q
    joint_limit_margin = torch.minimum(margin_to_lower, margin_to_upper) / joint_range

    r_action = -0.005 * torch.mean(torch.square(action), dim=1)
    r_smooth = -0.005 * torch.mean(torch.square(action - prev_action), dim=1)
    r_dq = -0.0005 * torch.mean(torch.square(dq), dim=1)
    margin_threshold = 0.08
    limit_violation = torch.relu(margin_threshold - joint_limit_margin)
    r_limit = -0.05 * torch.mean(torch.square(limit_violation), dim=1)

    # Per-env progress reward cache.
    if (
        not hasattr(env, "stage1_prev_pos_error")
        or env.stage1_prev_pos_error is None
        or env.stage1_prev_pos_error.shape != pos_error.shape
    ):
        env.stage1_prev_pos_error = pos_error.detach().clone()
    reset_mask = env.episode_length_buf == 0
    env.stage1_prev_pos_error[reset_mask] = pos_error.detach()[reset_mask]
    progress = env.stage1_prev_pos_error - pos_error.detach()
    progress = torch.clamp(progress, -0.05, 0.05)
    r_progress = 5.0 * progress
    env.stage1_prev_pos_error = pos_error.detach().clone()

    reward = 2.0 * r_pos + 1.0 * r_dist + r_success + r_progress + r_action + r_smooth + r_dq + r_limit

    # Stage-1 monitoring metrics for training logs.
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["stage1/mean_position_error"] = pos_error.mean()
    env.extras["log"]["stage1/success_rate"] = (pos_error < STAGE1_SUCCESS_THRESHOLD).float().mean()
    env.extras["log"]["stage1/progress_reward_mean"] = r_progress.mean()
    env.extras["log"]["stage1/success_threshold"] = torch.tensor(STAGE1_SUCCESS_THRESHOLD, device=pos_error.device)
    env.extras["log"]["stage1/mean_action_magnitude"] = torch.mean(torch.abs(action))
    env.extras["log"]["stage1/mean_action_smoothness"] = torch.mean(torch.square(action - prev_action))
    env.extras["log"]["stage1/mean_joint_velocity"] = torch.mean(torch.abs(dq))
    env.extras["log"]["stage1/mean_joint_limit_margin"] = joint_limit_margin.mean()

    return reward

