"""Shared utilities for Stage 1.1 constrained reach."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

STAGE1_SUCCESS_THRESHOLD = 0.08


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


def get_joint_limit_terms(
    robot: Articulation, joint_ids: slice | list[int] | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft joint lower/upper limits and safe range."""
    joint_lower = robot.data.soft_joint_pos_limits[:, joint_ids, 0]
    joint_upper = robot.data.soft_joint_pos_limits[:, joint_ids, 1]
    joint_range = torch.clamp(joint_upper - joint_lower, min=1e-6)
    return joint_lower, joint_upper, joint_range

