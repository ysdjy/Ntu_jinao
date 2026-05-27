"""Termination terms for Stage 1.1 constrained position reach."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .utils import STAGE1_SUCCESS_THRESHOLD, get_ee_and_target_position_env


def position_success(
    env,
    threshold: float = STAGE1_SUCCESS_THRESHOLD,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Success when EE-target position norm is below threshold."""
    ee_position, target_position = get_ee_and_target_position_env(env, asset_cfg, command_name)
    pos_error = torch.norm(target_position - ee_position, dim=1)
    return pos_error < threshold


def nan_or_inf_abort(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Optional safety termination for NaN/Inf state."""
    robot: Articulation = env.scene[asset_cfg.name]
    invalid_q = ~torch.isfinite(robot.data.joint_pos).all(dim=1)
    invalid_dq = ~torch.isfinite(robot.data.joint_vel).all(dim=1)
    return torch.logical_or(invalid_q, invalid_dq)

