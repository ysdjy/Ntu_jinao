"""Primitive skill executor for stage-2 Franka skill blueprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from orientation_control import (
    compute_orientation_delta_in_base_frame,
    compute_orientation_error,
    compute_target_orientation_w,
    orientation_converged,
)
from state_machine_cerebellum import CLOSE_GRIPPER, OPEN_GRIPPER, _first_bool, _first_scalar, _tensor_to_list


@dataclass
class PrimitiveSkillResult:
    """Execution result for one blueprint skill or parallel node."""

    node_id: str
    skill: str
    target: str
    params: dict[str, Any]
    start_step: int
    end_step: int
    pre_state: dict[str, Any]
    post_state: dict[str, Any]
    success: bool
    failure_reason: str | None
    timeout: bool
    num_steps: int
    target_pose: list[float] | None = None
    final_action: list[float] | None = None
    step_records: list[dict[str, Any]] = field(default_factory=list)
    node_type: str = "skill"
    position_converged: bool | None = None
    orientation_converged: bool | None = None
    final_ee_orientation_error: float | None = None
    orientation_error_type: str | None = None
    position_goal_success: bool | None = None
    orientation_goal_success: bool | None = None
    parallel_mode: str | None = None
    parallel_goal_count: int | None = None
    target_orientation: list[float] | None = None

    def to_trace(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("step_records", None)
        data["pre_state"] = self.pre_state
        data["post_state"] = self.post_state
        return data


class PrimitiveSkillExecutor:
    """Executes blueprint primitive skills through the existing IK-Rel environment."""

    def __init__(self, cerebellum, logger):
        self.cerebellum = cerebellum
        self.logger = logger
        self.device = cerebellum.device
        self.env = cerebellum.env
        self.unwrapped = cerebellum.unwrapped
        self.last_gripper_command = OPEN_GRIPPER

    def execute(self, node) -> PrimitiveSkillResult:
        if node.type == "parallel":
            return self.execute_parallel(node)
        skill = str(node.skill)
        if skill == "move_above":
            return self._execute_move_above(node)
        if skill == "reach":
            return self._execute_reach(node)
        if skill == "descend":
            return self._execute_descend(node)
        if skill == "grasp":
            return self._execute_grasp(node)
        if skill == "lift":
            return self._execute_lift(node)
        if skill == "place":
            return self._execute_place(node)
        if skill == "retreat":
            return self._execute_retreat(node)
        if skill == "wait":
            return self._execute_wait(node)
        if skill == "align_orientation":
            return self._execute_align_orientation(node)
        return self._empty_result(node, success=False, failure_reason="unsupported_skill")

    def execute_parallel(self, node) -> PrimitiveSkillResult:
        """Execute a parallel node: one env.step per loop, converging position and orientation."""

        goals = node.goals or {}
        position_goal = goals.get("position_goal")
        orientation_goal = goals.get("orientation_goal")
        node_params = dict(node.params)
        timeout_steps = int(node_params.get("timeout_steps", 400))
        gripper_mode = str(node_params.get("gripper", "keep"))
        gripper_command = self._resolve_gripper_command(gripper_mode)

        has_position = position_goal is not None
        has_orientation = (
            orientation_goal is not None
            and str((orientation_goal.get("params") or {}).get("orientation_mode", "")) != "none"
        )
        goal_count = int(position_goal is not None) + int(orientation_goal is not None)

        position_params = dict((position_goal or {}).get("params") or {})
        orientation_params = dict((orientation_goal or {}).get("params") or {})
        position_tolerance = float(position_params.get("position_tolerance", 0.025)) if has_position else 0.0
        position_speed = float(position_params.get("speed", 0.08)) if has_position else 0.0
        orientation_tolerance = float(orientation_params.get("orientation_tolerance", 0.08)) if orientation_goal else 0.0
        angular_speed = float(orientation_params.get("angular_speed", 0.08)) if orientation_goal else 0.0
        yaw_only = bool(orientation_params.get("keep_top_down", True)) if orientation_goal else False

        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        target_position = self._compute_position_target_from_goal(position_goal) if has_position else None
        target_orientation_w: torch.Tensor | None = None
        final_action: list[float] | None = None

        position_ok = not has_position
        orientation_ok = not has_orientation
        pos_error = 0.0
        orient_error: float | None = None
        orient_error_type: str | None = None
        timed_out = False
        failure_reason: str | None = None

        for _ in range(max(0, timeout_steps)):
            ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
            if has_position and target_position is not None:
                pos_error = float(torch.norm(target_position[:, :3] - ee_pos, dim=-1)[0].item())
                position_ok = pos_error <= position_tolerance
            if orientation_goal is not None:
                if not has_orientation:
                    orientation_ok = True
                else:
                    target_orientation_w = compute_target_orientation_w(
                        self.cerebellum,
                        str(orientation_goal.get("target", "")),
                        str(orientation_params.get("orientation_mode", "none")),
                        orientation_params,
                        ee_quat,
                        self._target_pose,
                    )
                    if target_orientation_w is None:
                        orientation_ok = True
                        orient_error = None
                    else:
                        orient_error, orient_error_type = compute_orientation_error(ee_quat, target_orientation_w, yaw_only=yaw_only)
                        orientation_ok = orient_error <= orientation_tolerance

            if position_ok and orientation_ok:
                break

            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            if has_position and target_position is not None and not position_ok:
                delta_b = self.cerebellum._compute_position_delta_in_base_frame(target_position[:, :3], ee_pos)
                action[:, :3] = torch.clamp(delta_b, min=-abs(position_speed), max=abs(position_speed))
            if has_orientation and target_orientation_w is not None and not orientation_ok:
                action[:, 3:6] = compute_orientation_delta_in_base_frame(
                    self.cerebellum, ee_quat, target_orientation_w, angular_speed
                )
            action[:, 6] = gripper_command
            final_action = _tensor_to_list(action[0])

            position_goal_log = None
            if has_position and target_position is not None:
                position_goal_log = {
                    "target": str((position_goal or {}).get("target", "")),
                    "target_position": _tensor_to_list(target_position[0, :3]),
                    "position_error": pos_error,
                }
            orientation_goal_log = None
            if orientation_goal is not None:
                orientation_goal_log = {
                    "target": str(orientation_goal.get("target", "")),
                    "orientation_mode": str(orientation_params.get("orientation_mode", "none")),
                    "target_orientation": _tensor_to_list(target_orientation_w[0]) if target_orientation_w is not None else None,
                    "orientation_error": orient_error,
                }

            self._step(
                action,
                node.node_id,
                "parallel",
                "PARALLEL",
                step_records,
                node_type="parallel",
                position_goal=position_goal_log,
                orientation_goal=orientation_goal_log,
            )
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                failure_reason = "env_done"
                break
        else:
            timed_out = not (position_ok and orientation_ok)
            if timed_out:
                if not orientation_ok and has_orientation:
                    failure_reason = "orientation_not_converged"
                elif not position_ok and has_position:
                    failure_reason = "position_not_converged"
                else:
                    failure_reason = "parallel_timeout"

        success = position_ok and orientation_ok and failure_reason is None
        post_state = self._scene_state()
        parallel_mode = str(node.parallel_mode or "all_success")
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill="parallel",
            target=self._parallel_target_label(goals),
            params=dict(node_params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=success,
            failure_reason=None if success else failure_reason,
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=_tensor_to_list(target_position[0]) if target_position is not None else None,
            target_orientation=_tensor_to_list(target_orientation_w[0]) if target_orientation_w is not None else None,
            final_action=final_action,
            step_records=step_records,
            node_type="parallel",
            position_converged=position_ok,
            orientation_converged=orientation_ok,
            final_ee_orientation_error=orient_error,
            orientation_error_type=orient_error_type,
            position_goal_success=position_ok,
            orientation_goal_success=orientation_ok if orientation_goal is not None else None,
            parallel_mode=parallel_mode,
            parallel_goal_count=goal_count,
        )

    def _execute_align_orientation(self, node) -> PrimitiveSkillResult:
        params = node.params
        orientation_mode = str(params.get("orientation_mode", "none"))
        if orientation_mode == "none":
            state = self._scene_state()
            step = self.cerebellum.global_step
            return PrimitiveSkillResult(
                node_id=node.node_id,
                skill=str(node.skill),
                target=str(node.target),
                params=dict(params),
                start_step=step,
                end_step=step,
                pre_state=state,
                post_state=state,
                success=True,
                failure_reason=None,
                timeout=False,
                num_steps=0,
                orientation_converged=True,
                orientation_goal_success=True,
                final_ee_orientation_error=None,
                orientation_error_type="none",
            )

        tolerance = float(params.get("orientation_tolerance", 0.08))
        angular_speed = float(params.get("angular_speed", 0.08))
        timeout_steps = int(params.get("timeout_steps", 200))
        yaw_only = bool(params.get("keep_top_down", True))
        gripper_command = self.last_gripper_command

        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        final_action: list[float] | None = None
        target_orientation_w: torch.Tensor | None = None
        orient_error: float | None = None
        orient_error_type: str | None = None
        converged = False
        timed_out = False

        for _ in range(max(0, timeout_steps)):
            ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
            target_orientation_w = compute_target_orientation_w(
                self.cerebellum,
                str(node.target),
                orientation_mode,
                params,
                ee_quat,
                self._target_pose,
            )
            if target_orientation_w is None or orientation_mode == "keep_current":
                converged = True
                break
            orient_error, orient_error_type = compute_orientation_error(ee_quat, target_orientation_w, yaw_only=yaw_only)
            if orient_error <= tolerance:
                converged = True
                break
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, 3:6] = compute_orientation_delta_in_base_frame(
                self.cerebellum, ee_quat, target_orientation_w, angular_speed
            )
            action[:, 6] = gripper_command
            final_action = _tensor_to_list(action[0])
            orientation_goal_log = {
                "target": str(node.target),
                "orientation_mode": orientation_mode,
                "target_orientation": _tensor_to_list(target_orientation_w[0]),
                "orientation_error": orient_error,
            }
            self._step(
                action,
                node.node_id,
                str(node.skill),
                "ALIGN_ORIENTATION",
                step_records,
                orientation_goal=orientation_goal_log,
            )
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                break
        else:
            timed_out = not converged

        post_state = self._scene_state()
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target),
            params=dict(params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=converged,
            failure_reason=None if converged else ("orientation_not_converged" if not timed_out else "timeout"),
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_orientation=_tensor_to_list(target_orientation_w[0]) if target_orientation_w is not None else None,
            final_action=final_action,
            step_records=step_records,
            orientation_converged=converged,
            orientation_goal_success=converged,
            final_ee_orientation_error=orient_error,
            orientation_error_type=orient_error_type,
        )

    def _execute_move_above(self, node) -> PrimitiveSkillResult:
        params = node.params
        target_pose = self._target_pose(str(node.target))
        xy_offset = params.get("xy_offset", [0.0, 0.0])
        target_pose[:, 0] += float(xy_offset[0])
        target_pose[:, 1] += float(xy_offset[1])
        target_pose[:, 2] += float(params.get("height_offset", 0.12))
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.025)),
            timeout_steps=int(params.get("timeout_steps", 250)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=self.last_gripper_command,
            state_name="MOVE_ABOVE",
        )

    def _execute_reach(self, node) -> PrimitiveSkillResult:
        params = node.params
        if "target_pose" in params:
            target_pose = self._pose_from_xyz(params["target_pose"][:3])
        else:
            target_ref = str(params.get("target_ref", node.target or "target"))
            target_pose = self._target_pose(target_ref)
            offset = params.get("offset", [0.0, 0.0, 0.0])
            target_pose[:, :3] += torch.tensor(offset[:3], device=self.device).reshape(1, 3)
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.025)),
            timeout_steps=int(params.get("timeout_steps", 250)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=self.last_gripper_command,
            state_name="REACH",
        )

    def _execute_descend(self, node) -> PrimitiveSkillResult:
        params = node.params
        ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
        target_pose = torch.cat([ee_pos.clone(), ee_quat.clone()], dim=-1)
        if params.get("target_height") is not None:
            target_pose[:, 2] = float(params["target_height"])
        elif params.get("relative_z") is not None:
            target_pose[:, 2] = ee_pos[:, 2] - abs(float(params["relative_z"]))
        else:
            ref_pose = self._target_pose(str(node.target))
            target_pose[:, 2] = ref_pose[:, 2]
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.015)),
            timeout_steps=int(params.get("timeout_steps", 150)),
            speed=float(params.get("speed", 0.035)),
            gripper_command=self.last_gripper_command,
            state_name="DESCEND",
        )

    def _execute_grasp(self, node) -> PrimitiveSkillResult:
        params = node.params
        wait_steps = min(int(params.get("close_wait_steps", 30)), int(params.get("timeout_steps", 80)))
        result = self._run_hold_skill(node, CLOSE_GRIPPER, wait_steps, "GRASP")
        self.last_gripper_command = CLOSE_GRIPPER
        return result

    def _execute_lift(self, node) -> PrimitiveSkillResult:
        params = node.params
        ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
        target_pose = torch.cat([ee_pos.clone(), ee_quat.clone()], dim=-1)
        target_pose[:, 2] = ee_pos[:, 2] + float(params.get("lift_height", 0.18))
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.025)),
            timeout_steps=int(params.get("timeout_steps", 200)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=CLOSE_GRIPPER,
            state_name="LIFT",
        )

    def _execute_place(self, node) -> PrimitiveSkillResult:
        params = node.params
        target_pose = self._target_pose(str(node.target))
        target_pose[:, 2] = target_pose[:, 2] + float(params.get("place_height", 0.045))
        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        final_action: list[float] | None = None
        timeout_steps = int(params.get("timeout_steps", 300))
        move_budget = max(1, timeout_steps - int(params.get("open_wait_steps", 30)))
        reached = self._move_to_pose(
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.035)),
            max_steps=move_budget,
            speed=float(params.get("speed", 0.08)),
            gripper_command=CLOSE_GRIPPER,
            node_id=node.node_id,
            active_skill=str(node.skill),
            state_name="PLACE_MOVE",
            step_records=step_records,
        )
        if step_records:
            final_action = step_records[-1].get("action")
        timed_out = not reached
        failure_reason = None if reached else "timeout"
        if reached:
            hold_steps = min(int(params.get("open_wait_steps", 30)), max(0, timeout_steps - len(step_records)))
            self.last_gripper_command = OPEN_GRIPPER
            final_action = self._hold(OPEN_GRIPPER, hold_steps, node.node_id, str(node.skill), "PLACE_OPEN", step_records)
        post_state = self._scene_state()
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target),
            params=dict(node.params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=reached,
            failure_reason=failure_reason,
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=_tensor_to_list(target_pose[0]),
            final_action=final_action,
            step_records=step_records,
        )

    def _execute_retreat(self, node) -> PrimitiveSkillResult:
        params = node.params
        ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
        target_pose = torch.cat([ee_pos.clone(), ee_quat.clone()], dim=-1)
        target_pose[:, 2] = ee_pos[:, 2] + float(params.get("retreat_height", 0.15))
        return self._run_motion_skill(
            node=node,
            target_pose=target_pose,
            tolerance=float(params.get("position_tolerance", 0.035)),
            timeout_steps=int(params.get("timeout_steps", 150)),
            speed=float(params.get("speed", 0.08)),
            gripper_command=OPEN_GRIPPER,
            state_name="RETREAT",
        )

    def _execute_wait(self, node) -> PrimitiveSkillResult:
        params = node.params
        gripper = str(params.get("gripper", "keep"))
        if gripper == "open":
            command = OPEN_GRIPPER
        elif gripper == "close":
            command = CLOSE_GRIPPER
        else:
            command = self.last_gripper_command
        return self._run_hold_skill(node, command, int(params.get("wait_steps", 30)), "WAIT")

    def _run_motion_skill(
        self,
        node,
        target_pose: torch.Tensor,
        tolerance: float,
        timeout_steps: int,
        speed: float,
        gripper_command: float,
        state_name: str,
    ) -> PrimitiveSkillResult:
        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        reached = self._move_to_pose(
            target_pose=target_pose,
            tolerance=tolerance,
            max_steps=timeout_steps,
            speed=speed,
            gripper_command=gripper_command,
            node_id=node.node_id,
            active_skill=str(node.skill),
            state_name=state_name,
            step_records=step_records,
        )
        self.last_gripper_command = gripper_command
        post_state = self._scene_state()
        timed_out = not reached
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target),
            params=dict(node.params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=reached,
            failure_reason=None if reached else "timeout",
            timeout=timed_out,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=_tensor_to_list(target_pose[0]),
            final_action=step_records[-1].get("action") if step_records else None,
            step_records=step_records,
        )

    def _run_hold_skill(self, node, gripper_command: float, steps: int, state_name: str) -> PrimitiveSkillResult:
        pre_state = self._scene_state()
        start_step = self.cerebellum.global_step
        step_records: list[dict[str, Any]] = []
        final_action = self._hold(gripper_command, steps, node.node_id, str(node.skill), state_name, step_records)
        post_state = self._scene_state()
        self.last_gripper_command = gripper_command
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target or "current"),
            params=dict(node.params),
            start_step=start_step,
            end_step=self.cerebellum.global_step,
            pre_state=pre_state,
            post_state=post_state,
            success=not self.cerebellum._episode_done(),
            failure_reason="env_done" if self.cerebellum._episode_done() else None,
            timeout=False,
            num_steps=self.cerebellum.global_step - start_step,
            target_pose=None,
            final_action=final_action,
            step_records=step_records,
        )

    def _move_to_pose(
        self,
        target_pose: torch.Tensor,
        tolerance: float,
        max_steps: int,
        speed: float,
        gripper_command: float,
        node_id: str,
        active_skill: str,
        state_name: str,
        step_records: list[dict[str, Any]],
    ) -> bool:
        for _ in range(max(0, max_steps)):
            ee_pos, _ = self.cerebellum._read_ee_pose_w()
            if torch.norm(target_pose[:, :3] - ee_pos, dim=-1)[0].item() < tolerance:
                return True
            delta_b = self.cerebellum._compute_position_delta_in_base_frame(target_pose[:, :3], ee_pos)
            delta_b = torch.clamp(delta_b, min=-abs(speed), max=abs(speed))
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, :3] = delta_b
            # Stage 2 first version keeps orientation deltas at zero as requested.
            action[:, 6] = gripper_command
            self._step(action, node_id, active_skill, state_name, step_records)
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                return False
        ee_pos, _ = self.cerebellum._read_ee_pose_w()
        return torch.norm(target_pose[:, :3] - ee_pos, dim=-1)[0].item() < tolerance

    def _hold(
        self,
        gripper_command: float,
        steps: int,
        node_id: str,
        active_skill: str,
        state_name: str,
        step_records: list[dict[str, Any]],
    ) -> list[float] | None:
        final_action = None
        for _ in range(max(0, steps)):
            action = torch.zeros((self.unwrapped.num_envs, 7), device=self.device)
            action[:, 6] = gripper_command
            final_action = _tensor_to_list(action[0])
            self._step(action, node_id, active_skill, state_name, step_records)
            if self.cerebellum._episode_done() or self.cerebellum.global_step >= self.cerebellum.max_episode_steps:
                break
        return final_action

    def _step(
        self,
        action: torch.Tensor,
        node_id: str,
        active_skill: str,
        state_name: str,
        step_records: list[dict[str, Any]],
        node_type: str | None = None,
        position_goal: dict[str, Any] | None = None,
        orientation_goal: dict[str, Any] | None = None,
    ) -> None:
        result = self.env.step(action)
        if len(result) == 5:
            _, reward, terminated, truncated, _ = result
        elif len(result) == 4:
            _, reward, dones, _ = result
            terminated = dones
            truncated = torch.zeros_like(dones, dtype=torch.bool)
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(result)}")
        self.cerebellum.last_reward = _first_scalar(reward)
        self.cerebellum.last_terminated = _first_bool(terminated)
        self.cerebellum.last_truncated = _first_bool(truncated)
        scene_state = self._scene_state()
        row = {
            "global_step": self.cerebellum.global_step,
            "node_id": node_id,
            "node_type": node_type,
            "active_skill": active_skill,
            "state_name": state_name,
            "ee_pose": scene_state["ee_pose"],
            "cube_pose": scene_state["cube_pose"],
            "target_pose": scene_state["target_pose"],
            "gripper_width": scene_state["gripper_width"],
            "action": _tensor_to_list(action[0]),
            "reward": self.cerebellum.last_reward,
            "terminated": self.cerebellum.last_terminated,
            "truncated": self.cerebellum.last_truncated,
            "position_goal": position_goal,
            "orientation_goal": orientation_goal,
        }
        step_records.append(row)
        self.logger.log_trajectory_step(row)
        self.cerebellum.global_step += 1

    def _compute_position_target_from_goal(self, position_goal: dict[str, Any] | None) -> torch.Tensor | None:
        if position_goal is None:
            return None
        skill = str(position_goal.get("skill", ""))
        target = str(position_goal.get("target", ""))
        params = dict(position_goal.get("params") or {})
        if skill == "move_above":
            target_pose = self._target_pose(target)
            xy_offset = params.get("xy_offset", [0.0, 0.0])
            target_pose[:, 0] += float(xy_offset[0])
            target_pose[:, 1] += float(xy_offset[1])
            target_pose[:, 2] += float(params.get("height_offset", 0.12))
            return target_pose
        if skill == "reach":
            if "target_pose" in params:
                return self._pose_from_xyz(params["target_pose"][:3])
            target_ref = str(params.get("target_ref", target))
            target_pose = self._target_pose(target_ref)
            offset = params.get("offset", [0.0, 0.0, 0.0])
            target_pose[:, :3] += torch.tensor(offset[:3], device=self.device).reshape(1, 3)
            return target_pose
        raise ValueError(f"Unsupported position_goal skill: {skill}")

    def _resolve_gripper_command(self, gripper_mode: str) -> float:
        if gripper_mode == "open":
            return OPEN_GRIPPER
        if gripper_mode == "close":
            return CLOSE_GRIPPER
        return self.last_gripper_command

    def _parallel_target_label(self, goals: dict[str, Any]) -> str:
        labels = []
        position_goal = goals.get("position_goal")
        orientation_goal = goals.get("orientation_goal")
        if position_goal:
            labels.append(str(position_goal.get("target", "")))
        if orientation_goal:
            labels.append(str(orientation_goal.get("target", "")))
        return "+".join(labels) if labels else "parallel"

    def _target_pose(self, target: str) -> torch.Tensor:
        if target == "cube":
            return self.cerebellum._read_cube_pose_w().clone()
        if target == "target":
            return self.cerebellum._get_target_pose_w().clone()
        if target == "current":
            ee_pos, ee_quat = self.cerebellum._read_ee_pose_w()
            return torch.cat([ee_pos, ee_quat], dim=-1)
        raise ValueError(f"Unsupported target reference: {target}")

    def _pose_from_xyz(self, xyz: list[float]) -> torch.Tensor:
        pose = torch.zeros((1, 7), device=self.device)
        pose[:, :3] = torch.tensor(xyz[:3], device=self.device).reshape(1, 3)
        pose[:, 3] = 1.0
        return pose

    def _scene_state(self) -> dict[str, Any]:
        state = self.cerebellum.get_scene_state()
        return {
            "ee_pose": state["ee_pose_w"],
            "cube_pose": state["cube_pose_w"],
            "target_pose": state["target_pose_w"],
            "gripper_width": state["gripper_width"],
            "robot_joint_pos": state["robot_joint_pos"],
            "step_index": state["step_index"],
        }

    def _empty_result(self, node, success: bool, failure_reason: str) -> PrimitiveSkillResult:
        state = self._scene_state()
        step = self.cerebellum.global_step
        return PrimitiveSkillResult(
            node_id=node.node_id,
            skill=str(node.skill),
            target=str(node.target or ""),
            params=dict(node.params),
            start_step=step,
            end_step=step,
            pre_state=state,
            post_state=state,
            success=success,
            failure_reason=failure_reason,
            timeout=False,
            num_steps=0,
        )
