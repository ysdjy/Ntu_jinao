"""Collect measured performance labels for stage-2 predictor datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any
import math

from skill_blueprint_schema import OPTIONAL_UNAVAILABLE_METRICS, PERFORMANCE_METRICS

_NULL_IS_VALID_METRICS = {"failure_reason"}


class PerformanceCollector:
    """Builds one predictor training sample per executed skill or parallel node."""

    def __init__(self, sim_dt: float, decimation: int):
        self.step_time = float(sim_dt) * int(decimation)
        self.missing_metric_counts: Counter[str] = Counter()

    def collect(self, episode_id: str, blueprint_id: str, node, result) -> dict[str, Any]:
        requested = list(node.performance_query)
        measured: dict[str, Any] = {}
        missing: dict[str, str] = {}
        available = self._compute_available_metrics(node, result)

        for metric in requested:
            if metric not in PERFORMANCE_METRICS:
                measured[metric] = None
                missing[metric] = "unsupported_metric"
            elif metric in available:
                measured[metric] = available[metric]
                if available[metric] is None and metric not in _NULL_IS_VALID_METRICS:
                    missing[metric] = self._missing_reason(metric, node, result)
            else:
                measured[metric] = None
                missing[metric] = self._missing_reason(metric, node, result)

        for metric in missing:
            self.missing_metric_counts[metric] += 1

        skill_name = node.skill if node.type == "skill" else "parallel"
        target = node.target if node.type == "skill" else getattr(result, "target", "parallel")
        skill_params = dict(node.params)
        if node.type == "parallel":
            skill_params = {
                **skill_params,
                "parallel_mode": node.parallel_mode,
                "goals": node.goals,
            }

        return {
            "sample_id": f"{episode_id}_{node.node_id}",
            "episode_id": episode_id,
            "blueprint_id": blueprint_id,
            "node_id": node.node_id,
            "node_type": node.type,
            "skill": skill_name,
            "target": target,
            "scene_state_before": result.pre_state,
            "skill_params": skill_params,
            "performance_query": requested,
            "measured_performance": measured,
            "measured_performance_missing": missing,
            "scene_state_after": result.post_state,
        }

    def _compute_available_metrics(self, node, result) -> dict[str, Any]:
        before = result.pre_state
        after = result.post_state
        steps = result.step_records
        cube_before = before["cube_pose"]
        cube_after = after["cube_pose"]
        target_after = after["target_pose"]
        ee_after = after["ee_pose"]
        ee_positions = [row["ee_pose"][:3] for row in steps]

        final_ee_error = None
        if result.target_pose is not None:
            final_ee_error = _distance(ee_after[:3], result.target_pose[:3])

        orientation_absent = self._orientation_goal_absent(node)
        orientation_mode_none = self._orientation_mode_none(node)

        final_ee_orientation_error = getattr(result, "final_ee_orientation_error", None)
        orientation_converged = getattr(result, "orientation_converged", None)
        position_converged = getattr(result, "position_converged", None)
        position_goal_success = getattr(result, "position_goal_success", None)
        orientation_goal_success = getattr(result, "orientation_goal_success", None)

        if orientation_absent or orientation_mode_none:
            orientation_converged = True if orientation_converged is None else orientation_converged
            orientation_goal_success = True if orientation_goal_success is None else orientation_goal_success
            if final_ee_orientation_error is None:
                final_ee_orientation_error = None

        parallel_mode = getattr(result, "parallel_mode", None)
        parallel_goal_count = getattr(result, "parallel_goal_count", None)
        if node.type == "parallel":
            parallel_mode = parallel_mode or node.parallel_mode or "all_success"
            if parallel_goal_count is None and node.goals:
                parallel_goal_count = int("position_goal" in node.goals) + int("orientation_goal" in node.goals)

        metrics = {
            "success": result.success,
            "execution_steps": result.num_steps,
            "execution_time": result.num_steps * self.step_time,
            "trajectory_length": _path_length(ee_positions),
            "final_ee_position_error": final_ee_error,
            "final_ee_orientation_error": final_ee_orientation_error,
            "position_converged": position_converged,
            "orientation_converged": orientation_converged,
            "parallel_mode": parallel_mode,
            "parallel_goal_count": parallel_goal_count,
            "position_goal_success": position_goal_success,
            "orientation_goal_success": orientation_goal_success,
            "timeout": result.timeout,
            "failure_reason": result.failure_reason,
            "object_lift_delta": float(cube_after[2]) - float(cube_before[2]),
            "ee_object_distance": _distance(ee_after[:3], cube_after[:3]),
            "min_ee_object_distance": _min_ee_object_distance(steps, before, after),
            "object_target_xy_distance": _distance(cube_after[:2], target_after[:2]),
            "final_position_error": _distance(cube_after[:3], target_after[:3]),
            "object_displacement": _distance(cube_after[:3], cube_before[:3]),
            "object_stability": _object_stability([row["cube_pose"][:3] for row in steps]),
            "gripper_width_start": before["gripper_width"],
            "gripper_width_end": after["gripper_width"],
            "gripper_command_final": result.final_action[-1] if result.final_action else None,
            "max_contact_force": None,
            "collision_count": None,
            "collision_risk": None,
            "object_drop_risk": None,
        }
        if getattr(result, "orientation_error_type", None):
            metrics["orientation_error_type"] = result.orientation_error_type
        return metrics

    def _orientation_goal_absent(self, node) -> bool:
        if node.type != "parallel":
            return False
        goals = node.goals or {}
        return goals.get("orientation_goal") is None

    def _orientation_mode_none(self, node) -> bool:
        if node.type == "parallel":
            orientation_goal = (node.goals or {}).get("orientation_goal")
            if orientation_goal is None:
                return False
            mode = str((orientation_goal.get("params") or {}).get("orientation_mode", ""))
            return mode == "none"
        if node.type == "skill" and node.skill == "align_orientation":
            return str(node.params.get("orientation_mode", "")) == "none"
        return False

    def _missing_reason(self, metric: str, node, result) -> str:
        if metric == "final_ee_orientation_error":
            if self._orientation_goal_absent(node):
                return "orientation_goal_absent"
            if self._orientation_mode_none(node):
                return "orientation_mode_none"
            if getattr(result, "final_ee_orientation_error", None) is None:
                return "orientation_error_unavailable"
        if metric == "orientation_converged" and self._orientation_goal_absent(node):
            return "orientation_goal_absent"
        if metric == "orientation_goal_success" and self._orientation_goal_absent(node):
            return "orientation_goal_absent"
        if metric in OPTIONAL_UNAVAILABLE_METRICS:
            return "sensor_unavailable_todo"
        if metric == "final_ee_position_error":
            return "skill_has_no_explicit_target_pose"
        if metric == "gripper_command_final":
            return "no_action_emitted"
        if metric in {"parallel_mode", "parallel_goal_count", "position_converged", "position_goal_success"}:
            if node.type != "parallel" and metric.startswith("parallel"):
                return "not_parallel_node"
        return "not_available"


def _distance(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _path_length(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_distance(prev, cur) for prev, cur in zip(points[:-1], points[1:]))


def _min_ee_object_distance(steps: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any]) -> float:
    distances = [_distance(before["ee_pose"][:3], before["cube_pose"][:3]), _distance(after["ee_pose"][:3], after["cube_pose"][:3])]
    distances.extend(_distance(row["ee_pose"][:3], row["cube_pose"][:3]) for row in steps)
    return min(distances)


def _object_stability(cube_positions: list[list[float]]) -> float:
    if not cube_positions:
        return 0.0
    tail = cube_positions[-min(10, len(cube_positions)) :]
    final = tail[-1]
    return max(_distance(pos, final) for pos in tail)
