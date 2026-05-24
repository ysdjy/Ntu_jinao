"""Run stage-2 skill blueprint executor for Franka pick/place data collection.

Example:
    ./isaaclab.sh -p source/standalone/franka_state_machine_cerebellum/run_skill_blueprint_executor.py \
        --task Isaac-Lift-Cube-Franka-IK-Rel-v0 \
        --num_episodes 5 \
        --headless \
        --output_dir ./datasets/state_machine_stage2_blueprint_test_5eps \
        --blueprint_path source/standalone/franka_state_machine_cerebellum/configs/example_skill_blueprint.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _bootstrap_repo_source_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    for package_dir in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
        source_path = repo_root / "source" / package_dir
        if source_path.exists() and source_path.as_posix() not in sys.path:
            sys.path.insert(0, source_path.as_posix())


_bootstrap_repo_source_paths()

from isaaclab.app import AppLauncher


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


parser = argparse.ArgumentParser(description="Stage-2 Franka skill blueprint executor.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0", help="Isaac Lab task id.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to execute.")
parser.add_argument("--output_dir", type=str, default="./datasets/state_machine_stage2_blueprint")
parser.add_argument(
    "--blueprint_path",
    type=str,
    default=str(Path(__file__).resolve().parent / "configs/example_skill_blueprint.json"),
    help="Path to stage-2 skill blueprint JSON.",
)
parser.add_argument("--watch_blueprint", type=_str_to_bool, default=False)
parser.add_argument("--reload_at", type=str, default="episode", choices=["episode", "skill"])
parser.add_argument("--target_mode", type=str, default="custom_tabletop", choices=["custom_tabletop", "official_command"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save_trajectory", type=_str_to_bool, default=True)
parser.add_argument("--max_episode_steps", type=int, default=1500)
parser.add_argument("--max_delta_pos", type=float, default=0.08)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

from skill_blueprint_loader import SkillBlueprint  # noqa: E402

blueprint_path = Path(args_cli.blueprint_path).expanduser()
try:
    parsed_blueprint = SkillBlueprint.from_json(blueprint_path)
except Exception as exc:
    parser.error(f"Failed to load --blueprint_path '{blueprint_path}': {exc}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from condition_evaluator import ConditionEvaluator  # noqa: E402
from logger import BlueprintDatasetLogger, BlueprintSummaryAccumulator  # noqa: E402
from performance_collector import PerformanceCollector  # noqa: E402
from primitive_skills import PrimitiveSkillExecutor  # noqa: E402
from skill_graph_executor import SkillGraphExecutor  # noqa: E402
from skill_interface import SkillPlan  # noqa: E402
from state_machine_cerebellum import StateMachineCerebellum  # noqa: E402


def main() -> None:
    if args_cli.watch_blueprint and args_cli.reload_at == "skill":
        print("[WARN] --reload_at skill is reserved; first version reloads only at episode boundaries.")

    torch.manual_seed(args_cli.seed)
    output_dir = Path(args_cli.output_dir).expanduser()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task and hasattr(env_cfg, "commands") and env_cfg.commands is not None:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        if args_cli.target_mode == "custom_tabletop":
            env_cfg.commands.object_pose.debug_vis = False
    env_cfg.episode_length_s = max(
        float(env_cfg.episode_length_s),
        args_cli.max_episode_steps * float(env_cfg.sim.dt) * int(env_cfg.decimation),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    target_marker = None
    if args_cli.target_mode == "custom_tabletop":
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Stage2BlueprintTabletopTarget"
        marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        target_marker = VisualizationMarkers(marker_cfg)
        target_marker.set_visibility(True)

    logger = BlueprintDatasetLogger(output_dir=output_dir, save_trajectory=args_cli.save_trajectory)
    summary = BlueprintSummaryAccumulator()
    blueprint = parsed_blueprint
    blueprint_mtime = blueprint_path.stat().st_mtime
    blueprint_reload_count = 0
    print(
        f"[INFO] Loaded blueprint: id={blueprint.blueprint_id} "
        f"start={blueprint.start} nodes={len(blueprint.nodes)} path={blueprint_path}"
    )

    try:
        for episode_idx in range(args_cli.num_episodes):
            if args_cli.watch_blueprint:
                current_mtime = blueprint_path.stat().st_mtime
                if current_mtime != blueprint_mtime:
                    blueprint = SkillBlueprint.from_json(blueprint_path)
                    blueprint_mtime = current_mtime
                    blueprint_reload_count += 1
                    print(f"[INFO] Reloaded blueprint at episode boundary: {blueprint_path}")

            episode_id = f"episode_{episode_idx:06d}"
            episode_seed = args_cli.seed + episode_idx
            dummy_plan = SkillPlan(task=blueprint.task, episode_id=episode_id, skill_plan=[])
            cerebellum = StateMachineCerebellum(
                env=env,
                skill_plan=dummy_plan,
                device=env.unwrapped.device,
                logger=logger,
                env_id=args_cli.task,
                target_mode=args_cli.target_mode,
                seed=args_cli.seed,
                max_episode_steps=args_cli.max_episode_steps,
                max_delta_pos=args_cli.max_delta_pos,
                max_delta_rot=0.0,
                target_marker=target_marker,
            )
            cerebellum.global_step = 0
            cerebellum.last_reward = None
            cerebellum.last_terminated = False
            cerebellum.last_truncated = False
            env.reset(seed=episode_seed)
            cerebellum._settle_scene()

            cube_pose = cerebellum._read_cube_pose_w()
            if args_cli.target_mode == "custom_tabletop":
                cerebellum.target_pose_w = cerebellum.sample_tabletop_target_pose(cube_pose, episode_seed)
            else:
                cerebellum.target_pose_w = cerebellum._read_official_command_pose_w()
            cerebellum._visualize_target_pose()

            trajectory_file = logger.start_trajectory(episode_id)
            initial_scene = _stage2_scene_state(cerebellum.get_scene_state())
            primitive_executor = PrimitiveSkillExecutor(cerebellum=cerebellum, logger=logger)
            condition_evaluator = ConditionEvaluator(cerebellum=cerebellum, episode_initial_state=initial_scene)
            performance_collector = PerformanceCollector(sim_dt=env_cfg.sim.dt, decimation=env_cfg.decimation)
            graph_executor = SkillGraphExecutor(
                blueprint=blueprint,
                cerebellum=cerebellum,
                logger=logger,
                primitive_executor=primitive_executor,
                condition_evaluator=condition_evaluator,
                performance_collector=performance_collector,
                episode_id=episode_id,
                max_episode_steps=args_cli.max_episode_steps,
            )
            episode = graph_executor.execute(
                initial_scene=initial_scene,
                env_id=args_cli.task,
                target_mode=args_cli.target_mode,
                seed=episode_seed,
                trajectory_file=trajectory_file,
            )
            logger.finish_trajectory()
            summary.add_episode(episode)
            summary.add_missing_metric_counts(performance_collector.missing_metric_counts)
            print(
                f"[INFO] {episode_id}: success={episode['final_result']['success']} "
                f"reason={episode['final_result']['failure_reason']} "
                f"steps={sum(item['num_steps'] for item in episode['skill_execution_trace'])} "
                f"samples={len(episode['skill_execution_trace'])}"
            )

        summary_dict = summary.to_dict()
        summary_dict["watch_blueprint"] = args_cli.watch_blueprint
        summary_dict["reload_at"] = args_cli.reload_at
        summary_dict["blueprint_reload_count"] = blueprint_reload_count
        logger.write_summary(summary_dict)
        print("[INFO] Stage-2 blueprint dataset summary:")
        print(f"  total episodes: {summary_dict['total_episodes']}")
        print(f"  episode success rate: {summary_dict['episode_success_rate']:.3f}")
        print(f"  predictor samples: {summary_dict['predictor_sample_count']}")
        print(f"  skill success rate: {summary_dict['skill_success_rate']}")
        print(f"  condition stats: {summary_dict['condition_stats']}")
        print(f"  missing metric counts: {summary_dict['missing_metric_counts']}")
        print(f"  blueprint reload count: {summary_dict['blueprint_reload_count']}")
        print(f"[INFO] Wrote dataset to: {output_dir.resolve()}")
    finally:
        logger.close()
        env.close()


def _stage2_scene_state(state: dict) -> dict:
    return {
        "cube_pose": state["cube_pose_w"],
        "target_pose": state["target_pose_w"],
        "ee_pose": state["ee_pose_w"],
        "gripper_width": state["gripper_width"],
        "robot_joint_pos": state["robot_joint_pos"],
        "step_index": state["step_index"],
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Skill blueprint executor failed: {exc}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
