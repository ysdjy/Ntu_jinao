"""Isaac scene construction hooks for the native camera backend.

The synthetic backend remains the deterministic acceptance path. This module isolates
native Isaac Sim startup and scene authoring so failures do not affect synthetic tests.
"""

from __future__ import annotations

import numpy as np

from .coordinate_transform import rotation_matrix_to_quat_wxyz


def _look_at_transform_cv(position_world, look_at_world, world_up=(0.0, 0.0, 1.0)):
    pos = np.asarray(position_world, dtype=np.float64)
    target = np.asarray(look_at_world, dtype=np.float64)
    forward = target - pos
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(world_up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = right.astype(np.float32)
    T[:3, 1] = down.astype(np.float32)
    T[:3, 2] = forward.astype(np.float32)
    T[:3, 3] = pos.astype(np.float32)
    return T


def isaac_available():
    try:
        import isaaclab  # noqa: F401
        import isaacsim  # noqa: F401

        return True
    except Exception:
        return False


class IsaacStereoSceneBuilder:
    """Build a minimal USD scene with ground, table, target object, and stereo cameras."""

    def __init__(self, config):
        self.config = config
        self.simulation_app = None
        self.world = None

    def launch(self):
        try:
            from isaacsim import SimulationApp
        except Exception:
            from isaacsim.simulation_app import SimulationApp

        self.simulation_app = SimulationApp({"headless": True})
        return self.simulation_app

    def _run_kit_coroutine(self, coroutine, max_updates=360):
        from omni.kit.async_engine import run_coroutine

        task = run_coroutine(coroutine)
        for _ in range(int(max_updates)):
            self.simulation_app.update()
            if task.done():
                return task.result()
        raise TimeoutError("Timed out waiting for Isaac scene initialization coroutine.")

    def build_stage(self, object_type="cube", object_position=(0.5, 0.0, 0.04)):
        """Create the USD scene primitives.

        This uses stable USD/Omniverse primitives instead of task-specific IsaacLab
        environments. RGB/depth/mask capture is handled by the collector.
        """

        if self.simulation_app is None:
            self.launch()

        try:
            return self._build_stage_with_world(object_type, object_position)
        except Exception:
            return self._build_stage_with_usd(object_type, object_position)

    def _build_stage_with_world(self, object_type, object_position):
        async def _build():
            import omni.kit.app
            import omni.usd
            from isaacsim.core.api import World
            from isaacsim.core.api.objects import FixedCuboid, VisualCuboid, VisualCylinder, VisualSphere
            from isaacsim.core.utils.semantics import add_labels
            from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
            from pxr import UsdGeom, UsdLux

            await create_new_stage_async()
            stage = omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            self.world = World(stage_units_in_meters=1.0)
            await self.world.initialize_simulation_context_async()
            await update_stage_async()

            self.world.scene.add_default_ground_plane()
            self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/Table",
                    name="table",
                    position=np.array([0.5, 0.0, 0.0]),
                    scale=np.array([0.7, 0.5, 0.04]),
                    size=1.0,
                    color=np.array([0.45, 0.38, 0.30]),
                )
            )

            pos = np.asarray(object_position, dtype=float)
            if object_type == "sphere":
                obj = self.world.scene.add(
                    VisualSphere(
                        prim_path="/World/target_sphere",
                        name="target_sphere",
                        position=pos,
                        radius=0.04,
                        color=np.array([0.10, 0.55, 1.0]),
                    )
                )
            elif object_type == "cylinder":
                obj = self.world.scene.add(
                    VisualCylinder(
                        prim_path="/World/target_cylinder",
                        name="target_cylinder",
                        position=pos,
                        radius=0.04,
                        height=0.08,
                        color=np.array([0.15, 0.8, 0.25]),
                    )
                )
            else:
                obj = self.world.scene.add(
                    VisualCuboid(
                        prim_path="/World/target_cube",
                        name="target_cube",
                        position=pos,
                        scale=np.array([0.08, 0.08, 0.08]),
                        size=1.0,
                        color=np.array([0.9, 0.1, 0.05]),
                    )
                )
            add_labels(obj.prim, labels=[f"target_{object_type}"], instance_name="class")

            light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
            light.CreateIntensityAttr(2500.0)
            light.CreateAngleAttr(0.6)

            await update_stage_async()
            await omni.kit.app.get_app().next_update_async()
            await self.world.reset_async()
            await update_stage_async()
            return stage

        return self._run_kit_coroutine(_build(), max_updates=600)

    def _build_stage_with_usd(self, object_type, object_position):
        import omni.usd
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf, UsdGeom, UsdLux

        create_new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        UsdGeom.Xform.Define(stage, "/World")
        light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        light.CreateIntensityAttr(2500.0)

        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, -0.01))
        ground.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.01))

        table = UsdGeom.Cube.Define(stage, "/World/Table")
        table.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
        table.AddScaleOp().Set(Gf.Vec3f(0.7, 0.5, 0.02))

        pos = Gf.Vec3d(*object_position)
        if object_type == "sphere":
            prim = UsdGeom.Sphere.Define(stage, "/World/target_sphere")
            prim.CreateRadiusAttr(0.04)
        elif object_type == "cylinder":
            prim = UsdGeom.Cylinder.Define(stage, "/World/target_cylinder")
            prim.CreateRadiusAttr(0.04)
            prim.CreateHeightAttr(0.08)
        else:
            prim = UsdGeom.Cube.Define(stage, "/World/target_cube")
            prim.AddScaleOp().Set(Gf.Vec3f(0.04, 0.04, 0.04))
        prim.AddTranslateOp().Set(pos)
        for _ in range(5):
            self.simulation_app.update()
        return stage

    def camera_pose(self):
        T_world_left = _look_at_transform_cv(
            self.config.get("camera_position_world", (0.65, 0.0, 0.80)),
            self.config.get("look_at_world", (0.45, 0.0, 0.05)),
        )
        baseline = float(self.config.get("baseline_m", 0.10))
        T_world_right = np.array(T_world_left, copy=True)
        T_world_right[:3, 3] += T_world_left[:3, 0] * baseline
        return T_world_left, T_world_right

    @staticmethod
    def transform_to_ros_camera_pose(T_world_cam):
        """Return an Isaac Camera pose using Isaac's ros camera axes.

        Project metadata uses OpenCV optical axes (x right, y down, z forward).
        Isaac Camera's ``camera_axes="ros"`` convention is x right, y up,
        z forward, so only the orientation passed to Isaac flips the image
        y axis. The saved ``T_world_left_cam`` remains the OpenCV transform
        used by the reconstruction pipeline.
        """

        T = np.asarray(T_world_cam, dtype=np.float32)
        R_ros = np.array(T[:3, :3], copy=True)
        R_ros[:, 1] *= -1.0
        return T[:3, 3].copy(), rotation_matrix_to_quat_wxyz(R_ros)

    def close(self):
        if self.world is not None:
            try:
                self.world.stop()
                self.world.clear_instance()
            except Exception:
                pass
            self.world = None
        if self.simulation_app is not None:
            self.simulation_app.close()
            self.simulation_app = None
