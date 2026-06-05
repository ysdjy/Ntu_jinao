"""Stereo rig helper for fixed left-reference stereo capture.

The rig convention is intentionally explicit:
- left RGB is the reference view for FoundationStereo disparity/depth.
- right RGB is only used for stereo matching.
- reconstructed point clouds start in the left camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass
class StereoRigController:
    rig_prim_path: str = "/World/StereoRig"
    baseline_m: float = 0.10
    resolution: tuple[int, int] = (640, 480)
    _rig_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    _rig_orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))
    _left_local_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    _right_local_position: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.0, 0.0], dtype=np.float64))
    _left_local_rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    _right_local_rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    def create_rig(self, rig_prim_path="/World/StereoRig", baseline_m=0.10, resolution=(640, 480), stage: Any = None):
        self.rig_prim_path = str(rig_prim_path)
        self.baseline_m = float(baseline_m)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.repair_rig()
        try:
            self._create_usd_rig(stage=stage)
        except Exception:
            pass
        return self

    def _create_usd_rig(self, stage=None):
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = stage or omni.usd.get_context().get_stage()
        if stage is None:
            return
        UsdGeom.Xform.Define(stage, self.rig_prim_path)
        for path, local_pos in (
            (self.get_left_camera_path(), self._left_local_position),
            (self.get_right_camera_path(), self._right_local_position),
        ):
            cam = UsdGeom.Camera.Define(stage, path)
            xform = UsdGeom.Xformable(cam.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*local_pos.tolist()))

    def get_left_camera_path(self):
        return f"{self.rig_prim_path}/left_camera"

    def get_right_camera_path(self):
        return f"{self.rig_prim_path}/right_camera"

    def set_rig_world_pose(self, position, orientation):
        self._rig_position = np.asarray(position, dtype=np.float64).reshape(3)
        self._rig_orientation = np.asarray(orientation, dtype=np.float64).reshape(4)

    def get_rig_world_pose(self):
        return self._rig_position.copy(), self._rig_orientation.copy()

    def repair_rig(self):
        self._left_local_position = np.zeros(3, dtype=np.float64)
        self._right_local_position = np.array([self.baseline_m, 0.0, 0.0], dtype=np.float64)
        self._left_local_rotation = np.eye(3, dtype=np.float64)
        self._right_local_rotation = np.eye(3, dtype=np.float64)
        return self.validate_baseline_and_orientation()

    def update_right_camera_from_left(self):
        self._right_local_position = self._left_local_position + np.array([self.baseline_m, 0.0, 0.0])
        self._right_local_rotation = np.array(self._left_local_rotation, copy=True)
        return self.validate_baseline_and_orientation()

    def validate_baseline_and_orientation(self):
        delta = self._right_local_position - self._left_local_position
        baseline_actual = float(np.linalg.norm(delta))
        baseline_error = baseline_actual - float(self.baseline_m)
        R = self._right_local_rotation @ self._left_local_rotation.T
        trace = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
        rotation_error_deg = float(math.degrees(math.acos(trace)))
        return {
            "rig_prim_path": self.rig_prim_path,
            "left_camera_prim_path": self.get_left_camera_path(),
            "right_camera_prim_path": self.get_right_camera_path(),
            "baseline_expected_m": float(self.baseline_m),
            "baseline_actual_m": baseline_actual,
            "baseline_error_m": baseline_error,
            "rotation_error_deg": rotation_error_deg,
            "is_rectified_like": bool(abs(baseline_error) < 1e-4 and rotation_error_deg < 1e-3),
            "stereo_rig_valid": bool(abs(baseline_error) < 1e-4 and rotation_error_deg < 1e-3),
        }

