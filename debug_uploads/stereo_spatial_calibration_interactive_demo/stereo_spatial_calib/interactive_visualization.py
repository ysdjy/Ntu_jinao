"""Isaac viewport markers for interactive localization results."""

from __future__ import annotations

import numpy as np


def add_result_markers(center_pred_world, center_gt_world=None, marker_radius=0.015):
    """Add predicted/GT center markers and an error line to the current stage.

    This function is best-effort and import-safe outside Isaac Sim.
    """

    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except Exception:
        return {"status": "skipped", "reason": "isaac_usd_unavailable"}

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {"status": "skipped", "reason": "no_stage"}

    root = "/World/Visualization"
    UsdGeom.Xform.Define(stage, root)

    pred = np.asarray(center_pred_world, dtype=float).reshape(3)
    pred_sphere = UsdGeom.Sphere.Define(stage, f"{root}/predicted_center")
    pred_sphere.CreateRadiusAttr(float(marker_radius))
    pred_sphere.AddTranslateOp().Set(Gf.Vec3d(*pred.tolist()))

    result = {"status": "success", "predicted_center_path": f"{root}/predicted_center"}
    if center_gt_world is not None:
        gt = np.asarray(center_gt_world, dtype=float).reshape(3)
        gt_sphere = UsdGeom.Sphere.Define(stage, f"{root}/gt_center")
        gt_sphere.CreateRadiusAttr(float(marker_radius))
        gt_sphere.AddTranslateOp().Set(Gf.Vec3d(*gt.tolist()))
        line = UsdGeom.BasisCurves.Define(stage, f"{root}/error_line")
        line.CreateTypeAttr("linear")
        line.CreateCurveVertexCountsAttr([2])
        line.CreatePointsAttr([Gf.Vec3f(*pred.tolist()), Gf.Vec3f(*gt.tolist())])
        result.update({"gt_center_path": f"{root}/gt_center", "error_line_path": f"{root}/error_line"})
    return result

