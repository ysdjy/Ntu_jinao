"""Marker helpers for constrained reach task."""

from __future__ import annotations

from isaaclab.markers.config import SPHERE_MARKER_CFG


TARGET_MARKER_CFG = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/CerebellumRL/target")
TARGET_MARKER_CFG.markers["sphere"].radius = 0.02
TARGET_MARKER_CFG.markers["sphere"].visual_material.diffuse_color = (0.0, 0.9, 0.2)

