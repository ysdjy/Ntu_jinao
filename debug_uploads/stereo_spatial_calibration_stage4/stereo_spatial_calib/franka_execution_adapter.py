"""Reserved adapter for downstream Franka execution."""

from __future__ import annotations


class FrankaExecutionAdapter:
    """TODO: send center_base_m to IK or a cerebellum/state-machine executor."""

    def send_target(self, center_base_m, metadata=None):
        raise NotImplementedError("Robot control is intentionally out of scope for v1.")

