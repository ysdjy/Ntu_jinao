import omni.ext
import omni.ui as ui

from .demo_controller import DemoController
from .ui_builder import build_ui


class StereoSpatialDemoExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._controller = DemoController()
        self._window = ui.Window("Stereo Spatial Calibration Demo", width=420, height=560)
        with self._window.frame:
            build_ui(self._controller)

    def on_shutdown(self):
        self._window = None
        self._controller = None

