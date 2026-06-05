from pathlib import Path

from stereo_spatial_calib.async_inference_worker import AsyncInferenceWorker
from stereo_spatial_calib.interactive_demo_runner import run_interactive_capture_and_infer


class DemoController:
    def __init__(self):
        self.status = "Idle"
        self.last_result = {}
        self.worker = AsyncInferenceWorker(self._run)

    def _run(self, **kwargs):
        self.status = "Running FoundationStereo..."
        result = run_interactive_capture_and_infer(**kwargs)
        self.last_result = result
        self.status = "Done"
        return result

    def run_full_localization(self, **kwargs):
        if not self.worker.start(**kwargs):
            self.status = "Busy"
        return self.worker

    def open_report_folder(self):
        path = Path(self.last_result.get("report_html_path", ".")).parent
        return str(path)

