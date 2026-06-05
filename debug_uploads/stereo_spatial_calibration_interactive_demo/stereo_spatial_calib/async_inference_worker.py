"""Small threaded worker for button-triggered single-shot inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import traceback
from typing import Callable, Optional


@dataclass
class AsyncInferenceWorker:
    target: Callable
    status: str = "Idle"
    result: Optional[dict] = None
    error: str = ""
    traceback_text: str = ""
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, *args, **kwargs):
        if self.running:
            self.status = "Busy"
            return False

        def _run():
            self.status = "Running FoundationStereo..."
            self.result = None
            self.error = ""
            self.traceback_text = ""
            try:
                self.result = self.target(*args, **kwargs)
                self.status = "Done"
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.traceback_text = traceback.format_exc()
                self.status = "Failed"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True

