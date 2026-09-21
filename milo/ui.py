from __future__ import annotations

import threading

from .cat_face import Face


class FaceUI:
    """Connect runtime face and voice events to the cat display."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.face = Face() if self.enabled else None
        self.thread = None

    def start(self):
        if not self.face:
            return
        self.thread = threading.Thread(target=self.face.run, name="milo-cat-face", daemon=True)
        self.thread.start()

    def stop(self):
        if self.face:
            self.face.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def set_gaze(self, x, y):
        if self.face:
            self.face.set_gaze(x, y)

    def set_speaking(self, value):
        if self.face:
            self.face.set_speaking_active(value)
            self.face.set_state("speaking" if value else "neutral")

    def set_status(self, value):
        return None
