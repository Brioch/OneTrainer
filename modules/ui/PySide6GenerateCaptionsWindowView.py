import threading
import traceback

from modules.ui.BaseGenerateCaptionsWindowView import BaseGenerateCaptionsWindowView
from modules.ui.GenerateCaptionsWindowController import GenerateCaptionsWindowController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QPushButton, QWidget

_PAD = pyside6_components.PAD


class PySide6GenerateCaptionsWindowView(BaseGenerateCaptionsWindowView, QDialog, metaclass=QtABCMeta):
    def __init__(self, parent, controller: GenerateCaptionsWindowController):
        QDialog.__init__(self, parent)
        BaseGenerateCaptionsWindowView.__init__(self, pyside6_components)

        self.controller = controller
        self._running = False
        ui_state = PySide6UIState(controller.data)

        self.setWindowTitle("Batch generate captions")
        self.setMinimumWidth(420)

        frame = QWidget(self)
        lo = pyside6_components._layout(self)
        lo.setContentsMargins(_PAD, _PAD, _PAD, _PAD)
        lo.addWidget(frame, 0, 0)
        frame_lo = pyside6_components._layout(frame)
        frame_lo.setColumnStretch(1, 1)
        self.build_form(frame, controller, ui_state)

        # Enter in the text fields must not trigger a dialog default button
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

    def schedule_on_main_thread(self, fn):
        QTimer.singleShot(0, self, fn)

    def run_job(self, fn):
        if self._running:
            return
        self._running = True
        self.create_button.setEnabled(False)

        def _run():
            try:
                fn()
            except Exception:
                traceback.print_exc()
            finally:
                self.schedule_on_main_thread(self._on_job_finished)

        threading.Thread(target=_run, daemon=True).start()

    def _on_job_finished(self):
        self._running = False
        self.create_button.setEnabled(True)

    def set_progress(self, value, max_value):
        # Called from the worker thread — defer to main thread
        self.schedule_on_main_thread(lambda: self._do_set_progress(value, max_value))

    def _do_set_progress(self, value, max_value):
        self.progress.setValue(int(value / max_value * 1000) if max_value else 0)
        self.progress_label.setText(f"Progress: {value}/{max_value}")

    def reject(self):
        # don't tear down the dialog while the worker thread is still using it
        if not self._running:
            super().reject()
