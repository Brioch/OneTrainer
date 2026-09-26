import contextlib
import tkinter as tk

from modules.ui.BaseGenerateMasksWindowView import BaseGenerateMasksWindowView
from modules.ui.GenerateMasksWindowController import GenerateMasksWindowController
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkGenerateMasksWindowView(BaseGenerateMasksWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: GenerateMasksWindowController, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseGenerateMasksWindowView.__init__(self, ctk_components)

        self.controller = controller
        ui_state = CtkUIState(self, controller.data)

        self.title("Batch generate masks")
        self.resizable(True, True)

        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        frame.grid_columnconfigure(1, weight=1)
        self.build_form(frame, controller, ui_state)
        self.progress.set(0)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def run_job(self, fn):
        fn()

    def set_progress(self, value, max_value):
        self.progress.set(value / max_value)
        self.progress_label.configure(text=f"Progress: {value}/{max_value}")
        self.progress.update()

    def destroy(self):
        with contextlib.suppress(tk.TclError):
            self.grab_release()

        super().destroy()
