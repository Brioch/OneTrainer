from abc import ABC, abstractmethod


class BaseGenerateMasksWindowView(ABC):
    def __init__(self, components):
        self.components = components
        self.progress = None
        self.progress_label = None
        self.create_button = None

    def build_form(self, frame, controller, ui_state):
        self.components.label(frame, 0, 0, "Model")
        self.components.options(frame, 0, 1, controller.models, ui_state, "model")

        self.components.label(frame, 1, 0, "Folder")
        self.components.path_entry(frame, 1, 1, ui_state, "path", mode="dir")

        self.components.label(frame, 2, 0, "Prompt",
                              tooltip="text prompt for ClipSeg, or a hex color (e.g. #00ff00) for Hex Color")
        self.components.entry(frame, 2, 1, ui_state, "prompt")

        self.components.label(frame, 3, 0, "Mode")
        self.components.options(frame, 3, 1, list(controller.modes.keys()), ui_state, "mode")

        self.components.label(frame, 4, 0, "Threshold")
        self.components.entry(frame, 4, 1, ui_state, "threshold", placeholder="0.0 - 1.0")

        self.components.label(frame, 5, 0, "Smooth")
        self.components.entry(frame, 5, 1, ui_state, "smooth", placeholder="5")

        self.components.label(frame, 6, 0, "Expand")
        self.components.entry(frame, 6, 1, ui_state, "expand", placeholder="10")

        self.components.label(frame, 7, 0, "Alpha")
        self.components.entry(frame, 7, 1, ui_state, "alpha", placeholder="1")

        self.components.label(frame, 8, 0, "Include subfolders")
        self.components.switch(frame, 8, 1, ui_state, "include_subdirectories")

        self.progress_label = self.components.label(frame, 9, 0, "Progress: 0/0")
        self.progress = self.components.progress(frame, 9, 1)

        self.create_button = self.components.button(frame, 10, 0, "Create Masks", self._on_create)

    def _on_create(self):
        self.run_job(self.controller.create_masks)

    @abstractmethod
    def run_job(self, fn):
        pass

    @abstractmethod
    def set_progress(self, value, max_value):
        pass
