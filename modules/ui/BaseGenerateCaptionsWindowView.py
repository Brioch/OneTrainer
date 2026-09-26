from abc import ABC, abstractmethod


class BaseGenerateCaptionsWindowView(ABC):
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

        self.components.label(frame, 2, 0, "Initial Caption")
        self.components.entry(frame, 2, 1, ui_state, "initial_caption")

        self.components.label(frame, 3, 0, "Caption Prefix")
        self.components.entry(frame, 3, 1, ui_state, "caption_prefix")

        self.components.label(frame, 4, 0, "Caption Postfix")
        self.components.entry(frame, 4, 1, ui_state, "caption_postfix")

        self.components.label(frame, 5, 0, "Mode")
        self.components.options(frame, 5, 1, list(controller.modes.keys()), ui_state, "mode")

        self.components.label(frame, 6, 0, "Include subfolders")
        self.components.switch(frame, 6, 1, ui_state, "include_subdirectories")

        self.progress_label = self.components.label(frame, 7, 0, "Progress: 0/0")
        self.progress = self.components.progress(frame, 7, 1)

        self.create_button = self.components.button(frame, 8, 0, "Create Captions", self._on_create)

    def _on_create(self):
        self.run_job(self.controller.create_captions)

    @abstractmethod
    def run_job(self, fn):
        pass

    @abstractmethod
    def set_progress(self, value, max_value):
        pass
