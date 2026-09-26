class GenerateCaptionsWindowController:
    models = ["Blip", "Blip2", "WD14 VIT v2"]
    modes = {
        "Replace all captions": "replace",
        "Create if absent": "fill",
        "Add as new line": "add",
    }

    def __init__(self, parent):
        self.parent = parent
        self.view = None
        self.data = {}

    def create_window(self, parent_window, path, parent_include_subdirectories, view_cls):
        self.data = {
            "model": "Blip",
            "path": path or "",
            "initial_caption": "",
            "caption_prefix": "",
            "caption_postfix": "",
            "mode": "Create if absent",
            "include_subdirectories": bool(parent_include_subdirectories),
        }
        self.view = view_cls(parent_window, self)
        return self.view

    def create_captions(self):
        self.parent.load_captioning_model(self.data["model"])

        self.parent.captioning_model.caption_folder(
            sample_dir=self.data["path"],
            initial_caption=self.data["initial_caption"],
            caption_prefix=self.data["caption_prefix"],
            caption_postfix=self.data["caption_postfix"],
            mode=self.modes[self.data["mode"]],
            progress_callback=self.view.set_progress,
            include_subdirectories=self.data["include_subdirectories"],
        )
