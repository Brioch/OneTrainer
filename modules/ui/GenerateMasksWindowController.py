class GenerateMasksWindowController:
    models = ["ClipSeg", "Rembg", "Rembg-Human", "Hex Color"]
    modes = {
        "Replace all masks": "replace",
        "Create if absent": "fill",
        "Add to existing": "add",
        "Subtract from existing": "subtract",
        "Blend with existing": "blend",
    }

    def __init__(self, parent):
        self.parent = parent
        self.view = None
        self.data = {}

    def create_window(self, parent_window, path, parent_include_subdirectories, view_cls):
        self.data = {
            "model": "ClipSeg",
            "path": path or "",
            "prompt": "",
            "mode": "Create if absent",
            "threshold": "0.3",
            "smooth": "5",
            "expand": "10",
            "alpha": "1",
            "include_subdirectories": bool(parent_include_subdirectories),
        }
        self.view = view_cls(parent_window, self)
        return self.view

    def create_masks(self):
        self.parent.load_masking_model(self.data["model"])

        self.parent.masking_model.mask_folder(
            sample_dir=self.data["path"],
            prompts=[self.data["prompt"]],
            mode=self.modes[self.data["mode"]],
            alpha=float(self.data["alpha"]),
            threshold=float(self.data["threshold"]),
            smooth_pixels=int(self.data["smooth"]),
            expand_pixels=int(self.data["expand"]),
            progress_callback=self.view.set_progress,
            include_subdirectories=self.data["include_subdirectories"],
        )
