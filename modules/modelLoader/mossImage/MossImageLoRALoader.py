from modules.model.MossImageModel import MossImageModel
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.ModelNames import ModelNames


class MossImageLoRALoader(
    LoRALoaderMixin
):
    def __init__(self):
        super().__init__()


    def load(
            self,
            model: MossImageModel,
            model_names: ModelNames,
    ):
        return self._load(model, model_names)
