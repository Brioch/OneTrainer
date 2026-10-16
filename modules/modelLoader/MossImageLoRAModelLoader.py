from modules.model.MossImageModel import MossImageModel
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mossImage.MossImageLoRALoader import MossImageLoRALoader
from modules.modelLoader.mossImage.MossImageModelLoader import MossImageModelLoader
from modules.util.enum.ModelType import ModelType

MossImageLoRAModelLoader = make_lora_model_loader(
    model_spec_map={ModelType.MOSS_IMAGE: "resources/sd_model_spec/moss_image-lora.json"},
    model_class=MossImageModel,
    model_loader_class=MossImageModelLoader,
    embedding_loader_class=None,
    lora_loader_class=MossImageLoRALoader,
)
