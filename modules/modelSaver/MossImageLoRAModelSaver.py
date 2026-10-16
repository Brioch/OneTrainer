from modules.model.MossImageModel import MossImageModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.mossImage.MossImageLoRASaver import MossImageLoRASaver
from modules.util.enum.ModelType import ModelType

MossImageLoRAModelSaver = make_lora_model_saver(
    ModelType.MOSS_IMAGE,
    model_class=MossImageModel,
    lora_saver_class=MossImageLoRASaver,
    embedding_saver_class=None,
)
