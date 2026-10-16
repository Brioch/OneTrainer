import os
import traceback

from modules.model.MossImageModel import MossImageModel
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.module.MossImageTransformer import MossImageTransformer
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer, Qwen3_5Model


class MossImageModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: MossImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, vae_model_name, quantization, stream_from_disk,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: MossImageModel,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        model.tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        # The repo's scheduler is a custom remote-code class. For training, it is a plain flow matching scheduler with
        # a static shift of 5. MossImageSampler builds the vendor's inference sigma schedule itself.
        model.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=5.0,
            use_dynamic_shifting=False,
        )

        # the full Qwen3.5 model is loaded so that the checkpoint keys match; only its language model is used
        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            Qwen3_5Model,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
            stream_from_disk=stream_from_disk,
        )

        model.vae = self._load_vae(
            AutoencoderKLFlux2,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            MossImageTransformer,
            weight_dtypes,
            base_model_name,
            "",
            quantization,
            stream_from_disk=stream_from_disk,
        )

    def load( #TODO share code between models
            self,
            model: MossImageModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
            stream_from_disk: bool = False,
    ):
        if model_names.transformer_model:
            raise NotImplementedError("Overriding the transformer is not supported for MOSS Image")

        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model, quantization,
                stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.vae_model, quantization,
                stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
