from contextlib import nullcontext
from random import Random

from modules.model.BaseModel import BaseModel
from modules.model.Flux2Model import Flux2Model
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.MossImageTransformer import MossImageTransformer
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor

from diffusers import AutoencoderKLFlux2, DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2Tokenizer, Qwen3_5Model

# Default token budget for the chat-formatted prompt, used when the "Text Encoder Sequence Length" setting is empty.
# The vendor allows up to 4096; captions rarely need more than a few hundred tokens, and every prompt is padded to
# this length before encoding, so a larger value makes text encoding slower.
DEFAULT_PROMPT_MAX_LENGTH = 512

# The vendor's inference default: the negative branch of CFG skips this main DiT block.
DEFAULT_UNCOND_SKIP_LAYERS = (22,)


def format_prompt(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# applied as tokenizer.apply_chat_template(format_prompt(text), tokenize=False, **CHAT_TEMPLATE_KWARGS)
CHAT_TEMPLATE_KWARGS = {"add_generation_prompt": True, "enable_thinking": True}


class MossImageModel(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3_5Model | None
    vae: AutoencoderKLFlux2 | None
    transformer: MossImageTransformer | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    # persistent lora training data
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.transformer_lora = None
        self.lora_state_dict = None

    def diffusers_to_original(self) -> list | None:
        # The native (ComfyUI) names are the vendor names, q/k/v split. No fusion_groups(): ComfyUI's MOSS LoRA
        # convention addresses the separate projections and maps them onto fused checkpoints itself, so split keys
        # work on both fused and unfused checkpoints, while fused proj.qkv keys would only match fused ones.
        return [
            ("{path}", "{path}"),
        ]

    def create_pipeline(self) -> DiffusionPipeline:
        # no diffusers pipeline for MOSS Image; MossImageSampler runs its own denoising loop
        raise NotImplementedError

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_sequence_length: int | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            text = [
                self.tokenizer.apply_chat_template(format_prompt(t), tokenize=False, **CHAT_TEMPLATE_KWARGS)
                for t in text
            ]
            tokenizer_output = self.tokenizer(
                text,
                max_length=text_encoder_sequence_length or DEFAULT_PROMPT_MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokenizer_output.input_ids.to(self.text_encoder.device)
            tokens_mask = tokenizer_output.attention_mask.bool().to(self.text_encoder.device)

        if text_encoder_output is None:
            with self.text_encoder_autocast_context:
                # Qwen3.5 is causal (full attention and gated-delta layers), so right padding does not change the
                # outputs of the real tokens. The last hidden state is the final-normed one the vendor uses.
                text_encoder_output = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask.float(),
                    use_cache=False,
                ).last_hidden_state

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        # Real tokens are a prefix of each row. Trim to this batch's longest prompt; the transformer masks the rest.
        tokens_mask = tokens_mask.bool()
        seq_lengths = tokens_mask.sum(dim=1)
        max_seq_length = int(seq_lengths.max())
        # when a mask is needed anyway (lengths differ), round up to a multiple of 16 to limit torch.compile
        # recompiles and cuDNN attention plan rebuilds from ever-changing sequence lengths. See Krea2Model.
        if max_seq_length % 16 != 0 and (seq_lengths != seq_lengths[0]).any():
            max_seq_length = min(max_seq_length + (16 - max_seq_length % 16), tokens_mask.shape[1])
        text_encoder_output = text_encoder_output[:, :max_seq_length]
        tokens_mask = tokens_mask[:, :max_seq_length]
        text_encoder_output = text_encoder_output * tokens_mask.unsqueeze(-1)

        return text_encoder_output, tokens_mask

    # The latent layout is FLUX.2's: the 32-channel VAE mean, patchified 2x2 to 128 channels, then normalized with
    # the VAE's batch-norm statistics.
    patchify_latents = staticmethod(Flux2Model.patchify_latents)
    unpatchify_latents = staticmethod(Flux2Model.unpatchify_latents)
    scale_latents = Flux2Model.scale_latents
    unscale_latents = Flux2Model.unscale_latents
