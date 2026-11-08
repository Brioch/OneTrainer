import copy
from abc import ABCMeta
from random import Random
from statistics import NormalDist

import modules.util.multi_gpu_util as multi
from modules.model.MossImageModel import MossImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_moss_image_transformer,
    enable_checkpointing_for_qwen35_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor


class BaseMossImageSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    # every Linear of the main blocks is either attention (proj.q/k/v/o) or mlp (proj.ffn_*)
    LAYER_PRESETS = {
        "blocks": ["layers."],
        "attn-only": ["proj.q", "proj.k", "proj.v", "proj.o"],
        "full": [],
    }

    # parsed Clean-End Schedule (None = off) and the timestep config derived from it for the current epoch
    _clean_end_schedule: tuple[float, ...] | None = None
    _clean_end_epoch: int | None = None
    _clean_end_timestep_config: TrainConfig | None = None

    @staticmethod
    def parse_clean_end_schedule(text: str) -> tuple[float, ...] | None:
        # Fizgig's clean-end share syntax: A (single), A:N:B (split), A:H:B:E (slide), A:H:B:E:R (slide with refresh)
        text = (text or "").replace("%", "").strip()
        if not text:
            return None
        try:
            parts = tuple(float(x) for x in text.split(":"))
        except ValueError:
            parts = ()
        if len(parts) not in (1, 3, 4, 5) or not all(0 < parts[i] < 100 for i in range(0, len(parts), 2) if i != 4) \
                or (len(parts) == 5 and parts[4] != int(parts[4])):
            raise ValueError(
                f"Invalid Clean-End Schedule '{text}'. Use a percentage (40), share:epochs:share (75:5:40 = 75% for "
                "the first 5 epochs, then 40%), share:epoch:share:epoch (60:7:30:15 = 60% until epoch 7, then down "
                "to 30% by epoch 15) or add a fifth number R (60:7:30:15:2 = afterwards, one epoch at 60% every 2nd "
                "epoch). Shares are percentages between 0 and 100.")
        return parts

    @staticmethod
    def clean_end_share(schedule: tuple[float, ...], epoch: int) -> float:
        # percentage of steps below sigma 0.5 for a 1-based epoch, as in Fizgig's MOSS-Image driver
        if len(schedule) == 1:
            return schedule[0]
        first, until, rest = schedule[:3]
        end = schedule[3] if len(schedule) >= 4 else until
        every = int(schedule[4]) if len(schedule) == 5 else 0
        if epoch <= until:
            share = first
        elif epoch >= end:
            share = rest
        else:
            share = first + (rest - first) * (epoch - until) / (end - until)
        if every > 0 and epoch > end and (epoch - end) % every == 0:
            share = first
        return share

    def _clean_end_config(self, config: TrainConfig, epoch: int) -> TrainConfig:
        # A logit-normal with std 1 and mean b puts Phi(-b) of the steps below sigma 0.5, so b = -Phi^-1(share).
        # This matches Fizgig's clean-end curve (its logit-normal has std 0.95 and a slightly truncated range).
        if self._clean_end_epoch != epoch:
            share = self.clean_end_share(self._clean_end_schedule, epoch)
            self._clean_end_epoch = epoch
            self._clean_end_timestep_config = copy.copy(config)
            self._clean_end_timestep_config.timestep_distribution = TimestepDistribution.LOGIT_NORMAL
            self._clean_end_timestep_config.noising_weight = 0.0
            self._clean_end_timestep_config.noising_bias = -NormalDist().inv_cdf(share / 100.0)
            print(f"MOSS Image clean-end schedule: {share:g}% of steps below half noise at epoch {epoch} "
                  f"(noising bias {self._clean_end_timestep_config.noising_bias:.2f})")
        return self._clean_end_timestep_config

    def setup_optimizations(
            self,
            model: MossImageModel,
            config: TrainConfig,
    ):
        super().setup_optimizations(model, config)
        self._setup_model_part(model, config, "transformer", config.transformer, enable_checkpointing_for_moss_image_transformer)
        self._setup_model_part(model, config, "text_encoder", config.text_encoder, enable_checkpointing_for_qwen35_encoder_layers, disable_fp16_autocast=True)
        self._setup_model_part(model, config, "vae", config.vae)

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)

        # parsed now so that an invalid schedule fails before caching and training start
        self._clean_end_schedule = self.parse_clean_end_schedule(config.clean_end_schedule)
        self._clean_end_epoch = None

    def predict(
            self,
            model: MossImageModel,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output, text_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            patchified_latent_image = model.patchify_latents(batch['latent_image'].float())
            scaled_latent_image = model.scale_latents(patchified_latent_image)

            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            if self._clean_end_schedule is not None and not deterministic:
                timestep_config = self._clean_end_config(config, train_progress.epoch + 1)
                timestep_shift = 1.0
            else:
                timestep_config = config
                timestep_shift = config.timestep_shift
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                timestep_config,
                shift=timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            if torch.all(text_attention_mask):
                text_attention_mask = None

            # the MOSS Image DiT takes the timestep in [0, 1000], not divided by 1000
            predicted_flow = model.transformer(
                hidden_states=scaled_noisy_latent_image.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                encoder_attention_mask=text_attention_mask,
                return_dict=False,
            )[0]

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                #unpatchify, to make the shape match the mask shape of masked training:
                'predicted': model.unpatchify_latents(predicted_flow),
                'target': model.unpatchify_latents(flow),
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: MossImageModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: MossImageModel, config: TrainConfig):
        model.materialize_only("text_encoder")
        model.eval()
