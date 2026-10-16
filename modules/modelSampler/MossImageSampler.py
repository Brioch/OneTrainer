from collections.abc import Callable

from modules.model.MossImageModel import DEFAULT_UNCOND_SKIP_LAYERS, MossImageModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat

import torch

from diffusers.image_processor import VaeImageProcessor

from tqdm import tqdm

# the vendor's inference schedule (MOSSImageScheduler): sigmas shifted by INFERENCE_SHIFT, then snapped to the nearest
# sigma of the (unshifted) training grid
INFERENCE_SHIFT = 5.0
TRAINING_SHIFT = 1.0


def moss_sigmas(num_steps: int, num_train_timesteps: int) -> torch.Tensor:
    def grid(count, shift):
        lower = shift / (num_train_timesteps + shift - 1)
        values = torch.linspace(1, lower, count)
        return shift * values / (1 + (shift - 1) * values)

    sigmas = grid(num_steps, INFERENCE_SHIFT)
    training = grid(num_train_timesteps, TRAINING_SHIFT).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=num_train_timesteps - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@factory.register(BaseModelSampler, ModelType.MOSS_IMAGE)
class MossImageSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: MossImageModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.image_processor = VaeImageProcessor(vae_scale_factor=16)

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            text_encoder_sequence_length: int | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            transformer = self.model.transformer
            vae = self.model.vae
            vae_scale_factor = 16  # 8x VAE, then 2x2 patchified
            num_latent_channels = transformer.config.latent_channels
            train_dtype = self.model.train_dtype.torch_dtype()

            # prepare prompt
            self.model.materialize_only("text_encoder")

            do_cfg = cfg_scale > 1.0
            prompt_embedding, prompt_attention_mask = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
                text_encoder_sequence_length=text_encoder_sequence_length,
            )
            if do_cfg:
                negative_prompt_embedding, negative_prompt_attention_mask = self.model.encode_text(
                    text=negative_prompt,
                    train_device=self.train_device,
                    text_encoder_sequence_length=text_encoder_sequence_length,
                )

            # prepare latent image, sampled directly in fp32
            latent_image = torch.randn(
                size=(1, num_latent_channels, height // vae_scale_factor, width // vae_scale_factor),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            num_train_timesteps = self.model.noise_scheduler.config['num_train_timesteps']
            sigmas = moss_sigmas(diffusion_steps, num_train_timesteps).to(self.train_device)
            num_steps = len(sigmas) - 1

            # denoising loop. CFG and the Euler update run in fp32
            self.model.materialize_only("transformer")
            for i in tqdm(range(num_steps), desc="sampling"):
                timestep = (sigmas[i] * num_train_timesteps).expand(1)

                noise_pred = transformer(
                    hidden_states=latent_image.to(dtype=train_dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embedding.to(dtype=train_dtype),
                    return_dict=False,
                )[0].float()

                if do_cfg:
                    # the vendor's default guidance: the negative branch skips a block of the DiT
                    noise_pred_negative = transformer(
                        hidden_states=latent_image.to(dtype=train_dtype),
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embedding.to(dtype=train_dtype),
                        skip_layers=DEFAULT_UNCOND_SKIP_LAYERS,
                        return_dict=False,
                    )[0].float()
                    noise_pred = noise_pred_negative + cfg_scale * (noise_pred - noise_pred_negative)

                latent_image = latent_image + (sigmas[i + 1] - sigmas[i]) * noise_pred

                on_update_progress(i + 1, num_steps)

            self.model.materialize_only("vae")

            latents = self.model.unscale_latents(latent_image.to(dtype=vae.dtype))
            latents = self.model.unpatchify_latents(latents)
            image = vae.decode(latents, return_dict=False)[0].clamp(-1, 1)

            do_denormalize = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=image[0],
            )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 64),
            width=self.quantize_resolution(sample_config.width, 64),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_sequence_length=sample_config.text_encoder_1_sequence_length,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)
