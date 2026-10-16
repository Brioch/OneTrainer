# MOSS Image DiT (OpenMOSS-Team/MOSS-Image-SFT / -Base).
#
# Port of MOSSImageTransformer from the vendor's pipeline_moss_image.py (Apache-2.0), with batching adapted from
# ostris/ai-toolkit (MIT). State-dict names match the vendor's transformer/ checkpoint exactly (q/k/v split), so it
# loads through the regular diffusers sub-module loader.
#
# One stream: image tokens followed by text tokens through 42 blocks that share one AdaLN modulation; the text first
# goes through a 2-block refiner. The vendor evaluates every sample alone at its true text length; here a batch runs
# together with padded text tokens masked out of every attention, which gives the same result.

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin


class _RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6)
        return y.to(x.dtype) * self.weight.to(x.dtype)


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    # x: (B, S, H, D), rope: (B, S, D) as [sin, cos] halves, rotate-half convention
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


class MossImageAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "MossImageAttention", q, k, v, rope, attention_mask):
        q = _rotate(q, rope).to(v.dtype)
        k = _rotate(k, rope).to(v.dtype)
        out = dispatch_attention_fn(
            q, k, v,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        return out.flatten(2)


class MossImageAttention(nn.Module, AttentionModuleMixin):
    # Parameter-free: only holds the processor, so ModelMixin.set_attention_backend reaches it. The projections stay
    # on the blocks to keep the vendor's state-dict names.
    _default_processor_cls = MossImageAttnProcessor
    _available_processors = [MossImageAttnProcessor]

    def __init__(self):
        super().__init__()
        self.set_processor(self._default_processor_cls())

    def forward(self, q, k, v, rope, attention_mask=None):
        return self.processor(self, q, k, v, rope, attention_mask)


class MossImageTextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)
        self.attn = MossImageAttention()

    def forward(self, hidden_states, rope, attention_mask=None):
        x = hidden_states
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1)))
        k = self.k_norm(k.unflatten(-1, (self.heads, -1)))
        v = v.unflatten(-1, (self.heads, -1))
        x = x + self.proj(self.attn(q, k, v, rope, attention_mask))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class MossImageTextEmbedder(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MossImageTextBlock(dim, ffn_dim, heads) for _ in range(2)])


class MossImageBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({
            name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')
        })
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)
        self.attn = MossImageAttention()

    def forward(self, hidden_states, rope, modulation, attention_mask=None):
        # modulation: (B, 6, D) [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]
        x = hidden_states
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.to(x.dtype).unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1)))
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1)))
        v = self.proj['v'](y).unflatten(-1, (self.heads, -1))
        x = x + self.proj['o'](self.attn(q, k, v, rope, attention_mask)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj['ffn_up_gate'](y).chunk(2, -1)
        return x + self.proj['ffn_down'](_situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    # The vendor checkpoint stores the band table as rope.bands; it is a derived constant, so it is kept only for
    # state-dict compatibility and recomputed in fp32 on use (a bf16-cast copy would lose precision).
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer('bands', self.compute_bands(torch.device('cpu')))

    def compute_bands(self, device: torch.device):
        exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)


class MossImageTransformer(ModelMixin, ConfigMixin):
    _no_split_modules = ["MossImageBlock", "MossImageTextBlock"]

    @register_to_config
    def __init__(
            self,
            dim: int = 3072,
            ffn_dim: int = 8192,
            num_heads: int = 24,
            num_layers: int = 42,
            text_dim: int = 4096,
            freq_dim: int = 256,
            latent_channels: int = 128,
            rope_base: float = 10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError('Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.')
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MossImageTextEmbedder(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MossImageBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def _rope(self, text_lengths: torch.Tensor, height: int, width: int, text_len: int, device: torch.device):
        # (B, H*W + S, head_dim) fp32 rope over (r, f, h, w) coordinates: image tokens at
        # (0, sample text length, centered h, centered w), text token i at (0, i, 0, 0)
        batch = text_lengths.shape[0]
        h, w = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij',
        )
        image = torch.zeros(batch, height * width, 4, device=device)
        image[..., 1] = text_lengths.float()[:, None]
        image[..., 2] = (h - (height - 1) / 2).flatten()
        image[..., 3] = (w - (width - 1) / 2).flatten()
        text = torch.zeros(batch, text_len, 4, device=device)
        text[..., 1] = torch.arange(text_len, device=device, dtype=torch.float32)
        phase = torch.cat((image, text), 1).unsqueeze(-1) * self.rope.compute_bands(device)
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    def _time_modulation(self, timestep: torch.Tensor, dtype: torch.dtype):
        # sinusoid computed in float64 like the vendor's training code, on t in [0, 1000]
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=timestep.device, dtype=torch.float64) / half)
        phase = timestep.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(dtype))
        modulation = self.shared_block_modulation(time).view(timestep.shape[0], 6, -1)
        head = self.video_head_modulation(time).view(timestep.shape[0], 2, -1)
        return modulation, head

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor | None = None,
            skip_layers: tuple[int, ...] | list[int] | None = None,
            return_dict: bool = True,
    ):
        """
        hidden_states: (B, latent_channels, H, W) patchified, BN-normalized latents
        timestep: (B,) or scalar, in [0, 1000] (not divided by 1000)
        encoder_hidden_states: (B, S, text_dim), right-padded
        encoder_attention_mask: (B, S) bool, True for valid tokens. None if all tokens are valid
        skip_layers: main block indices to skip (used for the negative branch of CFG at inference)
        """
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        num_image = height * width
        text_len = encoder_hidden_states.shape[1]

        timestep = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch)
        modulation, head = self._time_modulation(timestep, dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.bool()
            text_lengths = encoder_attention_mask.sum(1)
            # (B, 1, 1, S) key masks, broadcast over heads and queries
            text_mask = encoder_attention_mask[:, None, None, :]
            joint_mask = F.pad(encoder_attention_mask, (num_image, 0), value=True)[:, None, None, :]
        else:
            text_lengths = torch.full((batch,), text_len, device=device)
            text_mask = None
            joint_mask = None

        rope = self._rope(text_lengths, height, width, text_len, device).to(dtype)

        video = self.video_embedder(hidden_states.flatten(2).transpose(1, 2).to(dtype))
        text = self.text_embedder.proj(encoder_hidden_states.to(dtype))
        for block in self.text_embedder.blocks:
            text = block(text, rope[:, num_image:], text_mask)

        x = torch.cat((video, text), 1)
        skip_layers = frozenset(skip_layers or ())
        for index, layer in enumerate(self.layers):
            out = layer(x, rope, modulation, joint_mask)
            # skipped layers are still called so the layer offload conductor sees every layer in order
            if index not in skip_layers:
                x = out

        x = self.final_norm_video(x[:, :num_image])
        head = head.to(x.dtype)
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x)
        sample = x.transpose(1, 2).reshape(batch, -1, height, width)

        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)
