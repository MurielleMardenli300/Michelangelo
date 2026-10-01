from __future__ import annotations


"""
Mamba-conditioned DDIM diffusion for 3D DVF prediction with STRONG SPATIAL
REFERENCE CONDITIONING.

Core design
-----------
1) Mamba predicts future temporal tokens from the observed 2D respiratory sequence.
2) Vref is NEVER collapsed to a global vector. A trainable 3D reference encoder
   builds a spatial feature map at the diffusion resolution and a multiscale
   reference pyramid matching the denoising UNet.
3) The denoising UNet starts from Gaussian noise / noisy DVF latent and generates
   the complete DVF. There is NO deterministic DVF prior and NO residual over a
   deterministic DVF.
4) Temporal context + diffusion timestep condition EVERY residual block through
   FiLM.
5) Reference features are injected at EVERY UNet resolution on the down path,
   bottleneck, and up path.
6) CFG is kept, but it drops ONLY the temporal Mamba context. Vref is present in
   both conditional and unconditional branches.

This file is intended as the spatially-conditioned replacement for
`diffusion_mamba_ddim.py`.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/fellahr/4D_MoPred")  # the PARENT of temporal_predictor_pt
from models.spatial_transform import SpatialTransformer
from models.temporal.mamba.lung_edge_true_mamba import build_lung_edge_mamba_forecaster
from models.temporal.temporal_predictors import build_temporal_predictor
from utils.io import custom_load



# =============================================================================
# Utility blocks
# =============================================================================
def _groups(channels: int, maximum: int = 32) -> int:
    for g in range(min(maximum, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvNormAct3D(nn.Module):
    """Residual 3D convolution block used by the reference encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.SiLU(),
            nn.Linear(self.dim * 4, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / denom
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


# =============================================================================
# Strong spatial reference conditioning
# =============================================================================
class SpatialReferencePyramid3D(nn.Module):
    """Encode Vref into spatial features aligned with the diffusion UNet.

    The first stage is intentionally analogous to the successful deterministic
    model: full 3D Vref is processed by residual 3D blocks with real stride-2
    spatial downsampling. The deepest reference feature is resized, only if
    necessary, to the diffusion state's spatial resolution. From there we build
    a pyramid using the SAME stride-2 geometry as the denoising UNet.

    This preserves patient-specific spatial anatomy instead of compressing Vref
    to one global vector.
    """

    def __init__(
        self,
        unet_channels: Sequence[int],
        base_channels: int = 16,
        encoder_depth: int = 4,
        max_channels: int = 256,
    ):
        super().__init__()
        if len(unet_channels) < 1:
            raise ValueError("unet_channels must contain at least one level")
        if encoder_depth < 1:
            raise ValueError("encoder_depth must be >= 1")

        self.unet_channels = [int(c) for c in unet_channels]
        self.encoder_depth = int(encoder_depth)
        self.null_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, ch, 1, 1, 1)) for ch in self.unet_channels]
        )

        encoder = []
        in_ch = 1
        for level in range(self.encoder_depth):
            out_ch = min(int(base_channels) * (2 ** level), int(max_channels))
            encoder.append(ConvNormAct3D(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.encoder = nn.ModuleList(encoder)

        # Map the deep Vref representation to the first diffusion UNet width.
        self.to_level0 = nn.Sequential(
            nn.Conv3d(in_ch, self.unet_channels[0], 1),
            nn.GroupNorm(_groups(self.unet_channels[0]), self.unet_channels[0]),
            nn.SiLU(),
        )

        # Subsequent reference levels exactly mirror the diffusion downsampling.
        self.pyramid_down = nn.ModuleList(
            [
                ConvNormAct3D(
                    self.unet_channels[i],
                    self.unet_channels[i + 1],
                    stride=2,
                )
                for i in range(len(self.unet_channels) - 1)
            ]
        )

    def null_like(self, ref_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Learned null features matching the shape of a real ref_features list."""
        nulls = []
        for feat, null_param in zip(ref_features, self.null_tokens):
            B = feat.shape[0]
            spatial = feat.shape[2:]
            nulls.append(null_param.expand(B, -1, *spatial))
        return nulls

    def forward(
        self,
        reference: torch.Tensor,
        diffusion_shape: Sequence[int],
    ) -> list[torch.Tensor]:
        if reference.dim() != 5 or reference.shape[1] != 1:
            raise ValueError(
                "reference must be [B,1,D,H,W], "
                f"got {tuple(reference.shape)}"
            )

        h = reference
        for block in self.encoder:
            h = block(h)

        # Normally encoder_depth=4 maps 128x80x64 -> 8x5x4, matching the VAE
        # latent. Interpolation keeps the module robust to other VAE geometries.
        if tuple(h.shape[2:]) != tuple(diffusion_shape):
            h = F.interpolate(
                h,
                size=tuple(int(v) for v in diffusion_shape),
                mode="trilinear",
                align_corners=False,
            )

        h = self.to_level0(h)
        features = [h]

        for down in self.pyramid_down:
            h = down(h)
            features.append(h)

        return features


# =============================================================================
# Diffusion UNet blocks
# =============================================================================
class FiLMResBlock3D(nn.Module):
    """Residual block conditioned by diffusion time + temporal context via FiLM."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        scale, shift = self.cond_mlp(cond).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None, None])
        h = h + shift[:, :, None, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if channels % n_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = int(n_heads)
        self.head_dim = channels // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.to_qkv = nn.Conv3d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv3d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.to_qkv(h).chunk(3, dim=1)
        n = D * H * W
        q = q.view(B, self.n_heads, self.head_dim, n)
        k = k.view(B, self.n_heads, self.head_dim, n)
        v = v.view(B, self.n_heads, self.head_dim, n)
        attn = torch.softmax(
            torch.einsum("bhcn,bhcm->bhnm", q, k) * self.scale,
            dim=-1,
        )
        attn = self.dropout(attn)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(B, C, D, H, W)
        return x + self.to_out(out)


class RefTemporalDenoisingUNet3D(nn.Module):
    """True multiscale 3D UNet with spatial Vref + temporal FiLM conditioning.

    Reference conditioning is strong by construction:
      down level i: concat(x_i, ref_i) -> residual blocks
      bottleneck:   concat(x_L, ref_L) -> residual blocks
      up level i:   concat(up(x), skip_i, ref_i) -> residual blocks

    Temporal context is injected into every residual block together with the
    timestep embedding.
    """

    def __init__(
        self,
        in_ch: int,
        temporal_dim: int,
        base_ch: int = 128,
        ch_mults: Sequence[int] = (1, 2),
        time_dim: int = 128,
        n_attn_heads: int = 8,
        use_self_attn: bool = True,
        num_res_blocks: int = 2,
        res_dropout: float = 0.0,
    ):
        super().__init__()
        if len(ch_mults) < 1:
            raise ValueError("ch_mults must contain at least one value")

        self.channels = [int(base_ch) * int(m) for m in ch_mults]
        self.time_dim = int(time_dim)
        self.num_res_blocks = max(int(num_res_blocks), 1)

        self.time_embedding = SinusoidalTimestepEmbedding(self.time_dim)
        self.temporal_embedding = nn.Sequential(
            nn.LayerNorm(int(temporal_dim)),
            nn.Linear(int(temporal_dim), self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.input_proj = nn.Conv3d(in_ch, self.channels[0], 3, padding=1)

        # Down path. Each level sees its same-resolution Vref feature.
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for level in range(len(self.channels) - 1):
            ch = self.channels[level]
            blocks = nn.ModuleList()
            for block_idx in range(self.num_res_blocks):
                block_in = ch * 2 if block_idx == 0 else ch
                blocks.append(
                    FiLMResBlock3D(
                        block_in,
                        ch,
                        self.time_dim,
                        dropout=res_dropout,
                    )
                )
            self.down_res.append(blocks)
            self.down_attn.append(
                SelfAttention3D(ch, n_heads=n_attn_heads, dropout=res_dropout)
                if use_self_attn
                else nn.Identity()
            )
            self.downsample.append(
                nn.Conv3d(
                    ch,
                    self.channels[level + 1],
                    3,
                    stride=2,
                    padding=1,
                )
            )

        # Bottleneck: concatenate deepest spatial reference feature.
        deep_ch = self.channels[-1]
        middle = []
        for block_idx in range(self.num_res_blocks):
            middle.append(
                FiLMResBlock3D(
                    deep_ch * 2 if block_idx == 0 else deep_ch,
                    deep_ch,
                    self.time_dim,
                    dropout=res_dropout,
                )
            )
        self.middle_res = nn.ModuleList(middle)
        self.middle_attn = (
            SelfAttention3D(deep_ch, n_heads=n_attn_heads, dropout=res_dropout)
            if use_self_attn
            else nn.Identity()
        )

        # Up path. At every level concatenate decoder state + UNet skip + Vref.
        self.up_project = nn.ModuleList()
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()

        for level in reversed(range(len(self.channels) - 1)):
            in_level_ch = self.channels[level + 1]
            out_level_ch = self.channels[level]
            self.up_project.append(
                nn.Conv3d(in_level_ch, out_level_ch, 1)
            )

            blocks = nn.ModuleList()
            for block_idx in range(self.num_res_blocks):
                # First block sees: decoder + skip + spatial reference = 3*C.
                block_in = out_level_ch * 3 if block_idx == 0 else out_level_ch
                blocks.append(
                    FiLMResBlock3D(
                        block_in,
                        out_level_ch,
                        self.time_dim,
                        dropout=res_dropout,
                    )
                )
            self.up_res.append(blocks)
            self.up_attn.append(
                SelfAttention3D(
                    out_level_ch,
                    n_heads=n_attn_heads,
                    dropout=res_dropout,
                )
                if use_self_attn
                else nn.Identity()
            )

        self.output_proj = nn.Sequential(
            nn.GroupNorm(_groups(self.channels[0]), self.channels[0]),
            nn.SiLU(),
            nn.Conv3d(self.channels[0], in_ch, 3, padding=1),
        )

        # Stable start for x0/noise regression.
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def _condition(
        self,
        t: torch.Tensor,
        temporal_context: torch.Tensor,
    ) -> torch.Tensor:
        if temporal_context.dim() == 3:
            temporal_context = temporal_context.squeeze(1)
        if temporal_context.dim() != 2:
            raise ValueError(
                "temporal_context must be [B,D] or [B,1,D], "
                f"got {tuple(temporal_context.shape)}"
            )
        return self.time_embedding(t) + self.temporal_embedding(temporal_context)

    @staticmethod
    def _match_ref(ref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if tuple(ref.shape[2:]) == tuple(x.shape[2:]):
            return ref
        return F.interpolate(
            ref,
            size=x.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        temporal_context: torch.Tensor,
        ref_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(ref_features) != len(self.channels):
            raise ValueError(
                f"Expected {len(self.channels)} reference levels, "
                f"got {len(ref_features)}"
            )

        cond = self._condition(t, temporal_context)
        x = self.input_proj(x_t)      #. going from latent channels to UNet base channels (16 -> 128)
        skips = []

        # Down levels 0 .. L-2.
        for level, (blocks, attn, down) in enumerate(
            zip(self.down_res, self.down_attn, self.downsample)
        ):
            ref = self._match_ref(ref_features[level], x)
            x = torch.cat([x, ref], dim=1)
            for block in blocks:
                x = block(x, cond)
            x = attn(x)
            skips.append(x)
            x = down(x)

        # Deepest level.
        ref_deep = self._match_ref(ref_features[-1], x)
        x = torch.cat([x, ref_deep], dim=1)
        for block in self.middle_res:
            x = block(x, cond)
        x = self.middle_attn(x)

        # Up levels L-2 .. 0.
        for up_idx, level in enumerate(reversed(range(len(self.channels) - 1))):
            skip = skips[level]
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            x = self.up_project[up_idx](x)
            ref = self._match_ref(ref_features[level], x)
            x = torch.cat([x, skip, ref], dim=1)
            for block in self.up_res[up_idx]:
                x = block(x, cond)
            x = self.up_attn[up_idx](x)

        return self.output_proj(x)



# FOR ABLATION STUDY
class InterpolatedReferencePyramid3D(nn.Module):
    """
    Ablation baseline: NO learned 3D reference encoder and NO learned pyramid.

    The raw Vref is directly trilinearly resized to every diffusion-UNet
    resolution. Because the denoising UNet expects the reference tensor at
    level i to have unet_channels[i] channels, the single CT channel is simply
    expanded across channels (no trainable parameters in this branch).

    This keeps the exact same UNet concatenation locations and tensor widths as
    the full model while removing the learned spatial-reference module.
    """

    def __init__(self, unet_channels: Sequence[int]):
        super().__init__()
        if len(unet_channels) < 1:
            raise ValueError("unet_channels must contain at least one level")
        self.unet_channels = [int(c) for c in unet_channels]
        self.null_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, ch, 1, 1, 1)) for ch in self.unet_channels]
        )

    @staticmethod
    def _level_shape(diffusion_shape: Sequence[int], level: int) -> tuple[int, int, int]:
        # Mirror stride-2 Conv3d(k=3,p=1,s=2): output spatial size = ceil(input/2).
        shape = tuple(int(v) for v in diffusion_shape)
        for _ in range(level):
            shape = tuple((v + 1) // 2 for v in shape)
        return shape

    def null_like(self, ref_features: list[torch.Tensor]) -> list[torch.Tensor]:
        nulls = []
        for feat, null_param in zip(ref_features, self.null_tokens):
            B = feat.shape[0]
            spatial = feat.shape[2:]
            nulls.append(null_param.expand(B, -1, *spatial))
        return nulls

    def forward(
        self,
        reference: torch.Tensor,
        diffusion_shape: Sequence[int],
    ) -> list[torch.Tensor]:
        if reference.dim() != 5 or reference.shape[1] != 1:
            raise ValueError(
                "reference must be [B,1,D,H,W], "
                f"got {tuple(reference.shape)}"
            )

        features = []
        for level, ch in enumerate(self.unet_channels):
            target = self._level_shape(diffusion_shape, level)
            ref = F.interpolate(
                reference,
                size=target,
                mode="trilinear",
                align_corners=False,
            )

            # Pure channel expansion: no learned reference-feature extraction.
            # Shape: [B,1,D,H,W] -> [B,C,D,H,W]
            ref = ref.expand(-1, ch, -1, -1, -1)
            features.append(ref)

        return features


class ZeroReferencePyramid3D(nn.Module):
    """
    Ablation baseline: NO Vref information.

    Returns zero tensors at every UNet reference level. This is preferable to
    deleting the concatenations because it keeps the denoising UNet architecture,
    channel counts and parameter count identical to the full reference model.
    """

    def __init__(self, unet_channels: Sequence[int]):
        super().__init__()
        if len(unet_channels) < 1:
            raise ValueError("unet_channels must contain at least one level")
        self.unet_channels = [int(c) for c in unet_channels]

    @staticmethod
    def _level_shape(diffusion_shape: Sequence[int], level: int) -> tuple[int, int, int]:
        shape = tuple(int(v) for v in diffusion_shape)
        for _ in range(level):
            shape = tuple((v + 1) // 2 for v in shape)
        return shape

    def null_like(self, ref_features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.zeros_like(feat) for feat in ref_features]

    def forward(
        self,
        reference: torch.Tensor,
        diffusion_shape: Sequence[int],
    ) -> list[torch.Tensor]:
        if reference.dim() != 5:
            raise ValueError(
                "reference must be [B,1,D,H,W], "
                f"got {tuple(reference.shape)}"
            )

        B = reference.shape[0]
        features = []
        for level, ch in enumerate(self.unet_channels):
            target = self._level_shape(diffusion_shape, level)
            features.append(
                reference.new_zeros((B, ch, *target))
            )
        return features


# =============================================================================
# Cosine DDIM scheduler
# =============================================================================
class CosineDDIMScheduler:
    def __init__(self, T: int = 1000, s: float = 0.008, beta_max: float = 0.2):
        self.T = int(T)
        self.s = float(s)
        self.beta_max = float(beta_max)
        self.betas = self._cosine_beta_schedule(self.T, self.s, self.beta_max)
        self.alphas_bar = torch.cumprod(1.0 - self.betas, dim=0)
        print(
            f"Final alpha_bar: {self.alphas_bar[-1].item():.8e} "
            "(should be close to 0)"
        )

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float, beta_max: float) -> torch.Tensor:
        steps = T + 1
        t = torch.linspace(0, T, steps)
        f = torch.cos(((t / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_bar = f / f[0]
        betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(1e-8, beta_max)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_bar = self.alphas_bar.to(device)
        return self

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        ab = self.alphas_bar[t][:, None, None, None, None]
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,
        shape,
        device,
        ddim_steps: int = 50,
        eta: float = 0.0,
        prediction_type: str = "x0_small",
    ) -> torch.Tensor:
        timesteps = (
            torch.linspace(
                self.T - 1,
                0,
                steps=max(int(ddim_steps), 2),
                device=device,
            )
            .round()
            .long()
        )
        x_t = torch.randn(shape, device=device)

        for i, t_tensor in enumerate(timesteps):
            t_val = int(t_tensor.item())
            t_batch = torch.full(
                (shape[0],),
                t_val,
                device=device,
                dtype=torch.long,
            )
            alpha_bar_t = self.alphas_bar[t_val]
            model_out = model_fn(x_t, t_batch)

            if prediction_type == "noise":
                eps_pred = model_out
                x0_pred = (
                    x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred
                ) / (torch.sqrt(alpha_bar_t) + 1e-8)
            elif prediction_type == "x0_small":
                x0_pred = model_out
                eps_pred = (
                    x_t - torch.sqrt(alpha_bar_t) * x0_pred
                ) / (torch.sqrt(1.0 - alpha_bar_t) + 1e-8)
            else:
                raise ValueError(
                    f"Unsupported prediction_type: {prediction_type}"
                )

            if i == len(timesteps) - 1:
                x_t = x0_pred
                continue

            alpha_bar_prev = self.alphas_bar[timesteps[i + 1]]
            sigma_sq = (
                eta**2
                * (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            sigma = torch.sqrt(torch.clamp(sigma_sq, min=0.0))
            direction = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)
            )
            sampling_noise = (
                torch.randn_like(x_t)
                if eta > 0.0
                else torch.zeros_like(x_t)
            )
            x_t = (
                torch.sqrt(alpha_bar_prev) * x0_pred
                + direction * eps_pred
                + sigma * sampling_noise
            )

        return x_t


# =============================================================================
# Complete Mamba + spatial-reference diffusion model
# =============================================================================
class Diffusion_Mamba_SpatialRef(nn.Module):
    def __init__(
        self,
        num_frames: int,
        horizon: int,
        vol_size,
        pre_latent_dim: int,
        cond_net_checkpoint: str | None = None,
        freeze_cond_net: bool = True,
        T: int = 1000,
        ddim_steps: int = 100,
        beta_max: float = 0.2,
        eta: float = 0.0,
        prediction_type: str = "x0_small",
        temporal_predictor: str = "mamba",
        temporal_predictor_kwargs: dict | None = None,
        # diffusion UNet
        unet_base_ch: int = 128,
        unet_ch_mults: Sequence[int] = (1, 2),
        time_dim: int = 128,
        n_attn_heads: int = 8,
        use_self_attn: bool = True,
        num_res_blocks: int = 2,
        res_dropout: float = 0.0,
        # strong Vref encoder
        ref_mode: str = "learned",
        ref_base_channels: int = 16,
        ref_encoder_depth: int = 4,
        ref_max_channels: int = 256,
        # latent/direct diffusion
        use_latent_diffusion: bool = True,
        vae: nn.Module | None = None,
        vae_latent_channels: int | None = None,
        latent_use_mean: bool = True,
        freeze_vae: bool = True,
        dvf_downsample_factor: float = 1.0,
        # CFG -- temporal condition only
        cfg_dropout_temporal: float = 0.30,
        cfg_dropout_vref: float = 0.15,
        cfg_scale_temporal: float = 3.0,
        cfg_scale_vref: float = 1.5,
    ):
        super().__init__()

        self.num_frames = int(num_frames)
        self.horizon = int(horizon)
        self.vol_size = tuple(int(v) for v in vol_size)
        self.pre_latent_dim = int(pre_latent_dim)
        self.ddim_steps = int(ddim_steps)
        self.eta = float(eta)
        self.prediction_type = str(prediction_type)
        self.use_latent_diffusion = bool(use_latent_diffusion)
        self.vae = vae
        self.latent_channels = vae_latent_channels
        self.latent_use_mean = bool(latent_use_mean)
        self.freeze_vae = bool(freeze_vae)
        self.freeze_cond_net = bool(freeze_cond_net)
        self.dvf_downsample_factor = max(float(dvf_downsample_factor), 1.0)
        # self.cfg_dropout = float(cfg_dropout)
        # self.cfg_guidance_scale = float(cfg_guidance_scale)
        self.cfg_dropout_temporal = float(cfg_dropout_temporal)
        self.cfg_dropout_vref = float(cfg_dropout_vref)
        self.cfg_scale_temporal = float(cfg_scale_temporal)
        self.cfg_scale_vref = float(cfg_scale_vref)

        self.ref_mode = str(ref_mode).lower()
        if self.ref_mode not in {"learned", "interpolate", "none"}:
            raise ValueError(
                "ref_mode must be one of: 'learned', 'interpolate', 'none'"
            )

        if self.prediction_type not in {"noise", "x0_small"}:
            raise ValueError("prediction_type must be 'noise' or 'x0_small'")

        self.temporal_null = nn.Parameter(torch.zeros(1, self.pre_latent_dim))

        # ------------------------------------------------------------------
        # Frozen or trainable pretrained Mamba temporal predictor.
        # ------------------------------------------------------------------
        self.cond_net = build_temporal_predictor(
            temporal_predictor,
            num_frames=self.num_frames,
            horizon=self.horizon,
            pre_latent_dim=self.pre_latent_dim,
            checkpoint=cond_net_checkpoint,      # <- loading now happens correctly IN HERE, onto the raw net
            freeze=freeze_cond_net,
            **(temporal_predictor_kwargs or {}),
        )

        self.cond_net_checkpoint = cond_net_checkpoint or ""

        # ------------------------------------------------------------------
        # VAE for latent diffusion.
        # ------------------------------------------------------------------
        if self.use_latent_diffusion:
            if self.vae is None:
                raise ValueError("use_latent_diffusion=True requires a VAE")
            if hasattr(self.vae, "latent_channels"):
                self.latent_channels = int(self.vae.latent_channels)
            elif self.latent_channels is None:
                raise ValueError("vae_latent_channels must be provided")

            if self.freeze_vae:
                for p in self.vae.parameters():
                    p.requires_grad_(False)
                self.vae.eval()

        denoise_channels = self.latent_channels if self.use_latent_diffusion else 3
        unet_channels = [int(unet_base_ch) * int(m) for m in unet_ch_mults]

        # ------------------------------------------------------------------
        # Strong patient-specific spatial reference path.
        # ------------------------------------------------------------------
        # self.ref_encoder = SpatialReferencePyramid3D(
        #     unet_channels=unet_channels,
        #     base_channels=int(ref_base_channels),
        #     encoder_depth=int(ref_encoder_depth),
        #     max_channels=int(ref_max_channels),
        # )
        if self.ref_mode == "learned":
            self.ref_encoder = SpatialReferencePyramid3D(
                unet_channels=unet_channels,
                base_channels=int(ref_base_channels),
                encoder_depth=int(ref_encoder_depth),
                max_channels=int(ref_max_channels),
            )
        elif self.ref_mode == "interpolate":
            self.ref_encoder = InterpolatedReferencePyramid3D(
                unet_channels=unet_channels,
            )
        else:  # self.ref_mode == "none"
            self.ref_encoder = ZeroReferencePyramid3D(
                unet_channels=unet_channels,
            )
        print(f"[Reference conditioning] mode={self.ref_mode}")

        # ------------------------------------------------------------------
        # Diffusion denoiser. It generates the ENTIRE DVF latent from x_t.
        # ------------------------------------------------------------------
        self.unet = RefTemporalDenoisingUNet3D(
            in_ch=int(denoise_channels),
            temporal_dim=self.pre_latent_dim,
            base_ch=int(unet_base_ch),
            ch_mults=tuple(int(v) for v in unet_ch_mults),
            time_dim=int(time_dim),
            n_attn_heads=int(n_attn_heads),
            use_self_attn=bool(use_self_attn),
            num_res_blocks=int(num_res_blocks),
            res_dropout=float(res_dropout),
        )

        self.scheduler = CosineDDIMScheduler(T=T, beta_max=beta_max)
        print(f"Beta start: {self.scheduler.betas[0].item():.8e}")
        print(f"Beta end:   {self.scheduler.betas[-1].item():.8e}")
        self.spatial_transform = SpatialTransformer(self.vol_size)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_cond_net:
            self.cond_net.eval()
        if self.freeze_vae and self.vae is not None:
            self.vae.eval()
        return self

    # ------------------------------------------------------------------
    # Temporal conditioning
    # ------------------------------------------------------------------
    # def _run_mamba(self, Iseq: torch.Tensor) -> list[torch.Tensor]:
    #     if Iseq.dim() != 5:
    #         raise ValueError(
    #             f"Iseq must be [B,2,T,H,W], got {tuple(Iseq.shape)}"
    #         )
    #     past = Iseq[:, :, : self.num_frames].contiguous()
    #     first = next(self.cond_net.parameters())
    #     past = past.to(
    #         device=first.device,
    #         dtype=first.dtype,
    #         non_blocking=True,
    #     )

    #     ctx = torch.no_grad() if self.freeze_cond_net else torch.enable_grad()
    #     with ctx:
    #         tokens = self.cond_net(past)

    #     if tokens.ndim != 3 or tokens.shape[1] != self.horizon:
    #         raise RuntimeError(
    #             f"Mamba must return [B,{self.horizon},D], "
    #             f"got {tuple(tokens.shape)}"
    #         )
    #     return [tokens[:, i, :] for i in range(self.horizon)]

    def _run_temporal(self, Iseq: torch.Tensor):

        if Iseq.dim() != 5:
            raise ValueError(
                f"Iseq must be [B,2,T,H,W], got {tuple(Iseq.shape)}"
            )

        first = next(self.cond_net.parameters())

        Iseq = Iseq.to(
            device=first.device,
            dtype=first.dtype,
            non_blocking=True,
        )

        past = Iseq[:, :, :self.num_frames].contiguous()

        # Future is only needed by a trainable CondiNet posterior.
        needs_future = (
            self.cond_net.training
            and getattr(
                self.cond_net,
                "requires_future_for_training",
                False,
            )
        )

        future = None

        if needs_future:
            expected_T = self.num_frames + self.horizon

            if Iseq.shape[2] < expected_T:
                raise ValueError(
                    f"Temporal predictor requires future frames during training, "
                    f"but Iseq contains T={Iseq.shape[2]}; "
                    f"expected at least {expected_T}."
                )

            future = Iseq[
                :,
                :,
                self.num_frames:self.num_frames + self.horizon,
            ].contiguous()

        ctx = (
            torch.no_grad()
            if self.freeze_cond_net
            else torch.enable_grad()
        )

        with ctx:
            tokens, aux = self.cond_net(
                past,
                future=future,
            )

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Temporal predictor must return [B,H,D], "
                f"got {tuple(tokens.shape)}"
            )

        if tokens.shape[1] != self.horizon:
            raise RuntimeError(
                f"Expected horizon={self.horizon}, "
                f"got {tokens.shape[1]}"
            )

        if tokens.shape[2] != self.pre_latent_dim:
            raise RuntimeError(
                f"Expected temporal dim={self.pre_latent_dim}, "
                f"got {tokens.shape[2]}"
            )

        return [
            tokens[:, i, :]
            for i in range(self.horizon)
        ], aux
    # ------------------------------------------------------------------
    # Latent/DVF helpers
    # ------------------------------------------------------------------
    def _downsample_dvf(self, dvf: torch.Tensor) -> torch.Tensor:
        if self.dvf_downsample_factor <= 1.0:
            return dvf
        return F.interpolate(
            dvf,
            scale_factor=1.0 / self.dvf_downsample_factor,
            mode="trilinear",
            align_corners=False,
        )

    def _upsample_dvf(self, dvf: torch.Tensor, target_shape) -> torch.Tensor:
        if self.dvf_downsample_factor <= 1.0:
            return dvf
        return F.interpolate(
            dvf,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    def _downsample_shape(self, spatial_shape):
        if self.dvf_downsample_factor <= 1.0:
            return tuple(spatial_shape)
        return tuple(
            max(1, int(round(v / self.dvf_downsample_factor)))
            for v in spatial_shape
        )

    def _encode_dvf_latent(self, dvf: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_diffusion:
            return dvf
        ctx = torch.no_grad() if self.freeze_vae else torch.enable_grad()
        with ctx:
            z, mu, _ = self.vae.encode(dvf)
        return mu if self.latent_use_mean else z

    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.use_latent_diffusion:
            return latent
        # Do not wrap decode in no_grad: gradients wrt latent remain useful if
        # auxiliary decoded losses are added later, even with frozen VAE weights.
        return self.vae.decode(latent)

    def _reference_features(
        self,
        Vref: torch.Tensor,
        diffusion_shape: Sequence[int],
    ) -> list[torch.Tensor]:
        return self.ref_encoder(Vref, diffusion_shape=diffusion_shape)

    def dvf_epe_loss(self, pred, target, eps=1e-6):
        # pred,target: [B, 3, D, H, W]
        error = pred - target
        return torch.sqrt(
            torch.sum(error ** 2, dim=1) + eps
        ).mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward_train(
        self,
        Vref: torch.Tensor,
        Iseq: torch.Tensor,
        dvf_gt_list: list[torch.Tensor],
        return_components: bool = False,
    ):
        device = Vref.device
        self.scheduler.to(device)
        B = Vref.shape[0]

        temporal_features, temporal_aux = self._run_temporal(Iseq)
        kl_loss = None

        if temporal_aux:
            kl_loss = temporal_aux.get("kl_loss")
        total_loss = Vref.new_zeros(())
        component_losses = []


        # Reference features depend on the diffusion-state spatial size. Since all
        # horizon targets share geometry, compute them after obtaining the first
        # target latent and reuse them for every future step.
        ref_features = None
        ref_null = None

        for step in range(self.horizon):
            x0 = dvf_gt_list[step]
            x0_small = (
                self._encode_dvf_latent(x0)
                if self.use_latent_diffusion
                else self._downsample_dvf(x0)
            )
            # print(x0.shape, "dvf shape")
            # print(x0_small.shape, "latent shape")

            if ref_features is None:
                ref_features = self._reference_features(
                    Vref,
                    diffusion_shape=x0_small.shape[2:],
                )
                ref_null = self.ref_encoder.null_like(ref_features)

            temporal_cond = temporal_features[step]

            # ------------------------------------------------------------
            # INDEPENDENT CFG dropout: temporal axis
            # ------------------------------------------------------------
            if self.cfg_dropout_temporal > 0.0:
                keep_t = torch.rand(B, device=device) > self.cfg_dropout_temporal
                null_t = self.temporal_null.expand(B, -1).to(temporal_cond.dtype)
                temporal_cond = torch.where(keep_t[:, None], temporal_cond, null_t)

            # ------------------------------------------------------------
            # INDEPENDENT CFG dropout: Vref axis
            # ------------------------------------------------------------
            step_ref_features = ref_features
            if self.cfg_dropout_vref > 0.0:
                keep_v = torch.rand(B, device=device) > self.cfg_dropout_vref
                mask = keep_v[:, None, None, None, None]
                step_ref_features = [
                    torch.where(mask, real, null)
                    for real, null in zip(ref_features, ref_null)
                ]

            t = torch.randint(0, self.scheduler.T, (B,), device=device)
            noise = torch.randn_like(x0_small)
            x_t = self.scheduler.q_sample(x0_small, t, noise)

            model_out = self.unet(x_t, t, temporal_cond, step_ref_features)

            if self.prediction_type == "x0_small":
                pred_x0_small = model_out
            else:
                ab = self.scheduler.alphas_bar[t][:, None, None, None, None]

                pred_x0_small = (
                    x_t - torch.sqrt(1.0 - ab) * model_out
                ) / (torch.sqrt(ab) + 1e-8)

            if self.use_latent_diffusion:
                dvf_pred = self._decode_latent(pred_x0_small)
            else:
                dvf_pred = self._upsample_dvf(
                    pred_x0_small,
                    x0.shape[2:]
                )

            if dvf_pred.shape[2:] != x0.shape[2:]:
                dvf_pred = F.interpolate(
                    dvf_pred,
                    size=x0.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )

            loss_dvf = self.dvf_epe_loss(dvf_pred, x0)

            target = noise if self.prediction_type == "noise" else x0_small
            loss_diff = F.mse_loss(model_out, target)

            step_loss = loss_diff #+ 0.5 * loss_dvf

            total_loss = total_loss + step_loss
            component_losses.append(step_loss.detach())

        total_loss = total_loss / float(self.horizon)
        if kl_loss is not None and not self.freeze_cond_net:
            total_loss = total_loss + 1e-3 * kl_loss

        if not return_components:
            return total_loss

        return {
            "loss_diffusion": total_loss,
            "per_horizon_loss": torch.stack(component_losses),
        }

    # ------------------------------------------------------------------
    # Inference + CFG
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_inference(
        self,
        Vref: torch.Tensor,
        Iseq: torch.Tensor,
        # cfg_scale: float | None = None,
        cfg_scale_temporal: float | None = None,
        cfg_scale_vref: float | None = None,
    ):
        device = Vref.device
        self.scheduler.to(device)
        B = Vref.shape[0]
        # w = self.cfg_guidance_scale if cfg_scale is None else float(cfg_scale)
        w_t = self.cfg_scale_temporal if cfg_scale_temporal is None else float(cfg_scale_temporal)
        w_v = self.cfg_scale_vref if cfg_scale_vref is None else float(cfg_scale_vref)

        temporal_features, temporal_aux = self._run_temporal(Iseq)
        spatial_shape = tuple(Vref.shape[2:])

        if self.use_latent_diffusion:
            dummy = torch.zeros(
                (B, 3, *spatial_shape),
                device=device,
                dtype=Vref.dtype,
            )
            latent = self._encode_dvf_latent(dummy)
            low_shape = tuple(latent.shape[2:])
        else:
            low_shape = self._downsample_shape(spatial_shape)

        # Reference conditioning is deterministic and never dropped by CFG.
        ref_features = self._reference_features(Vref, diffusion_shape=low_shape)
        ref_null = self.ref_encoder.null_like(ref_features)
        temporal_null = self.temporal_null.expand(B, -1)

        generated_dvfs = []
        generated_volumes = []

        for step in range(self.horizon):
            temporal_cond = temporal_features[step]
            # temporal_uncond = torch.zeros_like(temporal_cond)

            def model_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                # If w==1, the guided result is exactly the conditional branch.
                # We still support batched CFG for any guidance scale.
                # if w == 1.0:
                #     return self.unet(
                #         x_t,
                #         t,
                #         temporal_cond,
                #         ref_features,
                #     )
                if w_t == 1.0 and w_v == 1.0:
                    return self.unet(x_t, t, temporal_cond, ref_features)
                x3 = torch.cat([x_t, x_t, x_t], dim=0)
                t3 = torch.cat([t, t, t], dim=0)
                temporal3 = torch.cat([temporal_null, temporal_null, temporal_cond], dim=0)
                ref3 = [
                    torch.cat([n, r, r], dim=0)
                    for n, r in zip(ref_null, ref_features)
                ]

                pred3 = self.unet(x3, t3, temporal3, ref3)
                pred_none, pred_vref_only, pred_full = pred3.chunk(3, dim=0)

                return (
                    pred_none
                    + w_v * (pred_vref_only - pred_none)
                    + w_t * (pred_full - pred_vref_only)
                )
            
                # x_both = torch.cat([x_t, x_t], dim=0)
                # t_both = torch.cat([t, t], dim=0)
                # temporal_both = torch.cat(
                #     [temporal_cond, temporal_uncond],
                #     dim=0,
                # )
                # # Crucial: Vref is present in BOTH CFG branches.
                # ref_both = [torch.cat([r, r], dim=0) for r in ref_features]

                # pred_both = self.unet(
                #     x_both,
                #     t_both,
                #     temporal_both,
                #     ref_both,
                # )
                # pred_cond, pred_uncond = pred_both.chunk(2, dim=0)
                # return pred_uncond + w * (pred_cond - pred_uncond)

            low_pred = self.scheduler.ddim_sample(
                model_fn=model_fn,
                shape=(
                    B,
                    self.latent_channels if self.use_latent_diffusion else 3,
                    *low_shape,
                ),
                device=device,
                ddim_steps=self.ddim_steps,
                eta=self.eta,
                prediction_type=self.prediction_type,
            )

            if self.use_latent_diffusion:
                dvf_pred = self._decode_latent(low_pred)
            else:
                dvf_pred = self._upsample_dvf(low_pred, spatial_shape)

            if tuple(dvf_pred.shape[2:]) != spatial_shape:
                dvf_pred = F.interpolate(
                    dvf_pred,
                    size=spatial_shape,
                    mode="trilinear",
                    align_corners=False,
                )

            generated_dvfs.append(dvf_pred)
            generated_volumes.append(
                self.spatial_transform(Vref, dvf_pred)
            )

        return generated_dvfs, generated_volumes

    def forward(
        self,
        Vref: torch.Tensor,
        Iseq: torch.Tensor,
        dvf_gt_list: list[torch.Tensor] | None = None,
        return_components: bool = False,
        cfg_scale_temporal: float | None = None,
        cfg_scale_vref: float | None = None,
    ):
        if dvf_gt_list is not None:
            return self.forward_train(
                Vref,
                Iseq,
                dvf_gt_list,
                return_components=return_components,
            )
        return self.forward_inference(Vref, Iseq, cfg_scale_temporal=cfg_scale_temporal, cfg_scale_vref=cfg_scale_vref)


# Drop-in alias for existing training code style.
Diffusion_TR = Diffusion_Mamba_SpatialRef