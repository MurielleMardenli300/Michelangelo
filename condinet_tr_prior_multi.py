# ---------------------------------------------------------------------------------
# Temporal predictor - multiple outputs using Conv. prior
# ---------------------------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import (
    PositionalEncoding1D,
    PositionalEncodingPermute3D,
)
from .transformer import Transformer
from .priors import kl_criterion, Gaussian_Enc, Gaussian_TR


# ---------------------------------------------------------------------------------
# Backbone ablation: multi-scale edge-aware encoder ported from the Mamba
# forecaster (lung_edge_mamba_forecaster.py). Only what's needed to run it
# with edge_mode="none" and use_spatial_mamba=False is included here, so this
# file stays self-contained and doesn't pull in the mamba_ssm dependency.
# If you already import this class elsewhere, feel free to delete this block
# and `from <your_mamba_module> import MultiScaleEdgeEncoder` instead -- the
# class below is byte-for-byte the same.
# ---------------------------------------------------------------------------------

class DropPath(nn.Module):
    """Per-sample stochastic depth without a timm dependency."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GRN(nn.Module):
    """Global Response Normalization used in ConvNeXt-V2-style blocks."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is channels-last: (B, H, W, C)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block2D(nn.Module):
    """Depthwise local mixer + pointwise MLP + GRN residual block."""

    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = expansion * dim
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return residual + self.drop_path(x)


class FixedEdgeBank(nn.Module):
    """Fixed Sobel/Laplacian bank applied to a one-channel motion image.

    Unused when edge_mode="none" (kept only so EdgeGuidedStem is a drop-in
    match with the Mamba forecaster's version).
    """

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        laplace = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("laplace", laplace.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"FixedEdgeBank expects (B,1,H,W), received {tuple(x.shape)}."
            )
        gx = F.conv2d(x, self.sobel_x.to(dtype=x.dtype), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        lap = F.conv2d(x, self.laplace.to(dtype=x.dtype), padding=1)
        return torch.cat([gx, gy, magnitude, lap], dim=1)


from typing import Literal

EdgeMode = Literal["full", "no_gate", "none"]


class EdgeGuidedStem(nn.Module):
    """Configurable stem. With edge_mode="none" this is anatomy-branch-only:
    no Sobel/Laplacian, no edge stem, no gating."""

    def __init__(
        self,
        out_dim: int,
        edge_mode: EdgeMode = "full",
    ) -> None:
        super().__init__()

        if edge_mode not in {"full", "no_gate", "none"}:
            raise ValueError(
                f"Unknown edge_mode={edge_mode!r}. "
                "Choose from: 'full', 'no_gate', 'none'."
            )

        self.edge_mode = edge_mode

        self.anatomy_stem = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=out_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GroupNorm(self._groups(out_dim), out_dim),
            nn.SiLU(),
        )

        if edge_mode != "none":
            self.edge_bank = FixedEdgeBank()

            self.edge_stem = nn.Sequential(
                nn.Conv2d(
                    in_channels=4,
                    out_channels=out_dim,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.GroupNorm(self._groups(out_dim), out_dim),
                nn.SiLU(),
            )

            if edge_mode == "full":
                self.edge_gate = nn.Sequential(
                    nn.Conv2d(
                        in_channels=out_dim,
                        out_channels=out_dim,
                        kernel_size=1,
                    ),
                    nn.Sigmoid(),
                )
            else:
                self.edge_gate = None

            self.fuse = nn.Conv2d(
                in_channels=out_dim * 2,
                out_channels=out_dim,
                kernel_size=1,
            )

        else:
            self.edge_bank = None
            self.edge_stem = None
            self.edge_gate = None
            self.fuse = nn.Conv2d(
                in_channels=out_dim,
                out_channels=out_dim,
                kernel_size=1,
            )

    @staticmethod
    def _groups(channels: int) -> int:
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def forward(self, frame_pair: torch.Tensor) -> torch.Tensor:
        if frame_pair.ndim != 4 or frame_pair.shape[1] < 2:
            raise ValueError(
                "EdgeGuidedStem expects (B,C,H,W) with at least "
                "[current, reference]."
            )

        current = frame_pair[:, 0:1]
        reference = frame_pair[:, 1:2]

        if frame_pair.shape[1] >= 3:
            lung_mask = frame_pair[:, 2:3].clamp(0.0, 1.0)
            current = current * lung_mask
            reference = reference * lung_mask

        difference = current - reference

        anatomy_input = torch.cat(
            [current, reference, difference, difference.abs()], dim=1
        )
        anatomy = self.anatomy_stem(anatomy_input)

        if self.edge_mode == "none":
            return self.fuse(anatomy)

        edge_input = self.edge_bank(difference)
        edges = self.edge_stem(edge_input)

        if self.edge_mode == "full":
            gate = self.edge_gate(edges)
            anatomy = anatomy * (1.0 + gate)

        fused_input = torch.cat([anatomy, edges], dim=1)
        return self.fuse(fused_input)


class MultiScaleEdgeEncoder(nn.Module):
    """Hierarchical edge-aware encoder producing one token per frame.

    With edge_mode="none" and use_spatial_mamba=False (the config used for
    this backbone ablation), this is a plain multi-scale ConvNeXt-V2-style
    encoder: no edge branch, no spatial-Mamba mixer at the bottleneck. It
    pools average+max statistics from every scale and projects them to a
    single `latent_dim` token per frame.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        dims=(32, 64, 128),
        depths=(1, 1, 2),
        use_spatial_mamba: bool = False,
        drop_path: float = 0.05,
        edge_mode: EdgeMode = "none",
    ) -> None:
        super().__init__()
        if len(dims) != 3 or len(depths) != 3:
            raise ValueError("dims and depths must each contain exactly 3 stages.")

        self.stem = EdgeGuidedStem(dims[0], edge_mode=edge_mode)
        total_blocks = sum(depths)
        rates = torch.linspace(0, drop_path, total_blocks).tolist()
        rate_index = 0

        stages = []
        downsamples = []
        for stage_index, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtV2Block2D(dim, drop_path=rates[rate_index]))
                rate_index += 1
            stages.append(nn.Sequential(*blocks))
            if stage_index < len(dims) - 1:
                downsamples.append(
                    nn.Sequential(
                        nn.GroupNorm(1, dim),
                        nn.Conv2d(dim, dims[stage_index + 1], kernel_size=2, stride=2),
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.downsamples = nn.ModuleList(downsamples)
        # use_spatial_mamba is forced False for this ablation, so the True
        # branch (which needs BidirectionalSpatialMamba2 / mamba_ssm) is
        # never evaluated -- kept as a flag only for parity with the
        # forecaster's version.
        self.spatial_mamba = (
            self._build_spatial_mamba(dims[-1], drop_path)
            if use_spatial_mamba
            else nn.Identity()
        )

        pooled_dim = 2 * sum(dims)
        self.output_projection = nn.Sequential(
            nn.Linear(pooled_dim, latent_dim * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(latent_dim),
        )

    @staticmethod
    def _build_spatial_mamba(dim: int, drop_path: float):
        # Intentionally not implemented here: this backbone ablation always
        # runs with use_spatial_mamba=False. Import BidirectionalSpatialMamba2
        # from the Mamba forecaster module if you want to enable it.
        raise NotImplementedError(
            "use_spatial_mamba=True requires BidirectionalSpatialMamba2 "
            "(and mamba_ssm) from the Mamba forecaster module; not ported "
            "into this ablation."
        )

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        maximum = F.adaptive_max_pool2d(x, output_size=1).flatten(1)
        return torch.cat([avg, maximum], dim=1)

    def forward(self, frame_pair: torch.Tensor) -> torch.Tensor:
        x = self.stem(frame_pair)
        scale_vectors = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index == len(self.stages) - 1:
                x = self.spatial_mamba(x)
            scale_vectors.append(self._pool(x))
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        return self.output_projection(torch.cat(scale_vectors, dim=1))


# ---------------------------------------------------------------------------------


class CondiNet_Tr_priormulti(nn.Module):
    def __init__(
        self,
        num_inputs,
        horizon,
        in_channels,
        out_channels,
        n_heads,
        enc_layers,
        dec_layers,
        normalize_before,
        output_dim,
        condi_type,
        prior_type,
        backbone_type: str = "conv",
        michelangelo_embed_dim: int = 64,
        michelangelo_num_latents: int = 256,
    ):
        """
        backbone_type:
            "conv"         -- original per-frame conv stack (unchanged).
            "edge_encoder" -- MultiScaleEdgeEncoder from the Mamba forecaster,
                               with edge_mode="none" and use_spatial_mamba=False.
                               Everything downstream (temporal transformer,
                               prior/posterior nets, decoder) is untouched, so
                               this isolates the effect of the backbone alone.
            "michelangelo" -- input is precomputed, pooled Michelangelo
                               latents: one (B, michelangelo_embed_dim) vector
                               per frame (mean-pooled over the encoder's
                               num_latents tokens upstream, in the data
                               pipeline -- NOT inside this model). A 1x1 Conv2d
                               projects that to hidden_dim, matching the exact
                               (B, hidden_dim, 1, 1) convention edge_encoder
                               already uses, so input_proj/pe3d/Gaussian_TR/
                               Gaussian_Enc (all parameterized by feat_size)
                               need no changes at all.

                               The OUTPUT, however, is NOT pooled: self.linear
                               is sized to produce a full
                               (num_latents * michelangelo_embed_dim) vector
                               per predicted timestep, reshaped in forward()
                               to (B, num_latents, michelangelo_embed_dim) --
                               exactly the shape ShapeAsLatentPerceiver.decode()
                               expects, with no separate expansion head. This
                               is a deliberate asymmetry: pooled in (cheap,
                               matches edge_encoder's pattern exactly), full
                               sequence out (so real-geometry decoding/
                               validation needs no extra learned component).
        """

        super().__init__()
        nb_convs = len(out_channels)
        self.horizon = horizon
        self.backbone_type = backbone_type
        hidden_dim = out_channels[-1]
        self.hidden_dim = hidden_dim
        self.michelangelo_embed_dim = michelangelo_embed_dim
        self.michelangelo_num_latents = michelangelo_num_latents
        norm = nn.BatchNorm2d
        cor_custom_stride = [2, 2, 2, (2, 1)]

        if backbone_type == "edge_encoder":
            # The encoder pools spatial dims away internally and returns a
            # single (B, hidden_dim) token per frame -- see _encode_frame().
            self.backbone = MultiScaleEdgeEncoder(
                latent_dim=hidden_dim,
                dims=(32, 64, 128),
                depths=(1, 1, 2),
                use_spatial_mamba=False,
                edge_mode="none",
            )
            # Downstream input_proj/pos-encoding treat the backbone output as
            # a (B, C, H, W) feature map; since this backbone already pooled
            # to a point, that "feature map" is 1x1.
            feat_h, feat_w = 1, 1

        elif backbone_type == "michelangelo":
            # Pooled (B, embed_dim, 1, 1) in -> (B, hidden_dim, 1, 1) out.
            # A 1x1 conv naturally preserves that shape, so _encode_frame
            # needs no branch for this backbone type (unlike edge_encoder,
            # which needs an explicit reshape since it pools INSIDE forward).
            self.backbone = nn.Conv2d(michelangelo_embed_dim, hidden_dim, kernel_size=1)
            feat_h, feat_w = 1, 1

        else:
            self.backbone = list()
            for i in range(nb_convs):
                if i == 0:
                    in_ch = in_channels
                else:
                    in_ch = out_channels[i - 1]

                if condi_type == "1":
                    self.backbone += [
                        nn.Conv2d(
                            in_ch, out_channels[i], kernel_size=3, padding=1, stride=2
                        )
                    ]
                else:
                    self.backbone += [
                        nn.Conv2d(
                            in_ch,
                            out_channels[i],
                            kernel_size=3,
                            padding=1,
                            stride=cor_custom_stride[i],
                        )
                    ]
                self.backbone += [norm(out_channels[i])]
                self.backbone += [nn.ReLU(True)]

                self.backbone += [
                    nn.Conv2d(
                        out_channels[i], out_channels[i], kernel_size=3, padding=1, stride=1
                    )
                ]
                self.backbone += [norm(out_channels[i])]
                self.backbone += [nn.ReLU(True)]
                self.backbone += [nn.Dropout(0.2)]
            self.backbone = nn.Sequential(*self.backbone)
            feat_h, feat_w = (8, 8) if condi_type == "2" else (5, 4)

        self.num_frames = num_inputs
        self.num_queries = horizon
        self.query_learn = nn.Embedding(horizon, hidden_dim)
        self.pe3d = PositionalEncodingPermute3D(hidden_dim)
        self.pe_temp = PositionalEncoding1D(
            hidden_dim
        )  # hidden_dim   hidden_dim*(self.num_frames+1)

        self.input_proj = nn.Conv3d(
            self.num_frames, self.num_frames, kernel_size=(1, feat_h, feat_w)
        )
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=normalize_before,
            return_intermediate_dec=False,
        )
        self.bn = nn.BatchNorm3d(out_channels[-1])

        # For "michelangelo", output_dim is overridden regardless of what was
        # passed in: the final projection must produce a FULL latent sequence
        # (num_latents * embed_dim) per predicted timestep, not a small
        # pooled vector, so decode() can consume it directly with no
        # separate expansion head.
        if backbone_type == "michelangelo":
            output_dim = michelangelo_num_latents * michelangelo_embed_dim
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.prior_type = prior_type

        self.prior_nets = nn.ModuleList()
        self.posterior_nets = nn.ModuleList()
        for i in range(self.num_queries):
            if prior_type == "learned_conv":
                self.prior_nets.append(
                    Gaussian_Enc(
                        in_ch=self.num_queries,
                        nb_in=self.num_frames,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                        channels=hidden_dim,
                        feat_size=(feat_h, feat_w),
                    )
                )
                self.posterior_nets.append(
                    Gaussian_Enc(
                        in_ch=self.num_queries + i + 1,
                        nb_in=self.num_frames,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                        channels=hidden_dim,
                        feat_size=(feat_h, feat_w),
                    )
                )
            else:
                self.prior_nets.append(
                    Gaussian_TR(
                        in_ch=self.num_frames,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                        n_layers=1,
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.1,
                        activation="relu",
                        normalize_before=normalize_before,
                        feat_size=(feat_h, feat_w),
                    )
                )
                self.posterior_nets.append(
                    Gaussian_TR(
                        in_ch=self.num_frames + i + 1,
                        d_model=hidden_dim,
                        output_size=hidden_dim,
                        n_layers=1,
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.1,
                        activation="relu",
                        normalize_before=normalize_before,
                        feat_size=(feat_h, feat_w),
                    )
                )

    def _encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Run one (B, C, H, W) frame pair through the backbone and return a
        (B, hidden_dim, H', W') map, whichever backbone is active.

        The conv and michelangelo backbones already return that shape
        directly (a 1x1 conv preserves (B,C,1,1) -> (B,hidden_dim,1,1)).
        MultiScaleEdgeEncoder pools straight to (B, hidden_dim); reshape it
        to (B, hidden_dim,1,1) so the rest of forward() (stacking, input_proj,
        positional encodings) doesn't need to branch on backbone_type.
        """
        feat = self.backbone(frame)
        if self.backbone_type == "edge_encoder":
            feat = feat[:, :, None, None]
        return feat

    def forward(self, Ipast, Ifuture=None):
        if Ifuture is not None:
            img_feats_past, img_feats_future = [], []
            # --------------- Backbone -------------------------------------
            for i in range(self.num_frames):
                img_feats_past.append(self._encode_frame(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            src_proj = self.input_proj(img_feats_past)  # [B, 3, 128, 1, 1]
            for j in range(self.num_queries):
                img_feats_future.append(self._encode_frame(Ifuture[:, :, j, :, :]))
            img_feats_future = torch.stack(img_feats_future, dim=2).permute(
                0, 2, 1, 3, 4
            )
            # ---------------- TR enc ----------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8] (BEFORE) now its # [B, 3, 128, 1, 1]
            src_proj = src_proj.flatten(-3)  # [B, 3, 128]
            pos = pos.flatten(-3)  # [bs, 3, 48, 64] (BEFORE, now its [B, 3, 128])
            # ----------------------------------------------------------------
            kl_loss = 0
            zts = []
            for k in range(self.num_queries):
                _, mu1, logvar1 = self.prior_nets[k](img_feats_past)
                who = torch.cat(
                    [img_feats_past, img_feats_future[:, : k + 1, :, :, :]], dim=1
                )
                z_t, mu2, logvar2 = self.posterior_nets[k](who)
                kl_loss += kl_criterion(mu1, logvar1, mu2, logvar2, Ipast.shape[0])
                zts.append(z_t)

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            zts = torch.stack(zts, dim=1)
            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            # ----------------------------------------------------------------

        else:
            img_feats_past = []
            for i in range(self.num_frames):
                img_feats_past.append(self._encode_frame(Ipast[:, :, i, :, :]))
            img_feats_past = torch.stack(img_feats_past, dim=2).permute(0, 2, 1, 3, 4)
            src_proj = self.input_proj(img_feats_past)
            # ----------------------------------------------------------------
            pos = self.pe3d(src_proj).to(src_proj.dtype)  # [bs, 3, 48, 8, 8]
            src_proj = src_proj.flatten(-3)
            pos = pos.flatten(-3)  # [bs, 3, 48, 64]
            # ----------------------------------------------------------------
            zts = []
            for k in range(self.num_queries):
                z_t, mu1, logvar1 = self.prior_nets[k](img_feats_past)
                zts.append(z_t)

            the_queries = self.query_learn.weight.unsqueeze(0).repeat(
                src_proj.shape[0], 1, 1
            )
            zts = torch.stack(zts, dim=1)
            if self.prior_type == "learned_conv":
                dec_input = torch.cat([the_queries, zts], dim=1)
            else:
                dec_input = torch.cat([the_queries, zts[:, :, 0, :]], dim=1)
            pos_dec_in = self.pe_temp(dec_input).to(dec_input.dtype)
            # ----------------------------------------------------------------
            kl_loss = None

        # ----------------------------------------------------------------
        hs = self.transformer(
            enc_in=src_proj, enc_pos=pos, dec_in=dec_input, dec_pos=pos_dec_in
        )
        hs = hs.permute(1, 0, 2)
        # ----------------------------------------------------------------
        out_feats = []
        for t in range(self.horizon):
            flat = self.linear(hs[:, t, :])
            if self.backbone_type == "michelangelo":
                # (B, num_latents * embed_dim) -> (B, num_latents, embed_dim),
                # a free reshape (no parameters) -- directly what
                # ShapeAsLatentPerceiver.decode() expects.
                flat = flat.view(
                    -1, self.michelangelo_num_latents, self.michelangelo_embed_dim
                )
            out_feats.append(flat)
        return out_feats, kl_loss