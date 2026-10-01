from __future__ import annotations

"""
Train / validate / test the Mamba-conditioned DDIM model with strong multiscale
3D reference-volume conditioning.

Compared with the previous diffusion training file:
- same lung dataset / 5-fold workflow
- same frozen Voxelmorph DVF targets
- same VAE latent-diffusion option
- same DDIM train/inference path
- same NCC / MSE / SSIM / geometric-error test outputs
- Mamba receives only observed input frames
- NEW: Vref is encoded spatially and injected at every UNet resolution
- NEW: temporal Mamba feature conditions every residual block through FiLM
- real stride-2 diffusion UNet hierarchy is used
"""

# =========================
# Standard library
# =========================
import argparse
import datetime
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "offline"

# =========================
# Third-party
# =========================
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from barbar import Bar
from skimage.metrics import structural_similarity as ssim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb

# =========================
# Project / local imports
# =========================
from diffusion_spatial_mamba_ref import (
    Diffusion_TR as DiffusionMambaSpatialRefModel,
)

import sys
sys.path.insert(0, "/home/fellahr/4D_MoPred") 

from data.data_loaders.navigator_4d import NAVIGATOR_4D_Dataset_multitime_v2
from models import SpatialTransformer, Voxelmorph
from models.diffusion.vae.dvf_vae_freq import FreqAware_DVF_VAE as DVF_VAE
from utils.early_stopping import EarlyStopping
from utils.io import (
    cond_mkdir,
    custom_load,
    custom_save,
    extract_phase_label,
    make_run_dirs,
    save_params_txt,
    save_tensor_as_nifti,
)
from utils.losses import ncc_loss
from utils.tre_helpers import _canonical_patient_id, load_gtv_reference_landmarks, gtv_tre_from_backward_dvfs


# ============================================================================
# Arguments
# ============================================================================
parser = argparse.ArgumentParser()

# ------------------
# Run mode and paths
# ------------------
parser.add_argument("--train_test", type=str, required=True, choices=["train", "test"])
parser.add_argument("--logging_dir", required=True)
parser.add_argument(
    "--data_dir",
    default="/home/fellahr/data/dataset_split",
)
parser.add_argument("--name", type=str, default="diffusion_mamba_spatial_ref")
parser.add_argument("--gpu_idx", type=int, default=0)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help=(
        "For test: parent run directory containing <fold>/model_best.pth. "
        "For train: optional single model checkpoint used to initialize each fold."
    ),
)

# -------------------------
# Training hyperparameters
# -------------------------
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--val_every", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=123)

# -------------------------
# Data / temporal setup
# -------------------------
parser.add_argument("--condi_type", type=str, default="2", choices=["1", "2"])
parser.add_argument("--nb_inputs", type=int, default=3)
parser.add_argument("--tp", type=int, default=1)

parser.add_argument("--temporal_predictor", type=str, default="mamba",
                     choices=["mamba", "condinet_tr", "convlstm", "lstm", "gru"])
# -------------------------
# Mamba conditioning network
# -------------------------
parser.add_argument("--preset", type=str, default="lite")
parser.add_argument("--temporal_backend", type=str, default="mamba2")
parser.add_argument("--max_history", type=int, default=10)
parser.add_argument("--temporal_depth", type=int, default=1)
parser.add_argument("--d_state", type=int, default=64)
parser.add_argument("--d_conv", type=int, default=4)
parser.add_argument("--mamba_expand", type=int, default=2)
parser.add_argument("--mamba_headdim", type=int, default=64)
parser.add_argument("--mamba3_mimo", action="store_true")
parser.add_argument("--mamba3_mimo_rank", type=int, default=4)
parser.add_argument(
    "--cond_net_checkpoint", # path to pretrained temporal predictor
    type=str,
    default="",
    help="Checkpoint for the pretrained Mamba temporal predictor.",
)
parser.add_argument(
    "--freeze_cond_net",
    dest="freeze_cond_net",
    action="store_true",
    default=True,
    help="Freeze the pretrained Mamba (default).",
)
parser.add_argument(
    "--finetune_cond_net",
    dest="freeze_cond_net",
    action="store_false",
    help="Fine-tune Mamba jointly with diffusion instead of freezing it.",
)


# -------------------------
# CondiNet Transformer conditionning
# -------------------------
parser.add_argument("--prelatent_size", type=int,  default=64)
parser.add_argument("--enc_layers",     type=int,  default=5)
parser.add_argument("--dec_layers",     type=int,  default=5)
parser.add_argument("--n_heads",        type=int,  default=8)
parser.add_argument("--norm_before",    type=bool, default=True)
parser.add_argument("--condi_channels", nargs="+", type=int, default=[16, 32, 64, 128])
parser.add_argument("--enc_channels",   nargs="+", type=int, default=[16, 32, 64, 128])
parser.add_argument("--prior_type",     type=str,  default="learned", choices=["learned", "none"])

# -------------------------
# LSTM temporal predictor
# -------------------------
parser.add_argument("--lstm_hidden_dim", type=int, default=128)
parser.add_argument("--lstm_num_layers", type=int, default=1)
parser.add_argument("--lstm_dropout", type=float, default=0.0)

# -------------------------
# GRU temporal predictor
# -------------------------
parser.add_argument("--gru_hidden_dim", type=int, default=128)
parser.add_argument("--gru_num_layers", type=int, default=1)
parser.add_argument("--gru_dropout", type=float, default=0.0)
# -------------------------
# Diffusion UNet
# -------------------------
parser.add_argument("--T", type=int, default=1000)
parser.add_argument(
    "--beta_max",
    type=float,
    default=0.2,
    help="Upper clamp for cosine betas. 0.2 gives a near-zero terminal alpha_bar.",
)
parser.add_argument("--ddim_steps", type=int, default=100)
parser.add_argument("--eta", type=float, default=0.0, help="0 = deterministic DDIM")
parser.add_argument("--unet_base_ch", type=int, default=128)
parser.add_argument(
    "--unet_ch_mults",
    nargs="+",
    type=int,
    default=[1, 2],
    help="Real multiscale levels. Example: --unet_ch_mults 1 2 4",
)
parser.add_argument("--time_dim", type=int, default=128)
parser.add_argument("--n_attn_heads", type=int, default=8)
parser.add_argument("--num_res_blocks", type=int, default=2)
parser.add_argument("--res_dropout", type=float, default=0.0)
parser.add_argument(
    "--use_self_attn",
    dest="use_self_attn",
    action="store_true",
    default=True,
)
parser.add_argument(
    "--no_self_attn",
    dest="use_self_attn",
    action="store_false",
)
parser.add_argument(
    "--diffusion_prediction_type",
    type=str,
    default="x0_small",
    choices=["noise", "x0_small"],
)

# -------------------------
# ConvLSTM temporal predictor
# -------------------------
parser.add_argument("--convlstm_hidden_ch", type=int, default=128)
parser.add_argument("--convlstm_kernel_size", type=int, default=3)
# -------------------------
# Strong spatial Vref encoder
# -------------------------
parser.add_argument(
    "--ref_base_channels",
    type=int,
    default=64,
    help="Base width of full-resolution 3D Vref encoder.",
)
parser.add_argument(
    "--ref_encoder_depth",
    type=int,
    default=3,
    help=(
        "Number of stride-2 Vref encoder stages before the diffusion pyramid. "
        "For 128x80x64 and VAE depth=4, depth=4 naturally gives 8x5x4."
    ),
)
parser.add_argument("--ref_max_channels", type=int, default=256)
parser.add_argument(
    "--ref_mode",
    type=str,
    default="learned",
    choices=["learned", "interpolate", "none"],
    help=(
        "How to encode the Vref reference volume. "
        "'learned' = 3D UNet encoder (default). "
        "'interpolate' = trilinear downsampling to each pyramid level. "
        "'none' = no Vref conditioning at all."
    ),
)

# -------------------------
# VAE / latent diffusion
# -------------------------
parser.add_argument(
    "--use_latent_diffusion",
    dest="use_latent_diffusion",
    action="store_true",
    default=True,
)
parser.add_argument(
    "--direct_dvf_diffusion",
    dest="use_latent_diffusion",
    action="store_false",
    help="Disable VAE and diffuse a downsampled DVF directly.",
)
parser.add_argument(
    "--vae_checkpoint",
    type=str,
    default="/home/fellahr/4D_MoPred/pretrained_models/VAE/2nd_vae_freq_32_16_4.pth",
)
parser.add_argument("--vae_base_channels", type=int, default=32)
parser.add_argument("--vae_latent_channels", type=int, default=16)
parser.add_argument("--vae_depth", type=int, default=4)
parser.add_argument(
    "--latent_use_mean",
    dest="latent_use_mean",
    action="store_true",
    default=True,
)
parser.add_argument("--use_alpf", dest="use_alpf", action="store_true", default=True)
parser.add_argument("--use_ahpf", dest="use_ahpf", action="store_true", default=True)
parser.add_argument(
    "--sample_vae_latent",
    dest="latent_use_mean",
    action="store_false",
)
parser.add_argument(
    "--freeze_vae",
    dest="freeze_vae",
    action="store_true",
    default=True,
)
parser.add_argument(
    "--finetune_vae",
    dest="freeze_vae",
    action="store_false",
)
parser.add_argument("--dvf_downsample_factor", type=float, default=1.0)
parser.add_argument("--gtv_landmarks_csv", type=str, default="/home/fellahr/data/gtv_landmarks_final_128x80x64_1x1x2.csv")
# -------------------------
# Classifier-free guidance
# -------------------------
# parser.add_argument(
#     "--cfg_dropout",
#     type=float,
#     default=0.30,
#     help="Drops only Mamba temporal context; Vref is always retained.",
# )
# parser.add_argument(
#     "--cfg_guidance_scale",
#     type=float,
#     default=3.0,
#     help="Start at 1.0 for quantitative DVF regression; sweep later if desired.",
# )

parser.add_argument(
    "--cfg_dropout_temporal", type=float, default=0.15,
    help="Prob. of replacing the Mamba temporal token with the learned null token in training.",
)
parser.add_argument(
    "--cfg_dropout_vref", type=float, default=0.15,
    help="Prob. of replacing Vref reference features with the learned null token in training.",
)
parser.add_argument(
    "--cfg_scale_temporal", type=float, default=2.0,
    help="Guidance scale on the temporal/phase axis at inference.",
)
parser.add_argument(
    "--cfg_scale_vref", type=float, default=2.0,
    help="Guidance scale on the Vref/anatomy axis at inference.",
)

opt = parser.parse_args()
print("\n".join([f"{k}: {v}" for k, v in vars(opt).items()]))


# ============================================================================
# Global config / reproducibility / device
# ============================================================================
VOL_SIZE = (128, 80, 64)
VM_CHECKPOINT = "/home/fellahr/4D_MoPred/pretrained_models/VM_trained_on_lungs.pth"

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)

if torch.cuda.is_available():
    visible_count = torch.cuda.device_count()
    if opt.gpu_idx < 0 or opt.gpu_idx >= visible_count:
        raise RuntimeError(
            f"Invalid --gpu_idx {opt.gpu_idx}: process sees {visible_count} CUDA device(s). "
            "If launching with CUDA_VISIBLE_DEVICES=1, use --gpu_idx 0."
        )
    # Important for Triton/Mamba2: selected device must be the current device.
    torch.cuda.set_device(opt.gpu_idx)
    device = torch.device(f"cuda:{opt.gpu_idx}")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")


# ============================================================================
# Frozen Voxelmorph target generator + evaluation STN
# ============================================================================
vm = Voxelmorph(
    VOL_SIZE,
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16],
    full_size=True,
).to(device)

stn = SpatialTransformer(VOL_SIZE).to(device)



# ============================================================================
# Model construction
# ============================================================================
def build_temporal_kwargs(opt):
    if opt.temporal_predictor == "mamba":
        return dict(
            preset=opt.preset,
            temporal_backend=opt.temporal_backend,
            max_history=opt.max_history,
            temporal_depth=opt.temporal_depth,
            d_state=opt.d_state,
            d_conv=opt.d_conv,
            expand=opt.mamba_expand,
            headdim=opt.mamba_headdim,
            mamba3_mimo=opt.mamba3_mimo,
            mamba3_mimo_rank=opt.mamba3_mimo_rank,
        )
    if opt.temporal_predictor == "condinet_tr":
        return dict(
            out_channels=opt.condi_channels,
            n_heads=opt.n_heads,
            enc_layers=opt.enc_layers,
            dec_layers=opt.dec_layers,
            normalize_before=opt.norm_before,
            condi_type=opt.condi_type,
            prior_type=opt.prior_type,
        )

    if opt.temporal_predictor == "convlstm":
        return dict(
            out_channels=opt.condi_channels,
            condi_type=opt.condi_type,
            hidden_ch=opt.convlstm_hidden_ch,
            kernel_size=opt.convlstm_kernel_size,
        )
    if opt.temporal_predictor == "lstm":
        return dict(
            out_channels=opt.condi_channels,
            condi_type=opt.condi_type,
            hidden_dim=opt.lstm_hidden_dim,
            num_layers=opt.lstm_num_layers,
            dropout=opt.lstm_dropout,
        )

    if opt.temporal_predictor == "gru":
        return dict(
            out_channels=opt.condi_channels,
            condi_type=opt.condi_type,
            hidden_dim=opt.gru_hidden_dim,
            num_layers=opt.gru_num_layers,
            dropout=opt.gru_dropout,
        )
    raise ValueError(opt.temporal_predictor)


def build_model() -> DiffusionMambaSpatialRefModel:
    """Build a fresh independent model for one cross-validation fold."""
    vae = None
    if opt.use_latent_diffusion:
        vae = DVF_VAE(
            in_channels=3,
            base_channels=opt.vae_base_channels,
            latent_channels=opt.vae_latent_channels,
            depth=opt.vae_depth,
            use_alpf = opt.use_alpf, # eventually add a param
            use_ahpf = opt.use_ahpf, # eventually add a param
        ).to(device)
        custom_load(vae, opt.vae_checkpoint, device)
        if opt.freeze_vae:
            for p in vae.parameters():
                p.requires_grad_(False)
            vae.eval()

    model = DiffusionMambaSpatialRefModel(
        num_frames=opt.nb_inputs,
        horizon=opt.tp,
        vol_size=VOL_SIZE,
        pre_latent_dim=opt.prelatent_size,
        cond_net_checkpoint=opt.cond_net_checkpoint,
        freeze_cond_net=opt.freeze_cond_net,
        T=opt.T,
        ddim_steps=opt.ddim_steps,
        beta_max=opt.beta_max,
        eta=opt.eta,
        prediction_type=opt.diffusion_prediction_type,
        unet_base_ch=opt.unet_base_ch,
        unet_ch_mults=tuple(opt.unet_ch_mults),
        time_dim=opt.time_dim,
        n_attn_heads=opt.n_attn_heads,
        use_self_attn=opt.use_self_attn,
        num_res_blocks=opt.num_res_blocks,
        res_dropout=opt.res_dropout,
        ref_mode=opt.ref_mode,
        ref_base_channels=opt.ref_base_channels,
        ref_encoder_depth=opt.ref_encoder_depth,
        ref_max_channels=opt.ref_max_channels,
        use_latent_diffusion=opt.use_latent_diffusion,
        vae=vae,
        vae_latent_channels=(
            opt.vae_latent_channels if opt.use_latent_diffusion else None
        ),
        latent_use_mean=opt.latent_use_mean,
        freeze_vae=opt.freeze_vae,
        dvf_downsample_factor=opt.dvf_downsample_factor,
        temporal_predictor=opt.temporal_predictor,
        temporal_predictor_kwargs=build_temporal_kwargs(opt),
        cfg_dropout_temporal=opt.cfg_dropout_temporal,
        cfg_dropout_vref=opt.cfg_dropout_vref,
        cfg_scale_temporal=opt.cfg_scale_temporal,
        cfg_scale_vref=opt.cfg_scale_vref,
    ).to(device)

    return model




# ============================================================================
# Build Mamba input sequence
# ============================================================================
# def build_Iseq(ref_volume, input_volume_list, opt, device):
#     """Build observed respiratory sequence for Mamba.

#     Returns [B, 2, nb_inputs, H, W].

#     Unlike the Transformer training script, no future target slices are appended.
#     Mamba receives only observed history and predicts horizon future tokens itself.
#     """
#     if opt.condi_type == "1":
#         z = ref_volume.shape[2] // 2
#         ref_slice = ref_volume[:, :, z, :, :]
#         seq = []
#         for q in range(len(input_volume_list)):
#             cur_slice = input_volume_list[q].unsqueeze(1)[:, :, z, :, :]
#             seq.append(
#                 torch.cat(
#                     [cur_slice.to(device), ref_slice.to(device)],
#                     dim=1,
#                 )
#             )
#     else:
#         y = ref_volume.shape[3] // 2
#         ref_slice = ref_volume[:, :, :, y, :]
#         seq = []
#         for q in range(len(input_volume_list)):
#             cur_slice = input_volume_list[q].unsqueeze(1)[:, :, :, y, :]
#             seq.append(
#                 torch.cat(
#                     [cur_slice.to(device), ref_slice.to(device)],
#                     dim=1,
#                 )
#             )

#     Iseq = torch.stack(seq, dim=2).contiguous().to(device)
#     return Iseq

def build_Iseq(ref_volume, input_volume_list, opt, device, future_volume_list=None):
    if opt.condi_type == "1":
        idx, dim = ref_volume.shape[2] // 2, 2
    else:
        idx, dim = ref_volume.shape[3] // 2, 3

    def _slice(vol):
        return vol.unsqueeze(1).narrow(dim, idx, 1).squeeze(dim)  # generalizes the old z/y indexing

    ref_slice = ref_volume.narrow(dim, idx, 1).squeeze(dim)
    seq = [torch.cat([_slice(v).to(device), ref_slice.to(device)], dim=1) for v in input_volume_list]

    if future_volume_list is not None:
        seq += [torch.cat([_slice(v).to(device), ref_slice.to(device)], dim=1) for v in future_volume_list]

    return torch.stack(seq, dim=2).contiguous().to(device)
# ============================================================================
# Training
# ============================================================================
def train(folds=None, dir_name=None):
    model = build_model()

    # Optional initialization checkpoint for training.
    if opt.checkpoint and os.path.isfile(opt.checkpoint):
        print(f"Initializing model from checkpoint: {opt.checkpoint}")
        custom_load(model, opt.checkpoint, device)

    custom_load(vm, VM_CHECKPOINT, device)
    vm.eval()
    for p in vm.parameters():
        p.requires_grad_(False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        threshold=1e-3,
        patience=2,
        min_lr=1e-7,
    )
    early_stopper = EarlyStopping(patience=8, verbose=True, delta=1e-4)

    log_dir, run_dir = make_run_dirs(opt.logging_dir, dir_name)
    save_params_txt(opt, log_dir)
    writer = SummaryWriter(run_dir)
    wandb.init(
        project="4DCT Lungs Motion Model",
        name=f"{dir_name}",
        config=vars(opt),
        dir=log_dir,
        reinit=True,
        mode="offline",
    )

    train_set = NAVIGATOR_4D_Dataset_multitime_v2(opt.data_dir, nb_inputs=opt.nb_inputs, sequence_list=folds[0], nb_pred=opt.tp)
    valid_set = NAVIGATOR_4D_Dataset_multitime_v2(opt.data_dir, nb_inputs=opt.nb_inputs, sequence_list=folds[1], nb_pred=opt.tp, mode="val")

    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    global_step = 0
    best_val_loss = np.inf
    val_every = max(opt.val_every, 1)

    print(f"Begin training {opt.temporal_predictor} + multiscale-Vref conditioned DDIM diffusion...")

    for epoch in range(opt.epochs):
        print(f"\nEpoch {epoch}")

        epoch_loss_sum = 0.0
        epoch_steps = 0
        model.train()

        for ref_volume, input_volume_list, current_volume_list, _ in Bar(train_loader):
            optimizer.zero_grad(set_to_none=True)

            ref_volume = ref_volume.unsqueeze(1).to(device)
            current_vols = [vol.unsqueeze(1).to(device) for vol in current_volume_list]

            # Ground-truth DVFs are generated by frozen Voxelmorph, exactly as in
            # the Transformer diffusion training script.
            with torch.no_grad():
                dvf_gt = [vm(ref_volume, v) for v in current_vols]

            # Mamba only receives the observed past sequence.
            needs_future = (
                model.cond_net.training
                and getattr(
                    model.cond_net,
                    "requires_future_for_training",
                    False,
                )
            )

            future_vols = (
                current_volume_list
                if needs_future
                else None
            )

            Iseq = build_Iseq(
                ref_volume,
                input_volume_list,
                opt,
                device,
                future_volume_list=future_vols,
            )

            diff_loss = model(
                ref_volume,
                Iseq,
                dvf_gt_list=dvf_gt,
                return_components=False,
            )

            loss = diff_loss
            loss.backward()

            if opt.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, opt.grad_clip)

            optimizer.step()

            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

            writer.add_scalar("train/diff_loss", float(diff_loss.item()), global_step)
            writer.add_scalar("train/total_loss", float(loss.item()), global_step)
            writer.add_scalar(
                "train/lr",
                float(optimizer.param_groups[0]["lr"]),
                global_step,
            )
            wandb.log(
                {
                    "train/diff_loss": float(diff_loss.item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=global_step,
            )
            global_step += 1

        epoch_loss = epoch_loss_sum / max(epoch_steps, 1)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        if epoch % val_every == 0:
            val_loss, val_metrics = validate(model, vm, valid_loader, opt, device)
            monitored_val = float(val_loss)

            old_best = best_val_loss
            if monitored_val < best_val_loss:
                best_val_loss = monitored_val
                custom_save(model, os.path.join(log_dir, "model_best.pth"))
                print(
                    f"\nVal loss improved from {old_best:.6f} to "
                    f"{monitored_val:.6f}. Saved model_best.pth"
                )
            else:
                print(
                    f"\nVal loss did not improve from {best_val_loss:.6f}; "
                    f"current={monitored_val:.6f}"
                )

            writer.add_scalar("val/loss", float(val_loss), global_step)
            writer.add_scalar(
                "val/mse_image",
                float(val_metrics["mse_image"]),
                global_step,
            )
            wandb.log(
                {
                    "val/loss": float(val_loss),
                    "val/mse_image": float(val_metrics["mse_image"]),
                },
                step=global_step,
            )

            scheduler.step(monitored_val)
            early_stopper(monitored_val)

        if early_stopper.early_stop:
            print("Early stopping")
            break

    writer.close()
    wandb.finish()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Validation
# ============================================================================
@torch.no_grad()
def validate(model, vm, valid_loader, opt, device):
    model.eval()
    vm.eval()

    val_mse_image = 0.0
    n = 0
    mse_metric = nn.MSELoss(reduction="mean").to(device)

    for ref_volume, input_volume_list, current_volume_list, _ in Bar(valid_loader):
        ref_volume = ref_volume.unsqueeze(1).to(device)
        current_vols = [vol.unsqueeze(1).to(device) for vol in current_volume_list]

        Iseq = build_Iseq(ref_volume, input_volume_list, opt, device)
        generated_dvf_list, generated_Vn_list = model.forward_inference(
            ref_volume,
            Iseq,
            # cfg_scale=opt.cfg_guidance_scale,
            cfg_scale_temporal=opt.cfg_scale_temporal,
            cfg_scale_vref=opt.cfg_scale_vref,
        )

        avg_mse_image = 0.0
        for tt in range(opt.tp):
            avg_mse_image += mse_metric(
                generated_Vn_list[tt],
                current_vols[tt],
            ).item()

        avg_mse_image /= opt.tp
        val_mse_image += avg_mse_image
        n += 1

    val_mse_image /= max(n, 1)
    return val_mse_image, {"mse_image": val_mse_image}


# ============================================================================
# Test
# ============================================================================
@torch.no_grad()
def test(fold=None, dir_name=None):
    model = build_model()

    model_ckpt = os.path.join(opt.checkpoint, "model_best.pth")
    if not os.path.isfile(model_ckpt):
        raise FileNotFoundError(f"Could not find test checkpoint: {model_ckpt}")

    custom_load(model, model_ckpt, device)
    custom_load(vm, VM_CHECKPOINT, device)

    model.eval()
    vm.eval()

    test_set = NAVIGATOR_4D_Dataset_multitime_v2(opt.data_dir, nb_inputs=opt.nb_inputs, sequence_list=fold, nb_pred=opt.tp, mode="test")
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
    )

    vol_dir = os.path.join(opt.logging_dir, "test", dir_name, "volumes")
    fig_dir = os.path.join(opt.logging_dir, "test", dir_name, "figures")
    cond_mkdir(vol_dir)
    cond_mkdir(fig_dir)

    MSE_loss, NCC_loss, SSIM_loss, geo_error, GTV_TRE = [], [], [], [], []
    mse = nn.MSELoss(reduction="mean").to(device)

    gtv_landmarks = load_gtv_reference_landmarks(opt.gtv_landmarks_csv)
    gtv_missing_patients = set()
    gtv_nonconverged_pred = 0
    gtv_nonconverged_gt = 0

    phase_labels = [str(p) for p in range(0, 100, 10)]
    phase_metrics = {
        phase: {
            "mse": [],
            "ncc": [],
            "ssim": [],
            "geo_error": [],
            "gtv_tre_mm": [],
        }
        for phase in phase_labels
    }

    for idx, [ref_volume, input_volume_list, current_volume_list, vol_file] in enumerate(
        Bar(test_loader)
    ):
        patient_no = vol_file[0][0].split("/")[-2]
        print(f"\nInference for patient #{patient_no}\n")

        ref_volume = ref_volume.unsqueeze(1).to(device)
        vmorph_volume, dvf = [], []

        for vol in range(len(current_volume_list)):
            current_volume_list[vol] = current_volume_list[vol].unsqueeze(1).to(device)
            dvf_vm = vm(ref_volume, current_volume_list[vol])
            dvf.append(dvf_vm)
            vmorph_volume.append(stn(ref_volume, dvf_vm))

        # Same observed Mamba input used in train/validation/test.
        Iseq = build_Iseq(ref_volume, input_volume_list, opt, device)

        generated_dvf_list, generated_current_volume = model.forward_inference(
            ref_volume,
            Iseq,
            # cfg_scale=opt.cfg_guidance_scale,
            cfg_scale_temporal=opt.cfg_scale_temporal,
            cfg_scale_vref=opt.cfg_scale_vref,
        )

        aff = nib.load(vol_file[0][0]).affine
        avg_ncc, avg_mse, avg_ssim = 0.0, 0.0, 0.0
        this_sample = []
        this_gtv_tre = []

        patient_key = _canonical_patient_id(patient_no)
        patient_gtv = gtv_landmarks.get(patient_key)
        if gtv_landmarks and patient_gtv is None:
            gtv_missing_patients.add(patient_key)

        for tp in range(opt.tp):
            saved_phase = vol_file[tp][0].split("/")[-1]
            volume_idx = vol_file[tp][0].split("/")[-2]
            patient_vol_dir = os.path.join(vol_dir, volume_idx)
            cond_mkdir(patient_vol_dir)

            save_tensor_as_nifti(
                vmorph_volume[tp][0, 0],
                "vm_volumes",
                patient_vol_dir,
                iter=saved_phase,
                aff=aff,
            )
            save_tensor_as_nifti(
                generated_current_volume[tp][0, 0],
                "generated_volumes",
                patient_vol_dir,
                iter=saved_phase,
                aff=aff,
            )

            # Uncomment if you also want DVFs saved.
            # save_tensor_as_nifti(
            #     generated_dvf_list[tp][0],
            #     "generated_dvfs",
            #     patient_vol_dir,
            #     iter=saved_phase,
            #     aff=aff,
            # )
            # save_tensor_as_nifti(
            #     dvf[tp][0],
            #     "DVFs_GT",
            #     patient_vol_dir,
            #     iter=saved_phase,
            #     aff=aff,
            # )

            phase_key = extract_phase_label(vol_file[tp][0])
            if phase_key not in phase_metrics:
                phase_metrics[phase_key] = {
                    "mse": [],
                    "ncc": [],
                    "ssim": [],
                    "geo_error": [],
                    "gtv_tre_mm": [],
                }

            ncc_val = ncc_loss(
                generated_current_volume[tp],
                vmorph_volume[tp],
                device=device,
            ).item()
            mse_val = mse(
                generated_current_volume[tp],
                vmorph_volume[tp],
            ).item()
            ssim_val = ssim(
                generated_current_volume[tp][0, 0].detach().cpu().numpy(),
                vmorph_volume[tp][0, 0].detach().cpu().numpy(),
                data_range=1.0,
            )

            avg_ncc += ncc_val
            avg_mse += mse_val
            avg_ssim += ssim_val

            phase_metrics[phase_key]["ncc"].append(ncc_val)
            phase_metrics[phase_key]["mse"].append(mse_val)
            phase_metrics[phase_key]["ssim"].append(ssim_val)

            np_gt = dvf[tp][0].detach().cpu().numpy()
            np_pred = generated_dvf_list[tp][0].detach().cpu().numpy()

            # Kept exactly from your original evaluation script.
            err = (
                (np_gt[0] - np_pred[0]) ** 2 * 1.0
                + (np_gt[1] - np_pred[1]) ** 2 * 1.0
                + (np_gt[2] - np_pred[2]) ** 2 * 2.0
            ) ** (1 / 3)

            this_sample.append(np.ravel(err))
            phase_metrics[phase_key]["geo_error"].append(float(np.mean(err)))

            # --------------------------------------------------------------
            # GTV target-registration error (VM-relative), in millimeters.
            #
            # The phase-50 CSV provides only the reference tumor centroid.
            # Both generated and GT VoxelMorph DVFs are backward warp fields,
            # so each field is inverted at that reference point to recover the
            # corresponding target-phase GTV position. Their Euclidean
            # difference is then measured in physical mm.
            # --------------------------------------------------------------
            if patient_gtv is not None:
                tre_mm, pred_ok, gt_ok, pred_res, gt_res = (
                    gtv_tre_from_backward_dvfs(
                        pred_dvf=generated_dvf_list[tp],
                        gt_dvf=dvf[tp],
                        ref_xyz=patient_gtv["xyz"],
                        spacing_xyz=patient_gtv["spacing"],
                        # max_iter=opt.gtv_inverse_max_iter,
                        # tol=opt.gtv_inverse_tol,
                    )
                )

                if not pred_ok:
                    gtv_nonconverged_pred += 1
                if not gt_ok:
                    gtv_nonconverged_gt += 1

                # Keep the value, but report non-convergence counts below.
                # The residuals make it easy to diagnose if inversion ever
                # becomes unreliable.
                this_gtv_tre.append(tre_mm)
                phase_metrics[phase_key]["gtv_tre_mm"].append(tre_mm)

        NCC_loss.append(avg_ncc / opt.tp)
        MSE_loss.append(avg_mse / opt.tp)
        SSIM_loss.append(avg_ssim / opt.tp)
        geo_error.append(this_sample)

        # Match the other test arrays: one scalar per test sample/patient
        # (averaged across the requested future horizons).
        if this_gtv_tre:
            GTV_TRE.append(float(np.mean(this_gtv_tre)))

    test_root = os.path.join(opt.logging_dir, "test", dir_name)
    for name, arr in [
        ("NCC_loss", NCC_loss),
        ("MSE_loss", MSE_loss),
        ("SSIM_loss", SSIM_loss),
        ("geo_error", geo_error),
        ("GTV_TRE", GTV_TRE),
    ]:
        np.save(os.path.join(test_root, f"{name}.npy"), np.asarray(arr))

    per_phase_arrays = {}
    for phase, metrics in phase_metrics.items():
        for name, values in metrics.items():
            key = f"phase_{phase}_{name}"
            per_phase_arrays[key] = np.asarray(values, dtype=np.float32)

    np.savez(
        os.path.join(test_root, "per_phase_metrics.npz"),
        **per_phase_arrays,
    )

    print(
        "\nTest set average — NCC: %0.4f  MSE: %0.4f  SSIM: %0.4f"
        % (np.mean(NCC_loss), np.mean(MSE_loss), np.mean(SSIM_loss))
    )

    if GTV_TRE:
        print(
            "GTV TRE (VM-relative): "
            f"mean={np.mean(GTV_TRE):.4f} mm | "
            f"std={np.std(GTV_TRE):.4f} mm | "
            f"n={len(GTV_TRE)}"
        )
        print(
            "[GTV TRE inversion] "
            f"non-converged predicted fields={gtv_nonconverged_pred}, "
            f"non-converged GT fields={gtv_nonconverged_gt}"
        )
    elif opt.gtv_landmarks_csv:
        print(
            "[GTV TRE] No evaluable landmark samples were found. "
            "Check that test patient folder names match the CSV patient IDs."
        )

    if gtv_missing_patients:
        print(
            f"[GTV TRE] Missing phase-50 landmarks for "
            f"{len(gtv_missing_patients)} test patient ID(s): "
            + ", ".join(sorted(gtv_missing_patients)[:20])
            + (" ..." if len(gtv_missing_patients) > 20 else "")
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    train_fold = sorted(os.listdir(os.path.join(opt.data_dir, "train")))
    valid_fold = sorted(os.listdir(os.path.join(opt.data_dir, "val")))
    test_fold = sorted(os.listdir(os.path.join(opt.data_dir, "test")))
    if opt.train_test == "train":
        dir_name = os.path.join(
            datetime.datetime.now().strftime("%m_%d"),
            datetime.datetime.now().strftime("%H.%M._") + "_" + opt.name,
        )

        train(
            (train_fold, valid_fold),
            dir_name=dir_name,
        )
    else:
        print("========TEST MODE========")
        if not opt.checkpoint:
            raise ValueError("--checkpoint is required in test mode.")

        dir_name = os.path.join(
            datetime.datetime.now().strftime("%m_%d"),
            datetime.datetime.now().strftime("%H.%M._")
            + "_"
            + "_".join(opt.checkpoint.strip("/").split("/")[-2:]),
        )

        test(
            test_fold,
            dir_name=dir_name,
        )

    print("\nDone.")