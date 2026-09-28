from __future__ import annotations

import argparse
import datetime
import os
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Sequence

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "offline"

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import trimesh
from barbar import Bar
from scipy.spatial import cKDTree
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb

# ── dataset: cached Michelangelo latents instead of raw image slices ────────────
from navigator_latent_dataset import NAVIGATOR_LATENT_Dataset_multitime

# ── model: unchanged import path/name, just instantiated with backbone_type="michelangelo" ──
from ..models.temporal.condiNet_Tr_prior_multi import CondiNet_Tr_priormulti

from ..utils.early_stopping import EarlyStopping
from ..utils.io import cond_mkdir, custom_load

# ── ShapeVAEModule + Michelangelo inference utils, for geometric validation ─────
sys.path.insert(0, str(Path(__file__).parent))
from train import ShapeVAEModule  # the ShapeVAEModule defined earlier in this project
from michelangelo.models.tsal.inference_utils import extract_geometry

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# Run mode / paths
parser.add_argument("--train_test",  type=str, default="train", choices=["train", "test"])
parser.add_argument("--logging_dir", required=True)
parser.add_argument("--latent_cache_dir", required=True,
                    help="Output of precompute_michelangelo_latents.py")
parser.add_argument("--ply_dir", required=True,
                    help="Original point cloud root (pc_dir passed to "
                         "precompute_michelangelo_latents.py) -- used only for "
                         "geometric validation, to load true future point clouds.")
parser.add_argument("--shapevae_ckpt", required=True,
                    help="Trained ShapeVAEModule checkpoint -- provides the decoder "
                         "used for periodic geometric validation.")
parser.add_argument("--train_participants_file", required=True,
                    help="Text file, one participant folder name per line.")
parser.add_argument("--val_participants_file", required=True)
parser.add_argument("--name",        type=str, default="condinet_michelangelo")
parser.add_argument("--gpu_idx",     type=str, default="0")
parser.add_argument(
    "--checkpoint", type=str, default="",
    help="Path to a CondiNetTrPrior checkpoint to resume from.",
)
parser.add_argument(
    "--save_path", type=str,
    default="./saved_models/condinet_tr_prior_michelangelo.pth",
)

# Data
parser.add_argument("--nb_inputs",   type=int, default=3)
parser.add_argument("--tp",          type=int, default=1)
parser.add_argument("--sample_stride", type=int, default=1,
                    help="Step between consecutive sampled windows within a participant.")

parser.add_argument("--condi_channels",   nargs="+", type=int, default=[16, 32, 64, 128])
parser.add_argument("--n_heads",          type=int,   default=8)
parser.add_argument("--enc_layers",       type=int,   default=5)
parser.add_argument("--dec_layers",       type=int,   default=5)
parser.add_argument("--norm_before", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--prior_type",       type=str,   default="learned",
                    choices=["learned", "none"])

# Michelangelo latent shape -- MUST match the ShapeVAEModule these latents came from
parser.add_argument("--michelangelo_embed_dim",   type=int, default=64)
parser.add_argument("--michelangelo_num_latents", type=int, default=256,
                    help="Number of latent tokens per frame. If your ShapeVAEModule's "
                         "encoder.query has num_latents+1 rows (a leading CLS-style "
                         "token), verify latents.shape from encode() and adjust this "
                         "(and precompute_michelangelo_latents.py's caching, sliced "
                         "accordingly) before trusting this value.")

# Hyper-parameters
parser.add_argument("--lr", type=float, default=1e-4, help="CondiNet learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--lambda_recon", type=float, default=1.0,
                    help="Weight for the latent MSE loss (CondiNet's predicted vs. "
                         "true future latent tokens).")
parser.add_argument("--lambda_kl", type=float, default=1e-3,
                    help="Weight for the CondiNet posterior/prior KL loss.")
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--early_stopping_patience", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=123)

parser.add_argument(
    "--backbone_type", type=str, default="michelangelo",
    choices=["conv", "edge_encoder", "michelangelo"],
    help="Kept as a real choice (not hardcoded) so you can still A/B against the "
         "other backbones on the same training loop if useful -- but 'conv'/"
         "'edge_encoder' expect image-shaped Ipast/Ifuture, which this script's "
         "data pipeline no longer produces, so only 'michelangelo' actually works "
         "with the dataset above.",
)

# Geometric validation (periodic, decodes real geometry -- expensive, so not every epoch)
parser.add_argument("--geometric_val_every_n_epochs", type=int, default=10)
parser.add_argument("--geometric_val_n_samples", type=int, default=4)
parser.add_argument("--geometric_val_octree_depth", type=int, default=6)
parser.add_argument("--geometric_val_surface_samples", type=int, default=5000)

opt = parser.parse_args()
print("\n".join([f"{k}: {v}" for k, v in vars(opt).items()]))

# Input channels to the backbone are irrelevant for backbone_type="michelangelo"
# (the 1x1 conv's in_channels is set from michelangelo_embed_dim instead), kept
# only because CondiNet_Tr_priormulti's signature still takes in_channels/
# out_channels/condi_type for the other backbone types.
SLICE_IN_CH = opt.michelangelo_embed_dim


# ---------------------------------------------------------------------------
# Latent pooling: CondiNet's "michelangelo" backbone input convention
# ---------------------------------------------------------------------------

def pool_latents_to_grid(latent_list: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    """list of T tensors, each (B, num_latents, embed_dim) once batched by the
    DataLoader -> (B, embed_dim, T, 1, 1), mean-pooled over num_latents. This
    is CondiNet's "michelangelo" backbone input shape -- see its docstring."""
    pooled = [latent.mean(dim=1).to(device, non_blocking=True) for latent in latent_list]  # each (B, embed_dim)
    stacked = torch.stack(pooled, dim=2)  # (B, embed_dim, T)
    return stacked.unsqueeze(-1).unsqueeze(-1)  # (B, embed_dim, T, 1, 1)


def compute_latent_prediction_loss(
    future_tokens: Sequence[torch.Tensor],
    target_latents: Sequence[torch.Tensor],
    mse_loss: nn.Module,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """future_tokens: list of tp tensors (B, num_latents, embed_dim) from CondiNet.
    target_latents: list of tp tensors (B, num_latents, embed_dim), the TRUE future
    frames' cached (full, unpooled) latents -- same convention as the input."""
    if len(future_tokens) != len(target_latents):
        raise ValueError(
            f"Predicted {len(future_tokens)} tokens but received "
            f"{len(target_latents)} target frames."
        )
    per_step_mse = [mse_loss(pred, target.to(pred.device)) for pred, target in
                     zip(future_tokens, target_latents)]
    recon = sum(per_step_mse) / len(per_step_mse)
    return recon, per_step_mse


# ---------------------------------------------------------------------------
# Geometric validation: decode a predicted latent, mesh it, compare to the
# TRUE future point cloud with a real geometric distance -- not just latent
# MSE. Expensive (meshing), so only run periodically on a handful of samples.
# ---------------------------------------------------------------------------

def latent_cache_path_to_ply_path(cache_path: str, cache_dir: Path, ply_dir: Path) -> Path:
    relative = Path(cache_path).relative_to(cache_dir)
    return (ply_dir / relative).with_suffix(".ply")


def load_ply_points_normalized(ply_path: Path) -> np.ndarray:
    """Same centroid/scale normalization as ShapeVAEModule's training data --
    the predicted mesh lives in that same normalized space, so the true
    future point cloud must be normalized identically before comparing."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    centroid = pts.mean(0)
    scale = np.abs(pts - centroid).max()
    return (pts - centroid) / (scale + 1e-8)


def bidirectional_nn_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> dict:
    tree_b = cKDTree(pts_b)
    d_a_to_b, _ = tree_b.query(pts_a, k=1, workers=-1)
    tree_a = cKDTree(pts_a)
    d_b_to_a, _ = tree_a.query(pts_b, k=1, workers=-1)
    d = np.concatenate([d_a_to_b, d_b_to_a])
    return {"mean": float(d.mean()), "rms": float(np.sqrt((d ** 2).mean())), "max": float(d.max())}


@torch.no_grad()
def geometric_validation_step(
    predicted_latent: torch.Tensor,   # (num_latents, embed_dim), single sample, no batch dim
    true_ply_path: Path,
    shape_model: ShapeVAEModule,
    device: torch.device,
    octree_depth: int,
    num_surface_samples: int,
) -> dict | None:
    latents = shape_model.model.decode(predicted_latent.unsqueeze(0).to(device))
    geometric_func = partial(shape_model.model.query_geometry, latents=latents)

    mesh_v_f, has_surface = extract_geometry(
        geometric_func=geometric_func,
        device=device,
        batch_size=1,
        bounds=(-1.1,) * 3 + (1.1,) * 3,
        octree_depth=octree_depth,
        num_chunks=10000,
        disable=True,
    )
    if not has_surface[0]:
        return None

    pred_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, num_surface_samples)

    true_pts = load_ply_points_normalized(true_ply_path)

    return bidirectional_nn_distance(np.asarray(pred_pts), true_pts)


def run_geometric_validation(
    condinet: nn.Module,
    shape_model: ShapeVAEModule,
    valid_set: NAVIGATOR_LATENT_Dataset_multitime,
    device: torch.device,
    n_samples: int,
    octree_depth: int,
    num_surface_samples: int,
) -> dict | None:
    """Samples up to n_samples items from valid_set, predicts (prior path,
    no Ifuture -- matching real inference), decodes the FIRST predicted
    future step, and compares it to the true future point cloud."""
    condinet.eval()
    cache_dir = Path(opt.latent_cache_dir)
    ply_dir = Path(opt.ply_dir)

    indices = np.linspace(0, len(valid_set) - 1, min(n_samples, len(valid_set)), dtype=int)
    all_stats = []

    for idx in indices:
        _, input_paths, output_paths = valid_set.samples[int(idx)]

        input_latents = [torch.load(p, map_location="cpu").unsqueeze(0) for p in input_paths]
        Ipast = pool_latents_to_grid(input_latents, device)

        out_feats, _ = condinet(Ipast, None)
        predicted_first_step = out_feats[0][0]  # drop batch dim -- (num_latents, embed_dim)

        true_ply_path = latent_cache_path_to_ply_path(output_paths[0], cache_dir, ply_dir)
        if not true_ply_path.exists():
            print(f"  [geometric val] missing true ply: {true_ply_path} -- skipping sample.")
            continue

        stats = geometric_validation_step(
            predicted_first_step, true_ply_path, shape_model, device,
            octree_depth, num_surface_samples,
        )
        if stats is None:
            print(f"  [geometric val] empty predicted mesh for sample {idx} -- skipping.")
            continue
        all_stats.append(stats)

    condinet.train()

    if not all_stats:
        return None
    return {
        "mean": float(np.mean([s["mean"] for s in all_stats])),
        "rms": float(np.mean([s["rms"] for s in all_stats])),
        "max": float(np.mean([s["max"] for s in all_stats])),
        "n_samples": len(all_stats),
    }


# ---------------------------------------------------------------------------
# Device / seeds
# ---------------------------------------------------------------------------

def resolve_device(gpu_idx: str) -> torch.device:
    if not torch.cuda.is_available():
        print("[Device] CUDA not available, using CPU.")
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_idx}")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dirs(logging_dir: str, dir_name: str):
    log_dir = os.path.join(logging_dir, "logs", dir_name)
    run_dir = os.path.join(logging_dir, "runs", dir_name)
    cond_mkdir(log_dir)
    cond_mkdir(run_dir)
    return log_dir, run_dir


def read_participant_list(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(folds, dir_name: str = ""):
    device = resolve_device(opt.gpu_idx)
    seed_everything(opt.seed)

    # ---- CondiNet ----
    condinet = CondiNet_Tr_priormulti(
        num_inputs=opt.nb_inputs,
        horizon=opt.tp,
        in_channels=SLICE_IN_CH,
        out_channels=opt.condi_channels,
        n_heads=opt.n_heads,
        enc_layers=opt.enc_layers,
        dec_layers=opt.dec_layers,
        normalize_before=opt.norm_before,
        output_dim=opt.condi_channels[-1],  # overridden internally for backbone_type="michelangelo"
        condi_type="2",  # unused by the michelangelo backbone; kept for signature compatibility
        prior_type=opt.prior_type,
        backbone_type=opt.backbone_type,
        michelangelo_embed_dim=opt.michelangelo_embed_dim,
        michelangelo_num_latents=opt.michelangelo_num_latents,
    ).to(device)

    if opt.checkpoint:
        custom_load(condinet, opt.checkpoint, device)
        print(f"Resumed CondiNetTrPrior from {opt.checkpoint}")

    # ---- ShapeVAEModule, for geometric validation only (frozen, eval mode) ----
    print(f"Loading ShapeVAEModule (frozen, for geometric validation) from {opt.shapevae_ckpt}")
    shape_model = ShapeVAEModule.load_from_checkpoint(opt.shapevae_ckpt)
    shape_model.eval().to(device)
    for param in shape_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        condinet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )
    early_stopper = EarlyStopping(
        patience=opt.early_stopping_patience, verbose=True, delta=1e-4
    )

    # ---- Data ----
    train_set = NAVIGATOR_LATENT_Dataset_multitime(
        opt.latent_cache_dir,
        nb_inputs=opt.nb_inputs,
        sequence_list=folds[0],
        nb_pred=opt.tp,
        mode="train",
        stride=opt.sample_stride,
    )
    valid_set = NAVIGATOR_LATENT_Dataset_multitime(
        opt.latent_cache_dir,
        nb_inputs=opt.nb_inputs,
        sequence_list=folds[1],
        nb_pred=opt.tp,
        mode="val",
        stride=opt.sample_stride,
    )
    print(len(train_set), "training samples")
    print(len(valid_set), "validation samples")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False,
        num_workers=opt.num_workers, pin_memory=pin_memory,
    )

    # ---- Logging ----
    log_dir, run_dir = make_run_dirs(opt.logging_dir, dir_name)
    writer = SummaryWriter(run_dir)
    wandb.init(
        project="Abdominal Surface Motion Model",
        name=f"{dir_name}-condinettrprior-michelangelo",
        config=vars(opt),
        dir=log_dir,
        reinit=True,
        mode="offline",
    )

    mse_loss = nn.MSELoss(reduction="mean")
    best_val = float("inf")
    best_epoch = -1
    global_step = 0

    print("Begin CondiNetTrPrior + Michelangelo latent training ...")
    for epoch in range(opt.epochs):
        condinet.train()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_count = 0

        for _, input_latent_list, output_latent_list, _ in Bar(train_loader):
            optimizer.zero_grad(set_to_none=True)

            Ipast = pool_latents_to_grid(input_latent_list, device)
            Ifuture = pool_latents_to_grid(output_latent_list, device)
            target_latents = [latent.to(device, non_blocking=True) for latent in output_latent_list]

            out_feats, kl_loss = condinet(Ipast, Ifuture)

            recon_loss, _ = compute_latent_prediction_loss(out_feats, target_latents, mse_loss)

            if kl_loss is None:
                kl_term = torch.zeros((), device=device)
            else:
                kl_term = kl_loss / Ipast.shape[0]

            loss = opt.lambda_recon * recon_loss + opt.lambda_kl * kl_term

            loss.backward()
            if opt.grad_clip > 0:
                nn.utils.clip_grad_norm_(condinet.parameters(), max_norm=opt.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_recon += float(recon_loss.item())
            epoch_kl += float(kl_term.item())
            epoch_count += 1

            writer.add_scalar("train/recon", float(recon_loss.item()), global_step)
            writer.add_scalar("train/kl", float(kl_term.item()), global_step)
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            wandb.log(
                {
                    "train/recon": float(recon_loss.item()),
                    "train/kl": float(kl_term.item()),
                    "train/loss": float(loss.item()),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=global_step,
            )
            global_step += 1

        denom = max(epoch_count, 1)
        train_epoch_loss = epoch_loss / denom
        train_epoch_recon = epoch_recon / denom
        train_epoch_kl = epoch_kl / denom

        val_loss, val_step_losses = validate(condinet, valid_loader, mse_loss, device)

        print(
            f"Epoch {epoch:03d} | total={train_epoch_loss:.5f} | "
            f"recon={train_epoch_recon:.5f} | kl={train_epoch_kl:.5f} | "
            f"val_mse={val_loss:.5f}"
        )

        writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch)
        writer.add_scalar("train/epoch_recon", train_epoch_recon, epoch)
        writer.add_scalar("train/epoch_kl", train_epoch_kl, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)

        epoch_log = {
            "train/epoch_loss": train_epoch_loss,
            "train/epoch_recon": train_epoch_recon,
            "train/epoch_kl": train_epoch_kl,
            "val/loss": val_loss,
        }
        for step, step_loss in enumerate(val_step_losses):
            writer.add_scalar(f"val/mse_t{step + 1}", step_loss, epoch)
            epoch_log[f"val/mse_t{step + 1}"] = step_loss

        # ---- periodic geometric validation (expensive -- meshing, not every epoch) ----
        if opt.geometric_val_every_n_epochs > 0 and (epoch + 1) % opt.geometric_val_every_n_epochs == 0:
            geo_stats = run_geometric_validation(
                condinet, shape_model, valid_set, device,
                opt.geometric_val_n_samples, opt.geometric_val_octree_depth,
                opt.geometric_val_surface_samples,
            )
            if geo_stats is not None:
                print(f"  [geometric val] epoch {epoch}: mean={geo_stats['mean']:.5f} "
                      f"rms={geo_stats['rms']:.5f} max={geo_stats['max']:.5f} "
                      f"(n={geo_stats['n_samples']})")
                writer.add_scalar("val/geometric_mean", geo_stats["mean"], epoch)
                writer.add_scalar("val/geometric_rms", geo_stats["rms"], epoch)
                writer.add_scalar("val/geometric_max", geo_stats["max"], epoch)
                epoch_log.update({
                    "val/geometric_mean": geo_stats["mean"],
                    "val/geometric_rms": geo_stats["rms"],
                    "val/geometric_max": geo_stats["max"],
                })
            else:
                print(f"  [geometric val] epoch {epoch}: no valid samples "
                      f"(empty predicted meshes or missing ply files).")

        wandb.log(epoch_log, step=global_step)

        scheduler.step(val_loss)
        early_stopper(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            wandb.run.summary["best_val_loss"] = best_val
            wandb.run.summary["best_epoch"] = best_epoch

            fold_save = os.path.join(log_dir, "condinet_best.pth")
            condinet_payload = {
                "model": condinet.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": vars(opt),
            }
            torch.save(condinet_payload, fold_save)

            save_parent = os.path.dirname(opt.save_path)
            if save_parent:
                os.makedirs(save_parent, exist_ok=True)
            torch.save(condinet_payload, opt.save_path)

            print(f"  -> Saved best CondiNet : {fold_save}")

        if early_stopper.early_stop:
            print("Early stopping.")
            break

    writer.close()
    wandb.finish()


# ---------------------------------------------------------------------------
# Validation (prior path — no Ifuture), latent MSE only (cheap, every epoch)
# ---------------------------------------------------------------------------

def validate(condinet, valid_loader, mse_loss, device):
    """Prior-path validation: latent MSE only. Real-geometry validation is
    handled separately by run_geometric_validation (periodic, not every epoch)."""
    condinet.eval()

    total = 0.0
    count = 0
    per_step_total = np.zeros(opt.tp, dtype=np.float64)

    with torch.no_grad():
        for _, input_latent_list, output_latent_list, _ in Bar(valid_loader):
            Ipast = pool_latents_to_grid(input_latent_list, device)
            target_latents = [latent.to(device, non_blocking=True) for latent in output_latent_list]

            out_feats, _ = condinet(Ipast, None)
            recon_loss, step_losses = compute_latent_prediction_loss(out_feats, target_latents, mse_loss)

            total += float(recon_loss.item())
            for step, step_loss in enumerate(step_losses):
                per_step_total[step] += float(step_loss.item())
            count += 1

    condinet.train()
    denominator = max(count, 1)
    return total / denominator, (per_step_total / denominator).tolist()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_participants = read_participant_list(opt.train_participants_file)
    val_participants = read_participant_list(opt.val_participants_file)
    print(f"Train participants ({len(train_participants)}): {train_participants}")
    print(f"Val participants ({len(val_participants)}): {val_participants}")

    train((train_participants, val_participants), dir_name=opt.name)