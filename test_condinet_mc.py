"""
test_condinet_mc.py

Evaluates a trained Michelangelo-adapted CondiNet, mirroring the original
image-based test()'s structure (posterior vs. prior comparison, floor
baseline, geometric error, cosine similarity, saved visual outputs) but in
latent/mesh terms instead of image-slice/DVF terms.

ARCHITECTURE: no longer taken from CLI args by default -- detect_condinet_config
(condinet_ckpt_config.py) reads it straight from the checkpoint (exactly, if
train_condinet_mc.py's saved "config" is present; via tensor-shape detection
otherwise). The CLI --nb_inputs/--tp/etc. flags below are now OPTIONAL
OVERRIDES -- leave them unset to always match whatever the checkpoint was
actually trained with, with zero copy-pasting.

Uses --train_participants_file's VALIDATION split (--split_mode temporal) as
the test set, since a proper held-out test set needs sequences never touched
during training/validation, which isn't available yet with a single sequence.
Treat these numbers as "does the model still do well on unseen frames of the
SAME recording", not "does it generalize to a new person" -- same caveat as
during training.

Metrics per test sample, per predicted timestep:
  mse_post / mse_prior      -- latent MSE vs. the true future latent
                                (posterior: given the real future frames as
                                context; prior: real inference conditions,
                                no future frames)
  geo_error_prior_vs_post   -- bidirectional NN distance (mm) between the
                                prior-decoded mesh and the posterior-decoded
                                mesh -- same ROLE as the original's DVF
                                endpoint error (how much does removing future
                                information change the prediction)
  geo_error_prior_vs_true   -- bidirectional NN distance (mm) between the
                                prior-decoded mesh and the ACTUAL future point
                                cloud -- a real accuracy number the original
                                script couldn't compute (no ground truth mesh
                                available at image-inference time)
  cos_sim / feat_l2         -- cosine similarity / L2 distance between the
                                flattened posterior and prior predicted
                                latents -- same as the original
  mse_copy_last             -- floor baseline: predict the future latent by
                                repeating the last observed input latent
                                (no Copy-Ref baseline here -- see module
                                docstring for why)

Visual outputs (first --n_visual_samples samples): the true future point
cloud (copied as-is) plus the prior- and posterior-decoded meshes, as .ply /
.obj files, for direct visual side-by-side comparison.

Usage
-----
# fully automatic -- architecture comes straight from the checkpoint
python test_condinet_mc.py \\
    --checkpoint logs/<run>/condinet_best.pth \\
    --latent_cache_dir output/latents --ply_dir /path/to/ply_folder \\
    --shapevae_ckpt /path/to/shapevae.ckpt \\
    --train_participants_file sequences.txt --split_mode temporal --val_frac 0.2 \\
    --logging_dir /path/to/logs --name my_run

# override one value if you know better than the detector (rare)
python test_condinet_mc.py ... --n_heads 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import trimesh
from barbar import Bar
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
# same flat-import layout as train_condinet_mc.py -- adjust if yours differs
from condinet_tr_prior_multi import CondiNet_Tr_priormulti
from detect_config import detect_condinet_config
from nav_lat_dataset import (
    NAVIGATOR_LATENT_Dataset_multitime,
    NAVIGATOR_LATENT_Dataset_temporal_split,
)
from vae_loading import load_shapevae
from michelangelo.models.tsal.inference_utils import extract_geometry


# ---------------------------------------------------------------------------
# Small self-contained utilities
# ---------------------------------------------------------------------------

def cond_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_participant_list(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def resolve_device(gpu_idx: str) -> torch.device:
    if not torch.cuda.is_available():
        print("[Device] CUDA not available, using CPU.")
        return torch.device("cpu")
    return torch.device(f"cuda:{gpu_idx}")


# ---------------------------------------------------------------------------
# Args -- architecture flags are now OPTIONAL overrides, default None
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--logging_dir", required=True)
parser.add_argument("--name", type=str, default="condinet_mc_test")
parser.add_argument("--gpu_idx", type=str, default="0")
parser.add_argument("--checkpoint", required=True, help="condinet_best.pth from training")

parser.add_argument("--latent_cache_dir", required=True)
parser.add_argument("--ply_dir", required=True,
                    help="Original point cloud root -- for the true future point cloud "
                         "(direct visual/metric comparison, no decoding needed for this side).")
parser.add_argument("--shapevae_ckpt", required=True)

parser.add_argument("--train_participants_file", required=True,
                    help="Same file used for training. --split_mode temporal: this is "
                         "the FULL sequence list; its validation split (see --val_frac) "
                         "becomes the test set here.")
parser.add_argument("--val_participants_file", default=None,
                    help="Only used with --split_mode participant.")
parser.add_argument("--split_mode", choices=["participant", "temporal"], default="temporal")
parser.add_argument("--val_frac", type=float, default=0.2,
                    help="Must match what training used, or the test set won't be the "
                         "same frames validation was computed on during training.")

# Architecture -- default None means "use whatever detect_condinet_config finds
# in the checkpoint"; set any of these explicitly only to OVERRIDE that.
parser.add_argument("--nb_inputs", type=int, default=None)
parser.add_argument("--tp", type=int, default=None)
parser.add_argument("--condi_channels", nargs="+", type=int, default=None)
parser.add_argument("--n_heads", type=int, default=None)
parser.add_argument("--enc_layers", type=int, default=None)
parser.add_argument("--dec_layers", type=int, default=None)
parser.add_argument("--norm_before", type=bool, default=None)
parser.add_argument("--prior_type", type=str, default=None, choices=[None, "learned", "none"])
parser.add_argument("--backbone_type", type=str, default=None,
                    choices=[None, "conv", "edge_encoder", "michelangelo"])
parser.add_argument("--michelangelo_embed_dim", type=int, default=None)
parser.add_argument("--michelangelo_num_latents", type=int, default=None)

parser.add_argument("--octree_depth", type=int, default=7)
parser.add_argument("--num_surface_samples", type=int, default=5000)
parser.add_argument("--n_visual_samples", type=int, default=5,
                    help="How many test samples to export meshes/point clouds for.")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=123)

opt = parser.parse_args()
print("\n".join(f"{k}: {v}" for k, v in vars(opt).items()))


# ---------------------------------------------------------------------------
# Latent pooling (same convention as training)
# ---------------------------------------------------------------------------

def pool_latents_to_grid(latent_list, device: torch.device) -> torch.Tensor:
    pooled = [latent.mean(dim=1).to(device, non_blocking=True) for latent in latent_list]
    stacked = torch.stack(pooled, dim=2)  # (B, embed_dim, T)
    return stacked.unsqueeze(-1).unsqueeze(-1)  # (B, embed_dim, T, 1, 1)


# ---------------------------------------------------------------------------
# Decoding + geometric comparison
# ---------------------------------------------------------------------------

def decode_to_mesh(latent: torch.Tensor, shape_model, device: torch.device,
                    octree_depth: int) -> trimesh.Trimesh | None:
    """latent: (num_latents, embed_dim), single sample, no batch dim."""
    with torch.no_grad():
        decoded = shape_model.model.decode(latent.unsqueeze(0).to(device))
        geometric_func = partial(shape_model.model.query_geometry, latents=decoded)
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func, device=device, batch_size=1,
            bounds=(-1.1,) * 3 + (1.1,) * 3, octree_depth=octree_depth,
            num_chunks=10000, disable=True,
        )
    if not has_surface[0]:
        return None
    return trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])


def load_true_ply_points(ply_path: Path) -> np.ndarray:
    """Same centroid/scale normalization as ShapeVAEModule training data,
    so this lives in the same normalized space as decoded meshes."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    centroid = pts.mean(0)
    scale = np.abs(pts - centroid).max()
    return (pts - centroid) / (scale + 1e-8)


def bidirectional_nn_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Mean bidirectional nearest-neighbour distance -- a Chamfer-style
    scalar geometric difference between two point sets."""
    tree_b = cKDTree(pts_b)
    d_a_to_b, _ = tree_b.query(pts_a, k=1, workers=-1)
    tree_a = cKDTree(pts_a)
    d_b_to_a, _ = tree_a.query(pts_b, k=1, workers=-1)
    return float(np.concatenate([d_a_to_b, d_b_to_a]).mean())


def mesh_surface_points(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts)


def latent_cache_path_to_ply_path(cache_path: str, cache_dir: Path, ply_dir: Path) -> Path:
    relative = Path(cache_path).relative_to(cache_dir)
    return (ply_dir / relative).with_suffix(".ply")


# ---------------------------------------------------------------------------
# Floor baseline
# ---------------------------------------------------------------------------

def copy_last_baseline(input_latent_list, target_latents, mse_loss: nn.Module) -> list[float]:
    """Predict each future latent by repeating the LAST observed input
    latent -- 'assume no further change since the last observation'. No
    model involved; if the trained model barely beats this, it isn't
    learning much about the actual dynamics."""
    last_observed = input_latent_list[-1]  # (B, num_latents, embed_dim)
    return [float(mse_loss(last_observed, target).item()) for target in target_latents]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = resolve_device(opt.gpu_idx)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # ---- architecture: from the checkpoint, with any explicit CLI values as overrides ----
    config = detect_condinet_config(opt.checkpoint)
    print(f"tp detected:", config["tp"])
    overrides = {
        "nb_inputs": opt.nb_inputs, "tp": opt.tp, "condi_channels": opt.condi_channels,
        "n_heads": opt.n_heads, "enc_layers": opt.enc_layers, "dec_layers": opt.dec_layers,
        "norm_before": opt.norm_before, "prior_type": opt.prior_type,
        "backbone_type": opt.backbone_type,
        "michelangelo_embed_dim": opt.michelangelo_embed_dim,
        "michelangelo_num_latents": opt.michelangelo_num_latents,
    }
    applied_overrides = {k: v for k, v in overrides.items() if v is not None}
    if applied_overrides:
        print(f"[override] Using explicit CLI values instead of the checkpoint's for: "
              f"{applied_overrides}")
        config.update(applied_overrides)

    condinet = CondiNet_Tr_priormulti(
        num_inputs=config["nb_inputs"], horizon=config["tp"],
        in_channels=config["michelangelo_embed_dim"], out_channels=config["condi_channels"],
        n_heads=config["n_heads"], enc_layers=config["enc_layers"], dec_layers=config["dec_layers"],
        normalize_before=config["norm_before"], output_dim=config["condi_channels"][-1],
        condi_type="2", prior_type=config["prior_type"], backbone_type=config["backbone_type"],
        michelangelo_embed_dim=config["michelangelo_embed_dim"],
        michelangelo_num_latents=config["michelangelo_num_latents"],
    ).to(device)

    payload = torch.load(opt.checkpoint, map_location=device, weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    condinet.load_state_dict(state_dict, strict=True)
    condinet.eval()
    print(f"Loaded CondiNet from {opt.checkpoint}")

    print(f"Loading ShapeVAEModule from {opt.shapevae_ckpt}")
    shape_model = load_shapevae(opt.shapevae_ckpt, device)

    sequences = read_participant_list(opt.train_participants_file)
    print(f"Sequences ({len(sequences)}): {sequences}")

    if opt.split_mode == "participant":
        if not opt.val_participants_file:
            raise SystemExit("--val_participants_file is required for --split_mode participant.")
        test_sequences = read_participant_list(opt.val_participants_file)
        test_set = NAVIGATOR_LATENT_Dataset_multitime(
            opt.latent_cache_dir, nb_inputs=config["nb_inputs"], sequence_list=test_sequences,
            nb_pred=config["tp"], mode="val",
        )
    else:
        print("[note] Using the VALIDATION split as the test set (--split_mode temporal) -- "
              "these are unseen frames from the SAME recording(s), not a held-out participant.")
        test_set = NAVIGATOR_LATENT_Dataset_temporal_split(
            opt.latent_cache_dir, nb_inputs=config["nb_inputs"], sequence_list=sequences,
            nb_pred=config["tp"], mode="val", val_frac=opt.val_frac,
        )
    print(f"{len(test_set)} test samples")

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    base_dir = os.path.join(opt.logging_dir, "test", opt.name)
    vol_dir = os.path.join(base_dir, "volumes")
    met_dir = os.path.join(base_dir, "metrics")
    cond_mkdir(vol_dir)
    cond_mkdir(met_dir)

    mse_loss = nn.MSELoss(reduction="mean")
    cache_dir = Path(opt.latent_cache_dir)
    ply_dir = Path(opt.ply_dir)

    records_post, records_prior, records_floor = [], [], []
    cos_sim_all = []
    n_empty_meshes = 0

    print("\nRunning test ...")
    with torch.no_grad():
        for sample_idx, (identifier, input_latent_list, output_latent_list, extra) in \
                enumerate(Bar(test_loader)):

            Ipast = pool_latents_to_grid(input_latent_list, device)
            Ifuture = pool_latents_to_grid(output_latent_list, device)
            target_latents = [t.to(device) for t in output_latent_list]

            out_feats_post, _ = condinet(Ipast, Ifuture)
            out_feats_prior, _ = condinet(Ipast, None)

            floor_mse = copy_last_baseline(
                [t.to(device) for t in input_latent_list], target_latents, mse_loss)

            for t in range(config["tp"]):
                pred_post = out_feats_post[t][0]    # (num_latents, embed_dim)
                pred_prior = out_feats_prior[t][0]
                true_target = target_latents[t][0]

                mse_post = float(mse_loss(pred_post, true_target).item())
                mse_prior = float(mse_loss(pred_prior, true_target).item())

                flat_post = pred_post.flatten()
                flat_prior = pred_prior.flatten()
                cos_sim = float(torch.nn.functional.cosine_similarity(
                    flat_post.unsqueeze(0), flat_prior.unsqueeze(0)).item())
                feat_l2 = float(torch.norm(flat_post - flat_prior).item())

                mesh_post = decode_to_mesh(pred_post, shape_model, device, opt.octree_depth)
                mesh_prior = decode_to_mesh(pred_prior, shape_model, device, opt.octree_depth)

                true_ply_path = latent_cache_path_to_ply_path(
                    extra["output_files"][t][0], cache_dir, ply_dir)
                true_pts = (load_true_ply_points(true_ply_path)
                            if true_ply_path.exists() else None)
                if true_pts is None:
                    print(f"  [warn] missing true ply for sample {sample_idx} tp{t}: "
                          f"{true_ply_path}")

                if mesh_post is not None and mesh_prior is not None:
                    geo_prior_vs_post = bidirectional_nn_distance(
                        mesh_surface_points(mesh_prior, opt.num_surface_samples),
                        mesh_surface_points(mesh_post, opt.num_surface_samples),
                    )
                else:
                    geo_prior_vs_post = float("nan")
                    n_empty_meshes += 1

                if mesh_prior is not None and true_pts is not None:
                    geo_prior_vs_true = bidirectional_nn_distance(
                        mesh_surface_points(mesh_prior, opt.num_surface_samples), true_pts)
                else:
                    geo_prior_vs_true = float("nan")

                records_post.append({"sample": sample_idx, "tp": t, "mse": mse_post})
                records_prior.append({"sample": sample_idx, "tp": t, "mse": mse_prior,
                                      "geo_prior_vs_post": geo_prior_vs_post,
                                      "geo_prior_vs_true": geo_prior_vs_true})
                cos_sim_all.append({"sample": sample_idx, "tp": t,
                                    "cos_sim": cos_sim, "feat_l2": feat_l2})
                records_floor.append({"sample": sample_idx, "tp": t,
                                      "mse_copy_last": floor_mse[t]})

                if sample_idx < opt.n_visual_samples:
                    sample_dir = os.path.join(vol_dir, f"sample_{sample_idx:04d}")
                    cond_mkdir(sample_dir)
                    if true_ply_path.exists():
                        shutil.copy(true_ply_path, os.path.join(sample_dir, f"target_tp{t}.ply"))
                    if mesh_prior is not None:
                        mesh_prior.export(os.path.join(sample_dir, f"prior_tp{t}.obj"))
                    if mesh_post is not None:
                        mesh_post.export(os.path.join(sample_dir, f"posterior_tp{t}.obj"))

    if n_empty_meshes:
        print(f"\n[warn] {n_empty_meshes} empty predicted mesh(es) encountered "
              f"(excluded from geo_prior_vs_post as NaN).")

    # ---- Aggregate and save ----
    mse_post_arr = np.array([r["mse"] for r in records_post])
    mse_prior_arr = np.array([r["mse"] for r in records_prior])
    geo_post_arr = np.array([r["geo_prior_vs_post"] for r in records_prior])
    geo_true_arr = np.array([r["geo_prior_vs_true"] for r in records_prior])
    cos_arr = np.array([r["cos_sim"] for r in cos_sim_all])
    feat_l2_arr = np.array([r["feat_l2"] for r in cos_sim_all])
    mse_copy_last_arr = np.array([r["mse_copy_last"] for r in records_floor])

    np.savez(
        os.path.join(met_dir, "test_metrics.npz"),
        mse_post=mse_post_arr, mse_prior=mse_prior_arr,
        geo_prior_vs_post=geo_post_arr, geo_prior_vs_true=geo_true_arr,
        cos_sim=cos_arr, feat_l2=feat_l2_arr, copy_last=mse_copy_last_arr,
    )

    def _fmt(arr: np.ndarray, name: str) -> str:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return f"  {name:<34s}  (no valid values)"
        return (f"  {name:<34s}  mean={np.nanmean(arr):.5f}  std={np.nanstd(arr):.5f}  "
                f"min={np.nanmin(arr):.5f}  max={np.nanmax(arr):.5f}  n={len(valid)}/{len(arr)}")

    summary_lines = [
        "=" * 76,
        "CondiNet (Michelangelo latents) test results",
        f"Checkpoint : {opt.checkpoint}",
        f"N samples  : {len(records_post)}",
        "=" * 76,
        "",
        "── Latent MSE ──────────────────────────────────────────────────────",
        _fmt(mse_post_arr, "MSE (posterior path)"),
        _fmt(mse_prior_arr, "MSE (prior path)"),
        "",
        "── Geometric error (mm, normalized space) ───────────────────────────",
        _fmt(geo_post_arr, "prior vs. posterior mesh"),
        _fmt(geo_true_arr, "prior vs. TRUE future point cloud"),
        "",
        "── Prior-posterior feature alignment ────────────────────────────────",
        _fmt(cos_arr, "Cosine similarity"),
        _fmt(feat_l2_arr, "Feature L2 distance"),
        "",
        "── Floor comparison ─────────────────────────────────────────────────",
        _fmt(mse_copy_last_arr, "MSE (Copy-Last floor)"),
        f"  Model (prior) vs Copy-Last floor = "
        f"{np.mean(mse_copy_last_arr) - np.mean(mse_prior_arr):+.5f}  "
        f"(positive = model beats floor)",
        f"  Posterior (ceiling) vs prior     = "
        f"{np.mean(mse_prior_arr) - np.mean(mse_post_arr):+.5f}  "
        f"(headroom if the model could see the future)",
        "",
        "[note] No Copy-Ref floor: the original image data had a baked-in "
        "reference channel per sample; latent samples here have no equivalent.",
        "=" * 76,
    ]
    summary_txt = "\n".join(summary_lines)
    print("\n" + summary_txt)
    with open(os.path.join(met_dir, "test_metrics_summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    print(f"\nAll artefacts saved to {base_dir}")


if __name__ == "__main__":
    main()