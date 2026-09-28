"""
precompute_michelangelo_latents.py

Encodes every point cloud frame with a trained ShapeVAEModule and caches the
FULL latent tensor (num_latents, embed_dim) to disk -- one .pt file per frame,
mirroring the pc_dir/<participant>/<frame>.ply structure. Nothing gets encoded
live during CondiNet training: the dataset (navigator_latent_dataset.py) just
loads these cached tensors and mean-pools them for CondiNet's input (see
CondiNet_Tr_priormulti's "michelangelo" backbone docstring for why input is
pooled but the cached/target tensor stays full).

Usage
-----
python precompute_michelangelo_latents.py \\
    --pc_dir /path/to/ply_folder_root \\
    --shapevae_ckpt /path/to/shapevae_best.ckpt \\
    --output_dir ./latent_cache
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

# ── Michelangelo / ShapeVAEModule import (same convention as train.py) ──────────
sys.path.insert(0, str(Path(__file__).parent))
from train import ShapeVAEModule  # the ShapeVAEModule from earlier in this project


def load_and_normalize_pointcloud(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Same normalization as AbdominalDataset.__getitem__ / evaluate.py:
    centroid/scale computed FROM THE POINT CLOUD, matching real inference."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)

    centroid = pts.mean(0)
    scale = np.abs(pts - centroid).max()

    pts_n = np.clip((pts - centroid) / (scale + 1e-8) * 0.9995, -0.9995, 0.9995)
    nrm_n = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
    return pts_n, nrm_n


def encode_one(model: ShapeVAEModule, ply_path: Path, n_surface: int,
                device: torch.device) -> torch.Tensor:
    pts_n, nrm_n = load_and_normalize_pointcloud(ply_path)

    N = len(pts_n)
    if N >= n_surface:
        idx = np.random.choice(N, n_surface, replace=False)
    else:
        idx = np.concatenate([np.arange(N), np.random.choice(N, n_surface - N, replace=True)])

    pc = torch.from_numpy(pts_n[idx]).float().unsqueeze(0).to(device)      # [1, n_surface, 3]
    feats = torch.from_numpy(nrm_n[idx]).float().unsqueeze(0).to(device)   # [1, n_surface, 3]

    with torch.no_grad():
        latents, _, _ = model.model.encode(pc, feats, sample_posterior=False)  # [1, num_latents, embed_dim]

    return latents.squeeze(0).cpu()  # [num_latents, embed_dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_dir", required=True,
                         help="Root folder containing <participant>/<frame>.ply files")
    parser.add_argument("--shapevae_ckpt", required=True)
    parser.add_argument("--output_dir", required=True,
                         help="Cache root; mirrors pc_dir's <participant>/<frame>.pt structure")
    parser.add_argument("--n_surface", type=int, default=1600,
                         help="Must match the value ShapeVAEModule was trained with")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-encode and overwrite cached .pt files that already exist")
    args = parser.parse_args()

    pc_dir = Path(args.pc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Loading ShapeVAEModule from {args.shapevae_ckpt}")
    model = ShapeVAEModule.load_from_checkpoint(args.shapevae_ckpt)
    model.eval().to(device)

    participant_dirs = sorted(p for p in pc_dir.iterdir() if p.is_dir())
    print(f"{len(participant_dirs)} participant folders found")

    total_encoded, total_skipped = 0, 0

    for participant_dir in participant_dirs:
        ply_files = sorted(participant_dir.glob("*.ply"))
        if not ply_files:
            continue

        out_participant_dir = output_dir / participant_dir.name
        out_participant_dir.mkdir(parents=True, exist_ok=True)

        for i, ply_path in enumerate(ply_files):
            out_path = out_participant_dir / f"{ply_path.stem}.pt"
            if out_path.exists() and not args.overwrite:
                total_skipped += 1
                continue

            latents = encode_one(model, ply_path, args.n_surface, device)
            torch.save(latents, out_path)
            total_encoded += 1

            if i % 50 == 0:
                print(f"  [{participant_dir.name}] {i + 1}/{len(ply_files)}")

    print(f"\nDone. Encoded {total_encoded} frames, skipped {total_skipped} "
          f"already-cached frames (use --overwrite to re-encode). Cache: {output_dir}")


if __name__ == "__main__":
    main()