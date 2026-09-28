"""
compute_mc_latents.py  (same script as precompute_michelangelo_latents.py)

Encodes every point cloud frame with a trained ShapeVAEModule and caches the
FULL latent tensor (num_latents, embed_dim) to disk -- one .pt file per frame.
Nothing gets encoded live during CondiNet / Mamba training: the dataset
(navigator_latent_dataset.py) just loads these cached tensors and mean-pools
them for the temporal model's input.

Input layouts (both handled, and they can be mixed):
  * FLAT:        --pc_dir contains the .ply files directly. The folder's own
                 name becomes the sequence ("participant") name.
  * PARTICIPANT: --pc_dir contains one subfolder per participant, each holding
                 that participant's .ply files.
Output mirrors that as  <output_dir>/<sequence_name>/<frame>.pt , which is what
navigator_latent_dataset.py expects (the sequence name is what goes in the
train/val participants files).

Checkpoint: --shapevae_ckpt can be either the original Michelangelo release
checkpoint or your own fine-tuned Lightning checkpoint; shapevae_loading.py
detects which. For the real pipeline use your fine-tuned one (under
output/<exp_name>/checkpoints/) so the encoder matches the decoder used for
geometric validation.

Usage
-----
python compute_mc_latents.py \\
    --pc_dir /path/to/ply_folder \\
    --shapevae_ckpt /path/to/checkpoint.ckpt \\
    --output_dir output/latents
"""

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

sys.path.insert(0, str(Path(__file__).parent))
from vae_loading import load_shapevae


def find_sequences(pc_dir: Path) -> dict[str, list[Path]]:
    """sequence name -> sorted .ply files. Handles a flat folder, participant
    subfolders, or both."""
    sequences: dict[str, list[Path]] = {}

    direct = sorted(pc_dir.glob("*.ply"))
    if direct:
        sequences[pc_dir.name] = direct

    for sub in sorted(p for p in pc_dir.iterdir() if p.is_dir()):
        plys = sorted(sub.glob("*.ply"))
        if plys:
            sequences[sub.name] = plys

    return sequences


def load_and_normalize_pointcloud(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Same normalization as AbdominalDataset.__getitem__ in train.py:
    centroid/scale computed FROM THE POINT CLOUD, as at real inference time."""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")
    if not pcd.has_normals():
        # The ShapeVAE was trained with normals as input features. Estimating
        # them here would silently shift the input distribution (orientation
        # conventions matter), so fail loudly instead.
        raise ValueError(
            f"{ply_path} has no normals. The ShapeVAE takes xyz + normals; "
            f"write normals into the .ply (e.g. average_clouds.py's save_ply does)."
        )
    nrm = np.asarray(pcd.normals, dtype=np.float32)

    centroid = pts.mean(0)
    scale = np.abs(pts - centroid).max()

    pts_n = np.clip((pts - centroid) / (scale + 1e-8) * 0.9995, -0.9995, 0.9995)
    nrm_n = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)
    return pts_n, nrm_n


def encode_one(model, ply_path: Path, n_surface: int, device: torch.device,
               seed: int) -> torch.Tensor:
    pts_n, nrm_n = load_and_normalize_pointcloud(ply_path)

    # Deterministic per-file subsampling, so re-running gives identical latents.
    rng = np.random.default_rng(zlib.crc32(str(ply_path).encode()) ^ seed)
    N = len(pts_n)
    if N >= n_surface:
        idx = rng.choice(N, n_surface, replace=False)
    else:
        idx = np.concatenate([np.arange(N), rng.choice(N, n_surface - N, replace=True)])

    pc = torch.from_numpy(pts_n[idx]).float().unsqueeze(0).to(device)      # [1, n_surface, 3]
    feats = torch.from_numpy(nrm_n[idx]).float().unsqueeze(0).to(device)   # [1, n_surface, 3]

    with torch.no_grad():
        latents, _, _ = model.model.encode(pc, feats, sample_posterior=False)  # [1, num_latents, embed_dim]

    return latents.squeeze(0).cpu()  # [num_latents, embed_dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_dir", required=True,
                        help="Folder of .ply files, or a folder of participant subfolders")
    parser.add_argument("--shapevae_ckpt", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Cache root: <output_dir>/<sequence_name>/<frame>.pt")
    parser.add_argument("--n_surface", type=int, default=1600,
                        help="Must match the value ShapeVAEModule was trained with")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode and overwrite cached .pt files that already exist")
    args = parser.parse_args()

    pc_dir = Path(args.pc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = find_sequences(pc_dir)
    if not sequences:
        raise SystemExit(
            f"No .ply files found in {pc_dir} or in any of its immediate subfolders."
        )
    print(f"Found {len(sequences)} sequence(s): "
          + ", ".join(f"{name} ({len(files)} frames)" for name, files in sequences.items()))

    device = torch.device(args.device)
    print(f"Loading ShapeVAEModule from {args.shapevae_ckpt}")
    model = load_shapevae(args.shapevae_ckpt, device)

    total_encoded, total_skipped = 0, 0
    shapes_seen: set[tuple[int, ...]] = set()

    for name, ply_files in sequences.items():
        out_sequence_dir = output_dir / name
        out_sequence_dir.mkdir(parents=True, exist_ok=True)

        for i, ply_path in enumerate(ply_files):
            out_path = out_sequence_dir / f"{ply_path.stem}.pt"
            if out_path.exists() and not args.overwrite:
                total_skipped += 1
                continue

            latents = encode_one(model, ply_path, args.n_surface, device, args.seed)
            if tuple(latents.shape) not in shapes_seen:
                shapes_seen.add(tuple(latents.shape))
                print(f"  [latent shape] {tuple(latents.shape)} = (num_latents, embed_dim) "
                      f"(from {ply_path.name})")
            torch.save(latents, out_path)
            total_encoded += 1

            if i % 50 == 0:
                print(f"  [{name}] {i + 1}/{len(ply_files)}")

    print(f"\nDone. Encoded {total_encoded} frames, skipped {total_skipped} "
          f"already-cached frames (use --overwrite to re-encode). Cache: {output_dir}")

    if len(shapes_seen) == 1:
        num_latents, embed_dim = next(iter(shapes_seen))
        print(f"\nFor the temporal-model training scripts, pass:\n"
              f"  --michelangelo_num_latents {num_latents} --michelangelo_embed_dim {embed_dim}\n"
              f"Sequence name(s) for the train/val participants files: "
              f"{', '.join(sequences)}")
    elif len(shapes_seen) > 1:
        print(f"\n[warn] Multiple latent shapes seen this run: {sorted(shapes_seen)} "
              f"-- something is inconsistent (mixed checkpoints?).")


if __name__ == "__main__":
    main()