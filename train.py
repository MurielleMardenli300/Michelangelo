"""
train.py  —  Fine-tune Michelangelo's ShapeAsLatentPerceiver on abdominal
             point-cloud → occupancy-field → mesh reconstruction.

Pipeline (matches the original SITA-VAE shape branch, image/text removed):
  PLY point cloud (xyz + normals)
      │
      ▼
  CrossAttentionEncoder  (perceiver, N_latents × width)
      │
      ▼  VAE bottleneck (pre_kl / post_kl)
  ShapeLatents  [B, num_latents, embed_dim]
      │
      ▼  Transformer decoder
  LatentFeatures  [B, num_latents, width]
      │
      ▼  CrossAttentionDecoder  (queries = volume sample points)
  Occupancy logits  [B, P]
      │
      ▼  BCE + KL loss  → marching cubes at inference

Usage
-----
python train.py \
    --pc_dir   /volatile/Datasets/Varian_Motion/point_clouds/norm_pc \
    --mesh_dir /volatile/Datasets/Varian_Motion/meshes \
    --ckpt_path /path/to/michelangelo_pretrained.ckpt \
    --output_dir ./output \
    --max_epochs 200 \
    --batch_size 4 \
    --lr 1e-4 \
    --n_surface 4096 \
    --n_volume  4096 \
    --n_near    2048
"""

import os
import sys
import argparse
import glob
import math
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple
import datetime

import wandb

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split

# ── Michelangelo imports (assumes repo root on PYTHONPATH) ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))  # adjust if needed
from michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiver
from michelangelo.models.tsal.inference_utils import extract_geometry
from michelangelo.models.tsal.tsal_base import Latent2MeshOutput
from michelangelo.models.modules.distributions import DiagonalGaussianDistribution


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════════════════════

class AbdominalDataset(Dataset):
    """
    Each sample:
      pc_normal   [n_surface, 6]  — xyz + normal, coords in [-0.9995, 0.9995]
      geo_points  [n_volume + n_near, 4]  — xyz + occupancy label (0/1)

    Point clouds come from *.ply files under pc_dir/<participant>/.
    Meshes (watertight STL) come from mesh_dir/<participant>/mesh_<frame>.stl.
    """

    def __init__(self,
                 pc_dir: str,
                 mesh_dir: str,
                 n_surface: int = 4096,
                 n_volume: int = 4096,
                 n_near: int = 2048,
                 augment: bool = True):

        self.n_surface = n_surface
        self.n_volume  = n_volume
        self.n_near    = n_near
        self.augment   = augment

        self.samples: List[Tuple[str, str]] = []   # (ply_path, stl_path)

        for participant_dir in sorted(Path(pc_dir).iterdir()):
            if not participant_dir.is_dir():
                continue
            participant = participant_dir.name
            mesh_part_dir = Path(mesh_dir) / participant
            if not mesh_part_dir.is_dir():
                continue

            for ply_path in sorted(participant_dir.glob("*.ply")):
                stem = ply_path.stem
                try:
                    frame_num = int("".join(filter(str.isdigit, stem)))
                except ValueError:
                    continue
                stl_path = mesh_part_dir / f"mesh_{frame_num}.stl"
                if stl_path.exists():
                    self.samples.append((str(ply_path), str(stl_path)))

        print(f"Dataset: {len(self.samples)} paired samples found.")

    def __len__(self):
        return len(self.samples)

    def _load_pointcloud(self, ply_path: str) -> np.ndarray:
        """Load PLY, FPS to n_surface, return [n_surface, 6] in [-0.9995, 0.9995]."""
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)

        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
            )
            pcd.orient_normals_consistent_tangent_plane(100)

        nrm = np.asarray(pcd.normals, dtype=np.float32)

        # normalise to [-0.9995, 0.9995]
        centroid = pts.mean(0)
        pts -= centroid
        scale = np.abs(pts).max()
        pts /= (scale + 1e-8)
        pts = np.clip(pts * 0.9995, -0.9995, 0.9995)
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)

        # FPS or random subsample to fixed n_surface
        N = len(pts)
        if N >= self.n_surface:
            idx = np.random.choice(N, self.n_surface, replace=False)
        else:
            idx = np.concatenate([np.arange(N),
                                   np.random.choice(N, self.n_surface - N, replace=True)])
        return np.concatenate([pts[idx], nrm[idx]], axis=-1)   # [n_surface, 6]

    def _sample_geo_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Sample volume (random bbox) + near-surface points and compute occupancy.
        Returns [n_volume + n_near, 4] where col-3 = {0: outside, 1: inside}.
        Mesh must be watertight for ray-cast occupancy to be reliable.
        """
        bounds = mesh.bounds  # [2, 3]
        extent = bounds[1] - bounds[0]
        centre = (bounds[0] + bounds[1]) / 2

        # --- volume points: uniform in a slightly enlarged bounding box ---
        vol_pts = (np.random.rand(self.n_volume, 3).astype(np.float32) - 0.5)
        vol_pts *= extent[None, :] * 1.1 + 0.05
        vol_pts += centre[None, :]

        # --- near-surface points: surface samples + small Gaussian noise ---
        surf_pts, _ = trimesh.sample.sample_surface(mesh, self.n_near)
        surf_pts = surf_pts.astype(np.float32)
        sigma = extent.mean() * 0.02
        near_pts = surf_pts + np.random.randn(self.n_near, 3).astype(np.float32) * sigma

        all_pts = np.concatenate([vol_pts, near_pts], axis=0)   # [n_volume+n_near, 3]

        # occupancy via ray-casting (reliable for watertight meshes)
        occ = mesh.contains(all_pts.astype(np.float64)).astype(np.float32)   # [P]

        # normalise query points to [-1.1, 1.1] matching the decoder bounds
        all_pts_n = (all_pts - centre[None, :]) / (extent.max() / 2 + 1e-8) * 1.1
        all_pts_n = np.clip(all_pts_n, -1.25, 1.25)

        return np.concatenate([all_pts_n, occ[:, None]], axis=-1)  # [P, 4]

    def __getitem__(self, idx: int):
        ply_path, stl_path = self.samples[idx]

        # point cloud
        pc_normal = self._load_pointcloud(ply_path)   # [n_surface, 6]

        # mesh  →  occupancy query points
        mesh = trimesh.load(stl_path, process=False, force="mesh")
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()

        # normalise mesh to unit sphere for consistent bounds
        centre = (mesh.bounds[0] + mesh.bounds[1]) / 2
        mesh.apply_translation(-centre)
        scale = mesh.extents.max() / 2 + 1e-8
        mesh.apply_scale(1.0 / scale)

        # optional augmentation: random Y-axis rotation
        if self.augment:
            angle = np.random.uniform(0, 2 * math.pi)
            rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(rot)
            # apply same rotation to point cloud xyz
            R = rot[:3, :3].astype(np.float32)
            pc_normal[:, :3] = pc_normal[:, :3] @ R.T
            pc_normal[:, 3:] = pc_normal[:, 3:] @ R.T

        geo_points = self._sample_geo_points(mesh)    # [P, 4]

        return {
            "surface":    torch.from_numpy(pc_normal).float(),   # [n_surface, 6]
            "geo_points": torch.from_numpy(geo_points).float(),  # [P, 4]
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Lightning Module  (shape-only, no image/text)
# ══════════════════════════════════════════════════════════════════════════════

class ShapeVAEModule(pl.LightningModule):
    """
    Fine-tunes ShapeAsLatentPerceiver from the Michelangelo pretrained checkpoint.
    Image and text encoders are dropped entirely — only the shape encoder,
    VAE bottleneck, transformer decoder, and geometry query decoder are used.

    Loss = BCE(vol) + near_weight * BCE(near) + kl_weight * KL
    """

    def __init__(self,
                 # model hyperparameters (must match pretrained ckpt)
                 num_latents: int = 256,
                 point_feats: int = 3,    # normals (3-channel)
                 embed_dim: int = 64,
                 num_freqs: int = 8,
                 width: int = 1024,
                 heads: int = 16,
                 num_encoder_layers: int = 8,
                 num_decoder_layers: int = 16,
                 use_checkpoint: bool = True,
                 flash: bool = False,
                 # loss weights
                 near_weight: float = 0.1,
                 kl_weight: float = 1e-3,
                 # optimiser
                 lr: float = 1e-4,
                 weight_decay: float = 1e-3,
                 warmup_steps: int = 500,
                 # checkpoint to initialise from
                 pretrained_ckpt: Optional[str] = None):

        super().__init__()
        self.save_hyperparameters()

        self.model = ShapeAsLatentPerceiver(
            device=None,
            dtype=None,
            num_latents=num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=True,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            use_ln_post=True,
            use_checkpoint=use_checkpoint,
            flash=flash,
        )

        self.near_weight = near_weight
        self.kl_weight   = kl_weight
        self.geo_criterion = torch.nn.BCEWithLogitsLoss()

        if pretrained_ckpt is not None:
            self._load_pretrained(pretrained_ckpt)

    # ── weight loading ────────────────────────────────────────────────────────

    def _load_pretrained(self, ckpt_path: str):
        """
        Load only the shape-model weights from the full Michelangelo checkpoint.
        The full checkpoint has keys like:
            model.shape_model.encoder.query
            model.shape_model.pre_kl.weight
            ...
        We strip the 'model.shape_model.' prefix and load into self.model.
        """
        print(f"Loading pretrained weights from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # handle both raw state_dict and Lightning checkpoint formats
        if "state_dict" in sd:
            sd = sd["state_dict"]

        prefix = "model.shape_model."
        shape_sd = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                shape_sd[k[len(prefix):]] = v

        if len(shape_sd) == 0:
            # try without the alignment wrapper prefix
            prefix = "shape_model."
            for k, v in sd.items():
                if k.startswith(prefix):
                    shape_sd[k[len(prefix):]] = v

        if len(shape_sd) == 0:
            print("WARNING: no shape_model keys found in checkpoint — loading nothing.")
            return

        missing, unexpected = self.model.load_state_dict(shape_sd, strict=False)
        print(f"Pretrained load: {len(shape_sd)} keys found, "
              f"{len(missing)} missing, {len(unexpected)} unexpected.")
        if missing:
            print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # ── forward / loss ────────────────────────────────────────────────────────

    def _step(self, batch, split: str):
        surface    = batch["surface"]                 # [B, N, 6]
        geo_points = batch["geo_points"]              # [B, P, 4]

        pc    = surface[..., :3]                      # xyz
        feats = surface[..., 3:6]                     # normals

        volume_queries = geo_points[..., :3]          # [B, P, 3]
        occ_labels     = geo_points[..., 3]           # [B, P]

        # forward through shape VAE
        logits, _, posterior = self.model(
            pc=pc,
            feats=feats,
            volume_queries=volume_queries,
            sample_posterior=(split == "train"),
        )                                             # logits: [B, P]

        n_vol  = self.hparams.get("n_volume", logits.shape[1] // 2) if hasattr(self, "hparams") else logits.shape[1] // 2
        # split logits into volume and near-surface regions
        # (they were concatenated in that order in the dataset)
        vol_logits  = logits[:, :n_vol]
        near_logits = logits[:, n_vol:]
        vol_labels  = occ_labels[:, :n_vol]
        near_labels = occ_labels[:, n_vol:]

        vol_bce  = self.geo_criterion(vol_logits.float(),  vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        if posterior is None:
            kl_loss = torch.tensor(0.0, device=logits.device)
        else:
            kl_loss = posterior.kl(dims=(1, 2)).mean()

        loss = vol_bce + self.near_weight * near_bce + self.kl_weight * kl_loss

        with torch.no_grad():
            acc = ((logits >= 0) == occ_labels.bool()).float().mean()
            pos_ratio = occ_labels.mean()

        self.log_dict({
            f"{split}/loss":      loss,
            f"{split}/vol_bce":   vol_bce,
            f"{split}/near_bce":  near_bce,
            f"{split}/kl":        kl_loss,
            f"{split}/accuracy":  acc,
            f"{split}/pos_ratio": pos_ratio,
        }, prog_bar=True, sync_dist=True, batch_size=surface.shape[0])

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # ── optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.99),
        )

        def lr_lambda(step):
            ws = self.hparams.warmup_steps
            if step < ws:
                return step / max(1, ws)
            # cosine decay to 10% of peak LR
            progress = (step - ws) / max(1, self.trainer.max_steps - ws)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [scheduler]

    # ── inference helper ──────────────────────────────────────────────────────

    @torch.no_grad()
    def reconstruct(self,
                    surface: torch.FloatTensor,
                    bounds: float = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000) -> List[Latent2MeshOutput]:
        """
        Full encode → decode → marching cubes for a batch of point clouds.

        Args:
            surface: [B, N, 6]  xyz + normals in [-0.9995, 0.9995]
        Returns:
            List of Latent2MeshOutput (one per sample in batch)
        """
        pc    = surface[..., :3]
        feats = surface[..., 3:6]

        latents, _, _ = self.model.encode(pc, feats, sample_posterior=False)
        latents = self.model.decode(latents)

        geometric_func = partial(self.model.query_geometry, latents=latents)

        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=surface.device,
            batch_size=surface.shape[0],
            bounds=(-bounds,) * 3 + (bounds,) * 3,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=True,
        )

        outputs = []
        for (v, f), ok in zip(mesh_v_f, has_surface):
            if not ok:
                outputs.append(None)
                continue
            out = Latent2MeshOutput()
            out.mesh_v = v
            out.mesh_f = f
            outputs.append(out)

        return outputs


# ══════════════════════════════════════════════════════════════════════════════
# 3. Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("ShapeVAE fine-tune on abdominal point clouds")

    # data
    p.add_argument("--pc_dir",   required=True, help="Root dir of point cloud PLYs")
    p.add_argument("--mesh_dir", required=True, help="Root dir of watertight STL meshes")
    p.add_argument("--n_surface", type=int, default=4096, help="Points sampled from PLY")
    p.add_argument("--n_volume",  type=int, default=4096, help="Volume query points per sample")
    p.add_argument("--n_near",    type=int, default=2048, help="Near-surface query points per sample")
    p.add_argument("--val_split", type=float, default=0.15, help="Fraction for validation")

    # model
    p.add_argument("--pretrained_ckpt", default=None, help="Path to Michelangelo pretrained .ckpt")
    p.add_argument("--num_latents",     type=int, default=256)
    p.add_argument("--embed_dim",       type=int, default=64)
    p.add_argument("--width",           type=int, default=1024)
    p.add_argument("--heads",           type=int, default=16)
    p.add_argument("--enc_layers",      type=int, default=8)
    p.add_argument("--dec_layers",      type=int, default=16)
    p.add_argument("--no_checkpoint",   action="store_true", help="Disable gradient checkpointing")

    # loss
    p.add_argument("--near_weight", type=float, default=0.1)
    p.add_argument("--kl_weight",   type=float, default=1e-3)

    # training
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--max_epochs",   type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--precision",    type=str,   default="bf16-mixed")

    # output
    p.add_argument("--output_dir",  default="./output")
    p.add_argument("--exp_name",    default="abdomen_shapevae")
    p.add_argument("--use_wandb",   action="store_true")
    p.add_argument("--seed",        type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # ── dataset ───────────────────────────────────────────────────────────────
    full_dataset = AbdominalDataset(
        pc_dir=args.pc_dir,
        mesh_dir=args.mesh_dir,
        n_surface=args.n_surface,
        n_volume=args.n_volume,
        n_near=args.n_near,
        augment=True,
    )
    
    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    wandb_name = "run" + "_" +cur_time
    
    # if args.use_wandb:
    wandb.init(
        project="Michelangelo",
        entity="MEDICAL_LAB",
        name=wandb_name,
        # id=WANDB_RUN_ID,
        config=vars(args),
        reinit=True,
        mode="online"
    )


    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    print(f"n_train, n_val: {len(full_dataset), n_train, n_val}")
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))

    # disable augmentation for validation subset
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── model ─────────────────────────────────────────────────────────────────
    model = ShapeVAEModule(
        num_latents=args.num_latents,
        point_feats=3,
        embed_dim=args.embed_dim,
        width=args.width,
        heads=args.heads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        use_checkpoint=not args.no_checkpoint,
        near_weight=args.near_weight,
        kl_weight=args.kl_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        pretrained_ckpt=args.pretrained_ckpt,
    )
    # store n_volume on model so _step can split logits correctly
    model.hparams["n_volume"] = args.n_volume

    # ── callbacks ─────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.output_dir) / args.exp_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ── logger ────────────────────────────────────────────────────────────────
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="Michelangelo",
            name=args.exp_name,
            save_dir=args.output_dir,
        )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,       # validate once per epoch
        gradient_clip_val=1.0,
        default_root_dir=args.output_dir,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining complete. Checkpoints saved to {ckpt_dir}")
    print("To reconstruct a mesh from a point cloud after training:")
    print("""
    from train import ShapeVAEModule
    import torch, open3d as o3d, numpy as np, trimesh

    model = ShapeVAEModule.load_from_checkpoint('path/to/last.ckpt')
    model.eval().cuda()

    pcd = o3d.io.read_point_cloud('your_cloud.ply')
    pts = np.asarray(pcd.points, np.float32)
    nrm = np.asarray(pcd.normals, np.float32)
    pts = np.clip(pts / (np.abs(pts).max() + 1e-8) * 0.9995, -0.9995, 0.9995)
    surface = torch.from_numpy(np.concatenate([pts, nrm], -1)).float().unsqueeze(0).cuda()

    outputs = model.reconstruct(surface, octree_depth=7)
    if outputs[0] is not None:
        mesh = trimesh.Trimesh(outputs[0].mesh_v, outputs[0].mesh_f)
        mesh.export('reconstruction.obj')
    """)


if __name__ == "__main__":
    main()