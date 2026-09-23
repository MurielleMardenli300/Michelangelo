import os
import sys
import argparse
import math
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import trimesh
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

# ── Michelangelo imports (assumes repo root on PYTHONPATH) ─────────────────────
sys.path.insert(0, str(Path(__file__).parent))  # adjust if needed
from michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiver
from michelangelo.models.tsal.inference_utils import extract_geometry
from michelangelo.models.tsal.tsal_base import Latent2MeshOutput

import datetime
import split_utils


class AbdominalDataset(Dataset):
    """
    Each sample:
      pc_normal   [n_surface, 6]  — xyz + normal, coords in [-0.9995, 0.9995]
      geo_points  [n_volume + n_near, 4]  — xyz + occupancy label (0/1)

    Point clouds come from *.ply files under pc_dir/<participant>/.
    Meshes (watertight STL) come from mesh_dir/<participant>/mesh_<frame>.stl.
    """

    def __init__(self,
                 sample_pairs: List[Tuple[str, str]],
                 n_surface: int = 1600,
                 n_volume: int = 2048,
                 n_near: int = 1536,
                 augment: bool = True):

        self.n_surface = n_surface
        self.n_volume = n_volume
        self.n_near = n_near
        self.augment = augment
        self.samples: List[Tuple[str, str]] = sample_pairs

        print(f"Dataset: {len(self.samples)} paired samples "
              f"(augment={self.augment}).")

    def __len__(self):
        return len(self.samples)

    def _sample_geo_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        bounds = mesh.bounds
        extent = bounds[1] - bounds[0]
        centre = (bounds[0] + bounds[1]) / 2

        # split volume points: half uniform (global far-field coverage),
        # half biased near the surface (much more likely to actually land
        # inside a thin shell, giving the model real positive-label signal)
        n_uniform = self.n_volume // 2
        n_biased = self.n_volume - n_uniform

        uniform_pts = (np.random.rand(n_uniform, 3).astype(np.float32) - 0.5)
        uniform_pts *= extent[None, :] * 1.1 + 0.05
        uniform_pts += centre[None, :]

        surf_seed, _ = trimesh.sample.sample_surface(mesh, n_biased)
        surf_seed = surf_seed.astype(np.float32)
        biased_sigma = extent.mean() * 0.1
        biased_pts = surf_seed + np.random.randn(n_biased, 3).astype(np.float32) * biased_sigma

        vol_pts = np.concatenate([uniform_pts, biased_pts], axis=0)

        surf_pts, _ = trimesh.sample.sample_surface(mesh, self.n_near)
        surf_pts = surf_pts.astype(np.float32)
        sigma = extent.mean() * 0.02
        near_pts = surf_pts + np.random.randn(self.n_near, 3).astype(np.float32) * sigma

        all_pts = np.concatenate([vol_pts, near_pts], axis=0)
        occ = mesh.contains(all_pts.astype(np.float64)).astype(np.float32)

        all_pts_n = (all_pts - centre[None, :]) / (extent.max() / 2 + 1e-8) * 1.1
        all_pts_n = np.clip(all_pts_n, -1.25, 1.25)

        return np.concatenate([all_pts_n, occ[:, None]], axis=-1)

    def __getitem__(self, idx: int):
        ply_path, stl_path = self.samples[idx]

        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        nrm = np.asarray(pcd.normals, dtype=np.float32)

        # ONE normalization, computed from the point cloud (since that's what's
        # available at inference time too -- the mesh won't exist at inference)
        centroid = pts.mean(0)
        scale = np.abs(pts - centroid).max()

        pts_n = np.clip((pts - centroid) / (scale + 1e-8) * 0.9995, -0.9995, 0.9995)
        nrm_n = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)

        N = len(pts_n)
        if N >= self.n_surface:
            idx_s = np.random.choice(N, self.n_surface, replace=False)
        else:
            idx_s = np.concatenate([np.arange(N), np.random.choice(N, self.n_surface - N, replace=True)])
        pc_normal = np.concatenate([pts_n[idx_s], nrm_n[idx_s]], axis=-1)

        # apply the SAME transform to the mesh -- not its own independent bbox normalization
        mesh = trimesh.load(stl_path, process=False, force="mesh")
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.apply_translation(-centroid)
        mesh.apply_scale(1.0 / (scale + 1e-8))

        if self.augment:
            angle = np.random.uniform(0, 2 * math.pi)
            rot = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(rot)
            R = rot[:3, :3].astype(np.float32)
            pc_normal[:, :3] = pc_normal[:, :3] @ R.T
            pc_normal[:, 3:] = pc_normal[:, 3:] @ R.T

        geo_points = self._sample_geo_points(mesh)

        return {
            "surface": torch.from_numpy(pc_normal).float(),
            "geo_points": torch.from_numpy(geo_points).float(),
        }


# Lightning Module  (shape-only, no image/text input of Michelangelo)
class ShapeVAEModule(pl.LightningModule):
    """
    Fine-tunes ShapeAsLatentPerceiver from the Michelangelo pretrained checkpoint.
    VAE bottleneck, transformer decoder, and geometry query decoder are used.

    Loss = BCE(vol) + near_weight * BCE(near) + kl_weight * KL
    """

    def __init__(self,
                 num_latents: int = 257,
                 point_feats: int = 3,    # normals (3-channel)
                 embed_dim: int = 64,
                 num_freqs: int = 8,
                 width: int = 768,
                 heads: int = 12,
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
                 warmup_steps: int = 50,
                 mesh_export_every_n_epochs: int = 5,
                 pretrained_ckpt: Optional[str] = None):

        super().__init__()
        self.save_hyperparameters()

        print(f"ShapeVAEModule: num_latents={num_latents}, point_feats={point_feats}, ")

        self.model = ShapeAsLatentPerceiver(
            device=None,
            dtype=None,
            num_latents=num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=False,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            use_ln_post=True,
            use_checkpoint=use_checkpoint,
            init_scale=0.25,
            qkv_bias=False,
            flash=flash,
        )

        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.geo_criterion = torch.nn.BCEWithLogitsLoss()

        self._fixed_val_sample = None

        if pretrained_ckpt is not None:
            self._load_pretrained(pretrained_ckpt)

    def on_validation_epoch_end(self):
        """Every N epochs, reconstruct a fixed validation sample and save as .stl,
        so you can visually track reconstruction quality over training."""
        every_n = self.hparams.mesh_export_every_n_epochs
        if every_n <= 0:
            return
        if (self.current_epoch + 1) % every_n != 0:
            return
        if self._fixed_val_sample is None:
            print("[mesh export] no fixed validation sample set, skipping export.")
            return
        if not self.trainer.is_global_zero:
            return

        surface = self._fixed_val_sample["surface"].unsqueeze(0).to(self.device)

        print(f"[mesh export] epoch {self.current_epoch}: "
              f"surface xyz range = "
              f"[{surface[..., :3].min().item():.4f}, {surface[..., :3].max().item():.4f}]")

        self.eval()
        with torch.no_grad():
            outputs = self.reconstruct(surface, octree_depth=7)
        self.train()

        out_dir = Path('./output')
        out_dir.mkdir(parents=True, exist_ok=True)

        if outputs[0] is None:
            print(f"[mesh export] epoch {self.current_epoch}: EMPTY MESH (has_surface=False)")
            return

        mesh = trimesh.Trimesh(outputs[0].mesh_v, outputs[0].mesh_f)
        out_path = out_dir / f"epoch_{self.current_epoch:04d}.stl"
        mesh.export(str(out_path))
        print(f"[mesh export] epoch {self.current_epoch}: saved "
              f"{len(mesh.vertices)} verts / {len(mesh.faces)} faces -> {out_path}")

    # weight loading

    def _load_pretrained(self, ckpt_path: str):
        print(f"Loading pretrained weights from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "state_dict" in sd:
            sd = sd["state_dict"]

        prefix = "model.shape_model."
        shape_sd = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                shape_sd[k[len(prefix):]] = v

        if len(shape_sd) == 0:
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
            print("\nMISSING KEYS:")
            for k in missing:
                print("  ", k)
        if unexpected:
            print("\nUNEXPECTED KEYS:")
            for k in unexpected:
                print("  ", k)

    def _step(self, batch, split: str):
        surface = batch["surface"]                 # [B, N, 6]
        geo_points = batch["geo_points"]            # [B, P, 4]

        pc = surface[..., :3]                       # xyz
        feats = surface[..., 3:6]                    # normals

        volume_queries = geo_points[..., :3]         # [B, P, 3]
        occ_labels = geo_points[..., 3]               # [B, P]

        logits, _, posterior = self.model(
            pc=pc,
            feats=feats,
            volume_queries=volume_queries,
            sample_posterior=(split == "train"),
        )                                             # logits: [B, P]

        n_vol = self.hparams.get("n_volume", logits.shape[1] // 2) if hasattr(self, "hparams") else logits.shape[1] // 2
        vol_logits = logits[:, :n_vol]
        near_logits = logits[:, n_vol:]
        vol_labels = occ_labels[:, :n_vol]
        near_labels = occ_labels[:, n_vol:]

        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
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
            f"{split}/loss": loss,
            f"{split}/vol_bce": vol_bce,
            f"{split}/near_bce": near_bce,
            f"{split}/kl": kl_loss,
            f"{split}/accuracy": acc,
            f"{split}/pos_ratio": pos_ratio,
        }, prog_bar=True, sync_dist=True, batch_size=surface.shape[0])

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

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
            progress = (step - ws) / max(1, self.trainer.max_steps - ws)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [scheduler]

    # inference

    @torch.no_grad()
    def reconstruct(self,
                     surface: torch.FloatTensor,
                     bounds: float = 1.1,
                     octree_depth: int = 7,
                     num_chunks: int = 10000) -> List[Latent2MeshOutput]:
        pc = surface[..., :3]
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


def parse_args():
    p = argparse.ArgumentParser("ShapeVAE fine-tune on abdominal point clouds")

    p.add_argument("--pc_dir", required=True)
    p.add_argument("--mesh_dir", required=True)
    p.add_argument("--n_surface", type=int, default=1600)
    p.add_argument("--n_volume", type=int, default=2048)
    p.add_argument("--n_near", type=int, default=1536)

    # ── split management ────────────────────────────────────────────────
    p.add_argument("--split_json", default="./output/split.json",
                    help="Path to the train/val/test split file. If it "
                         "already exists it is loaded as-is (this is what "
                         "makes the split reproducible run to run). If it "
                         "doesn't exist, it is created from a fresh scan of "
                         "pc_dir/mesh_dir and saved here.")
    p.add_argument("--val_split", type=float, default=0.15,
                    help="Only used when --split_json doesn't exist yet.")
    p.add_argument("--test_split", type=float, default=0.15,
                    help="Only used when --split_json doesn't exist yet. "
                         "Set to 0 if you don't want a held-out test set.")

    p.add_argument("--pretrained_ckpt", default=None)
    p.add_argument("--num_latents", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--enc_layers", type=int, default=8)
    p.add_argument("--dec_layers", type=int, default=16)
    p.add_argument("--no_checkpoint", action="store_true")

    p.add_argument("--near_weight", type=float, default=0.1)
    p.add_argument("--kl_weight", type=float, default=1e-3)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="bf16-mixed")

    p.add_argument("--output_dir", default="./output")
    p.add_argument("--exp_name", default="abdomen_shapevae")
    p.add_argument("--use_wandb", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--mesh_export_every_n_epochs", type=int, default=20)

    return p.parse_args()


def detect_model_config_from_ckpt(ckpt_path: str) -> dict:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    prefix = "model.shape_model."
    shape_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not shape_sd:
        prefix = "shape_model."
        shape_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    width = shape_sd["encoder.query"].shape[1]
    num_latents_raw = shape_sd["encoder.query"].shape[0]
    num_latents = num_latents_raw - 1
    embed_dim = shape_sd["pre_kl.weight"].shape[0] // 2 if "pre_kl.weight" in shape_sd else 0

    enc_layer_indices = set(
        int(k.split(".")[3]) for k in shape_sd
        if k.startswith("encoder.self_attn.resblocks.")
    )
    num_encoder_layers = len(enc_layer_indices)

    transformer_layer_indices = set(
        int(k.split(".")[2]) for k in shape_sd
        if k.startswith("transformer.resblocks.")
    )
    num_decoder_layers = len(transformer_layer_indices)

    heads = width // 64

    input_proj_in = shape_sd["encoder.input_proj.weight"].shape[1]
    num_freqs = 8
    fourier_dim = num_freqs * 2 * 3 + 3
    point_feats = input_proj_in - fourier_dim

    config = {
        "width": width,
        "num_latents": num_latents + 1,
        "embed_dim": embed_dim,
        "heads": heads,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "point_feats": point_feats,
    }

    print(f"[ckpt] Detected architecture: {config}")
    return config


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # ── build / load the reproducible split ─────────────────────────────
    split = split_utils.build_or_load_split(
        pc_dir=args.pc_dir,
        mesh_dir=args.mesh_dir,
        split_json=args.split_json,
        val_frac=args.val_split,
        test_frac=args.test_split,
        seed=args.seed,
    )

    train_pairs = split_utils.keys_to_paths(split["train"], args.pc_dir, args.mesh_dir)
    val_pairs = split_utils.keys_to_paths(split["val"], args.pc_dir, args.mesh_dir)
    test_pairs = split_utils.keys_to_paths(split["test"], args.pc_dir, args.mesh_dir)

    # separate dataset INSTANCES per split -- fixes the original bug where
    # train_ds and val_ds were Subsets of the same underlying object, so
    # `val_ds.dataset.augment = False` also silently turned off augmentation
    # for training.
    train_ds = AbdominalDataset(train_pairs, n_surface=args.n_surface,
                                 n_volume=args.n_volume, n_near=args.n_near,
                                 augment=True)
    val_ds = AbdominalDataset(val_pairs, n_surface=args.n_surface,
                               n_volume=args.n_volume, n_near=args.n_near,
                               augment=False)
    test_ds = AbdominalDataset(test_pairs, n_surface=args.n_surface,
                                n_volume=args.n_volume, n_near=args.n_near,
                                augment=False)

    if len(train_ds) > 0:
        sample = train_ds[0]
        pos_ratio = sample["geo_points"][:, 3].mean()
        print(f"pos_ratio: {pos_ratio:.5f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    # test_loader isn't used during training -- evaluate.py reads split.json
    # directly and runs inference/metrics on the "test" keys after training.

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples  |  "
          f"Test: {len(test_ds)} samples (held out, not used until evaluate.py)")

    if args.pretrained_ckpt is not None:
        detected = detect_model_config_from_ckpt(args.pretrained_ckpt)
        args.width = detected["width"]
        args.num_latents = detected["num_latents"]
        args.embed_dim = detected["embed_dim"]
        args.heads = detected["heads"]
        args.enc_layers = detected["num_encoder_layers"]
        args.dec_layers = detected["num_decoder_layers"]
        point_feats = detected["point_feats"]
        print("[ckpt] Using detected architecture instead of CLI defaults.")
    else:
        point_feats = 3

    model = ShapeVAEModule(
        num_latents=args.num_latents,
        point_feats=point_feats,
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
        mesh_export_every_n_epochs=args.mesh_export_every_n_epochs,
    )

    if len(val_ds) > 0:
        model._fixed_val_sample = val_ds[0]

    model.hparams["n_volume"] = args.n_volume

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

    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    wandb_name = "run_" + cur_time
    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="Michelangelo",
            name=wandb_name,
            save_dir=args.output_dir,
        )
        # record which split file this run used, directly in the wandb config
        logger.experiment.config.update({"split_json": str(Path(args.split_json).resolve())})

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=2,
        gradient_clip_val=1.0,
        default_root_dir=args.output_dir,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining complete. Checkpoints saved to {ckpt_dir}")
    print(f"Split file used: {Path(args.split_json).resolve()}")


if __name__ == "__main__":
    main()