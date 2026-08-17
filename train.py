"""
Patch notes vs your original train.py — search for "# >>> PATCH" markers.

1. n_volume is now passed into __init__ directly (not bolted onto self.hparams
   after save_hyperparameters()), so _step's split of vol/near logits is
   guaranteed correct rather than relying on a post-hoc hparams mutation.

2. _step now logs extra diagnostics every step:
     - fraction of predictions that are positive (pred_pos_ratio)
     - whether accuracy == (1 - pos_ratio), which is the fingerprint of a
       collapsed model that just predicts "outside" everywhere
     - logit mean/std, to catch saturation (e.g. everything strongly negative)

3. New on_validation_epoch_end hook: every 10 epochs, runs reconstruct() on a
   FIXED validation sample (same one each time) and exports the mesh to
   <output_dir>/<exp_name>/meshes/epoch_XXX.obj — plus prints vertex/face
   counts and a clear "EMPTY MESH" warning if has_surface was False, so you
   can watch reconstruction quality (or lack of it) evolve during training
   instead of only finding out at the very end via a separate script.

4. reconstruct() now optionally accepts already-normalized surface tensors
   AND prints diagnostic stats about the input point cloud range, so a
   normalization mismatch (like the one in your inference script) is
   immediately visible in the logs rather than silently returning None.
"""

import argparse
import datetime
import math
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import torch
import trimesh
import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, random_split

# these imports assume the same locations as your original file
from michelangelo.models.tsal.sal_perceiver import ShapeAsLatentPerceiver
from michelangelo.models.tsal.tsal_base import Latent2MeshOutput
from michelangelo.models.tsal.inference_utils import extract_geometry



# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset  (unchanged from your version)
# ══════════════════════════════════════════════════════════════════════════════

class AbdominalDataset(Dataset):
    """
    Each sample:
      pc_normal   [n_surface, 6]  — xyz + normal, coords in [-0.9995, 0.9995]
      geo_points  [n_volume + n_near, 4]  — xyz + occupancy label (0/1)
    """

    def __init__(self,
                 pc_dir: str,
                 mesh_dir: str,
                 n_surface: int = 4096,
                 n_volume: int = 4096,
                 n_near: int = 2048,
                 augment: bool = True):

        self.n_surface = n_surface
        self.n_volume = n_volume
        self.n_near = n_near
        self.augment = augment

        self.samples: List[Tuple[str, str]] = []

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

        centroid = pts.mean(0)
        pts -= centroid
        scale = np.abs(pts).max()
        pts /= (scale + 1e-8)
        pts = np.clip(pts * 0.9995, -0.9995, 0.9995)
        nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-8)

        N = len(pts)
        if N >= self.n_surface:
            idx = np.random.choice(N, self.n_surface, replace=False)
        else:
            idx = np.concatenate([np.arange(N),
                                   np.random.choice(N, self.n_surface - N, replace=True)])
        return np.concatenate([pts[idx], nrm[idx]], axis=-1)

    def _sample_geo_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        bounds = mesh.bounds
        extent = bounds[1] - bounds[0]
        centre = (bounds[0] + bounds[1]) / 2

        vol_pts = (np.random.rand(self.n_volume, 3).astype(np.float32) - 0.5)
        vol_pts *= extent[None, :] * 1.1 + 0.05
        vol_pts += centre[None, :]

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

        pc_normal = self._load_pointcloud(ply_path)

        mesh = trimesh.load(stl_path, process=False, force="mesh")
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()

        centre = (mesh.bounds[0] + mesh.bounds[1]) / 2
        mesh.apply_translation(-centre)
        scale = mesh.extents.max() / 2 + 1e-8
        mesh.apply_scale(1.0 / scale)

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


# ══════════════════════════════════════════════════════════════════════════════
# 2. Lightning Module
# ══════════════════════════════════════════════════════════════════════════════

class ShapeVAEModule(pl.LightningModule):
    """
    Fine-tunes ShapeAsLatentPerceiver from the Michelangelo pretrained checkpoint.
    """

    def __init__(self,
                 num_latents: int = 256,
                 point_feats: int = 3,
                 embed_dim: int = 64,
                 num_freqs: int = 8,
                 width: int = 1024,
                 heads: int = 16,
                 num_encoder_layers: int = 8,
                 num_decoder_layers: int = 16,
                 use_checkpoint: bool = True,
                 flash: bool = False,
                 near_weight: float = 0.1,
                 kl_weight: float = 1e-3,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-3,
                 warmup_steps: int = 500,
                 pretrained_ckpt: Optional[str] = None,
                 n_volume: int = 4096,          # >>> PATCH: now a real constructor arg
                 output_dir: str = "./output",  # >>> PATCH: needed for periodic mesh export
                 exp_name: str = "run",
                 mesh_export_every_n_epochs: int = 10,
                 ):

        super().__init__()
        self.save_hyperparameters()   # >>> PATCH: n_volume/output_dir/exp_name are now
                                       # correctly captured here, not bolted on afterward

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
        self.kl_weight = kl_weight
        self.geo_criterion = torch.nn.BCEWithLogitsLoss()

        # >>> PATCH: fixed validation sample cached for periodic mesh export,
        # set externally right after dataset construction (see main())
        self._fixed_val_sample = None

        if pretrained_ckpt is not None:
            print(f"pretrained_ckpt is there")
            self._load_pretrained(pretrained_ckpt)

    # ── weight loading ────────────────────────────────────────────────────────

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
            print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # ── forward / loss ────────────────────────────────────────────────────────

    def _step(self, batch, split: str):
        surface = batch["surface"]                    # [B, N, 6]
        geo_points = batch["geo_points"]               # [B, P, 4]

        pc = surface[..., :3]
        feats = surface[..., 3:6]

        volume_queries = geo_points[..., :3]
        occ_labels = geo_points[..., 3]

        logits, _, posterior = self.model(
            pc=pc,
            feats=feats,
            volume_queries=volume_queries,
            sample_posterior=(split == "train"),
        )

        # >>> PATCH: n_volume now comes straight from self.hparams, which is
        # guaranteed correct since it was passed into __init__ before
        # save_hyperparameters() ran, instead of mutated afterward.
        n_vol = self.hparams.n_volume

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
            pred_pos = (logits >= 0)
            acc = (pred_pos == occ_labels.bool()).float().mean()
            pos_ratio = occ_labels.mean()
            pred_pos_ratio = pred_pos.float().mean()          # >>> PATCH
            logit_mean = logits.mean()                         # >>> PATCH
            logit_std = logits.std()                           # >>> PATCH
            # >>> PATCH: fingerprint of a collapsed "always predict outside" model
            collapsed_to_outside = (pred_pos_ratio < 0.01) and (pos_ratio > 0.01)

        self.log_dict({
            f"{split}/loss": loss,
            f"{split}/vol_bce": vol_bce,
            f"{split}/near_bce": near_bce,
            f"{split}/kl": kl_loss,
            f"{split}/accuracy": acc,
            f"{split}/pos_ratio": pos_ratio,
            f"{split}/pred_pos_ratio": pred_pos_ratio,   # >>> PATCH
            f"{split}/logit_mean": logit_mean,           # >>> PATCH
            f"{split}/logit_std": logit_std,             # >>> PATCH
        }, prog_bar=True, sync_dist=True, batch_size=surface.shape[0])

        if collapsed_to_outside and self.global_step % 50 == 0:
            print(f"[WARN][{split}] step={self.global_step}: model predicts "
                  f"almost nothing as 'inside' (pred_pos_ratio={pred_pos_ratio:.4f}) "
                  f"while {pos_ratio:.4f} of labels ARE inside. "
                  f"This is the fingerprint of a collapsed / undertrained model.")

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # >>> PATCH: periodic mesh export so you can watch reconstruction quality
    # evolve during training, using a FIXED sample each time for comparability.
    def on_validation_epoch_end(self):
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
              f"[{surface[..., :3].min().item():.4f}, {surface[..., :3].max().item():.4f}] "
              f"(expect approx [-1, 1] — if this is way off, normalization is broken)")

        self.eval()
        with torch.no_grad():
            outputs = self.reconstruct(surface, octree_depth=7)
        self.train()

        out_dir = Path(self.hparams.output_dir) / self.hparams.exp_name / "meshes"
        out_dir.mkdir(parents=True, exist_ok=True)

        if outputs[0] is None:
            print(f"[mesh export] epoch {self.current_epoch}: EMPTY MESH "
                  f"(has_surface=False) — model did not predict any occupied "
                  f"region. See logit_mean/pred_pos_ratio in the logs above.")
            return

        mesh = trimesh.Trimesh(outputs[0].mesh_v, outputs[0].mesh_f)
        out_path = out_dir / f"epoch_{self.current_epoch:04d}.obj"
        mesh.export(str(out_path))
        print(f"[mesh export] epoch {self.current_epoch}: saved "
              f"{len(mesh.vertices)} verts / {len(mesh.faces)} faces -> {out_path}")

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
        Full encode -> decode -> marching cubes for a batch of point clouds.

        Args:
            surface: [B, N, 6]  xyz + normals — MUST already be normalized to
                      roughly [-0.9995, 0.9995], matching training's
                      AbdominalDataset._load_pointcloud. If you feed raw-scale
                      coordinates here, the model has never seen that
                      distribution and will very likely predict "outside"
                      everywhere, giving you an empty mesh (outputs[i] is None).
        """
        # >>> PATCH: surface range sanity check, printed unconditionally so a
        # normalization mismatch is visible immediately rather than silently
        # returning None three function calls later.
        xyz = surface[..., :3]
        xyz_min, xyz_max = xyz.min().item(), xyz.max().item()
        if xyz_min < -1.5 or xyz_max > 1.5:
            print(f"[reconstruct] WARNING: input xyz range is "
                  f"[{xyz_min:.4f}, {xyz_max:.4f}], far outside the expected "
                  f"[-1, 1]-ish range the model was trained on. This is very "
                  f"likely why reconstruction is failing — normalize the point "
                  f"cloud the same way AbdominalDataset._load_pointcloud does "
                  f"(center on centroid, divide by max abs coordinate, scale "
                  f"by 0.9995) before calling reconstruct().")

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
        for i, ((v, f), ok) in enumerate(zip(mesh_v_f, has_surface)):
            if not ok:
                print(f"[reconstruct] sample {i}: has_surface=False -> empty mesh")
                outputs.append(None)
                continue
            print(f"[reconstruct] sample {i}: {len(v)} verts / {len(f)} faces")
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

    p.add_argument("--pc_dir", required=True)
    p.add_argument("--mesh_dir", required=True)
    p.add_argument("--n_surface", type=int, default=4096)
    p.add_argument("--n_volume", type=int, default=4096)
    p.add_argument("--n_near", type=int, default=2048)
    p.add_argument("--val_split", type=float, default=0.15)

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
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--max_epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="bf16-mixed")

    p.add_argument("--output_dir", default="./output")
    p.add_argument("--exp_name", default="abdomen_shapevae")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # >>> PATCH: new arg
    p.add_argument("--mesh_export_every_n_epochs", type=int, default=10,
                    help="Export a reconstructed mesh from a fixed val sample "
                         "every N epochs. Set 0 to disable.")

    return p.parse_args()


def detect_model_config_from_ckpt(ckpt_path: str) -> dict:
    """
    Read the actual architecture hyperparameters from the pretrained checkpoint
    by inspecting tensor shapes, so the model we build always matches exactly.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]

    # strip to shape_model prefix
    prefix = "model.shape_model."
    shape_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not shape_sd:
        prefix = "shape_model."
        shape_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    # width: from the encoder query parameter shape [num_latents, width]
    width = shape_sd["encoder.query"].shape[1]

    # num_latents: same parameter, first dim
    # AlignedShapeLatentPerceiver adds 1 extra latent for shape_embed,
    # so real num_latents = total - 1
    num_latents_raw = shape_sd["encoder.query"].shape[0]
    num_latents = num_latents_raw - 1

    # embed_dim: from pre_kl weight [embed_dim*2, width] -> embed_dim*2 is out_features
    embed_dim = shape_sd["pre_kl.weight"].shape[0] // 2 if "pre_kl.weight" in shape_sd else 0

    # heads: from cross_attn in_proj shape [3*width, width] — infer from typical ratios
    # easier: read from transformer layer count
    encoder_layers = sum(1 for k in shape_sd if k.startswith("encoder.self_attn.resblocks."))
    
    # print a few transformer keys to confirm structure
    sample_transformer_keys = [k for k in shape_sd if k.startswith("transformer.")][:5]
    print(f"[ckpt] sample transformer keys: {sample_transformer_keys}")

    sample_encoder_keys = [k for k in shape_sd if k.startswith("encoder.self_attn.")][:5]
    print(f"[ckpt] sample encoder keys: {sample_encoder_keys}")
    
    # each resblock has 4 keys (attn.in_proj, attn.out_proj, ln_*, mlp)
    # count unique layer indices
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

    # heads: width / 64 is standard for these models
    heads = width // 64

    # point_feats: from encoder input_proj weight [width, fourier_dim + point_feats]
    # fourier_dim = num_freqs * 2 * 3 (with pi) + 3 = num_freqs*6 + 3
    # default num_freqs=8 -> fourier_dim = 51
    input_proj_in = shape_sd["encoder.input_proj.weight"].shape[1]
    num_freqs = 8
    fourier_dim = num_freqs * 2 * 3 + 3  # include_pi=True
    point_feats = input_proj_in - fourier_dim

    config = {
        "width": width,
        "num_latents": num_latents +1,
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

    full_dataset = AbdominalDataset(
        pc_dir=args.pc_dir,
        mesh_dir=args.mesh_dir,
        n_surface=args.n_surface,
        n_volume=args.n_volume,
        n_near=args.n_near,
        augment=True,
    )

    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    wandb_name = "run" + "_" + cur_time

    wandb.init(
        project="Michelangelo",
        entity="MEDICAL_LAB",
        name=wandb_name,
        config=vars(args),
        reinit=True,
        mode="online",
    )

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    print(f"n_train, n_val: {len(full_dataset), n_train, n_val}")
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(args.seed))

    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    
    if args.pretrained_ckpt is not None:
        detected = detect_model_config_from_ckpt(args.pretrained_ckpt)
        args.width        = detected["width"]
        args.num_latents  = detected["num_latents"]
        args.embed_dim    = detected["embed_dim"]
        args.heads        = detected["heads"]
        args.enc_layers   = detected["num_encoder_layers"]
        args.dec_layers   = detected["num_decoder_layers"]
        # point_feats stays 3 (normals) unless checkpoint says otherwise
        point_feats       = detected["point_feats"]
        print(f"[ckpt] Using detected architecture instead of CLI defaults.")
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
        n_volume=args.n_volume,                                   # >>> PATCH
        output_dir=args.output_dir,                                # >>> PATCH
        exp_name=args.exp_name,                                    # >>> PATCH
        mesh_export_every_n_epochs=args.mesh_export_every_n_epochs,  # >>> PATCH
    )

    # >>> PATCH: cache a FIXED validation sample (index 0 of val_ds) for
    # periodic mesh export, so every export uses the same input and results
    # are directly comparable epoch to epoch.
    model._fixed_val_sample = val_ds[0]

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

    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            project="Michelangelo",
            name=args.exp_name,
            save_dir=args.output_dir,
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        default_root_dir=args.output_dir,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()