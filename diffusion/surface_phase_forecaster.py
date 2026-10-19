from __future__ import annotations

"""
Camera-surface respiratory PHASE forecaster.

Bridges unpaired data:
    camera surfaces (patients A)  --> future respiratory phase
    4D volumes      (patients B)  --> diffusion model conditioned on phase

Pipeline
--------
1) Each camera recording is reduced to a 1D breathing signal (mean surface
   height per frame). End-inhale peaks are detected and phase rises linearly
   from 0 (end-inhale) to 1 between consecutive peaks -- the same phase-based
   binning convention used for 4D phases (0% = end-inhale). No manual labels.
2) Your pretrained surface temporal model (frozen by default) turns past
   surfaces into horizon tokens [B,H,D]; a small head maps each token to
   (sin 2*pi*phase, cos 2*pi*phase).
3) At deployment:
       phase = forecaster.predict_phase(surface_past)[:, step]   # [B]
       dvfs, vols = diffusion_model(Vref, phase=phase[:, None])

Data convention (edit load_recordings if yours differs)
-------------------------------------------------------
--surface_dir contains one file per camera recording (.npy or .pt), shape
[T, *frame], where one frame is exactly what the surface model received per
time step during its own training (e.g. [C,H,W]). Use --add_channel_dim if
frames are stored as [H,W] but the model expects [1,H,W].

Example
-------
python surface_phase_forecaster.py --surface_dir /path/to/camera_surfaces \\
    --surface_checkpoint /path/to/surface_best.pth --fps 30 \\
    --out_dir /path/to/phase_forecaster
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/fellahr/4D_MoPred")
from models.temporal.temporal_predictors import (  # noqa: E402
    build_temporal_predictor,
    read_checkpoint_config,
)


# =============================================================================
# Breathing signal -> phase labels
# =============================================================================
def breathing_signal(recording: np.ndarray) -> np.ndarray:
    """[T, *frame] -> [T] mean surface value per frame (NaN-safe, gaps filled)."""
    T = recording.shape[0]
    with np.errstate(all="ignore"):
        signal = np.nanmean(recording.reshape(T, -1).astype(np.float64), axis=1)
    bad = ~np.isfinite(signal)
    if bad.all():
        raise ValueError("Recording contains no finite values.")
    if bad.any():
        idx = np.arange(T)
        signal[bad] = np.interp(idx[bad], idx[~bad], signal[~bad])
    return signal


def phase_from_signal(
    signal: np.ndarray,
    fps: float,
    inhale_is_peak: bool = True,
    min_period_s: float = 1.5,
    smooth_s: float = 0.25,
    prominence: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear phase between end-inhale peaks. Returns (phase[T], peaks).

    Frames before the first / after the last detected peak get NaN.
    """
    s = signal if inhale_is_peak else -signal
    width = max(1, int(round(smooth_s * fps)))
    if width > 1:
        s = np.convolve(s, np.ones(width) / width, mode="same")

    peaks, _ = find_peaks(
        s,
        distance=max(1, int(round(min_period_s * fps))),
        prominence=prominence * float(np.std(s)),
    )

    phase = np.full(len(signal), np.nan, dtype=np.float32)
    for a, b in zip(peaks[:-1], peaks[1:]):
        phase[a:b] = (np.arange(a, b) - a) / float(b - a)
    return phase, peaks


def load_recordings(surface_dir: str, add_channel_dim: bool) -> tuple[list[str], list[np.ndarray]]:
    files = sorted(
        glob.glob(os.path.join(surface_dir, "*.npy"))
        + glob.glob(os.path.join(surface_dir, "*.pt"))
    )
    if not files:
        raise FileNotFoundError(f"No .npy/.pt recordings found in {surface_dir}")

    recordings = []
    for path in files:
        if path.endswith(".npy"):
            arr = np.load(path)
        else:
            arr = torch.load(path, map_location="cpu")
            arr = arr.numpy() if torch.is_tensor(arr) else np.asarray(arr)
        arr = np.asarray(arr, dtype=np.float32)
        if add_channel_dim:
            arr = arr[:, None]
        recordings.append(arr)
    return files, recordings


def save_qc_plot(signal, phase, peaks, fps, path):
    """Plot to check that peaks really are end-inhale (flip --inhale_is_peak if not)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[QC] matplotlib not available; skipping plot.")
        return
    t = np.arange(len(signal)) / fps
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(t, signal, lw=0.8)
    axes[0].plot(t[peaks], signal[peaks], "rv", label="detected end-inhale")
    axes[0].set_ylabel("breathing signal")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, phase, lw=0.8)
    axes[1].set_ylabel("phase [0,1)")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[QC] Saved {path}")


# =============================================================================
# Dataset of (past surfaces, future phases) windows
# =============================================================================
class SurfacePhaseWindows(Dataset):
    def __init__(self, recordings, phases, periods, nb_inputs, horizon, stride, time_dim):
        if time_dim < 1:
            raise ValueError("time_dim counts the batch axis and must be >= 1")
        self.recordings = recordings
        self.phases = phases
        self.periods = periods
        self.nb_inputs = nb_inputs
        self.horizon = horizon
        self.time_dim = time_dim
        self.items = []
        for r, (rec, ph) in enumerate(zip(recordings, phases)):
            T = rec.shape[0]
            for s in range(0, T - nb_inputs - horizon + 1, stride):
                window = ph[s + nb_inputs - 1: s + nb_inputs + horizon]
                if np.isfinite(window).all():
                    self.items.append((r, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, s = self.items[i]
        nb, H = self.nb_inputs, self.horizon
        past = torch.from_numpy(self.recordings[r][s: s + nb])  # [T_in, *frame]
        past = torch.movedim(past, 0, self.time_dim - 1)         # time at time_dim once batched
        target = torch.from_numpy(self.phases[r][s + nb: s + nb + H])
        last = torch.tensor(float(self.phases[r][s + nb - 1]))
        period = torch.tensor(float(self.periods[r]))
        return past, target, last, period


# =============================================================================
# Model
# =============================================================================
def phase_to_vec(phase: torch.Tensor) -> torch.Tensor:
    angle = 2.0 * math.pi * phase
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def vec_to_phase(vec: torch.Tensor) -> torch.Tensor:
    return (torch.atan2(vec[..., 0], vec[..., 1]) / (2.0 * math.pi)) % 1.0


def circular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute phase error in [0, 0.5] cycles."""
    return ((pred - target + 0.5) % 1.0 - 0.5).abs()


class SurfacePhaseForecaster(nn.Module):
    """Pretrained surface temporal model + small phase head."""

    def __init__(self, surface_predictor: nn.Module, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = surface_predictor
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, surface_past: torch.Tensor) -> torch.Tensor:
        """[B, ...surface sequence...] -> [B, horizon, 2] (sin, cos)."""
        frozen = getattr(self.backbone, "frozen", False)
        with torch.no_grad() if frozen else torch.enable_grad():
            tokens, _ = self.backbone(surface_past, future=None)  # prior path
        return self.head(tokens)

    @torch.no_grad()
    def predict_phase(self, surface_past: torch.Tensor) -> torch.Tensor:
        """[B, horizon] future phases in [0,1)."""
        return vec_to_phase(self(surface_past))


def resolve_surface_dims(cfg: dict) -> dict:
    """Fill num_frames / horizon / latent_dim of the PRETRAINED surface model
    from its checkpoint config when not given (keys used by your scripts)."""
    ckpt_cfg = read_checkpoint_config(cfg["surface_checkpoint"])
    for key, ckpt_key in [
        ("surface_nb_inputs", "nb_inputs"),
        ("surface_horizon", "tp"),
        ("surface_latent_dim", "prelatent_size"),
    ]:
        if cfg.get(key, -1) in (-1, None):
            if ckpt_key not in ckpt_cfg:
                raise ValueError(
                    f"--{key} not given and '{ckpt_key}' not in the surface "
                    "checkpoint config; pass it explicitly."
                )
            cfg[key] = int(ckpt_cfg[ckpt_key])
    return cfg


def build_forecaster(cfg: dict, device: torch.device) -> SurfacePhaseForecaster:
    predictor = build_temporal_predictor(
        "condinet_surface",
        num_frames=cfg["surface_nb_inputs"],
        horizon=cfg["surface_horizon"],
        pre_latent_dim=cfg["surface_latent_dim"],
        checkpoint=cfg["surface_checkpoint"],
        freeze=not cfg["finetune_backbone"],
        input_time_dim=cfg["surface_time_dim"],
        **(json.loads(cfg["surface_kwargs"]) if cfg.get("surface_kwargs") else {}),
    )
    return SurfacePhaseForecaster(
        predictor,
        latent_dim=cfg["surface_latent_dim"],
        hidden=cfg["head_hidden"],
    ).to(device)


def load_forecaster(path: str, device: torch.device) -> SurfacePhaseForecaster:
    """Rebuild a trained forecaster for deployment."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_forecaster(payload["config"], device)
    model.head.load_state_dict(payload["head"])
    if payload.get("backbone") is not None:
        model.backbone.net.load_state_dict(payload["backbone"])
    model.eval()
    return model


# =============================================================================
# Training
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, horizon, device):
    model.eval()
    err_model = torch.zeros(horizon, dtype=torch.float64)
    err_base = torch.zeros(horizon, dtype=torch.float64)
    n = 0
    steps = torch.arange(1, horizon + 1, dtype=torch.float32)
    for past, target, last, period in loader:
        past, target = past.to(device), target.to(device)
        pred = model.predict_phase(past)
        # Constant-rate baseline: advance the last observed phase at the
        # recording's median breathing rate.
        base = (last[:, None] + steps[None] / period[:, None]) % 1.0
        err_model += circular_error(pred, target).sum(0).double().cpu()
        err_base += circular_error(base, target.cpu()).sum(0).double()
        n += target.shape[0]
    return (err_model / max(n, 1)).tolist(), (err_base / max(n, 1)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface_dir", required=True)
    parser.add_argument("--surface_checkpoint", required=True,
                        help="Your pretrained surface temporal model.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", type=float, required=True, help="Camera frame rate.")
    parser.add_argument("--inhale_is_peak", action=argparse.BooleanOptionalAction, default=True,
                        help="True if the signal is HIGHEST at end-inhale (e.g. surface "
                             "height). Use --no-inhale_is_peak for camera distance maps.")
    parser.add_argument("--min_period_s", type=float, default=1.5)
    parser.add_argument("--smooth_s", type=float, default=0.25)
    parser.add_argument("--prominence", type=float, default=0.3)
    parser.add_argument("--add_channel_dim", action="store_true")

    # Pretrained surface model geometry (-1 = read from its checkpoint config)
    parser.add_argument("--surface_nb_inputs", type=int, default=-1)
    parser.add_argument("--surface_horizon", type=int, default=-1)
    parser.add_argument("--surface_latent_dim", type=int, default=-1)
    parser.add_argument("--surface_time_dim", type=int, default=2)
    parser.add_argument("--surface_kwargs", type=str, default="",
                        help="Optional JSON overrides of the surface model constructor args.")
    parser.add_argument("--finetune_backbone", action="store_true",
                        help="Also train the surface model (default: frozen, head only).")

    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of RECORDINGS held out (no window leakage).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu_idx", type=int, default=0)
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    rng = np.random.default_rng(opt.seed)
    device = torch.device(f"cuda:{opt.gpu_idx}" if torch.cuda.is_available() else "cpu")
    os.makedirs(opt.out_dir, exist_ok=True)

    cfg = resolve_surface_dims(vars(opt).copy())
    print(json.dumps({k: cfg[k] for k in
                      ("surface_nb_inputs", "surface_horizon", "surface_latent_dim")}, indent=2))

    # ---- Phase labels from the camera signal ----
    files, recordings = load_recordings(opt.surface_dir, opt.add_channel_dim)
    phases, periods, keep = [], [], []
    for i, (path, rec) in enumerate(zip(files, recordings)):
        signal = breathing_signal(rec)
        phase, peaks = phase_from_signal(
            signal, opt.fps, opt.inhale_is_peak,
            opt.min_period_s, opt.smooth_s, opt.prominence,
        )
        if len(peaks) < 3:
            print(f"[Labels] {os.path.basename(path)}: only {len(peaks)} peaks, skipped.")
            continue
        period = float(np.median(np.diff(peaks)))
        print(f"[Labels] {os.path.basename(path)}: {len(peaks)} cycles, "
              f"median period {period / opt.fps:.2f}s")
        if not keep:
            save_qc_plot(signal, phase, peaks, opt.fps,
                         os.path.join(opt.out_dir, "qc_phase_labels.png"))
        phases.append(phase)
        periods.append(period)
        keep.append(i)
    recordings = [recordings[i] for i in keep]
    if len(recordings) < 2:
        raise RuntimeError("Need at least 2 usable recordings (train + val).")

    # ---- Split by recording ----
    order = rng.permutation(len(recordings))
    n_val = max(1, int(round(opt.val_fraction * len(recordings))))
    val_ids, train_ids = set(order[:n_val].tolist()), order[n_val:].tolist()

    def subset(ids):
        return SurfacePhaseWindows(
            [recordings[i] for i in ids], [phases[i] for i in ids],
            [periods[i] for i in ids], cfg["surface_nb_inputs"],
            cfg["surface_horizon"], opt.stride, opt.surface_time_dim,
        )

    train_set, val_set = subset(train_ids), subset(sorted(val_ids))
    print(f"{len(train_set)} train windows / {len(val_set)} val windows")
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers)

    # ---- Model ----
    model = build_forecaster(cfg, device)
    groups = [{"params": model.head.parameters(), "lr": opt.lr}]
    if opt.finetune_backbone:
        groups.append({"params": [p for p in model.backbone.parameters() if p.requires_grad],
                       "lr": opt.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    best, bad_epochs = float("inf"), 0
    for epoch in range(opt.epochs):
        model.train()
        total, count = 0.0, 0
        for past, target, _, _ in train_loader:
            past, target = past.to(device), target.to(device)
            loss = nn.functional.mse_loss(model(past), phase_to_vec(target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * target.shape[0]
            count += target.shape[0]

        err_model, err_base = evaluate(model, val_loader, cfg["surface_horizon"], device)
        mean_err = float(np.mean(err_model))
        print(
            f"Epoch {epoch:03d} | train_loss={total / max(count, 1):.4f} | "
            "val phase error per step (% cycle): "
            + ", ".join(f"{100 * e:.1f}" for e in err_model)
            + " | constant-rate baseline: "
            + ", ".join(f"{100 * e:.1f}" for e in err_base)
        )

        if mean_err < best:
            best, bad_epochs = mean_err, 0
            torch.save(
                {
                    "head": model.head.state_dict(),
                    "backbone": model.backbone.net.state_dict() if opt.finetune_backbone else None,
                    "config": cfg,
                    "val_phase_error": err_model,
                    "val_baseline_error": err_base,
                },
                os.path.join(opt.out_dir, "phase_forecaster_best.pth"),
            )
            print("  -> saved phase_forecaster_best.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= opt.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()