"""
hp_search.py — Hyperparameter search for ShapeVAEModule fine-tuning.

Searches training-relevant hyperparameters only (NOT architecture params,
since those are auto-detected from the pretrained checkpoint via
detect_model_config_from_ckpt() in train.py and must stay fixed to match it).

Given the diagnostics discussed (pos_ratio imbalance, vol_bce vs near_bce
tradeoff), this search specifically includes near_weight and the
n_volume/n_near sampling balance as tunable parameters, since those directly
affect the coarse-shape vs fine-detail tradeoff identified as the likely
quality bottleneck.

Usage:
    pip install optuna
    python hp_search.py \
        --pc_dir /path/to/point_clouds \
        --mesh_dir /path/to/meshes \
        --pretrained_ckpt /path/to/michelangelo_pretrained.ckpt \
        --n_trials 30 \
        --epochs_per_trial 15
"""

import argparse
import math
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader, random_split


class OptunaPruningCallback(Callback):
    """
    Minimal, version-safe replacement for optuna_integration's
    PyTorchLightningPruningCallback. That package has had repeated
    compatibility breaks against pytorch_lightning's internal
    `is_overridden("state_dict", ...)` check across version combinations
    -- rather than chase exact compatible pins, this reimplements only the
    small amount of behavior actually needed: report the monitored metric
    to Optuna after each validation epoch, and raise TrialPruned if Optuna
    decides this trial should stop early.
    """

    def __init__(self, trial: optuna.Trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self._pruned = False
        self._pruned_message = ""

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # avoid pruning during sanity-check validation before real training starts
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)
        if current_score is None:
            return

        current_score = current_score.item() if torch.is_tensor(current_score) else current_score
        step = trainer.current_epoch

        self.trial.report(current_score, step=step)
        if self.trial.should_prune():
            self._pruned = True
            self._pruned_message = f"Trial pruned at epoch {step} with {self.monitor}={current_score:.6f}"
            # stop training immediately rather than waiting for should_prune
            # to be rechecked elsewhere
            trainer.should_stop = True

    def check_pruned(self) -> None:
        if self._pruned:
            raise optuna.TrialPruned(self._pruned_message)

# Reuse the dataset/model classes from your existing train.py
from train import AbdominalDataset, ShapeVAEModule, detect_model_config_from_ckpt


def build_dataloaders(args, n_volume: int, n_near: int):
    full_dataset = AbdominalDataset(
        pc_dir=args.pc_dir,
        mesh_dir=args.mesh_dir,
        n_surface=args.n_surface,
        n_volume=n_volume,
        n_near=n_near,
        augment=True,
    )

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def objective(trial: optuna.Trial, args, detected_config: dict):
    # ── search space ────────────────────────────────────────────────────────
    lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    warmup_steps = trial.suggest_int("warmup_steps", 20, 500, log=True)

    # near_weight controls the coarse-shape (vol_bce) vs fine-detail (near_bce)
    # tradeoff flagged in the log-dict diagnostics -- worth searching a wide range
    near_weight = trial.suggest_float("near_weight", 0.05, 2.0, log=True)
    kl_weight = trial.suggest_float("kl_weight", 1e-5, 1e-2, log=True)

    # occupancy sampling balance -- directly affects pos_ratio and the
    # coarse/fine supervision tradeoff discussed
    n_volume = trial.suggest_categorical("n_volume", [1024, 1536, 2048, 3072])
    n_near = trial.suggest_categorical("n_near", [1024, 1536, 2048, 3072])

    batch_size_override = 1
    args.batch_size = 1

    print(f"\n[trial {trial.number}] lr={lr:.2e} wd={weight_decay:.2e} "
          f"warmup={warmup_steps} near_w={near_weight:.3f} kl_w={kl_weight:.2e} "
          f"n_volume={n_volume} n_near={n_near} batch_size={batch_size_override}")

    train_loader, val_loader = build_dataloaders(args, n_volume=n_volume, n_near=n_near)

    model = ShapeVAEModule(
        num_latents=detected_config["num_latents"],
        point_feats=detected_config["point_feats"],
        embed_dim=detected_config["embed_dim"],
        width=detected_config["width"],
        heads=detected_config["heads"],
        num_encoder_layers=detected_config["num_encoder_layers"],
        num_decoder_layers=detected_config["num_decoder_layers"],
        use_checkpoint=True,
        near_weight=near_weight,
        kl_weight=kl_weight,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        pretrained_ckpt=args.pretrained_ckpt,
    )
    model.hparams["n_volume"] = n_volume

    pruning_callback = OptunaPruningCallback(trial, monitor="val/loss")
    early_stop = EarlyStopping(monitor="val/loss", patience=5, mode="min")

    trial_dir = Path(args.output_dir) / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs_per_trial,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        callbacks=[pruning_callback, early_stop],
        logger=False,          # keep search runs lightweight; no wandb per-trial
        enable_checkpointing=False,
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        default_root_dir=str(trial_dir),
        enable_progress_bar=False,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    except optuna.TrialPruned:
        raise
    except Exception as e:
        # a bad hyperparameter combo (e.g. NaN loss, OOM) shouldn't kill the
        # whole search -- report a very high loss so optuna treats it as bad
        # and moves on, rather than crashing the study.
        print(f"[trial {trial.number}] failed with: {e}")
        return float("inf")

    # PyTorchLightningPruningCallback sets an internal flag rather than
    # always raising during fit() -- must check and re-raise explicitly,
    # or a pruned trial can silently be scored as if it completed normally.
    pruning_callback.check_pruned()

    val_loss = trainer.callback_metrics.get("val/loss")
    if val_loss is None or not math.isfinite(val_loss.item()):
        return float("inf")

    return val_loss.item()


def main():
    parser = argparse.ArgumentParser("Hyperparameter search for ShapeVAEModule")

    parser.add_argument("--pc_dir", required=True)
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--pretrained_ckpt", required=True,
                         help="Required -- architecture is auto-detected from this checkpoint")
    parser.add_argument("--n_surface", type=int, default=1600)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--epochs_per_trial", type=int, default=15,
                         help="Kept short deliberately -- this is a search, not a full run. "
                              "Re-train the best config for many more epochs afterward.")
    parser.add_argument("--output_dir", default="./hp_search_output")
    parser.add_argument("--study_name", default="shapevae_search")
    parser.add_argument("--storage", default=None,
                         help="Optional optuna storage URL (e.g. sqlite:///hp_search.db) "
                              "to persist/resume the study across runs")

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # detect architecture ONCE up front -- fixed across all trials, since it
    # must match the pretrained checkpoint regardless of training hparams
    detected_config = detect_model_config_from_ckpt(args.pretrained_ckpt)
    print(f"Fixed architecture (from checkpoint, not searched): {detected_config}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(trial, args, detected_config),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val/loss: {study.best_value:.6f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 70)

    # save results
    import json
    results_path = Path(args.output_dir) / "best_hyperparameters.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "detected_architecture": detected_config,
        }, f, indent=2)
    print(f"\nSaved best hyperparameters to {results_path}")

    # print exact CLI command to re-run train.py with the winning config
    p = study.best_params
    print("\nSuggested full training command with best hyperparameters:")
    print(
        f"python train.py "
        f"--pc_dir {args.pc_dir} --mesh_dir {args.mesh_dir} "
        f"--pretrained_ckpt {args.pretrained_ckpt} "
        f"--lr {p['lr']:.6e} --weight_decay {p['weight_decay']:.6e} "
        f"--warmup_steps {p['warmup_steps']} "
        f"--near_weight {p['near_weight']:.4f} --kl_weight {p['kl_weight']:.6e} "
        f"--n_volume {p['n_volume']} --n_near {p['n_near']} "
        f"--batch_size {p['batch_size']} "
        f"--max_epochs 200"
    )


if __name__ == "__main__":
    main()