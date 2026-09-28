import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import ShapeVAEModule, detect_model_config_from_ckpt

_RELEASE_PREFIXES = ("model.shape_model.", "shape_model.")


def _extract_state_dict(raw) -> dict:
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    return raw


def _is_release_checkpoint(state_dict: dict) -> bool:
    return any(k.startswith(_RELEASE_PREFIXES) for k in state_dict)


def load_shapevae(ckpt_path: str, device) -> ShapeVAEModule:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(raw)

    if _is_release_checkpoint(state_dict):
        # --- original Michelangelo release checkpoint ---
        print(f"[load_shapevae] {ckpt_path}: original Michelangelo release checkpoint "
              f"-- detecting architecture from tensor shapes.")
        del raw, state_dict  # detect_model_config_from_ckpt / _load_pretrained re-read the file
        config = detect_model_config_from_ckpt(ckpt_path)
        model = ShapeVAEModule(
            num_latents=config["num_latents"],
            point_feats=config["point_feats"],
            embed_dim=config["embed_dim"],
            width=config["width"],
            heads=config["heads"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            use_checkpoint=False,
            pretrained_ckpt=ckpt_path,
        )

    elif isinstance(raw, dict) and "hyper_parameters" in raw:
        # --- Lightning checkpoint saved by our own ShapeVAEModule training ---
        print(f"[load_shapevae] {ckpt_path}: Lightning checkpoint from ShapeVAEModule training.")
        valid_args = set(inspect.signature(ShapeVAEModule.__init__).parameters) - {"self"}
        # train.py adds model.hparams["n_volume"] after construction; it is not
        # an __init__ argument, so keep only real ones.
        hparams = {k: v for k, v in dict(raw["hyper_parameters"]).items() if k in valid_args}
        hparams["pretrained_ckpt"] = None   # weights come from state_dict below
        hparams["use_checkpoint"] = False   # gradient checkpointing is pointless at inference
        model = ShapeVAEModule(**hparams)
        model.load_state_dict(state_dict, strict=True)

    else:
        sample_keys = list(state_dict.keys())[:5] if isinstance(state_dict, dict) else []
        raise ValueError(
            f"{ckpt_path} is neither a Michelangelo release checkpoint (keys starting "
            f"with {_RELEASE_PREFIXES}) nor a ShapeVAEModule Lightning checkpoint "
            f"(needs 'hyper_parameters'). First keys seen: {sample_keys}"
        )

    return model.eval().to(device)