"""
condinet_ckpt_config.py

Reproducible architecture recovery for CondiNet_Tr_priormulti checkpoints, so
test_condinet_mc.py (or anything else loading one of these) never needs
architecture args copy-pasted by hand.

Two layers, tried in order:

1. EXACT: train_condinet_mc.py saves {"model": ..., "config": vars(opt), ...}
   -- if "config" is present and has everything needed, that's used directly.
   No guessing, no ambiguity.

2. FALLBACK (shape detection): for checkpoints saved without a config (e.g.
   from a hand-modified training run, or someone else's checkpoint). Most
   architecture values genuinely ARE recoverable from tensor shapes for the
   "michelangelo" backbone:
     - nb_inputs             = input_proj.weight.shape[0]      (Conv3d in==out==num_frames)
     - tp (horizon)          = query_learn.weight.shape[0]     (nn.Embedding(horizon, hidden_dim))
     - michelangelo_embed_dim, hidden_dim = backbone.weight.shape[:2]  (Conv2d(embed_dim, hidden_dim, 1))
     - michelangelo_num_latents = linear.weight.shape[0] // embed_dim
     - enc_layers / dec_layers = count of distinct transformer.encoder/decoder.layers.N indices present

   n_heads is NOT recoverable this way: nn.MultiheadAttention's parameter
   shapes (in_proj_weight etc.) are identical regardless of head count, so
   there is no tensor shape that encodes it. Same situation as the width//64
   heuristic in detect_model_config_from_ckpt (train.py) -- that isn't a true
   detection either, just a convention. This module does the same: falls
   back to hidden_dim // 64 and prints a loud warning, since a wrong guess
   here won't raise an error, it will just silently give numerically wrong
   (poorly-initialized-equivalent) results.

   norm_before and prior_type also aren't shape-detectable (they change how
   forward() USES the layers, not what parameters exist), so the fallback
   path defaults them and prints the same kind of warning.
"""

from __future__ import annotations

import torch

# Args required to reconstruct CondiNet_Tr_priormulti exactly
REQUIRED_KEYS = [
    "nb_inputs", "tp", "condi_channels", "n_heads", "enc_layers", "dec_layers",
    "norm_before", "prior_type", "backbone_type",
    "michelangelo_embed_dim", "michelangelo_num_latents",
]


def _extract_state_dict(payload) -> dict:
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _detect_from_shapes(state_dict: dict) -> dict:
    if "backbone.weight" not in state_dict or state_dict["backbone.weight"].dim() != 4:
        raise ValueError(
            "Can't detect architecture: 'backbone.weight' (a plain Conv2d weight -- the "
            "'michelangelo' backbone's signature) isn't present as a top-level key in this "
            "checkpoint's state_dict. It may use a different backbone_type ('conv' or "
            "'edge_encoder' have differently-named/nested backbone keys), or this isn't a "
            "CondiNet_Tr_priormulti checkpoint at all."
        )

    hidden_dim, embed_dim = state_dict["backbone.weight"].shape[:2]

    linear_out = state_dict["linear.weight"].shape[0]
    if linear_out % embed_dim != 0:
        raise ValueError(
            f"linear.weight's output size ({linear_out}) isn't a multiple of the "
            f"detected michelangelo_embed_dim ({embed_dim}) -- can't recover "
            f"michelangelo_num_latents from tensor shapes."
        )

    enc_layer_indices = {
        int(k.split(".")[3]) for k in state_dict if k.startswith("transformer.encoder.layers.")
    }
    dec_layer_indices = {
        int(k.split(".")[3]) for k in state_dict if k.startswith("transformer.decoder.layers.")
    }

    heuristic_heads = max(1, int(hidden_dim) // 64)
    print(f"[ckpt][warn] n_heads can't be recovered from tensor shapes "
          f"(nn.MultiheadAttention's parameter shapes don't depend on head count) -- "
          f"guessing hidden_dim // 64 = {heuristic_heads} by convention, same approach as "
          f"detect_model_config_from_ckpt's 'heads = width // 64' in train.py. If this "
          f"checkpoint used a non-default --n_heads, results will be silently wrong -- "
          f"pass the real value as an override if you know it.")
    print("[ckpt][warn] norm_before/prior_type also aren't shape-detectable (they change "
          "how forward() uses the layers, not what parameters exist) -- defaulting to "
          "True/'learned'. Override if your training run used different values.")

    return {
        "nb_inputs": int(state_dict["input_proj.weight"].shape[0]),
        "tp": int(state_dict["query_learn.weight"].shape[0]),
        "condi_channels": [int(hidden_dim)],  # only the LAST element matters for this backbone
        "n_heads": heuristic_heads,
        "enc_layers": len(enc_layer_indices),
        "dec_layers": len(dec_layer_indices),
        "norm_before": True,
        "prior_type": "learned",
        "backbone_type": "michelangelo",
        "michelangelo_embed_dim": int(embed_dim),
        "michelangelo_num_latents": int(linear_out // embed_dim),
    }


def detect_condinet_config(ckpt_path: str) -> dict:
    """Returns a dict with all of REQUIRED_KEYS, ready to pass straight into
    CondiNet_Tr_priormulti(**config) alongside num_inputs/horizon (which are
    just config["nb_inputs"]/config["tp"] under CondiNet's own arg names)."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(payload)

    if isinstance(payload, dict) and "config" in payload:
        saved = payload["config"]
        config = {k: saved.get(k) for k in REQUIRED_KEYS}
        missing = [k for k, v in config.items() if v is None]
        if not missing:
            print(f"[ckpt] Exact config recovered from checkpoint's saved 'config' "
                  f"(epoch={payload.get('epoch')}, val_loss={payload.get('val_loss')}): {config}")
            return config
        print(f"[ckpt] Saved config present but missing {missing} -- "
              f"falling back to shape detection for those.")
    else:
        config = {}
        print("[ckpt] No saved 'config' in this checkpoint -- detecting architecture "
              "from tensor shapes instead (see warnings for what can't be recovered exactly).")

    detected = _detect_from_shapes(state_dict)
    for key, value in detected.items():
        if config.get(key) is None:
            config[key] = value

    print(f"[ckpt] Final config: {config}")
    return config