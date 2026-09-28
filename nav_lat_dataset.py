"""
navigator_latent_dataset.py

Replacement for NAVIGATOR_2D_Dataset_multitime, serving cached Michelangelo
latents (from precompute_michelangelo_latents.py) instead of image slices.
Same sample interface the training loop iterates over:
    identifier, input_latent_list, output_latent_list, extra

input_latent_list  : list of nb_inputs tensors, each (num_latents, embed_dim)
output_latent_list : list of nb_pred   tensors, each (num_latents, embed_dim)

Each list entry is the FULL cached latent (not pooled) -- pooling for
CondiNet's input happens in the training script's collate/forward step, right
before the model call, so the dataset itself stays representation-agnostic
and the same cached files also serve as ground truth for the loss and for
geometric validation decoding.

Sequence construction: for each participant, frames are sorted by the last
number in their filename (matching precompute_michelangelo_latents.py's
mirrored <frame>.pt naming), and every valid sliding window of length
nb_inputs + nb_pred becomes one sample. Adjust `_list_frames` if your
frame-numbering convention differs.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _frame_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else -1


def _list_frames(cache_dir: Path, participant: str) -> list[Path]:
    participant_dir = cache_dir / participant
    if not participant_dir.is_dir():
        return []
    return sorted(participant_dir.glob("*.pt"), key=_frame_number)


class NAVIGATOR_LATENT_Dataset_multitime(Dataset):
    def __init__(
        self,
        cache_dir: str,
        nb_inputs: int,
        sequence_list: list[str],
        nb_pred: int,
        mode: str = "train",
        stride: int = 1,
    ):
        """
        cache_dir     : root of precompute_michelangelo_latents.py's output
                         (contains one subfolder per participant).
        sequence_list : which participant folders belong to this split --
                         same role/semantics as the original dataset's param.
        stride        : step between consecutive sampled windows within a
                         participant (1 = every possible window; raise this
                         to reduce overlap/redundancy in the sample list).
        """
        self.cache_dir = Path(cache_dir)
        self.nb_inputs = nb_inputs
        self.nb_pred = nb_pred
        self.mode = mode

        self.samples: list[tuple[str, list[Path], list[Path]]] = []
        for participant in sequence_list:
            frames = _list_frames(self.cache_dir, participant)
            window = nb_inputs + nb_pred
            if len(frames) < window:
                print(f"  [warn] {participant}: only {len(frames)} cached frames, "
                      f"need >= {window} for one sample -- skipping.")
                continue
            for start in range(0, len(frames) - window + 1, stride):
                input_paths = frames[start:start + nb_inputs]
                output_paths = frames[start + nb_inputs:start + window]
                self.samples.append((participant, input_paths, output_paths))

        print(f"NAVIGATOR_LATENT_Dataset_multitime[{mode}]: "
              f"{len(self.samples)} samples from {len(sequence_list)} participants")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        participant, input_paths, output_paths = self.samples[idx]

        input_latents = [torch.load(p, map_location="cpu") for p in input_paths]    # each (num_latents, embed_dim)
        output_latents = [torch.load(p, map_location="cpu") for p in output_paths]  # each (num_latents, embed_dim)

        identifier = f"{participant}:{input_paths[0].stem}-{output_paths[-1].stem}"
        extra = {
            "participant": participant,
            "input_files": [str(p) for p in input_paths],
            "output_files": [str(p) for p in output_paths],
        }
        return identifier, input_latents, output_latents, extra