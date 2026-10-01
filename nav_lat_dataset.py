"""
navigator_latent_dataset.py

Two ways to split cached Michelangelo latents into train/val, both producing
the same sample interface the training loop iterates over:
    identifier, input_latent_list, output_latent_list, extra

1. NAVIGATOR_LATENT_Dataset_multitime -- PARTICIPANT-level split (original
   behaviour). sequence_list picks whole sequences for this split; use this
   when you have >= 2 sequences, via make_participant_splits.py.

2. NAVIGATOR_LATENT_Dataset_temporal_split -- TEMPORAL (within-sequence) split,
   for when you only have ONE sequence (one participant/acquisition). Frame
   INDICES are split instead of whole sequences: e.g. the first 80% of frames
   for train, the last 20% for val, with a gap between them so no sliding
   window straddles the boundary and leaks train frames into a val sample (or
   vice versa). This is a materially weaker validation signal than a held-out
   participant -- it tells you "does the model generalize to a LATER part of
   the SAME recording", not "does it generalize to a different person/session"
   -- worth keeping in mind when reading val_loss, and revisiting once you
   have more than one sequence.

Both classes share the frame-listing/-numbering helpers below.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _frame_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else -1


def _list_frames(cache_dir: Path, sequence: str) -> list[Path]:
    sequence_dir = cache_dir / sequence
    if not sequence_dir.is_dir():
        return []
    return sorted(sequence_dir.glob("*.pt"), key=_frame_number)


def _windows_from_frames(frames: list[Path], nb_inputs: int, nb_pred: int,
                         stride: int) -> list[tuple[list[Path], list[Path]]]:
    window = nb_inputs + nb_pred
    return [
        (frames[start:start + nb_inputs], frames[start + nb_inputs:start + window])
        for start in range(0, len(frames) - window + 1, stride)
    ]


class _BaseLatentDataset(Dataset):
    samples: list[tuple[str, list[Path], list[Path]]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sequence, input_paths, output_paths = self.samples[idx]

        input_latents = [torch.load(p, map_location="cpu") for p in input_paths]
        output_latents = [torch.load(p, map_location="cpu") for p in output_paths]

        identifier = f"{sequence}:{input_paths[0].stem}-{output_paths[-1].stem}"
        extra = {
            "sequence": sequence,
            "input_files": [str(p) for p in input_paths],
            "output_files": [str(p) for p in output_paths],
        }
        return identifier, input_latents, output_latents, extra


class NAVIGATOR_LATENT_Dataset_multitime(_BaseLatentDataset):
    """Participant-level split: sequence_list picks whole sequences for this
    split (train xor val, never both). Use when you have >= 2 sequences."""

    def __init__(
        self,
        cache_dir: str,
        nb_inputs: int,
        sequence_list: list[str],
        nb_pred: int,
        mode: str = "train",
        stride: int = 1,
    ):
        self.cache_dir = Path(cache_dir)
        self.nb_inputs = nb_inputs
        self.nb_pred = nb_pred
        self.mode = mode

        self.samples = []
        for sequence in sequence_list:
            frames = _list_frames(self.cache_dir, sequence)
            if len(frames) < nb_inputs + nb_pred:
                print(f"  [warn] {sequence}: only {len(frames)} cached frames, "
                      f"need >= {nb_inputs + nb_pred} for one sample -- skipping.")
                continue
            for input_paths, output_paths in _windows_from_frames(frames, nb_inputs, nb_pred, stride):
                self.samples.append((sequence, input_paths, output_paths))

        print(f"NAVIGATOR_LATENT_Dataset_multitime[{mode}]: "
              f"{len(self.samples)} samples from {len(sequence_list)} sequence(s)")


class NAVIGATOR_LATENT_Dataset_temporal_split(_BaseLatentDataset):
    """Temporal split WITHIN one (or more) sequences: frame INDICES are split
    into an early train range and a late val range, with a gap so no window
    straddles the boundary. See the module docstring for the validity caveat.
    """

    def __init__(
        self,
        cache_dir: str,
        nb_inputs: int,
        sequence_list: list[str],
        nb_pred: int,
        mode: str,                      # "train" or "val" -- REQUIRED here (no default)
        val_frac: float = 0.2,
        stride: int = 1,
    ):
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        if not 0.0 < val_frac < 1.0:
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

        self.cache_dir = Path(cache_dir)
        self.nb_inputs = nb_inputs
        self.nb_pred = nb_pred
        self.mode = mode
        self.val_frac = val_frac

        window = nb_inputs + nb_pred
        gap = window  # frames straddling the boundary are excluded from both sides

        self.samples = []
        for sequence in sequence_list:
            frames = _list_frames(self.cache_dir, sequence)
            n = len(frames)
            if n < 2 * window + gap:
                print(f"  [warn] {sequence}: only {n} cached frames, need >= "
                      f"{2 * window + gap} to form both a train and a val range "
                      f"with a gap between them -- skipping.")
                continue

            split_point = int(round(n * (1 - val_frac)))
            if mode == "train":
                sub_frames = frames[:split_point]
            else:
                sub_frames = frames[split_point + gap:]

            if len(sub_frames) < window:
                print(f"  [warn] {sequence}: {mode} range has only {len(sub_frames)} "
                      f"frames after the split (need >= {window}) -- skipping.")
                continue

            for input_paths, output_paths in _windows_from_frames(sub_frames, nb_inputs, nb_pred, stride):
                self.samples.append((sequence, input_paths, output_paths))

        print(f"NAVIGATOR_LATENT_Dataset_temporal_split[{mode}]: "
              f"{len(self.samples)} samples from {len(sequence_list)} sequence(s) "
              f"(val_frac={val_frac}, gap={gap} frames at the boundary)")