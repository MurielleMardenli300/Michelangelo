"""
make_participant_splits.py

Creates the --train_participants_file / --val_participants_file text files the
temporal-model training scripts expect: one sequence ("participant") folder name
per line, each matching a subfolder of --latent_cache_dir.

The split is at the participant/sequence level: no sequence appears in more than
one file, so validation never sees frames from a training sequence. It is seeded
and, once written, will not be overwritten unless you pass --overwrite -- same
idea as split.json for the ShapeVAE: the split stays fixed across runs, so
results stay comparable.

Sequences with too few cached frames to form even one (nb_inputs + tp) window are
excluded and reported. With fewer than 2 usable sequences no files are written,
because a participant-level split is impossible (see the message it prints).

Usage
-----
python make_participant_splits.py --cache_dir output/latents --output_dir splits \\
    --val_frac 0.2 --nb_inputs 3 --tp 3
"""

import argparse
import random
from pathlib import Path


def count_frames(sequence_dir: Path) -> int:
    return len(list(sequence_dir.glob("*.pt")))


def write_list(names: list[str], path: Path) -> None:
    path.write_text("".join(f"{n}\n" for n in names))
    print(f"  wrote {path}  ({len(names)}): {', '.join(names)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True,
                        help="Same folder you pass as --latent_cache_dir to training "
                             "(the PARENT of the per-sequence folders)")
    parser.add_argument("--output_dir", default="splits")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.0,
                        help="Optionally also write test_participants.txt (the training "
                             "scripts don't read it; it's for later evaluation)")
    parser.add_argument("--nb_inputs", type=int, default=3)
    parser.add_argument("--tp", type=int, default=3,
                        help="Prediction horizon; use the larger of the values you plan to train with")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    min_frames = args.nb_inputs + args.tp

    if not cache_dir.is_dir():
        raise SystemExit(f"{cache_dir} is not a directory.")

    counts = {p.name: count_frames(p) for p in sorted(cache_dir.iterdir()) if p.is_dir()}
    # if not counts:
    #     raise SystemExit(f"No sequence subfolders found in {cache_dir}.")

    print(f"Sequences in {cache_dir}:")
    for name, n in counts.items():
        print(f"  {name}: {n} cached frames" + ("" if n >= min_frames else f"  (< {min_frames}, excluded)"))

    # usable = [name for name, n in counts.items() if n >= min_frames]
    # if len(usable) < 2:
    #     raise SystemExit(
    #         f"\nOnly {len(usable)} usable sequence(s) (need >= {min_frames} frames each). "
    #         f"A participant-level split needs at least 2, otherwise the same frames would "
    #         f"end up in both train and validation and the validation loss would be "
    #         f"meaningless. Nothing written.\n"
    #         f"Options: add more sequences to the cache, or split a single sequence by "
    #         f"time instead (first part train, last part validation) -- that needs a small "
    #         f"change to the dataset class."
    #     )

    train_path = output_dir / "train_participants.txt"
    val_path = output_dir / "val_participants.txt"
    test_path = output_dir / "test_participants.txt"
    targets = [train_path, val_path] + ([test_path] if args.test_frac > 0 else [])
    existing = [p for p in targets if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"\nAlready exist: {', '.join(map(str, existing))}. Refusing to overwrite so the "
            f"split stays fixed across runs; pass --overwrite if you really want a new one."
        )

    rng = random.Random(args.seed)
    shuffled = sorted(usable)      # sort first so the shuffle doesn't depend on filesystem order
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(round(n * args.test_frac)) if args.test_frac > 0 else 0
    n_val = max(1, int(round(n * args.val_frac)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise SystemExit(f"Not enough sequences ({n}) for these fractions -- train would be empty.")

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSplit (seed={args.seed}):")
    write_list(train, train_path)
    write_list(val, val_path)
    if test:
        write_list(test, test_path)

    print(f"\nPass to training:\n  --train_participants_file {train_path} "
          f"--val_participants_file {val_path}")


if __name__ == "__main__":
    main()