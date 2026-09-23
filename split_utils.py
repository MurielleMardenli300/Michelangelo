import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def resolve_paths(key: str, pc_dir: str, mesh_dir: str) -> Tuple[str, str]:
    """'<participant>/pointcloud_00001.ply' -> (ply_path, stl_path), using
    the same frame-number convention as the original AbdominalDataset scan.
    """
    participant, ply_name = key.split("/", 1)
    ply_path = Path(pc_dir) / participant / ply_name
    frame_num = int("".join(filter(str.isdigit, ply_path.stem)))
    stl_path = Path(mesh_dir) / participant / f"mesh_{frame_num}.stl"
    return str(ply_path), str(stl_path)


def scan_paired_samples(pc_dir: str, mesh_dir: str) -> List[str]:
    """Scan disk for every (ply, stl) pair that exists on both sides.
    Identical traversal order to the original AbdominalDataset (sorted
    iterdir, sorted glob) so this is a drop-in replacement.
    """
    keys = []
    for participant_dir in sorted(Path(pc_dir).iterdir()):
        if not participant_dir.is_dir():
            continue
        participant = participant_dir.name
        mesh_part_dir = Path(mesh_dir) / participant
        if not mesh_part_dir.is_dir():
            continue
        for ply_path in sorted(participant_dir.glob("*.ply")):
            try:
                frame_num = int("".join(filter(str.isdigit, ply_path.stem)))
            except ValueError:
                continue
            stl_path = mesh_part_dir / f"mesh_{frame_num}.stl"
            if stl_path.exists():
                keys.append(f"{participant}/{ply_path.name}")
    return keys


def build_or_load_split(pc_dir: str,
                         mesh_dir: str,
                         split_json: str,
                         val_frac: float = 0.15,
                         test_frac: float = 0.15,
                         seed: int = 42) -> Dict[str, List[str]]:
    """
    If split_json already exists: load it as-is and return it. This is what
    makes the split reproducible across runs, machines, and checkpoints --
    once created, the split file is the source of truth, not the directory
    scan or the seed.

    If split_json does not exist yet: scan pc_dir/mesh_dir, shuffle
    deterministically with `seed`, cut into train/val/test by the given
    fractions, write the result to split_json, and return it.

    If split_json exists but new paired samples exist on disk that aren't in
    any of its three lists (e.g. new participants added after the split was
    created), they're appended to "train" and the file is rewritten, with a
    printed warning -- so growth of the dataset never silently changes val/test.
    """
    split_path = Path(split_json)
    current_keys = set(scan_paired_samples(pc_dir, mesh_dir))

    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)

        known = (set(split.get("train", [])) | set(split.get("val", []))
                 | set(split.get("test", [])))

        missing_on_disk = known - current_keys
        if missing_on_disk:
            print(f"[split] WARNING: {len(missing_on_disk)} samples listed in "
                  f"{split_json} no longer exist on disk (deleted/renamed?). "
                  f"First few: {sorted(missing_on_disk)[:5]}")

        new_on_disk = current_keys - known
        if new_on_disk:
            print(f"[split] WARNING: {len(new_on_disk)} paired samples found on "
                  f"disk that are NOT in {split_json} (new data since the split "
                  f"was created). Adding them to 'train' and rewriting the file. "
                  f"First few: {sorted(new_on_disk)[:5]}")
            split.setdefault("train", []).extend(sorted(new_on_disk))
            with open(split_path, "w") as f:
                json.dump(split, f, indent=2)

        print(f"[split] Loaded existing split from {split_json}: "
              f"train={len(split.get('train', []))}, "
              f"val={len(split.get('val', []))}, "
              f"test={len(split.get('test', []))}")
        return split

    # -- no split file yet: create one --
    keys = sorted(current_keys)          # sort first: shuffle must not depend
    rng = random.Random(seed)            # on filesystem iteration order
    rng.shuffle(keys)

    n_total = len(keys)
    n_val = int(round(n_total * val_frac))
    n_test = int(round(n_total * test_frac))
    n_train = n_total - n_val - n_test

    split = {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
        "_meta": {"seed": seed, "val_frac": val_frac, "test_frac": test_frac,
                  "n_total": n_total},
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"[split] Created new split, saved to {split_json}: "
          f"train={n_train}, val={n_val}, test={n_test} (seed={seed})")
    return split


def keys_to_paths(keys: List[str], pc_dir: str, mesh_dir: str) -> List[Tuple[str, str]]:
    """Resolve a list of split keys to (ply_path, stl_path) pairs, skipping
    (with a warning) any that no longer exist on disk."""
    pairs = []
    for k in keys:
        ply_path, stl_path = resolve_paths(k, pc_dir, mesh_dir)
        if not Path(ply_path).exists() or not Path(stl_path).exists():
            print(f"[split] WARNING: sample '{k}' missing on disk, skipping "
                  f"(ply exists={Path(ply_path).exists()}, "
                  f"stl exists={Path(stl_path).exists()})")
            continue
        pairs.append((ply_path, stl_path))
    return pairs


def reconstruct_legacy_random_split(pc_dir: str, mesh_dir: str,
                                     seed: int = 42,
                                     val_frac: float = 0.15) -> Dict[str, List[str]]:
    """
    Recreates the exact train/val split produced by the ORIGINAL script's

        torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed))

    for checkpoints trained before split.json existed (e.g. last-v47.ckpt).

    IMPORTANT: only valid if pc_dir/mesh_dir have NOT changed (same files,
    same directory listing) since that checkpoint was trained -- the legacy
    random_split's result depends entirely on len(dataset) and the index
    order iterdir()/glob() produced at that time. There is no way to verify
    this after the fact; treat the result as "best reconstruction", not a
    guarantee, and re-validate the recovered checkpoint against it once.
    """
    import torch

    keys = scan_paired_samples(pc_dir, mesh_dir)  # same order as old AbdominalDataset
    n_total = len(keys)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    return {
        "train": [keys[i] for i in train_idx],
        "val": [keys[i] for i in val_idx],
        "test": [],
        "_meta": {"reconstructed_legacy": True, "seed": seed, "val_frac": val_frac},
    }