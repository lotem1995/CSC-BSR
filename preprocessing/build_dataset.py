import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

import yaml

import torch
from torch.utils.data import Dataset

from handle_fen import fen_to_board_int
from handle_game_CSV import pair_images_with_fens
from splitting_images import slice_image_with_coordinates

NUM_CLASSES = 17  # 0 is empty, 1-16 are piece IDs
CLASS_MAP = {
    0: "empty",
    1: "white_pawn",
    2: "white_knight",
    3: "white_bishop",
    4: "white_rook",
    5: "white_queen",
    6: "white_king",
    11: "black_pawn",
    12: "black_knight",
    13: "black_bishop",
    14: "black_rook",
    15: "black_queen",
    16: "black_king",
    17: "OOD"     #handles hands.
}
DEFAULT_SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}


def _load_config(path: Path) -> Dict:
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs. Install pyyaml or use JSON.")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_split(split: Dict[str, Any]) -> Dict[str, Any]:
    # Check if this is a manual split (values are lists of game names)
    is_manual = all(isinstance(v, list) for v in split.values())

    if is_manual:
        # Just ensure the required keys exist
        for key in ("train", "val", "test"):
            if key not in split:
                raise ValueError("Manual split dict must contain train, val, and test keys")
        return split

    # Existing logic for percentage-based splits
    total = sum(split.values())
    if not math.isclose(total, 1.0, rel_tol=1e-3):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")
    for key in ("train", "val", "test"):
        if key not in split:
            raise ValueError("Split dict must contain train, val, and test keys")
        if split[key] < 0:
            raise ValueError("Split ratios must be positive")
    return split


def _extract_label(image_path: Path) -> Optional[int]:
    match = re.search(r"_class(\d+)", image_path.stem)
    if not match:
        return None
    return int(match.group(1))


def _board_id_from_tile(image_path: Path) -> str:
    # Tiles are named like original_tile_rowX_columnY_classZ
    parts = image_path.stem.split("_tile")[0]
    return parts




def _desired_class_counts(global_counts: np.ndarray, split: Dict[str, float]) -> Dict[str, np.ndarray]:
    desired = {}
    for name, ratio in split.items():
        desired[name] = global_counts * ratio
    return desired


def _score_split(
        prospective: np.ndarray,
        desired: np.ndarray,
        board_load: int,
        target_boards: float,
) -> float:
    class_penalty = np.linalg.norm(prospective - desired)
    load_penalty = max(0.0, (board_load - target_boards))
    return class_penalty + load_penalty


def _identify_game(board_id: str, known_games: List[str]) -> str:
    """
    Attempts to match a board_id to a specific game name.
    If no game matches (or no games known), treats the board as its own game.
    """
    if not known_games:
        return board_id

    # Sort games by length descending to match specific prefixes first
    # e.g., match "Game_01_Part2" before "Game_01"
    for game in sorted(known_games, key=len, reverse=True):
        if board_id.startswith(game):
            return game

    return board_id  # Fallback: treat the board as the atomic unit


def _group_stratified_split(
        tiles: List[Dict],
        split: Dict[str, Any],  # Changed type hint to Any
        known_games: List[str],
        num_classes: int = NUM_CLASSES,
        seed: int = 42,
) -> Dict[str, List[str]]:
    random.seed(seed)

    # 1. Group tiles by Board
    boards = _group_tiles_by_board(tiles)

    # 2. Group Boards by Game
    games: Dict[str, List[str]] = defaultdict(list)
    for board_id in boards:
        game_id = _identify_game(board_id, known_games)
        games[game_id].append(board_id)

    # CHECK: Is this a manual split (lists) or automatic (ratios)?
    is_manual = all(isinstance(v, list) for v in split.values())

    if is_manual:
        # --- Manual Assignment Logic ---
        game_assignments = {}
        for game_id in games:
            assigned_split = None

            # Check which list this game belongs to in the yaml
            for split_name, game_list in split.items():
                if game_id in game_list:
                    assigned_split = split_name
                    break

            # If game found on disk but not in yaml, warn and default to train
            if assigned_split is None:
                print(f"[WARNING] Game '{game_id}' found on disk but not in YAML split config. Defaulting to 'train'.")
                assigned_split = "train"

            game_assignments[game_id] = assigned_split

    else:
        # --- Existing Automatic Stratification Logic ---
        # 3. Compute Class Counts per Game
        game_class_counts = _compute_game_class_counts(games, boards, num_classes)

        total_counts = sum(game_class_counts.values())
        desired = _desired_class_counts(total_counts, split)

        # 4. Assign Games to Splits
        game_assignments = _assign_groups(
            game_class_counts,
            split,
            len(games),
            seed,
        )

    # 5. Expand back to tiles (Common for both methods)
    return _collect_split_tiles_by_game(games, boards, game_assignments, split)


def _group_tiles_by_board(tiles: List[Dict]) -> Dict[str, List[Dict]]:
    boards: Dict[str, List[Dict]] = defaultdict(list)
    for tile in tiles:
        boards[tile["board_id"]].append(tile)
    return boards



def _compute_game_class_counts(
        games: Dict[str, List[str]],
        boards: Dict[str, List[Dict]],
        num_classes: int
) -> Dict[str, np.ndarray]:
    game_counts: Dict[str, np.ndarray] = {}

    for game_id, board_ids in games.items():
        counts = np.zeros(num_classes, dtype=float)
        for bid in board_ids:
            for item in boards[bid]:
                label = item["label"]
                if label < num_classes:
                    counts[label] += 1
        game_counts[game_id] = counts

    return game_counts


def _assign_groups(
        group_counts: Dict[str, np.ndarray],
        split: Dict[str, float],
        num_groups: int,
        seed: int,
) -> Dict[str, str]:
    total_counts = sum(group_counts.values())
    ordered_ids = _order_groups_by_rarity(group_counts, total_counts)

    rng = random.Random(seed)
    rng.shuffle(ordered_ids)

    assignments: Dict[str, str] = {}
    split_names = sorted(split.keys(), key=lambda k: split[k], reverse=True)
    split_sizes = {name: int(num_groups * ratio) for name, ratio in split.items()}

    remainder = num_groups - sum(split_sizes.values())
    split_sizes[split_names[0]] += remainder

    current_idx = 0
    for split_name in split_names:
        size = split_sizes[split_name]
        for _ in range(size):
            if current_idx < len(ordered_ids):
                assignments[ordered_ids[current_idx]] = split_name
                current_idx += 1

    return assignments


def _collect_split_tiles_by_game(
        games: Dict[str, List[str]],
        boards: Dict[str, List[Dict]],
        game_assignments: Dict[str, str],
        split: Dict[str, float],
) -> Dict[str, List[str]]:
    split_tiles: Dict[str, List[str]] = {name: [] for name in split}

    for game_id, board_ids in games.items():
        destination = game_assignments[game_id]
        for bid in board_ids:
            split_tiles[destination].extend(tile["image"] for tile in boards[bid])

    return split_tiles


def _order_groups_by_rarity(group_counts: Dict[str, np.ndarray], total_counts: np.ndarray) -> List[str]:
    def rarity_score(counts: np.ndarray) -> float:
        # Heavily weight presence of rare classes
        rare_weights = np.where(total_counts > 0, 1.0 / (total_counts + 1e-6), 0.0)
        return float((counts * rare_weights).sum())

    return sorted(group_counts, key=lambda g: rarity_score(group_counts[g]), reverse=True)

def _collect_hand_tile_names(hands_dir: Path) -> Set[str]:
    hands_dir = hands_dir.expanduser().resolve()

    print("[DEBUG] hands_dir =", hands_dir)
    print("[DEBUG] hands_dir exists =", hands_dir.exists(), "is_dir =", hands_dir.is_dir())

    if not hands_dir.exists() or not hands_dir.is_dir():
        print("[DEBUG] hands_dir invalid -> collected 0 hands names")
        return set()

    imgs = list(hands_dir.rglob("*.png"))
    names = {p.name for p in imgs if p.is_file()}

    print("[DEBUG] hands png files found =", len(imgs))
    print("[DEBUG] unique hands names =", len(names))
    print("[DEBUG] sample hands names =", list(sorted(names))[:8])

    return names


def _gather_tiles(
    raw_tiles_dir: Path,
    hands_dir: Path,
    embedding_dir: Optional[Path],
    embedding_ext: str,
) -> List[Dict]:
    raw_tiles_dir = raw_tiles_dir.expanduser().resolve()
    hands_dir = hands_dir.expanduser().resolve()

    print("[DEBUG] raw_tiles_dir =", raw_tiles_dir)
    print("[DEBUG] raw_tiles_dir exists =", raw_tiles_dir.exists(), "is_dir =", raw_tiles_dir.is_dir())

    hand_tile_names = _collect_hand_tile_names(hands_dir)

    raw_png_paths = list(raw_tiles_dir.rglob("*.png"))
    raw_names = {p.name for p in raw_png_paths}
    inter = raw_names & hand_tile_names

    print("[DEBUG] raw tiles png files scanned =", len(raw_png_paths))
    print("[DEBUG] unique raw tile names =", len(raw_names))
    print("[DEBUG] intersection(raw_names, hands_names) =", len(inter))
    print("[DEBUG] sample intersections =", list(sorted(inter))[:8])

    dropped_inside_hands = 0
    skipped_no_label = 0
    kept = 0
    relabeled_to_ood = 0

    tiles: List[Dict] = []
    for image_path in sorted(raw_png_paths):
        # Safety: if hands_dir is inside raw_tiles_dir, do NOT ingest the copies inside hands_dir
        try:
            image_path.resolve().relative_to(hands_dir)
            dropped_inside_hands += 1
            continue
        except ValueError:
            pass

        is_hand = image_path.name in hand_tile_names

        # If it's a hand tile -> force OOD label, do NOT require _class(\d+)
        if is_hand:
            label = 17
            relabeled_to_ood += 1
        else:
            label = _extract_label(image_path)
            if label is None:
                skipped_no_label += 1
                continue

        board_id = _board_id_from_tile(image_path)

        embedding_path = None
        if embedding_dir:
            candidate = embedding_dir / f"{image_path.stem}{embedding_ext}"
            if candidate.exists():
                embedding_path = str(candidate)

        tiles.append(
            {
                "image": str(image_path),
                "label": int(label),
                "board_id": board_id,
                "embedding": embedding_path,
            }
        )
        kept += 1

    print("[DEBUG] skipped_no_label =", skipped_no_label)
    print("[DEBUG] dropped_inside_hands =", dropped_inside_hands)
    print("[DEBUG] relabeled_to_ood =", relabeled_to_ood)
    print("[DEBUG] kept tiles =", kept)

    if not tiles:
        raise RuntimeError(
            "No tiles were found. Check raw_tiles_dir/hands_dir paths and naming."
        )

    return tiles



def build_manifest(config_path: Path) -> Dict:
    config = _load_config(config_path)

    # config is: ...\CSC-BSR\preprocessing\config.yaml
    # so project root is: ...\CSC-BSR
    project_root = config_path.parents[1].expanduser().resolve()

    # hands is a folder in CSC-BSR root
    hands_dir = Path(config.get("hands_dir", project_root / "hands")).expanduser().resolve()

    print("[DEBUG] config_path =", config_path)
    print("[DEBUG] project_root =", project_root)
    print("[DEBUG] hands_dir =", hands_dir)

    raw_tiles_dir, data_root_path, embedding_dir_path = _resolve_paths(config)
    path_root = _compute_path_root(config_path, raw_tiles_dir, data_root_path, embedding_dir_path)
    final_size, overlap_percent, zero_padding = _parse_tile_params(config)
    embedding_ext = config.get("embedding_ext", ".npy")
    split = _validate_split(config.get("split", DEFAULT_SPLIT))
    seed = int(config.get("seed", 42))

    known_game_names: List[str] = []

    if data_root_path:
        games_found = _discover_games(data_root_path)
        known_game_names = [g_name for g_name, _, _ in games_found]

        print("\n" + "=" * 60)
        print(f" [INFO] DETECTED {len(known_game_names)} GAMES IN '{data_root_path}':")
        print(" (Use these exact names in your dataset_config.yaml)")
        for name in known_game_names:
            print(f"   - {name}")
        print("=" * 60 + "\n")

        _generate_tiles_from_games(
            games_found,
            raw_tiles_dir,
            overlap_percent,
            final_size,
            zero_padding,
        )

    # ✅ hands filtering happens BEFORE splitting and before manifest creation
    tiles = _gather_tiles(raw_tiles_dir, hands_dir, embedding_dir_path, embedding_ext)

    split_tiles = _group_stratified_split(tiles, split, known_game_names, NUM_CLASSES, seed)

    manifest = {
        "config": {
            "raw_tiles_dir": str(raw_tiles_dir),
            "data_root": str(data_root_path) if data_root_path else None,
            "embedding_dir": str(embedding_dir_path) if embedding_dir_path else None,
            "embedding_ext": embedding_ext,
            "split": split,
            "seed": seed,
            "tile_size": list(final_size),
            "tile_overlap": overlap_percent,
            "zero_padding": zero_padding,
            "path_root": str(path_root),
            "hands_dir": str(hands_dir),
        },
        "classes": CLASS_MAP,
        "splits": {name: [] for name in split},
    }

    tiles_by_path = {tile["image"]: tile for tile in tiles}
    for split_name, image_paths in split_tiles.items():
        for image_path in image_paths:
            tile = tiles_by_path[image_path]
            manifest["splits"][split_name].append(_relativize_sample(tile, path_root))

    print("[DEBUG] manifest split sizes:", {k: len(v) for k, v in manifest["splits"].items()})

    return manifest


def _compute_path_root(
        config_path: Path,
        raw_tiles_dir: Path,
        data_root: Optional[Path],
        embedding_dir: Optional[Path],
) -> Path:
    candidates = [raw_tiles_dir]
    if data_root:
        candidates.append(data_root)
    if embedding_dir:
        candidates.append(embedding_dir)
    candidates.append(config_path.parent)
    common = os.path.commonpath([str(p) for p in candidates])
    return Path(common)


def _relativize_sample(sample: Dict, root: Path) -> Dict:
    new_sample = dict(sample)
    new_sample["image"] = os.path.relpath(sample["image"], root)
    if sample.get("embedding"):
        new_sample["embedding"] = os.path.relpath(sample["embedding"], root)
    return new_sample


def _resolve_paths(config: Dict) -> Tuple[Path, Optional[Path], Optional[Path]]:
    raw_tiles_dir = Path(config["raw_tiles_dir"]).expanduser().resolve()
    data_root = config.get("data_root")
    data_root_path = Path(data_root).expanduser().resolve() if data_root else None
    embedding_dir = config.get("embedding_dir")
    embedding_dir_path = Path(embedding_dir).expanduser().resolve() if embedding_dir else None
    return raw_tiles_dir, data_root_path, embedding_dir_path


def _parse_tile_params(config: Dict) -> Tuple[Tuple[int, int], float, bool]:
    tile_size_cfg = config.get("tile_size", [224, 224])
    if len(tile_size_cfg) != 2:
        raise ValueError("tile_size must be a 2-element list like [224, 224]")
    final_size: Tuple[int, int] = (int(tile_size_cfg[0]), int(tile_size_cfg[1]))
    overlap_percent = float(config.get("tile_overlap", 0.7))
    zero_padding = bool(config.get("zero_padding", True))
    return final_size, overlap_percent, zero_padding


def save_manifest(manifest: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def save_split_indices(manifest: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in manifest["splits"].items():
        csv_path = output_dir / f"{split_name}.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("image,label,board_id,embedding\n")
            for sample in samples:
                embedding = sample.get("embedding") or ""
                handle.write(f"{sample['image']},{sample['label']},{sample['board_id']},{embedding}\n")


def _generate_tiles_from_games(
        games_list: List[Tuple[str, Path, Path]],
        raw_tiles_dir: Path,
        overlap_percent: float,
        final_size: Tuple[int, int],
        zero_padding: bool,
) -> None:
    raw_tiles_dir.mkdir(parents=True, exist_ok=True)
    # games_list now contains (game_name, csv_path, images_dir)
    for game_name, csv_path, images_dir in games_list:
        pairs = pair_images_with_fens(str(csv_path), str(images_dir))
        print("Generating tiles for game", game_name)
        for image_path, fen in pairs:
            board = fen_to_board_int(fen)
            slice_image_with_coordinates(
                game_name,
                image_path,
                str(raw_tiles_dir),
                board,
                overlap_percent=overlap_percent,
                final_size=final_size,
                zero_padding=zero_padding,
            )


def _discover_games(data_root: Path) -> List[Tuple[str, Path, Path]]:
    """
    Returns list of (game_name, csv_path, images_dir)
    """
    games: List[Tuple[str, Path, Path]] = []
    for game_dir in sorted(data_root.iterdir()):
        if not game_dir.is_dir():
            continue
        csv_files = sorted(game_dir.glob("*.csv"))
        images_dir = game_dir / "tagged_images"
        if not csv_files or not images_dir.exists():
            continue
        # Use folder name or CSV stem as game name
        game_name = csv_files[0].stem
        games.append((game_name, csv_files[0], images_dir))
    if not games:
        raise RuntimeError(f"No game folders with CSV and tagged_images found under {data_root}")
    return games


class ChessSquaresDataset(Dataset):
    """
    Torch Dataset for chess square tiles.
    """

    def __init__(
            self,
            manifest_path: Path,
            split: str = "train",
            transform=None,
            use_embeddings: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.data = json.loads(self.manifest_path.read_text())
        if split not in self.data["splits"]:
            raise ValueError(f"Split {split} not found in manifest")
        self.samples = self.data["splits"][split]
        self.transform = transform
        self.use_embeddings = use_embeddings
        self.path_root = Path(self.data["config"].get("path_root", ".")).expanduser().resolve()
        print("[DEBUG] loading manifest:", Path(manifest_path).resolve())
        print("[DEBUG] samples in split", split, "=", len(self.data["splits"][split]))


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = self.path_root / sample["image"]
        label = int(sample["label"])
        board_id = sample["board_id"]
        embedding_path = sample.get("embedding")
        if embedding_path:
            embedding_path = self.path_root / embedding_path

        if self.use_embeddings and embedding_path:
            features = np.load(embedding_path)
            image_tensor = torch.as_tensor(features)
        else:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    image_tensor = self.transform(img)
                else:
                    image_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return {
            "image": image_tensor,
            "label": label,
            "board_id": board_id,
            "path": str(path),
        }


def main():
    parser = argparse.ArgumentParser(description="Build stratified game-level train/val/test splits.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output manifest path (default comes from config or data/manifest.json).",
    )
    parser.add_argument(
        "--split-indices-dir",
        default=None,
        help="Where to write per-split CSV indices (default: alongside manifest in 'splits/').",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    manifest = build_manifest(config_path)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(config_path).with_suffix("").with_name("dataset_manifest.json")
    )
    save_manifest(manifest, output_path)
    splits_dir = (
        Path(args.split_indices_dir).expanduser().resolve()
        if args.split_indices_dir
        else output_path.parent / "splits"
    )
    save_split_indices(manifest, splits_dir)
    print(f"Saved manifest to {output_path} and split CSVs to {splits_dir}")


if __name__ == "__main__":
    main()
