import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import yaml
except ImportError:
    yaml = None

from handle_fen import fen_to_board_int
from handle_game_CSV import pair_images_with_fens
from splitting_images import slice_image_with_coordinates

# =========================
# CONSTANT RULES
# =========================
FORCED_VAL_GAME = "game2"
FORCED_TEST_GAME = "game4"

PIECE_LABELS = {
    0: "empty",
    1: "white_pawn", 11: "black_pawn",
    2: "white_knight", 12: "black_knight",
    3: "white_bishop", 13: "black_bishop",
    4: "white_rook", 14: "black_rook",
    5: "white_queen", 15: "black_queen",
    6: "white_king", 16: "black_king",
}

# =========================
# PATHS (AUTO)
# =========================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

DATA_DIR = ROOT / "data"
HANDS_DIR = DATA_DIR / "hands"
RAW_TILES_DIR = DATA_DIR / "preprocessed_data"
SPLITS_DIR = DATA_DIR / "splits"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

# accept either dataset_config.yaml OR dataset_config.example.yaml automatically
PREPROC_DIR = ROOT / "preprocessing"
CONFIG_CANDIDATES = [
    PREPROC_DIR / "dataset_config.yaml",
    PREPROC_DIR / "dataset_config.yml",
    PREPROC_DIR / "dataset_config.example.yaml",
    PREPROC_DIR / "dataset_config.example.yml",
]


# =========================
# HELPERS
# =========================
def load_config() -> Tuple[Dict, Path]:
    cfg_path = next((p for p in CONFIG_CANDIDATES if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(
            "Config not found. Tried:\n" + "\n".join(str(p) for p in CONFIG_CANDIDATES)
        )

    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs (pip install pyyaml).")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg, cfg_path


def reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def extract_label(p: Path) -> Optional[int]:
    m = re.search(r"_class(\d+)", p.stem)
    return int(m.group(1)) if m else None


def board_id_from_tile(p: Path) -> str:
    return p.stem.split("_tile")[0]


def discover_games(data_root: Path) -> List[Tuple[str, Path, Path]]:
    """
    Returns list of (game_dir_name, csv_path, images_dir)
    Assumes folders like data/game2_per_frame/ containing images/ + gt.csv (or any .csv fallback).
    """
    games: List[Tuple[str, Path, Path]] = []
    for gdir in sorted(data_root.iterdir()):
        if not gdir.is_dir():
            continue

        images = gdir / "images"
        if not images.exists():
            images = gdir / "tagged_images"
        if not images.exists():
            continue

        csvs = list(gdir.glob("gt.csv")) or sorted(gdir.glob("*.csv"))
        if not csvs:
            continue

        games.append((gdir.name, csvs[0], images))

    if not games:
        raise RuntimeError("No games found under ROOT/data/ (expected folders with images/ + gt.csv).")
    return games


def canonical_game_name(game_dir_name: str) -> str:
    """
    Convert 'game2_per_frame' -> 'game2'
    Convert 'game11_per_frame' -> 'game11'
    Otherwise keep as-is.
    """
    if game_dir_name.endswith("_per_frame"):
        return game_dir_name[:-len("_per_frame")]
    return game_dir_name


def generate_tiles(games, tile_size, overlap, zero_padding):
    RAW_TILES_DIR.mkdir(parents=True, exist_ok=True)
    for game_dir_name, csv_path, img_dir in games:
        game = canonical_game_name(game_dir_name)  # ✅ IMPORTANT: NO _per_frame IN TILE NAMES
        pairs = pair_images_with_fens(str(csv_path), str(img_dir))
        print(f"[TILES] {game_dir_name} -> prefix '{game}': {len(pairs)} frames")

        for img, fen in pairs:
            board = fen_to_board_int(fen)
            slice_image_with_coordinates(
                game,                    # ✅ tile prefix becomes game2, game4, ...
                img,
                str(RAW_TILES_DIR),
                board,
                overlap_percent=overlap,
                final_size=tile_size,
                zero_padding=zero_padding,
            )


def collect_hands() -> set:
    if not HANDS_DIR.exists():
        print(f"[HANDS] not found: {HANDS_DIR} (OOD will be 0)")
        return set()
    names = {p.name for p in HANDS_DIR.rglob("*.png")}
    print(f"[HANDS] found {len(names)} hand tiles in {HANDS_DIR}")
    if names:
        print(f"[HANDS] sample: {list(sorted(names))[:3]}")
    return names


def gather_tiles(embedding_dir: Optional[Path], embedding_ext: str) -> List[Dict]:
    hand_names = collect_hands()
    tiles: List[Dict] = []

    if not RAW_TILES_DIR.exists():
        raise RuntimeError(f"Missing tiles dir: {RAW_TILES_DIR}")

    for p in RAW_TILES_DIR.rglob("*.png"):
        label = extract_label(p)
        if label is None:
            continue
        if label not in PIECE_LABELS:
            continue

        is_ood = p.name in hand_names  # ✅ now matches exactly
        board_id = board_id_from_tile(p)

        emb = None
        if embedding_dir:
            cand = embedding_dir / f"{p.stem}{embedding_ext}"
            if cand.exists():
                emb = str(cand)

        tiles.append({
            "image": str(p),
            "label": int(label),
            "is_ood": bool(is_ood),
            "board_id": board_id,
            "embedding": emb,
        })

    print(f"[TILES] total={len(tiles)}  ood={sum(1 for t in tiles if t['is_ood'])}")
    if not tiles:
        raise RuntimeError("No tiles collected. Something is wrong with tile generation paths.")
    return tiles


def split_tiles(tiles: List[Dict]) -> Dict[str, List[str]]:
    boards = defaultdict(list)
    for t in tiles:
        boards[t["board_id"]].append(t)

    # game_id is the prefix before '_' in board_id (e.g. game2_frame_000856 -> game2)
    game_to_board_ids = defaultdict(list)
    for bid in boards.keys():
        game_id = bid.split("_")[0]
        game_to_board_ids[game_id].append(bid)

    splits = {"train": [], "val": [], "test": []}
    for game_id, bids in game_to_board_ids.items():
        if game_id == FORCED_TEST_GAME:
            dest = "test"
        elif game_id == FORCED_VAL_GAME:
            dest = "val"
        else:
            dest = "train"

        for b in bids:
            splits[dest].extend(t["image"] for t in boards[b])

    return splits


def relativize(sample: Dict) -> Dict:
    out = dict(sample)
    out["image"] = os.path.relpath(out["image"], ROOT)
    if out.get("embedding"):
        out["embedding"] = os.path.relpath(out["embedding"], ROOT)
    return out


def save_splits(manifest: Dict):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for s, rows in manifest["splits"].items():
        with (SPLITS_DIR / f"{s}.csv").open("w", encoding="utf-8") as f:
            f.write("image,label,is_ood,board_id,embedding\n")
            for r in rows:
                emb = r.get("embedding") or ""
                f.write(f"{r['image']},{r['label']},{int(bool(r['is_ood']))},{r['board_id']},{emb}\n")


# =========================
# MAIN
# =========================
def main():
    print("[BUILD] starting")

    cfg, cfg_path = load_config()
    print(f"[BUILD] using config: {cfg_path}")

    # reset outputs every run
    DATA_DIR.mkdir(exist_ok=True)
    reset_dir(RAW_TILES_DIR)
    reset_dir(SPLITS_DIR)

    # read tile params
    tile_size_list = cfg.get("tile_size", [224, 224])
    tile_size = (int(tile_size_list[0]), int(tile_size_list[1]))
    overlap = float(cfg.get("tile_overlap", 0.7))
    zero_padding = bool(cfg.get("zero_padding", True))

    emb_dir_cfg = cfg.get("embedding_dir")
    emb_dir = (ROOT / emb_dir_cfg).resolve() if emb_dir_cfg else None
    emb_ext = cfg.get("embedding_ext", ".npy")

    # discover and generate tiles from ROOT/data/*
    games = discover_games(DATA_DIR)
    generate_tiles(games, tile_size, overlap, zero_padding)

    # gather tiles and mark OOD based on data/hands filenames
    tiles = gather_tiles(emb_dir, emb_ext)

    # forced splits
    splits = split_tiles(tiles)

    by_path = {t["image"]: t for t in tiles}

    manifest = {
        "config": {
            "path_root": str(ROOT),
            "data_dir": str(DATA_DIR),
            "hands_dir": str(HANDS_DIR),
            "raw_tiles_dir": str(RAW_TILES_DIR),
            "forced_val": FORCED_VAL_GAME,
            "forced_test": FORCED_TEST_GAME,
            "note": "Tiles are generated with canonical game prefix (no _per_frame) to match hands naming.",
        },
        "classes": PIECE_LABELS,
        "splits": {"train": [], "val": [], "test": []},
    }

    for s, imgs in splits.items():
        for img in imgs:
            manifest["splits"][s].append(relativize(by_path[img]))

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_splits(manifest)

    print("[DONE]")
    print(" train:", len(manifest["splits"]["train"]))
    print(" val:  ", len(manifest["splits"]["val"]), f"(forced {FORCED_VAL_GAME})")
    print(" test: ", len(manifest["splits"]["test"]), f"(forced {FORCED_TEST_GAME})")
    print(f" manifest: {MANIFEST_PATH}")
    print(f" splits:   {SPLITS_DIR}")
    print(f" tiles:    {RAW_TILES_DIR}")


class ChessSquaresDataset(Dataset):
    """
    Torch Dataset for chess square tiles.
    Reads from data/dataset_manifest.json.
    """

    def __init__(self, manifest_path: Path, split: str = "train", transform=None, use_embeddings: bool = False):
        self.manifest_path = Path(manifest_path)
        self.data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if split not in self.data["splits"]:
            raise ValueError(f"Split {split} not found in manifest")
        self.samples = self.data["splits"][split]
        self.transform = transform
        self.use_embeddings = use_embeddings
        self.path_root = Path(self.data["config"].get("path_root", ".")).expanduser().resolve()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = self.path_root / sample["image"]
        label = int(sample["label"])
        is_ood = bool(sample.get("is_ood", False))
        board_id = sample["board_id"]

        embedding_path = sample.get("embedding")
        if embedding_path:
            embedding_path = self.path_root / embedding_path

        if self.use_embeddings and embedding_path and Path(embedding_path).exists():
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
            "is_ood": is_ood,
            "board_id": board_id,
            "path": str(path),
        }


if __name__ == "__main__":
    main()
