#!/usr/bin/env python3
"""dataset_stats_for_docs.py

Docs-first dataset statistics generator for CSC-BSR.

Outputs:
  1) PNG figures (BoardState dark styling)
  2) A Markdown snippet (Just-the-Docs callouts + tables) you can include
     in preprocessing.md
  3) stats.json (machine-readable)

Typical usage:
  python dataset_stats_for_docs.py \
    --splits-dir data/splits \
    --root . \
    --out-dir docs/assets/preprocessing_stats \
    --md-out docs/_includes/preprocessing_stats.md \
    --config preprocessing/dataset_config.yaml

Then in docs/preprocessing.md add:
  {% include preprocessing_stats.md %}

Notes:
- This script is careful about "empty" (class 0) dominating plots.
- It checks split hygiene (board/game overlap), tile completeness (expect 64 tiles per board),
  missing files, and embedding coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional imports (plots are optional but recommended)
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# -----------------------------
# Domain constants
# -----------------------------

CLASS_MAP: Dict[int, str] = {
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
}

# BoardState colors (dark)
COLORS = {
    "bg": "#0F1115",
    "surface": "#151A22",
    "surface2": "#1B2230",
    "text": "#E8EDF7",
    "muted": "#A9B4C7",
    "border": "#2A3446",
    "accent": "#7C5CFF",
    "success": "#2ECC71",
    "warn": "#F2C94C",
    "info": "#56CCF2",
    "danger": "#EB5757",
    "purple2": "#BB6BD9",
}

DEFAULT_EXPECTED_TILES_PER_BOARD = 64


# -----------------------------
# Helpers
# -----------------------------


def _safe_pct(n: float, d: float) -> float:
    return (100.0 * n / d) if d else 0.0


def parse_game_id(board_id: str) -> str:
    """Infer game id from board_id.

    Works well for ids like: game5_frame_001908, Game_01_frame_000123, etc.
    Falls back to the prefix before the first underscore.
    """
    if not isinstance(board_id, str):
        return "unknown"

    # Most common format
    if "_frame_" in board_id:
        return board_id.split("_frame_")[0]

    # Other common separators
    m = re.match(r"^(.*?)(?:_frame|_img|_image|_move|_turn|_t\d+)_", board_id)
    if m:
        return m.group(1)

    # Fallback
    return board_id.split("_")[0] if "_" in board_id else board_id


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JSD(P||Q) in bits (0..1 for two distributions).

    Uses base-2 logarithms.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * (np.log2(a) - np.log2(b))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def setup_boardstate_matplotlib(style_path: Optional[Path] = None) -> None:
    """Apply BoardState styling.

    If style_path exists, uses it. Otherwise sets rcParams directly.
    """
    if plt is None:
        return

    if style_path and style_path.exists():
        try:
            plt.style.use(str(style_path))
            return
        except Exception as e:
            print(f"Warning: Could not load style file {style_path}: {e}", file=sys.stderr)

    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["surface"],
            "savefig.facecolor": COLORS["bg"],
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.edgecolor": COLORS["border"],
            "grid.color": COLORS["border"],
            "grid.alpha": 0.6,
            "axes.grid": True,
            "axes.axisbelow": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
            "legend.frameon": True,
            "legend.facecolor": COLORS["surface"],
            "legend.edgecolor": COLORS["border"],
            "legend.framealpha": 0.9,
        }
    )


def boardstate_axes(ax):
    if ax is None:
        return ax
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


@dataclass
class SplitStats:
    name: str
    tiles: int
    boards: int
    games: int
    ood_tiles: int
    nonempty_tiles: int
    tiles_per_board_min: int
    tiles_per_board_median: float
    tiles_per_board_mean: float
    tiles_per_board_max: int
    boards_complete_pct: float
    missing_images: int
    embedding_present: int


@dataclass
class DataloaderStats:
    """Statistics about dataloader sampling and weighting."""
    split_name: str
    batch_size: int
    num_samples: int
    class_weights: Dict[int, float]
    weighted_distribution: Dict[int, float]
    effective_class_balance: Dict[int, float]  # After weighting


@dataclass
class GlobalStats:
    total_tiles: int
    total_ood_tiles: int
    total_boards: int
    total_games: int
    overall_class_counts: Dict[int, int]
    per_split: Dict[str, SplitStats]
    split_hygiene: Dict[str, int]
    jsd_train_test: Optional[float]
    config: Dict[str, object]
    dataloader_stats: Optional[Dict[str, DataloaderStats]] = None


def read_yaml_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    if yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_split_csv(csv_path: Path, root: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize types
    df["label"] = df["label"].astype(int)
    df["board_id"] = df["board_id"].astype(str)

    # Normalize OOD flag (now separate from classes)
    if "is_ood" not in df.columns:
        df["is_ood"] = 0
    df["is_ood"] = df["is_ood"].fillna(0).astype(int)

    # Normalize embedding presence: empty string should count as missing
    if "embedding" not in df.columns:
        df["embedding"] = ""
    else:
        df["embedding"] = df["embedding"].fillna("")

    # Absolute paths (for file existence checks)
    df["image_abs"] = df["image"].apply(lambda p: str((root / str(p)).resolve()))
    df["game_id"] = df["board_id"].apply(parse_game_id)

    return df


def compute_split_stats(name: str, df: pd.DataFrame, root: Path, expected_tiles_per_board: int) -> SplitStats:
    tiles = int(len(df))
    boards = int(df["board_id"].nunique())
    games = int(df["game_id"].nunique())

    ood_mask = df["is_ood"].astype(bool) if "is_ood" in df.columns else pd.Series(False, index=df.index)
    ood_tiles = int(ood_mask.sum())
    nonempty_tiles = int(((df["label"] != 0) & (~ood_mask)).sum())

    # tiles per board
    tpb = df.groupby("board_id").size().astype(int)
    tiles_per_board_min = int(tpb.min()) if len(tpb) else 0
    tiles_per_board_max = int(tpb.max()) if len(tpb) else 0
    tiles_per_board_mean = float(tpb.mean()) if len(tpb) else 0.0
    tiles_per_board_median = float(tpb.median()) if len(tpb) else 0.0
    boards_complete_pct = _safe_pct(int((tpb == expected_tiles_per_board).sum()), int(len(tpb)))

    # file existence checks (cheap)
    missing_images = int((~df["image_abs"].apply(lambda p: Path(p).exists())).sum())

    # embedding presence: non-empty path AND file exists
    def _emb_ok(x: str) -> bool:
        if not isinstance(x, str) or not x:
            return False
        return (root / x).exists()

    embedding_present = int(df["embedding"].apply(_emb_ok).sum())

    return SplitStats(
        name=name,
        tiles=tiles,
        boards=boards,
        games=games,
        ood_tiles=ood_tiles,
        nonempty_tiles=nonempty_tiles,
        tiles_per_board_min=tiles_per_board_min,
        tiles_per_board_median=tiles_per_board_median,
        tiles_per_board_mean=tiles_per_board_mean,
        tiles_per_board_max=tiles_per_board_max,
        boards_complete_pct=boards_complete_pct,
        missing_images=missing_images,
        embedding_present=embedding_present,
    )


def class_counts(df: pd.DataFrame) -> Dict[int, int]:
    counts = {cid: 0 for cid in CLASS_MAP.keys()}
    if "is_ood" in df.columns:
        df = df[df["is_ood"] == 0]
    vc = df["label"].value_counts().to_dict()
    for cid in counts.keys():
        counts[cid] = int(vc.get(cid, 0))
    return counts


def compute_global_stats(
    dfs: Dict[str, pd.DataFrame],
    root: Path,
    expected_tiles_per_board: int,
    config: Dict[str, object],
) -> GlobalStats:
    # Split stats
    per_split: Dict[str, SplitStats] = {}
    for name, df in dfs.items():
        per_split[name] = compute_split_stats(name, df, root, expected_tiles_per_board)

    # Totals
    all_df = pd.concat(list(dfs.values()), ignore_index=True)
    total_tiles = int(len(all_df))
    total_boards = int(all_df["board_id"].nunique())
    total_games = int(all_df["game_id"].nunique())
    total_ood_tiles = int(all_df["is_ood"].sum()) if "is_ood" in all_df.columns else 0
    overall_counts = class_counts(all_df)
    # Split hygiene
    train = dfs.get("train")
    val = dfs.get("val")
    test = dfs.get("test")

    def _overlap(a: Optional[pd.DataFrame], b: Optional[pd.DataFrame], col: str) -> int:
        if a is None or b is None:
            return 0
        return int(len(set(a[col].unique()) & set(b[col].unique())))

    split_hygiene = {
        "board_overlap_train_val": _overlap(train, val, "board_id"),
        "board_overlap_train_test": _overlap(train, test, "board_id"),
        "board_overlap_val_test": _overlap(val, test, "board_id"),
        "game_overlap_train_val": _overlap(train, val, "game_id"),
        "game_overlap_train_test": _overlap(train, test, "game_id"),
        "game_overlap_val_test": _overlap(val, test, "game_id"),
    }

    # Distribution distance (train vs test)
    jsd = None
    if train is not None and test is not None:
        # Compare class distribution excluding empty (0) to get a more meaningful distance
        train_id = train[train["is_ood"] == 0] if "is_ood" in train.columns else train
        test_id = test[test["is_ood"] == 0] if "is_ood" in test.columns else test
        train_counts = np.array([class_counts(train_id)[cid] for cid in CLASS_MAP.keys() if cid != 0], dtype=float)
        test_counts = np.array([class_counts(test_id)[cid] for cid in CLASS_MAP.keys() if cid != 0], dtype=float)
        jsd = jensen_shannon_divergence(train_counts, test_counts)

    # Dataloader analysis (if train split exists)
    dataloader_stats = None
    if train is not None:
        dataloader_stats = compute_dataloader_stats(train)

    return GlobalStats(
        total_tiles=total_tiles,
        total_ood_tiles=total_ood_tiles,
        total_boards=total_boards,
        total_games=total_games,
        overall_class_counts=overall_counts,
        per_split=per_split,
        split_hygiene=split_hygiene,
        jsd_train_test=jsd,
        config=config,
        dataloader_stats=dataloader_stats,
    )


def compute_dataloader_stats(train_df: pd.DataFrame) -> Dict[str, DataloaderStats]:
    """
    Analyze class weighting and effective sampling distribution.
    
    Mimics the behavior of get_train_dataloader from load_dataset.py
    with WeightedRandomSampler.
    """
    results = {}

    if "is_ood" in train_df.columns:
        train_df = train_df[train_df["is_ood"] == 0]
    if len(train_df) == 0:
        return results
    
    # Compute class weights (inverse of class counts)
    class_counts_dict = class_counts(train_df)
    total_samples = len(train_df)
    
    class_weights = {}
    for class_id, count in class_counts_dict.items():
        if count > 0:
            class_weights[class_id] = 1.0 / count
        else:
            class_weights[class_id] = 0.0
    
    # Compute per-sample weights and aggregate
    sample_weights_list = [class_weights[label] for label in train_df["label"].values]
    total_weight = sum(sample_weights_list)
    
    # Compute effective distribution after weighting
    weighted_dist = {}
    
    for class_id, count in class_counts_dict.items():
        weight_sum = sum(w for label, w in zip(train_df["label"].values, sample_weights_list) 
                        if label == class_id)
        weighted_dist[class_id] = (weight_sum / total_weight * 100) if total_weight > 0 else 0.0
    
    # Effective class balance (how many samples each class gets in expectation)
    effective_balance = {}
    for class_id in class_counts_dict.keys():
        # In WeightedRandomSampler with replacement, probability is proportional to weight
        effective_balance[class_id] = weighted_dist[class_id]
    
    results["train_weighted"] = DataloaderStats(
        split_name="train_weighted",
        batch_size=32,  # Default from get_train_dataloader
        num_samples=total_samples,
        class_weights=class_weights,
        weighted_distribution=weighted_dist,
        effective_class_balance=effective_balance,
    )
    
    return results


# -----------------------------
# Plotting
# -----------------------------


def _save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    facecolor = plt.rcParams.get("savefig.facecolor", COLORS["bg"])
    fig.savefig(out_path, dpi=150, facecolor=facecolor, bbox_inches="tight")
    plt.close(fig)


def plot_overview(dfs: Dict[str, pd.DataFrame], stats: GlobalStats, out_dir: Path) -> Optional[Path]:
    if plt is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    splits = [s for s in ["train", "val", "test"] if s in stats.per_split]
    split_colors = {
        "train": COLORS["accent"],
        "val": COLORS["warn"],
        "test": COLORS["danger"],
    }

    # 1) Tiles per split
    ax = axes[0, 0]
    vals = [stats.per_split[s].tiles for s in splits]
    bars = ax.bar(splits, vals, color=[split_colors[s] for s in splits])
    ax.set_title("Tiles per split")
    ax.set_ylabel("# tiles")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height()):,}", ha="center", va="bottom")
    boardstate_axes(ax)

    # 2) OOD share
    ax = axes[0, 1]
    ood_pct = [
        _safe_pct(stats.per_split[s].ood_tiles, stats.per_split[s].tiles)
        for s in splits
    ]
    bars = ax.bar(splits, ood_pct, color=COLORS["danger"])
    ax.set_title("OOD share")
    ax.set_ylabel("% of tiles")
    ax.set_ylim(0, max(5.0, max(ood_pct) * 1.25 if ood_pct else 5.0))
    for b, p in zip(bars, ood_pct):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{p:.2f}%", ha="center", va="bottom")
    boardstate_axes(ax)

    # 3) Tiles-per-board histogram
    ax = axes[1, 0]
    all_df = pd.concat(list(dfs.values()), ignore_index=True)
    tpb = all_df.groupby("board_id").size().astype(int)
    ax.hist(tpb.values, bins=range(0, max(int(tpb.max()), 64) + 2), color=COLORS["info"], edgecolor=COLORS["border"])
    ax.set_title("Tiles per board (sanity check)")
    ax.set_xlabel("tiles per board_id")
    ax.set_ylabel("# boards")
    boardstate_axes(ax)

    # 4) Piece distribution excluding empty
    ax = axes[1, 1]
    counts = stats.overall_class_counts
    piece_ids = [cid for cid in CLASS_MAP.keys() if cid not in (0,)]
    piece_counts = np.array([counts[cid] for cid in piece_ids], dtype=float)

    labels = [CLASS_MAP[cid].replace("_", " ") for cid in piece_ids]
    order = np.argsort(piece_counts)[::-1]
    top_k = min(8, len(order))
    order = order[:top_k]

    ax.barh([labels[i] for i in order][::-1], piece_counts[order][::-1], color=COLORS["accent"])
    ax.set_title("Top classes (excluding empty)")
    ax.set_xlabel("# tiles")
    boardstate_axes(ax)

    out_path = out_dir / "overview.png"
    _save_fig(fig, out_path)
    return out_path


def plot_class_heatmap(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> Optional[Path]:
    if plt is None:
        return None

    splits = [s for s in ["train", "val", "test"] if s in dfs]
    class_ids = list(CLASS_MAP.keys())

    # percentages per split
    mat = []
    for s in splits:
        c = class_counts(dfs[s])
        tot = sum(c.values())
        mat.append([_safe_pct(c[cid], tot) for cid in class_ids])
    mat = np.array(mat, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.imshow(mat.T, aspect="auto")

    ax.set_title("Class distribution per split (%)")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits)
    ax.set_yticks(range(len(class_ids)))
    ax.set_yticklabels([CLASS_MAP[cid] for cid in class_ids])

    # annotate only non-trivial values
    for i in range(len(class_ids)):
        for j in range(len(splits)):
            v = mat[j, i]
            if v >= 1.0 or class_ids[i] == 0:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)

    boardstate_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("%")

    out_path = out_dir / "class_heatmap.png"
    _save_fig(fig, out_path)
    return out_path


def plot_games(dfs: Dict[str, pd.DataFrame], out_dir: Path) -> Optional[Path]:
    if plt is None:
        return None

    all_df = pd.concat(list(dfs.values()), ignore_index=True)

    # boards per game
    boards_per_game = all_df.groupby("game_id")["board_id"].nunique().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    top = boards_per_game.head(15)
    ax.barh(top.index[::-1], top.values[::-1], color=COLORS["purple2"])
    ax.set_title("Top games by #frames (board_ids)")
    ax.set_xlabel("# frames")
    boardstate_axes(ax)

    out_path = out_dir / "games_top15.png"
    _save_fig(fig, out_path)
    return out_path


def plot_dataloader_weights(
    dataloader_stats: Optional[Dict[str, DataloaderStats]], out_dir: Path
) -> Optional[Path]:
    """Plot class weights and effective distribution after weighting."""
    if plt is None or dataloader_stats is None:
        return None

    # Use train_weighted if available
    if "train_weighted" not in dataloader_stats:
        return None

    dl_stats = dataloader_stats["train_weighted"]
    
    class_ids = sorted(dl_stats.class_weights.keys())
    weights = [dl_stats.class_weights[cid] for cid in class_ids]
    effective = [dl_stats.effective_class_balance[cid] for cid in class_ids]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    # 1. Class weights
    ax = axes[0]
    labels = [CLASS_MAP.get(cid, f"class_{cid}") for cid in class_ids]
    colors = [COLORS["accent"] if w > 0 else COLORS["border"] for w in weights]
    ax.bar(range(len(class_ids)), weights, color=colors)
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
    ax.set_title("Class Weights (1/count)")
    ax.set_ylabel("Weight")
    boardstate_axes(ax)
    
    # 2. Effective sampling distribution
    ax = axes[1]
    colors = [COLORS["success"] if e > 0 else COLORS["border"] for e in effective]
    ax.bar(range(len(class_ids)), effective, color=colors)
    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels([f"{cid}" for cid in class_ids], rotation=45)
    ax.set_title("Effective Distribution after Weighting")
    ax.set_ylabel("% of samples")
    boardstate_axes(ax)
    
    out_path = out_dir / "dataloader_weights.png"
    _save_fig(fig, out_path)
    return out_path


# -----------------------------
# Markdown generation
# -----------------------------


def relpath_for_docs(asset_path: Path, md_out: Path) -> str:
    """Return a forward-slash relative path from md_out parent to asset_path."""
    try:
        rel = os.path.relpath(asset_path, md_out.parent)
    except Exception:
        rel = str(asset_path)
    return rel.replace("\\", "/")


def format_int(n: int) -> str:
    return f"{n:,}"


def write_markdown(
    md_out: Path,
    stats: GlobalStats,
    asset_paths: Dict[str, Path],
    assets_url_prefix: str | None = None,
) -> None:
    md_out.parent.mkdir(parents=True, exist_ok=True)

    # Convenience
    s_train = stats.per_split.get("train")
    s_val = stats.per_split.get("val")
    s_test = stats.per_split.get("test")

    # Derive a few headline numbers
    empty = stats.overall_class_counts.get(0, 0)
    ood = stats.total_ood_tiles
    nonempty = stats.total_tiles - empty - ood

    jsd_line = ""
    if stats.jsd_train_test is not None:
        jsd_line = f"Train↔Test class distribution shift (JSD, excluding empty): **{stats.jsd_train_test:.3f} bits**."

    # Split hygiene line
    hygiene = stats.split_hygiene
    hygiene_ok = all(v == 0 for v in hygiene.values())

    cfg = stats.config
    tile_size = cfg.get("tile_size")
    tile_overlap = cfg.get("tile_overlap")
    zero_padding = cfg.get("zero_padding")
    seed = cfg.get("seed")

    lines: List[str] = []
    lines.append("## Dataset statistics")
    lines.append("")

    # Config block
    lines.append("{: .repro }")
    cfg_bits = []
    if tile_size is not None:
        cfg_bits.append(f"tile_size={tile_size}")
    if tile_overlap is not None:
        cfg_bits.append(f"tile_overlap={tile_overlap}")
    if zero_padding is not None:
        cfg_bits.append(f"zero_padding={zero_padding}")
    if seed is not None:
        cfg_bits.append(f"seed={seed}")
    if cfg_bits:
        lines.append("**Config:** " + ", ".join(cfg_bits))
    else:
        lines.append("**Config:** (not provided to script)")
    lines.append("")

    # Headline summary
    lines.append("{: .result }")
    lines.append(
        f"**Size:** {format_int(stats.total_tiles)} tiles from {format_int(stats.total_boards)} board frames across {format_int(stats.total_games)} games. "
        f"Non-empty (in-distribution) tiles: **{_safe_pct(nonempty, stats.total_tiles):.1f}%**; OOD tiles: **{_safe_pct(ood, stats.total_tiles):.2f}%**."  # noqa
    )
    if jsd_line:
        lines.append(jsd_line)
    lines.append("")

    # Hygiene
    if hygiene_ok:
        lines.append("{: .result }")
        lines.append("**Split hygiene:** no board_id / game_id overlap between train/val/test (✅).")
    else:
        lines.append("{: .warning }")
        lines.append("**Split hygiene issue detected:** overlaps exist between splits (❌).")
        lines.append("")
        lines.append("| overlap | count |")
        lines.append("|---|---:|")
        for k, v in hygiene.items():
            lines.append(f"| {k} | {v} |")
    lines.append("")

    # Per split table
    lines.append("### Split breakdown")
    lines.append("")
    lines.append("| split | tiles | boards | games | non-empty tiles | OOD tiles | boards complete (64 tiles) | missing images | embeddings present |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for s in [s_train, s_val, s_test]:
        if s is None:
            continue
        lines.append(
            "| {name} | {tiles} | {boards} | {games} | {nonempty} ({nonempty_pct:.1f}%) | {ood} ({ood_pct:.2f}%) | {complete:.1f}% | {missing} | {emb} ({emb_pct:.1f}%) |".format(
                name=s.name,
                tiles=format_int(s.tiles),
                boards=format_int(s.boards),
                games=format_int(s.games),
                nonempty=format_int(s.nonempty_tiles),
                nonempty_pct=_safe_pct(s.nonempty_tiles, s.tiles),
                ood=format_int(s.ood_tiles),
                ood_pct=_safe_pct(s.ood_tiles, s.tiles),
                complete=s.boards_complete_pct,
                missing=format_int(s.missing_images),
                emb=format_int(s.embedding_present),
                emb_pct=_safe_pct(s.embedding_present, s.tiles),
            )
        )

    lines.append("")

    # Class distribution highlights
    lines.append("### Class balance (high level)")
    lines.append("")
    lines.append("{: .decision }")
    lines.append(
        f"The dataset is **empty-dominant**: class `empty` is {format_int(empty)} tiles (**{_safe_pct(empty, stats.total_tiles):.1f}%**). "
        f"If you train a classifier directly on raw tiles, use class balancing (sampler / loss weights)."
    )
    lines.append("")

    # Top + rare classes
    counts = stats.overall_class_counts
    # exclude empty for ranking
    ranked = sorted([cid for cid in CLASS_MAP.keys() if cid != 0], key=lambda c: counts.get(c, 0), reverse=True)
    top = ranked[:8]
    rare = [cid for cid in ranked if _safe_pct(counts.get(cid, 0), stats.total_tiles) < 0.75]

    lines.append("**Top classes (excluding empty):** " + ", ".join([f"{CLASS_MAP[c]} ({format_int(counts[c])})" for c in top]))
    if rare:
        lines.append("**Rare (<0.75%)**: " + ", ".join([f"{CLASS_MAP[c]} ({format_int(counts[c])})" for c in rare]))
    lines.append("")

    # Dataloader section
    if stats.dataloader_stats and "train_weighted" in stats.dataloader_stats:
        lines.append("### Training dataloader (WeightedRandomSampler)")
        lines.append("")
        dl = stats.dataloader_stats["train_weighted"]
        lines.append(f"**Class weights:** Inverse class frequency (rarer classes get higher weight)")
        lines.append("")
        lines.append("| class | weight | effective % |")
        lines.append("|---|---:|---:|")
        for cid in sorted(dl.class_weights.keys()):
            w = dl.class_weights[cid]
            eff = dl.effective_class_balance[cid]
            lines.append(f"| {CLASS_MAP[cid]} | {w:.4f} | {eff:.2f}% |")
        lines.append("")
        lines.append("{: .info }")
        lines.append(f"The training dataloader uses `WeightedRandomSampler` to balance class representation during training, giving rarer classes more frequent sampling.")
        lines.append("")

    # Figures
    if asset_paths:
        lines.append("### Visual summary")
        lines.append("")
        for title, p in asset_paths.items():
            if assets_url_prefix:
                rel = f"{assets_url_prefix.rstrip('/')}/{p.name}"
            else:
                rel = relpath_for_docs(p, md_out)
            lines.append(f"**{title}**")
            lines.append("")
            lines.append(f"![]({rel})")
            lines.append("")

    md_out.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate dataset stats + markdown snippet for docs")
    ap.add_argument("--splits-dir", type=Path, default=Path("data/splits"), help="Directory with train.csv/val.csv/test.csv")
    ap.add_argument("--root", type=Path, default=Path("."), help="Project root for relative image/embedding paths")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis_results_for_docs"), help="Where to write PNGs")
    ap.add_argument("--md-out", type=Path, default=Path("preprocessing_stats.md"), help="Markdown snippet output")
    ap.add_argument(
        "--assets-url-prefix",
        type=str,
        default=None,
        help=(
            "If set, Markdown will reference figures as '<prefix>/<filename>'. "
            "Example for GitHub Pages: 'assets/preprocessing_stats'"
        ),
    )
    ap.add_argument("--json-out", type=Path, default=None, help="Optional JSON stats output (default: out-dir/stats.json)")
    ap.add_argument("--config", type=Path, default=None, help="Optional dataset_config.yaml to document tile_size/tile_overlap/seed")
    ap.add_argument("--style", type=Path, default=None, help="Optional matplotlib style")
    ap.add_argument("--expected-tiles-per-board", type=int, default=DEFAULT_EXPECTED_TILES_PER_BOARD)
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = ap.parse_args()

    # Load splits
    root = args.root.expanduser().resolve()
    
    # Resolve style path - if provided, make it absolute; otherwise use default
    if args.style is None:
        args.style = Path(__file__).parent / "styles" / "boardstate-dark.mplstyle"
    else:
        args.style = args.style.expanduser().resolve()
    splits_dir = args.splits_dir.expanduser().resolve()

    dfs: Dict[str, pd.DataFrame] = {}
    for name in ["train", "val", "test"]:
        p = splits_dir / f"{name}.csv"
        if p.exists():
            dfs[name] = load_split_csv(p, root)

    if not dfs:
        raise SystemExit(f"No split CSVs found in {splits_dir} (expected train.csv/val.csv/test.csv)")

    # Load config (optional)
    config: Dict[str, object] = {}
    if args.config is not None:
        config = read_yaml_config(args.config)

        # Normalize keys to match manifest naming
        # dataset_config.example.yaml uses tile_size/tile_overlap/zero_padding/seed.
        # We'll keep only what we need.
        keep = {"tile_size", "tile_overlap", "zero_padding", "seed"}
        config = {k: config.get(k) for k in keep if k in config}

    # Stats
    stats = compute_global_stats(dfs, root, args.expected_tiles_per_board, config)

    # Save JSON
    json_out = args.json_out or (args.out_dir / "stats.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(obj):
        if isinstance(obj, SplitStats):
            return obj.__dict__
        if isinstance(obj, GlobalStats):
            return {
                "total_tiles": obj.total_tiles,
                "total_ood_tiles": obj.total_ood_tiles,
                "total_boards": obj.total_boards,
                "total_games": obj.total_games,
                "overall_class_counts": obj.overall_class_counts,
                "per_split": {k: _to_dict(v) for k, v in obj.per_split.items()},
                "split_hygiene": obj.split_hygiene,
                "jsd_train_test": obj.jsd_train_test,
                "config": obj.config,
            }
        return obj

    json_out.write_text(json.dumps(_to_dict(stats), indent=2), encoding="utf-8")

    # Plots
    asset_paths: Dict[str, Path] = {}
    if not args.no_plots and plt is not None:
        setup_boardstate_matplotlib(args.style)
        args.out_dir.mkdir(parents=True, exist_ok=True)

        p1 = plot_overview(dfs, stats, args.out_dir)
        if p1:
            asset_paths["Overview"] = p1

        p2 = plot_class_heatmap(dfs, args.out_dir)
        if p2:
            asset_paths["Class heatmap"] = p2

        p3 = plot_games(dfs, args.out_dir)
        if p3:
            asset_paths["Top games by frames"] = p3

        p4 = plot_dataloader_weights(stats.dataloader_stats, args.out_dir)
        if p4:
            asset_paths["Dataloader weights"] = p4

    # Markdown snippet
    write_markdown(args.md_out, stats, asset_paths, assets_url_prefix=args.assets_url_prefix)

    print(f"✓ Wrote markdown: {args.md_out}")
    print(f"✓ Wrote stats JSON: {json_out}")
    if asset_paths:
        print(f"✓ Wrote {len(asset_paths)} figure(s) to: {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
