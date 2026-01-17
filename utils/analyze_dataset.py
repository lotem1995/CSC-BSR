"""
Dataset Analysis Script

Analyzes the chess tile dataset created by build_dataset.py.
Generates comprehensive visualizations following the boardstate design language.

Features:
  - Class distribution analysis (overall, per split)
  - Dataset size and statistics
  - Image sample visualization
  - Board ID frequency analysis
  - Split composition analysis
  - Embedding coverage (if available)
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Dataset configuration
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
    17: "OOD"
}

# Design language colors (from boardstate-dark.mplstyle)
COLORS = {
    "background": "#0F1115",
    "sidebar": "#151A22",
    "text": "#E8EDF7",
    "heading": "#F7F9FF",
    "border": "#2A3446",
    "accent": "#7C5CFF",
    "accent_dark": "#5B46D6",
    "success": "#2ECC71",
    "warning": "#F2C94C",
    "danger": "#EB5757",
}

# Class-specific colors (piece colors)
CLASS_COLORS = {
    0: "#808080",      # empty - gray
    1: "#E8E8E8",      # white_pawn - light
    2: "#E8E8E8",      # white_knight
    3: "#E8E8E8",      # white_bishop
    4: "#E8E8E8",      # white_rook
    5: "#E8E8E8",      # white_queen
    6: "#E8E8E8",      # white_king
    11: "#303030",     # black_pawn - dark
    12: "#303030",     # black_knight
    13: "#303030",     # black_bishop
    14: "#303030",     # black_rook
    15: "#303030",     # black_queen
    16: "#303030",     # black_king
    17: "#EB5757",     # OOD - red/danger
}


def setup_matplotlib_style():
    """Load boardstate design language style."""
    plt.style.use("styles/boardstate-dark.mplstyle")
    plt.rcParams["figure.figsize"] = (7.2, 4.0)   # ~16:9-ish
    plt.rcParams["figure.constrained_layout.use"] = True


def boardstate_axes(ax):
    """Apply boardstate styling to axes (hide top/right spines)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def load_dataset(csv_path: Path, root: Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(csv_path)
    return df


def load_manifest(manifest_path: Path) -> Dict:
    """Load manifest JSON."""
    with manifest_path.open("r") as f:
        return json.load(f)


def analyze_class_distribution(
    dfs: Dict[str, pd.DataFrame]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Analyze class distribution across splits.
    
    Returns:
        tuple: (global_counts, split_counts_dict)
    """
    num_classes = max(CLASS_MAP.keys()) + 1  # 0-17 requires size 18
    global_counts = np.zeros(num_classes)
    split_counts = {}
    
    for split_name, df in dfs.items():
        counts = np.zeros(num_classes)
        for label in df["label"].values:
            label_int = int(label)
            if label_int in CLASS_MAP:
                counts[label_int] += 1
        split_counts[split_name] = counts
        global_counts += counts
    
    return global_counts, split_counts


def plot_class_distribution(
    output_dir: Path,
    global_counts: np.ndarray,
    split_counts: Dict[str, np.ndarray],
):
    """Plot class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall distribution
    ax = axes[0]
    classes = [CLASS_MAP.get(i, f"unknown_{i}") for i in range(len(global_counts))]
    colors = [CLASS_COLORS.get(i, "#808080") for i in range(len(global_counts))]
    
    bars = ax.bar(range(len(global_counts)), global_counts, color=colors, edgecolor=COLORS["border"])
    ax.set_xlabel("Class", color=COLORS["text"])
    ax.set_ylabel("Count", color=COLORS["text"])
    ax.set_title("Overall Class Distribution", color=COLORS["heading"])
    ax.set_xticks(range(len(global_counts)))
    ax.set_xticklabels([f"{i}" for i in range(len(global_counts))], rotation=45)
    boardstate_axes(ax)
    
    # Per-split distribution
    ax = axes[1]
    splits = list(split_counts.keys())
    x = np.arange(len(splits))
    width = 0.15
    
    for i in sorted(CLASS_MAP.keys())[:6]:  # Show top 6 classes
        class_counts = [split_counts[s][i] for s in splits]
        ax.bar(x + i * width, class_counts, width, label=CLASS_MAP[i], color=CLASS_COLORS[i])
    
    ax.set_xlabel("Split", color=COLORS["text"])
    ax.set_ylabel("Count", color=COLORS["text"])
    ax.set_title("Distribution per Split (Top 6 Classes)", color=COLORS["heading"])
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(splits)
    ax.legend(fontsize=8, loc="upper right")
    boardstate_axes(ax)
    
    plt.tight_layout()
    output_path = output_dir / "01_class_distribution.png"
    plt.savefig(output_path, dpi=150, facecolor=COLORS["background"])
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_split_composition(
    output_dir: Path,
    dfs: Dict[str, pd.DataFrame],
):
    """Plot split composition."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    split_sizes = {name: len(df) for name, df in dfs.items()}
    total = sum(split_sizes.values())
    
    # Create pie chart
    labels = [f"{name}\n({count:,} samples)" for name, count in split_sizes.items()]
    sizes = list(split_sizes.values())
    colors_pie = [COLORS["accent"], COLORS["warning"], COLORS["danger"]][:len(sizes)]
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
    )
    
    # Style text
    for text in texts:
        text.set_color(COLORS["text"])
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color(COLORS["background"])
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)
    
    ax.set_title("Dataset Split Composition", color=COLORS["heading"], fontsize=12, pad=20)
    
    plt.tight_layout()
    output_path = output_dir / "02_split_composition.png"
    plt.savefig(output_path, dpi=150, facecolor=COLORS["background"])
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_dataset_statistics(
    output_dir: Path,
    dfs: Dict[str, pd.DataFrame],
    root: Path,
):
    """Plot dataset statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Samples per split
    ax = axes[0, 0]
    splits = list(dfs.keys())
    counts = [len(dfs[s]) for s in splits]
    bars = ax.bar(splits, counts, color=[COLORS["accent"], COLORS["warning"], COLORS["danger"]][:len(splits)])
    ax.set_ylabel("Number of Samples", color=COLORS["text"])
    ax.set_title("Samples per Split", color=COLORS["heading"])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', color=COLORS["text"], fontsize=10)
    boardstate_axes(ax)
    
    # 2. OOD distribution
    ax = axes[0, 1]
    ood_counts = []
    for split_name in splits:
        ood_count = len(dfs[split_name][dfs[split_name]["label"] == 17])
        ood_counts.append(ood_count)
    
    bars = ax.bar(splits, ood_counts, color=COLORS["danger"])
    ax.set_ylabel("OOD Samples", color=COLORS["text"])
    ax.set_title("Out-of-Distribution Samples", color=COLORS["heading"])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', color=COLORS["text"], fontsize=10)
    boardstate_axes(ax)
    
    # 3. Board ID diversity
    ax = axes[1, 0]
    board_diversity = []
    labels_div = []
    for split_name in splits:
        unique_boards = dfs[split_name]["board_id"].nunique()
        board_diversity.append(unique_boards)
        labels_div.append(f"{split_name}\n({unique_boards} boards)")
    
    bars = ax.bar(splits, board_diversity, color=[COLORS["accent"], COLORS["warning"], COLORS["danger"]][:len(splits)])
    ax.set_ylabel("Unique Board IDs", color=COLORS["text"])
    ax.set_title("Board ID Diversity", color=COLORS["heading"])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', color=COLORS["text"], fontsize=10)
    boardstate_axes(ax)
    
    # 4. Embedding coverage
    ax = axes[1, 1]
    embedding_coverage = []
    for split_name in splits:
        df = dfs[split_name]
        with_emb = df["embedding"].notna().sum()
        total = len(df)
        coverage = (with_emb / total * 100) if total > 0 else 0
        embedding_coverage.append(coverage)
    
    bars = ax.bar(splits, embedding_coverage, color=[COLORS["success"], COLORS["warning"], COLORS["danger"]][:len(splits)])
    ax.set_ylabel("Coverage (%)", color=COLORS["text"])
    ax.set_title("Embedding Coverage", color=COLORS["heading"])
    ax.set_ylim([0, 105])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', color=COLORS["text"], fontsize=10)
    boardstate_axes(ax)
    
    plt.tight_layout()
    output_path = output_dir / "03_dataset_statistics.png"
    plt.savefig(output_path, dpi=150, facecolor=COLORS["background"])
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_image_samples(
    output_dir: Path,
    dfs: Dict[str, pd.DataFrame],
    root: Path,
    samples_per_class: int = 2,
):
    """Plot sample images from each class."""
    # Get one sample from each class
    fig = plt.figure(figsize=(16, 10))
    
    sample_count = 0
    for class_id in sorted(CLASS_MAP.keys()):
        # Find a sample from this class across all splits
        sample = None
        for split_name in dfs.keys():
            matches = dfs[split_name][dfs[split_name]["label"] == class_id]
            if len(matches) > 0:
                sample = matches.iloc[0]
                break
        
        if sample is None:
            continue
        
        sample_count += 1
        ax = plt.subplot(5, 4, sample_count)
        
        # Load and display image
        img_path = root / sample["image"]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\n{img_path.name}", 
                   ha='center', va='center', color=COLORS["danger"])
        
        # Style
        ax.axis("off")
        title = f"{CLASS_MAP[class_id]}\n(class {class_id})"
        ax.set_title(title, color=COLORS["text"], fontsize=9, pad=5)
        
        if sample_count >= 20:
            break
    
    fig.suptitle("Dataset Sample Images", fontsize=14, color=COLORS["heading"], y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / "04_sample_images.png"
    plt.savefig(output_path, dpi=100, facecolor=COLORS["background"], bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def plot_board_frequency(
    output_dir: Path,
    dfs: Dict[str, pd.DataFrame],
    top_n: int = 15,
):
    """Plot most frequent board IDs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate board IDs across all splits
    all_board_counts = {}
    for df in dfs.values():
        board_counts = df["board_id"].value_counts()
        for board_id, count in board_counts.items():
            all_board_counts[board_id] = all_board_counts.get(board_id, 0) + count
    
    # Get top N
    top_boards = sorted(all_board_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    boards, counts = zip(*top_boards)
    
    bars = ax.barh(range(len(boards)), counts, color=COLORS["accent"])
    ax.set_yticks(range(len(boards)))
    ax.set_yticklabels(boards, fontsize=9)
    ax.set_xlabel("Sample Count", color=COLORS["text"])
    ax.set_title(f"Top {top_n} Board IDs by Sample Count", color=COLORS["heading"])
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f" {int(count)}", va="center", color=COLORS["text"], fontsize=9)
    
    boardstate_axes(ax)
    plt.tight_layout()
    
    output_path = output_dir / "05_board_frequency.png"
    plt.savefig(output_path, dpi=150, facecolor=COLORS["background"])
    plt.close()
    print(f"✓ Saved: {output_path.name}")


def generate_summary_report(
    output_dir: Path,
    dfs: Dict[str, pd.DataFrame],
    global_counts: np.ndarray,
    root: Path,
):
    """Generate a text summary report."""
    report = []
    report.append("=" * 80)
    report.append("DATASET ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    total_samples = sum(len(df) for df in dfs.values())
    report.append(f"Total Samples: {total_samples:,}")
    report.append("")
    
    # Per-split statistics
    report.append("SPLIT STATISTICS:")
    report.append("-" * 80)
    for split_name, df in dfs.items():
        report.append(f"\n{split_name.upper()}:")
        report.append(f"  Samples: {len(df):,}")
        report.append(f"  Unique Boards: {df['board_id'].nunique()}")
        report.append(f"  OOD Samples: {len(df[df['label'] == 17])}")
        report.append(f"  Embedding Coverage: {df['embedding'].notna().sum() / len(df) * 100:.1f}%")
    
    report.append("")
    report.append("CLASS DISTRIBUTION:")
    report.append("-" * 80)
    for class_id in sorted(CLASS_MAP.keys()):
        count = global_counts[class_id]
        pct = count / total_samples * 100 if total_samples > 0 else 0
        report.append(f"{CLASS_MAP[class_id]:20s}: {int(count):6,} ({pct:5.2f}%)")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    report_path = output_dir / "dataset_analysis_report.txt"
    with report_path.open("w") as f:
        f.write(report_text)
    print(f"✓ Saved: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize chess tile dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing split CSVs (train.csv, val.csv, test.csv)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory for image paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip sample image visualization (faster)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest.json for additional analysis",
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_matplotlib_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Analyzing dataset from {args.data_dir}")
    print(f"📁 Saving results to {args.output_dir}")
    print()
    
    # Load data
    dfs = {}
    for split_file in ["train.csv", "val.csv", "test.csv"]:
        split_path = args.data_dir / split_file
        if split_path.exists():
            split_name = split_file.replace(".csv", "")
            dfs[split_name] = load_dataset(split_path, args.root)
            print(f"✓ Loaded {split_name}: {len(dfs[split_name])} samples")
    
    if not dfs:
        print(f"❌ No CSV files found in {args.data_dir}")
        return
    
    print()
    
    # Analysis
    global_counts, split_counts = analyze_class_distribution(dfs)
    
    # Plotting
    print("📈 Generating visualizations...")
    plot_class_distribution(args.output_dir, global_counts, split_counts)
    plot_split_composition(args.output_dir, dfs)
    plot_dataset_statistics(args.output_dir, dfs, args.root)
    
    if not args.no_images:
        plot_image_samples(args.output_dir, dfs, args.root)
    
    plot_board_frequency(args.output_dir, dfs)
    
    # Report
    print()
    print("📝 Generating report...")
    generate_summary_report(args.output_dir, dfs, global_counts, args.root)
    
    print()
    print(f"✅ Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
