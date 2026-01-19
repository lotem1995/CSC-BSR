#!/usr/bin/env python3
"""finetuning_stats_for_docs.py

Fine-tuning statistics and visualization generator for CSC-BSR.

Outputs:
  1) PNG figures (BoardState dark styling) comparing fine-tuned models
  2) A Markdown snippet (Just-the-Docs callouts + tables) for docs
  3) finetuning_stats.json (machine-readable summary)

Typical usage:
  python utils/finetuning_stats_for_docs.py \
    --metrics-dir embedding \
    --out-dir docs/assets/finetuning_stats \
    --md-out docs/_includes/finetuning_stats.md \
    --style-path utils/styles/boardstate-dark.mplstyle

Then in docs/*.md add:
  {% include finetuning_stats.md %}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Optional imports (plots are optional but recommended)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:
    plt = None

# BoardState colors (dark theme)
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

# Model display names and colors
MODEL_DISPLAY = {
    "dino-small_backbone": {"name": "DinoV2 (Backbone)", "color": COLORS["accent"]},
    "dino-small_head-only": {"name": "DinoV2 (Head Only)", "color": COLORS["info"]},
    "qwen_head-only": {"name": "Qwen2-VL (Head Only)", "color": COLORS["warn"]},
    "qwen_lora": {"name": "Qwen2-VL (LoRA)", "color": COLORS["success"]},
}


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class ModelMetrics:
    """Metrics for a single fine-tuned model."""

    model_key: str
    embedding_model: str
    strategy: str
    epochs: int
    batch_size: int
    train_loss: List[float]
    val_loss: List[float]
    val_balanced_accuracy: List[float]
    val_f1_score: List[float]

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss) if self.val_loss else float("inf")

    @property
    def best_balanced_accuracy(self) -> float:
        return max(self.val_balanced_accuracy) if self.val_balanced_accuracy else 0.0

    @property
    def best_f1_score(self) -> float:
        return max(self.val_f1_score) if self.val_f1_score else 0.0

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY.get(self.model_key, {}).get("name", self.model_key)

    @property
    def color(self) -> str:
        return MODEL_DISPLAY.get(self.model_key, {}).get("color", COLORS["muted"])


@dataclass
class GlobalStats:
    """Overall statistics across all fine-tuned models."""

    models: Dict[str, ModelMetrics]
    best_model_by_accuracy: Optional[str] = None
    best_model_by_f1: Optional[str] = None
    best_model_by_val_loss: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------


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
        except Exception:
            pass

    # Fallback to manual styling
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
    """Style axes with BoardState theme."""
    if ax is None:
        return None
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def _save_fig(fig, out_path: Path):
    """Save figure with tight layout."""
    if fig is None or plt is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Data loading
# -----------------------------


def load_metrics_json(json_path: Path) -> Optional[ModelMetrics]:
    """Load metrics from a JSON file."""
    if not json_path.exists():
        return None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Build model key
        embedding_model = data.get("embedding_model", "unknown")
        strategy = data.get("strategy", "unknown")
        model_key = f"{embedding_model}_{strategy}"

        return ModelMetrics(
            model_key=model_key,
            embedding_model=embedding_model,
            strategy=strategy,
            epochs=data.get("epochs", 0),
            batch_size=data.get("batch_size", 0),
            train_loss=data.get("train_loss", []),
            val_loss=data.get("val_loss", []),
            val_balanced_accuracy=data.get("val_balanced_accuracy", []),
            val_f1_score=data.get("val_f1_score", []),
        )
    except Exception as e:
        print(f"Warning: Failed to load {json_path}: {e}", file=sys.stderr)
        return None


def load_all_metrics(metrics_dir: Path) -> Dict[str, ModelMetrics]:
    """Load all metrics JSON files from the directory."""
    models = {}

    # Look for metrics_*.json files
    for json_path in sorted(metrics_dir.glob("metrics_*.json")):
        metrics = load_metrics_json(json_path)
        if metrics:
            models[metrics.model_key] = metrics

    return models


def compute_global_stats(models: Dict[str, ModelMetrics]) -> GlobalStats:
    """Compute global statistics across all models."""
    if not models:
        return GlobalStats(models={})

    # Find best models by each metric
    best_acc_key = max(models.keys(), key=lambda k: models[k].best_balanced_accuracy)
    best_f1_key = max(models.keys(), key=lambda k: models[k].best_f1_score)
    best_loss_key = min(models.keys(), key=lambda k: models[k].best_val_loss)

    return GlobalStats(
        models=models,
        best_model_by_accuracy=best_acc_key,
        best_model_by_f1=best_f1_key,
        best_model_by_val_loss=best_loss_key,
    )


# -----------------------------
# Plotting functions
# -----------------------------


def plot_model_comparison(stats: GlobalStats, out_dir: Path) -> Optional[Path]:
    """
    Main comparison plot for presentation.
    
    Shows all models with their balanced accuracy and F1 score as grouped bars.
    """
    if plt is None or not stats.models:
        return None

    models = stats.models
    model_keys = sorted(models.keys())

    # Extract metrics
    accuracies = [models[k].best_balanced_accuracy * 100 for k in model_keys]
    f1_scores = [models[k].best_f1_score * 100 for k in model_keys]
    display_names = [models[k].display_name for k in model_keys]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(model_keys))
    width = 0.35

    # Plot bars
    bars1 = ax.bar(
        [i - width / 2 for i in x],
        accuracies,
        width,
        label="Balanced Accuracy",
        color=COLORS["accent"],
        alpha=0.9,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        f1_scores,
        width,
        label="F1 Score",
        color=COLORS["success"],
        alpha=0.9,
    )

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["text"],
            )

    add_labels(bars1)
    add_labels(bars2)

    # Styling
    ax.set_xlabel("Fine-Tuning Strategy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Fine-Tuned Model Performance Comparison",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=10)
    boardstate_axes(ax)

    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    out_path = out_dir / "model_comparison.png"
    _save_fig(fig, out_path)
    return out_path


def plot_training_curves(stats: GlobalStats, out_dir: Path) -> Optional[Path]:
    """
    Plot training and validation loss curves for all models.
    """
    if plt is None or not stats.models:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_key, metrics in stats.models.items():
        epochs = list(range(1, len(metrics.train_loss) + 1))
        color = metrics.color

        # Training loss
        ax1.plot(
            epochs,
            metrics.train_loss,
            marker="o",
            label=metrics.display_name,
            color=color,
            linewidth=2,
        )

        # Validation loss
        ax2.plot(
            epochs,
            metrics.val_loss,
            marker="s",
            label=metrics.display_name,
            color=color,
            linewidth=2,
        )

    # Styling for training loss
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    boardstate_axes(ax1)

    # Styling for validation loss
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    boardstate_axes(ax2)

    fig.suptitle(
        "Fine-Tuning Loss Curves",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    out_path = out_dir / "training_curves.png"
    _save_fig(fig, out_path)
    return out_path


def plot_metrics_heatmap(stats: GlobalStats, out_dir: Path) -> Optional[Path]:
    """
    Heatmap showing all metrics for all models.
    """
    if plt is None or not stats.models:
        return None

    import numpy as np

    models = stats.models
    model_keys = sorted(models.keys())
    display_names = [models[k].display_name for k in model_keys]

    # Build metrics matrix (rows=models, cols=metrics)
    metrics_data = []
    for k in model_keys:
        m = models[k]
        metrics_data.append(
            [
                m.best_val_loss,
                m.best_balanced_accuracy * 100,
                m.best_f1_score * 100,
            ]
        )

    data = np.array(metrics_data).T  # Transpose for better visualization

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create heatmap
    im = ax.imshow(data, aspect="auto", cmap="viridis", interpolation="nearest")

    # Set ticks
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=15, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Val Loss", "Balanced Acc (%)", "F1 Score (%)"])

    # Add text annotations
    for i in range(3):
        for j in range(len(model_keys)):
            value = data[i, j]
            text = ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=COLORS["text"] if value < data[i].mean() else COLORS["bg"],
                fontsize=10,
                fontweight="bold",
            )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=9, colors=COLORS["muted"])

    ax.set_title(
        "Fine-Tuning Metrics Heatmap",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = out_dir / "metrics_heatmap.png"
    _save_fig(fig, out_path)
    return out_path


def plot_strategy_comparison(stats: GlobalStats, out_dir: Path) -> Optional[Path]:
    """
    Compare strategies (backbone vs head-only vs LoRA) across models.
    """
    if plt is None or not stats.models:
        return None

    # Group by strategy
    strategies = {}
    for model_key, metrics in stats.models.items():
        strategy = metrics.strategy
        if strategy not in strategies:
            strategies[strategy] = []
        strategies[strategy].append(metrics)

    if len(strategies) <= 1:
        return None  # Not enough strategies to compare

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    strategy_names = sorted(strategies.keys())
    x = range(len(strategy_names))

    # Average metrics per strategy
    avg_accuracy = [
        sum(m.best_balanced_accuracy for m in strategies[s]) / len(strategies[s]) * 100
        for s in strategy_names
    ]
    avg_f1 = [
        sum(m.best_f1_score for m in strategies[s]) / len(strategies[s]) * 100
        for s in strategy_names
    ]
    avg_val_loss = [
        sum(m.best_val_loss for m in strategies[s]) / len(strategies[s])
        for s in strategy_names
    ]

    # Plot 1: Accuracy and F1
    bars1 = ax1.bar(
        [i - 0.2 for i in x],
        avg_accuracy,
        0.4,
        label="Avg Balanced Accuracy",
        color=COLORS["accent"],
        alpha=0.9,
    )
    bars2 = ax1.bar(
        [i + 0.2 for i in x],
        avg_f1,
        0.4,
        label="Avg F1 Score",
        color=COLORS["success"],
        alpha=0.9,
    )

    ax1.set_xlabel("Strategy", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Average Performance by Strategy", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_names)
    ax1.legend(loc="best", fontsize=9)
    boardstate_axes(ax1)

    # Plot 2: Validation loss
    bars3 = ax2.bar(x, avg_val_loss, color=COLORS["warn"], alpha=0.9)

    ax2.set_xlabel("Strategy", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Validation Loss", fontsize=11, fontweight="bold")
    ax2.set_title("Average Validation Loss by Strategy", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_names)
    boardstate_axes(ax2)

    fig.suptitle(
        "Fine-Tuning Strategy Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    out_path = out_dir / "strategy_comparison.png"
    _save_fig(fig, out_path)
    return out_path


# -----------------------------
# Markdown generation
# -----------------------------


def relpath_for_docs(asset_path: Path, md_out: Path) -> str:
    """Compute relative path from markdown file to asset."""
    try:
        return "../" + str(asset_path.relative_to(md_out.parent.parent))
    except ValueError:
        return str(asset_path)


def write_markdown(
    md_out: Path,
    stats: GlobalStats,
    asset_paths: Dict[str, Optional[Path]],
) -> None:
    """Generate a Markdown snippet for Just-the-Docs."""
    md_out.parent.mkdir(parents=True, exist_ok=True)

    with md_out.open("w", encoding="utf-8") as f:
        f.write("<!-- Auto-generated by finetuning_stats_for_docs.py -->\n\n")
        f.write("## Fine-Tuning Results\n\n")

        if not stats.models:
            f.write("{: .warning }\n")
            f.write("> No fine-tuning metrics found.\n\n")
            return

        # Summary callout
        best_model = stats.models.get(stats.best_model_by_accuracy)
        if best_model:
            f.write("{: .highlight }\n")
            f.write(
                f"> **Best Model**: {best_model.display_name} achieved "
                f"**{best_model.best_balanced_accuracy*100:.2f}%** balanced accuracy "
                f"and **{best_model.best_f1_score*100:.2f}%** F1 score.\n\n"
            )

        # Main comparison figure
        if asset_paths.get("model_comparison"):
            img_url = relpath_for_docs(asset_paths["model_comparison"], md_out)
            f.write("### Model Comparison\n\n")
            f.write(f"![Model Comparison]({img_url})\n\n")

        # Summary table
        f.write("### Performance Summary\n\n")
        f.write("| Model | Strategy | Balanced Acc | F1 Score | Val Loss |\n")
        f.write("|-------|----------|--------------|----------|----------|\n")

        for model_key in sorted(stats.models.keys()):
            m = stats.models[model_key]
            f.write(
                f"| {m.display_name} | {m.strategy} | "
                f"{m.best_balanced_accuracy*100:.2f}% | "
                f"{m.best_f1_score*100:.2f}% | "
                f"{m.best_val_loss:.4f} |\n"
            )

        f.write("\n")

        # Training curves
        if asset_paths.get("training_curves"):
            img_url = relpath_for_docs(asset_paths["training_curves"], md_out)
            f.write("### Training Curves\n\n")
            f.write(f"![Training Curves]({img_url})\n\n")

        # Metrics heatmap
        if asset_paths.get("metrics_heatmap"):
            img_url = relpath_for_docs(asset_paths["metrics_heatmap"], md_out)
            f.write("### Metrics Heatmap\n\n")
            f.write(f"![Metrics Heatmap]({img_url})\n\n")

        # Strategy comparison
        if asset_paths.get("strategy_comparison"):
            img_url = relpath_for_docs(asset_paths["strategy_comparison"], md_out)
            f.write("### Strategy Comparison\n\n")
            f.write(f"![Strategy Comparison]({img_url})\n\n")

        # Technical details callout
        f.write("{: .note }\n")
        f.write("> **Technical Details**\n")
        f.write(">\n")
        f.write(f"> - **Models evaluated**: {len(stats.models)}\n")
        f.write(
            f"> - **Best by accuracy**: {stats.models[stats.best_model_by_accuracy].display_name}\n"
        )
        f.write(
            f"> - **Best by F1 score**: {stats.models[stats.best_model_by_f1].display_name}\n"
        )
        f.write(
            f"> - **Best by val loss**: {stats.models[stats.best_model_by_val_loss].display_name}\n"
        )

    print(f"✓ Markdown written to {md_out}")


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning statistics and figures for CSC-BSR docs."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("embedding"),
        help="Directory containing metrics_*.json files (default: embedding)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/assets/finetuning_stats"),
        help="Output directory for figures (default: docs/assets/finetuning_stats)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("docs/_includes/finetuning_stats.md"),
        help="Output markdown file (default: docs/_includes/finetuning_stats.md)",
    )
    parser.add_argument(
        "--style-path",
        type=Path,
        default=Path("utils/styles/boardstate-dark.mplstyle"),
        help="Matplotlib style file (default: utils/styles/boardstate-dark.mplstyle)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional: output JSON summary file",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.metrics_dir.exists():
        print(f"Error: Metrics directory not found: {args.metrics_dir}", file=sys.stderr)
        return 1

    # Setup matplotlib styling
    if plt is not None:
        setup_boardstate_matplotlib(args.style_path)
    else:
        print("Warning: matplotlib not available, skipping plots", file=sys.stderr)

    # Load metrics
    print(f"Loading metrics from {args.metrics_dir}...")
    models = load_all_metrics(args.metrics_dir)

    if not models:
        print("Error: No valid metrics files found", file=sys.stderr)
        return 1

    print(f"Found {len(models)} fine-tuned models")

    # Compute statistics
    stats = compute_global_stats(models)

    # Generate figures
    args.out_dir.mkdir(parents=True, exist_ok=True)
    asset_paths = {}

    print("Generating figures...")
    asset_paths["model_comparison"] = plot_model_comparison(stats, args.out_dir)
    asset_paths["training_curves"] = plot_training_curves(stats, args.out_dir)
    asset_paths["metrics_heatmap"] = plot_metrics_heatmap(stats, args.out_dir)
    asset_paths["strategy_comparison"] = plot_strategy_comparison(stats, args.out_dir)

    # Write markdown
    if args.md_out:
        write_markdown(args.md_out, stats, asset_paths)

    # Write JSON summary
    if args.json_out:
        json_data = {
            "models": {
                k: {
                    "display_name": v.display_name,
                    "embedding_model": v.embedding_model,
                    "strategy": v.strategy,
                    "epochs": v.epochs,
                    "batch_size": v.batch_size,
                    "best_val_loss": v.best_val_loss,
                    "best_balanced_accuracy": v.best_balanced_accuracy,
                    "best_f1_score": v.best_f1_score,
                }
                for k, v in stats.models.items()
            },
            "best_model_by_accuracy": stats.best_model_by_accuracy,
            "best_model_by_f1": stats.best_model_by_f1,
            "best_model_by_val_loss": stats.best_model_by_val_loss,
        }

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ JSON summary written to {args.json_out}")

    print("\n✅ Done! Generated:")
    for name, path in asset_paths.items():
        if path:
            print(f"   - {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
