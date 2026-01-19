#!/usr/bin/env python3
"""Generate fine-tuning figures for the GitHub site (docs) and slides.

Assumptions (per user request):
- This script will live in: ./utils
- Matplotlib style files live in: ./styles
- GitHub Pages site lives in: ./docs
- Fine-tune metrics JSON files live in: ./embedding

Outputs:
- Docs images:   ./docs/assets/fine_tuning/<theme>/...
- Slides image:  ./docs/assets/presentation/<theme>/finetune_model_comparison.png
- Optional docs include snippet: ./docs/_includes/fine_tuning_figures_<theme>.md

Run from repo root, e.g.:
  python utils/generate_finetune_figures.py --theme both

Notes:
- This script is robust to metrics files named like:
    metrics_<model>_<strategy>_{time}.json   (your current fine_tune.py behavior)
    metrics_<model>_<strategy>_2026-01-19.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class Run:
    path: Path
    embedding_model: str
    strategy: str
    epochs_declared: Optional[int]
    train_loss: List[float]
    val_loss: List[float]
    val_balanced_accuracy: List[float]
    val_f1_score: List[float]

    @property
    def label(self) -> str:
        # Short, slide-friendly label
        return f"{self.embedding_model} · {self.strategy}"

    @property
    def slug(self) -> str:
        s = (self.embedding_model + "_" + self.strategy).lower()
        s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
        return s

    def _best_epoch_idx(self) -> int:
        """Return 0-based epoch index of best run, primarily by val_balanced_accuracy."""
        if not self.val_balanced_accuracy:
            return 0
        best = 0
        for i in range(len(self.val_balanced_accuracy)):
            a = self.val_balanced_accuracy[i]
            b = self.val_f1_score[i] if i < len(self.val_f1_score) else float("-inf")
            l = self.val_loss[i] if i < len(self.val_loss) else float("inf")

            a_best = self.val_balanced_accuracy[best]
            b_best = self.val_f1_score[best] if best < len(self.val_f1_score) else float("-inf")
            l_best = self.val_loss[best] if best < len(self.val_loss) else float("inf")

            # Compare tuples: max acc, then max f1, then min loss
            if (a, b, -l) > (a_best, b_best, -l_best):
                best = i
        return best

    def best_epoch(self) -> int:
        return self._best_epoch_idx() + 1

    def best_val_bal_acc(self) -> Optional[float]:
        if not self.val_balanced_accuracy:
            return None
        return self.val_balanced_accuracy[self._best_epoch_idx()]

    def best_val_f1(self) -> Optional[float]:
        if not self.val_f1_score:
            return None
        idx = min(self._best_epoch_idx(), len(self.val_f1_score) - 1)
        return self.val_f1_score[idx]

    def final_val_loss(self) -> Optional[float]:
        if not self.val_loss:
            return None
        return self.val_loss[-1]


def _safe_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]


def load_runs(metrics_dir: Path) -> List[Run]:
    runs: List[Run] = []
    for p in sorted(metrics_dir.glob("metrics_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        embedding_model = str(data.get("embedding_model", "unknown"))
        strategy = str(data.get("strategy", "unknown"))

        runs.append(
            Run(
                path=p,
                embedding_model=embedding_model,
                strategy=strategy,
                epochs_declared=(int(data["epochs"]) if "epochs" in data else None),
                train_loss=_safe_list(data.get("train_loss")),
                val_loss=_safe_list(data.get("val_loss")),
                val_balanced_accuracy=_safe_list(data.get("val_balanced_accuracy")),
                val_f1_score=_safe_list(data.get("val_f1_score")),
            )
        )

    return runs


def apply_style(theme: str, styles_dir: Path) -> None:
    """Apply BoardState style if possible; otherwise fall back gracefully."""
    style_file = styles_dir / f"boardstate-{theme}.mplstyle"
    try:
        if style_file.exists():
            plt.style.use(str(style_file))
    except Exception as e:
        print(f"[WARN] Failed to apply style {style_file}: {e}")

    # Consistent layout defaults
    plt.rcParams.update(
        {
            "figure.constrained_layout.use": True,
        }
    )


def savefig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _epochs_for(run: Run) -> List[int]:
    n = max(len(run.train_loss), len(run.val_loss), len(run.val_balanced_accuracy), len(run.val_f1_score), 1)
    return list(range(1, n + 1))


def plot_run_learning_curves(run: Run, out: Path, title_prefix: str = "") -> None:
    ep = _epochs_for(run)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    # Losses
    ax = axes[0]
    if run.train_loss:
        ax.plot(ep[: len(run.train_loss)], run.train_loss, marker="o", label="train_loss")
    if run.val_loss:
        ax.plot(ep[: len(run.val_loss)], run.val_loss, marker="o", label="val_loss")
    ax.set_title(f"{title_prefix}{run.label} — Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")

    # Metrics
    ax = axes[1]
    if run.val_balanced_accuracy:
        ax.plot(ep[: len(run.val_balanced_accuracy)], run.val_balanced_accuracy, marker="o", label="val_bal_acc")
    if run.val_f1_score:
        ax.plot(ep[: len(run.val_f1_score)], run.val_f1_score, marker="o", label="val_f1")
    ax.set_title(f"{run.label} — Validation metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")

    savefig(fig, out)


def plot_overview_curves(runs: List[Run], out_dir: Path) -> None:
    # Loss overview: train + val separately to avoid clutter
    # Train loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        if r.train_loss:
            ep = _epochs_for(r)
            ax.plot(ep[: len(r.train_loss)], r.train_loss, marker="o", label=r.label)
    ax.set_title("Train loss across strategies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.legend(loc="best")
    savefig(fig, out_dir / "ft_train_loss_curves.png")

    # Val loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        if r.val_loss:
            ep = _epochs_for(r)
            ax.plot(ep[: len(r.val_loss)], r.val_loss, marker="o", label=r.label)
    ax.set_title("Validation loss across strategies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.legend(loc="best")
    savefig(fig, out_dir / "ft_val_loss_curves.png")

    # Val balanced accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        if r.val_balanced_accuracy:
            ep = _epochs_for(r)
            ax.plot(ep[: len(r.val_balanced_accuracy)], r.val_balanced_accuracy, marker="o", label=r.label)
    ax.set_title("Validation balanced accuracy across strategies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    savefig(fig, out_dir / "ft_val_bal_acc_curves.png")

    # Val F1
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        if r.val_f1_score:
            ep = _epochs_for(r)
            ax.plot(ep[: len(r.val_f1_score)], r.val_f1_score, marker="o", label=r.label)
    ax.set_title("Validation F1 (weighted) across strategies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (weighted)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    savefig(fig, out_dir / "ft_val_f1_curves.png")


def plot_model_comparison(runs: List[Run], out: Path, metric: str = "val_balanced_accuracy") -> None:
    # Bar chart of best metric per run
    rows: List[Tuple[str, float, int]] = []
    for r in runs:
        if metric == "val_balanced_accuracy":
            v = r.best_val_bal_acc()
        elif metric == "val_f1_score":
            v = r.best_val_f1()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if v is None:
            continue
        rows.append((r.label, float(v), r.best_epoch()))

    rows.sort(key=lambda t: t[1], reverse=True)

    labels = [t[0] for t in rows]
    vals = [t[1] for t in rows]
    epochs = [t[2] for t in rows]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.6 * len(labels))))

    # Use color cycle by plotting bars one-by-one
    y = list(range(len(labels)))
    for i, (lab, val, ep) in enumerate(zip(labels, vals, epochs)):
        ax.barh(i, val, label=lab)
        ax.text(val + 0.01, i, f"{val:.3f} (ep {ep})", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    if metric == "val_balanced_accuracy":
        ax.set_title("Fine-tuning comparison — best validation balanced accuracy")
        ax.set_xlabel("Best validation balanced accuracy")
    else:
        ax.set_title("Fine-tuning comparison — best validation F1 (weighted)")
        ax.set_xlabel("Best validation F1 (weighted)")

    ax.set_xlim(0.0, 1.05)

    savefig(fig, out)


def write_docs_include(runs: List[Run], theme: str, out_md: Path, assets_url_prefix: str) -> None:
    # Summary table and a couple of key callouts (Just the Docs style)
    scored = [(r, r.best_val_bal_acc() or float("-inf")) for r in runs]
    scored.sort(key=lambda t: t[1], reverse=True)

    best_run = scored[0][0] if scored else None

    lines: List[str] = []
    if best_run is not None and best_run.best_val_bal_acc() is not None:
        lines += [
            "{: .result }",
            f"**Best model (validation balanced accuracy):** **{best_run.label}** — {best_run.best_val_bal_acc():.3f} at epoch {best_run.best_epoch()}.",
            "",
        ]

    lines += [
        "### Fine-tuning summary (validation)",
        "",
        "| Model | Strategy | Best epoch | Best val bal. acc | Best val F1 | Final val loss |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for r, _ in scored:
        ba = r.best_val_bal_acc()
        f1 = r.best_val_f1()
        vl = r.final_val_loss()
        lines.append(
            "| "
            + f"{r.embedding_model} | {r.strategy} | {r.best_epoch()} | "
            + (f"{ba:.3f}" if ba is not None else "-")
            + " | "
            + (f"{f1:.3f}" if f1 is not None else "-")
            + " | "
            + (f"{vl:.3f}" if vl is not None else "-")
            + " |"
        )

    lines += [
        "",
        "### Figures",
        "",
        f"![Model comparison]({{\% raw \%}}{{{{ site.baseurl }}}}{{\% endraw \%}}/{assets_url_prefix}/{theme}/finetune_model_comparison.png)",
        f"![Validation balanced accuracy curves]({{\% raw \%}}{{{{ site.baseurl }}}}{{\% endraw \%}}/{assets_url_prefix}/{theme}/ft_val_bal_acc_curves.png)",
        f"![Validation loss curves]({{\% raw \%}}{{{{ site.baseurl }}}}{{\% endraw \%}}/{assets_url_prefix}/{theme}/ft_val_loss_curves.png)",
        "",
        "{: .repro }",
        "These figures are generated automatically from `embedding/metrics_*.json` (produced by `fine_tune.py`).",
        "",
    ]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fine-tuning figures for docs + slides")

    parser.add_argument("--metrics-dir", default="embedding", help="Directory containing metrics_*.json")
    parser.add_argument("--styles-dir", default="styles", help="Directory containing boardstate-*.mplstyle")
    parser.add_argument("--docs-dir", default="docs", help="GitHub Pages site directory")
    parser.add_argument("--theme", choices=["dark", "light", "both"], default="both")

    parser.add_argument(
        "--assets-subdir",
        default="assets/fine_tuning",
        help="Subdir inside docs/ for site figures",
    )
    parser.add_argument(
        "--presentation-subdir",
        default="assets/presentation",
        help="Subdir inside docs/ for presentation figures",
    )
    parser.add_argument(
        "--write-include",
        action="store_true",
        help="Write docs/_includes/fine_tuning_figures_<theme>.md snippet",
    )
    parser.add_argument(
        "--assets-url-prefix",
        default="assets/fine_tuning",
        help="URL prefix used in generated markdown (relative to site root)",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    metrics_dir = (repo_root / args.metrics_dir).resolve()
    styles_dir = (repo_root / args.styles_dir).resolve()
    docs_dir = (repo_root / args.docs_dir).resolve()

    if not metrics_dir.exists():
        raise SystemExit(f"metrics dir not found: {metrics_dir}")

    runs = load_runs(metrics_dir)
    if not runs:
        raise SystemExit(f"No metrics_*.json found in {metrics_dir}")

    themes = [args.theme] if args.theme != "both" else ["dark", "light"]

    for theme in themes:
        apply_style(theme, styles_dir)

        docs_out = docs_dir / args.assets_subdir / theme
        pres_out = docs_dir / args.presentation_subdir / theme

        docs_out.mkdir(parents=True, exist_ok=True)
        pres_out.mkdir(parents=True, exist_ok=True)

        # 1) Docs overview figures
        plot_overview_curves(runs, docs_out)

        # 2) Per-run learning curves (useful for the site)
        per_run_dir = docs_out / "per_run"
        for r in runs:
            plot_run_learning_curves(r, per_run_dir / f"learning_curves_{r.slug}.png")

        # 3) One comparison figure (docs + presentation)
        plot_model_comparison(runs, docs_out / "finetune_model_comparison.png", metric="val_balanced_accuracy")
        plot_model_comparison(runs, pres_out / "finetune_model_comparison.png", metric="val_balanced_accuracy")

        # Optional include snippet
        if args.write_include:
            include_path = docs_dir / "_includes" / f"fine_tuning_figures_{theme}.md"
            write_docs_include(
                runs=runs,
                theme=theme,
                out_md=include_path,
                assets_url_prefix=args.assets_url_prefix,
            )

    print("✓ Done")
    print(f"- Docs figures: {docs_dir / args.assets_subdir}")
    print(f"- Presentation figure: {docs_dir / args.presentation_subdir}")


if __name__ == "__main__":
    main()