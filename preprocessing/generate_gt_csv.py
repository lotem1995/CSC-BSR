import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from handle_game_CSV import pair_images_with_fens

DEFAULT_VIEW = "white_bottom"


def _select_csv(game_dir: Path) -> Optional[Path]:
    gt_path = game_dir / "gt.csv"
    if gt_path.exists():
        return gt_path
    csv_candidates = sorted(p for p in game_dir.glob("*.csv") if p.name != "gt.csv")
    return csv_candidates[0] if csv_candidates else None


def _select_images_dir(game_dir: Path) -> Optional[Path]:
    for candidate in (game_dir / "images", game_dir / "tagged_images"):
        if candidate.exists():
            return candidate
    return None


def _derive_game_name(game_dir: Path) -> str:
    name = game_dir.name
    if name.endswith("_per_frame"):
        name = name.replace("_per_frame", "")
    return name


def build_gt_for_game(game_dir: Path, view: str, create_symlink: bool) -> None:
    csv_path = _select_csv(game_dir)
    images_dir = _select_images_dir(game_dir)

    if not csv_path or not images_dir:
        print(f"[WARN] Skipping {game_dir}: missing CSV or images directory")
        return

    pairs = pair_images_with_fens(str(csv_path), str(images_dir))
    if not pairs:
        print(f"[WARN] No image/FEN pairs produced for {game_dir}")
        return

    target_images_dir = game_dir / "images"
    if create_symlink and not target_images_dir.exists():
        try:
            target_images_dir.symlink_to(images_dir, target_is_directory=True)
            print(f"[INFO] Created symlink {target_images_dir} -> {images_dir}")
        except FileExistsError:
            pass

    rows = [{"image_name": Path(img_path).name, "fen": fen, "view": view} for img_path, fen in pairs]
    df = pd.DataFrame(rows)
    output_path = game_dir / "gt.csv"
    df.to_csv(output_path, index=False)
    print(f"[OK] Wrote {len(rows)} rows to {output_path}")



def generate_all(data_root: Path, view: str, create_symlink: bool) -> None:
    game_dirs: List[Path] = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    if not game_dirs:
        print(f"[WARN] No game directories found under {data_root}")
        return

    for game_dir in game_dirs:
        print(f"\n=== Processing {game_dir} ===")
        build_gt_for_game(game_dir, view=view, create_symlink=create_symlink)



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gt.csv files from legacy per-frame CSVs")
    parser.add_argument("--data-root", default="./data", help="Root directory containing game folders")
    parser.add_argument("--view", default=DEFAULT_VIEW, help="View specification string to store in gt.csv")
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Create an 'images' symlink pointing at 'tagged_images' when needed",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    generate_all(data_root, view=args.view, create_symlink=args.symlink_images)


if __name__ == "__main__":
    main()
