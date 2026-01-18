import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def _extract_frame_number(filename: str) -> Optional[int]:
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def _sort_key_with_frame(path: Path) -> Tuple[int, int, str]:
    frame = _extract_frame_number(path.name)
    # (0/1 flag for whether a frame was found, frame value, fallback name)
    return (0 if frame is not None else 1, frame or -1, path.name)


def pair_images_with_fens(csv_path: str, images_folder: str) -> List[Tuple[str, str]]:
    """
    Pair images with FEN strings, skipping missing or duplicated frames.

    Supports two CSV schemas:
    1) gt.csv format: columns [image_name, fen, view]
    2) Legacy format: columns [from_frame, to_frame, fen]
    """

    csv_path = Path(csv_path)
    images_dir = Path(images_folder)

    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' not found.")
        return []
    if not images_dir.exists():
        print(f"Error: Image folder '{images_dir}' not found.")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: Failed reading '{csv_path}': {exc}")
        return []

    # Handle modern gt.csv layout
    if {"image_name", "fen"}.issubset(df.columns):
        df = df.dropna(subset=["image_name", "fen"])
        df = df.drop_duplicates(subset=["image_name", "fen"])

        results: List[Tuple[str, str]] = []
        skipped_missing = 0
        for _, row in df.iterrows():
            image_path = images_dir / str(row["image_name"])
            if not image_path.exists():
                skipped_missing += 1
                continue
            results.append((str(image_path), str(row["fen"])))

        print(
            f"Matched {len(results)} rows from gt.csv format; "
            f"skipped {skipped_missing} missing images."
        )
        return results

    required_cols = {"from_frame", "to_frame", "fen"}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV is missing columns. Found: {df.columns}")
        return []

    df = df.dropna(subset=["from_frame", "to_frame", "fen"])
    df = df.drop_duplicates(subset=["from_frame", "to_frame", "fen"]).reset_index(drop=True)

    image_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=_sort_key_with_frame,
    )

    print(f"Found {len(image_files)} images. Matching with CSV...")

    matched_count = 0
    skipped_no_number = 0
    skipped_duplicates = 0
    skipped_no_match = 0
    results: List[Tuple[str, str]] = []

    for img_path in image_files:
        frame_number = _extract_frame_number(img_path.name)
        if frame_number is None:
            skipped_no_number += 1
            continue

        matching_rows = df[(df["from_frame"] <= frame_number) & (df["to_frame"] >= frame_number)]
        if matching_rows.empty:
            skipped_no_match += 1
            continue

        if len(matching_rows) > 1:
            unique_fens = matching_rows["fen"].dropna().unique()
            if len(unique_fens) != 1:
                skipped_duplicates += 1
                continue
            fen = unique_fens[0]
        else:
            fen = matching_rows.iloc[0]["fen"]

        results.append((str(img_path), str(fen)))
        matched_count += 1

    print(
        "Successfully matched {matched} images; skipped {no_match} with no FEN, "
        "{no_number} without frame numbers, {dups} with duplicate/conflicting rows.".format(
            matched=matched_count,
            no_match=skipped_no_match,
            no_number=skipped_no_number,
            dups=skipped_duplicates,
        )
    )
    return results


