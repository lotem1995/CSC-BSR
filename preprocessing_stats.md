## Dataset statistics

{: .repro }
**Config:** (not provided to script)

{: .result }
**Size:** 38,144 tiles from 596 board frames across 10 games. Non-empty (in-distribution) tiles: **32.5%**; OOD tiles: **1.47%**.
Train↔Test class distribution shift (JSD, excluding empty): **0.026 bits**.

{: .result }
**Split hygiene:** no board_id / game_id overlap between train/val/test (✅).

### Split breakdown

| split | tiles | boards | games | non-empty tiles | OOD tiles | boards complete (64 tiles) | missing images | embeddings present |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 16,384 | 256 | 3 | 5,788 (35.3%) | 140 (0.85%) | 100.0% | 0 | 0 (0.0%) |
| val | 7,424 | 116 | 4 | 2,907 (39.2%) | 63 (0.85%) | 100.0% | 0 | 0 (0.0%) |
| test | 14,336 | 224 | 3 | 3,716 (25.9%) | 359 (2.50%) | 100.0% | 0 | 0 (0.0%) |

### Class balance (high level)

{: .decision }
The dataset is **empty-dominant**: class `empty` is 25,171 tiles (**66.0%**). If you train a classifier directly on raw tiles, use class balancing (sampler / loss weights).

**Top classes (excluding empty):** white_pawn (3,305), black_pawn (3,130), black_rook (844), white_rook (843), black_bishop (667), white_knight (658), white_bishop (614), white_king (587)

### Visual summary

**Overview**

![](analysis_results_for_docs/overview.png)

**Class heatmap**

![](analysis_results_for_docs/class_heatmap.png)

**Top games by frames**

![](analysis_results_for_docs/games_top15.png)
