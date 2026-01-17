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

### Training dataloader (WeightedRandomSampler)

**Class weights:** Inverse class frequency (rarer classes get higher weight)

| class | weight | effective % |
|---|---:|---:|
| empty | 0.0001 | 7.14% |
| white_pawn | 0.0006 | 7.14% |
| white_knight | 0.0039 | 7.14% |
| white_bishop | 0.0029 | 7.14% |
| white_rook | 0.0029 | 7.14% |
| white_queen | 0.0054 | 7.14% |
| white_king | 0.0040 | 7.14% |
| black_pawn | 0.0007 | 7.14% |
| black_knight | 0.0043 | 7.14% |
| black_bishop | 0.0031 | 7.14% |
| black_rook | 0.0029 | 7.14% |
| black_queen | 0.0054 | 7.14% |
| black_king | 0.0040 | 7.14% |
| OOD | 0.0071 | 7.14% |

{: .info }
The training dataloader uses `WeightedRandomSampler` to balance class representation during training, giving rarer classes more frequent sampling.

### Visual summary

**Overview**

![](assets/preprocessing_stats/overview.png)

**Class heatmap**

![](assets/preprocessing_stats/class_heatmap.png)

**Top games by frames**

![](assets/preprocessing_stats/games_top15.png)

**Dataloader weights**

![](assets/preprocessing_stats/dataloader_weights.png)
