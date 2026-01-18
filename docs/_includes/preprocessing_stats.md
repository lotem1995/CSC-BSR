## Dataset statistics

{: .repro }
**Config:** (not provided to script)

{: .result }
**Size:** 75,279 tiles from 1,170 board frames across 11 games. Non-empty (in-distribution) tiles: **33.1%**; OOD tiles: **0.00%**.
Train↔Test class distribution shift (JSD, excluding empty): **0.500 bits**.

{: .result }
**Split hygiene:** no board_id / game_id overlap between train/val/test (✅).

### Split breakdown

| split | tiles | boards | games | non-empty tiles | OOD tiles | boards complete (64 tiles) | missing images | embeddings present |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 75,279 | 1,170 | 11 | 24,886 (33.1%) | 0 (0.00%) | 98.4% | 0 | 0 (0.0%) |
| val | 0 | 0 | 0 | 0 (0.0%) | 0 (0.00%) | 0.0% | 0 | 0 (0.0%) |
| test | 0 | 0 | 0 | 0 (0.0%) | 0 (0.00%) | 0.0% | 0 | 0 (0.0%) |

### Class balance (high level)

{: .decision }
The dataset is **empty-dominant**: class `empty` is 50,393 tiles (**66.9%**). If you train a classifier directly on raw tiles, use class balancing (sampler / loss weights).

**Top classes (excluding empty):** white_pawn (6,582), black_pawn (6,270), black_rook (1,691), white_rook (1,673), black_bishop (1,358), white_knight (1,325), white_bishop (1,238), black_king (1,180)
**Rare (<0.75%)**: OOD (0)

### Training dataloader (WeightedRandomSampler)

**Class weights:** Inverse class frequency (rarer classes get higher weight)

| class | weight | effective % |
|---|---:|---:|
| empty | 0.0000 | 7.69% |
| white_pawn | 0.0002 | 7.69% |
| white_knight | 0.0008 | 7.69% |
| white_bishop | 0.0008 | 7.69% |
| white_rook | 0.0006 | 7.69% |
| white_queen | 0.0014 | 7.69% |
| white_king | 0.0008 | 7.69% |
| black_pawn | 0.0002 | 7.69% |
| black_knight | 0.0010 | 7.69% |
| black_bishop | 0.0007 | 7.69% |
| black_rook | 0.0006 | 7.69% |
| black_queen | 0.0014 | 7.69% |
| black_king | 0.0008 | 7.69% |
| OOD | 0.0000 | 0.00% |

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
