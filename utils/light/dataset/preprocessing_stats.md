## Dataset statistics

{: .repro }
**Config:** (not provided to script)

{: .result }
**Size:** 38,016 tiles from 594 board frames across 10 games. Non-empty (in-distribution) tiles: **32.6%**; OOD tiles: **1.48%**.
Train↔Test class distribution shift (JSD, excluding empty): **0.025 bits**.

{: .result }
**Split hygiene:** no board_id / game_id overlap between train/val/test (✅).

### Split breakdown

| split | tiles | boards | games | non-empty tiles | OOD tiles | boards complete (64 tiles) | missing images | embeddings present |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 21,312 | 333 | 8 | 8,119 (38.1%) | 182 (0.85%) | 100.0% | 0 | 0 (0.0%) |
| val | 4,928 | 77 | 1 | 1,734 (35.2%) | 45 (0.91%) | 100.0% | 0 | 0 (0.0%) |
| test | 11,776 | 184 | 1 | 2,528 (21.5%) | 335 (2.84%) | 100.0% | 0 | 0 (0.0%) |

### Class balance (high level)

{: .decision }
The dataset is **empty-dominant**: class `empty` is 25,073 tiles (**66.0%**). If you train a classifier directly on raw tiles, use class balancing (sampler / loss weights).

**Top classes (excluding empty):** white_pawn (3,297), black_pawn (3,123), white_rook (841), black_rook (841), black_bishop (666), white_knight (658), white_bishop (611), white_king (585)

### Training dataloader (WeightedRandomSampler)

**Class weights:** Inverse class frequency (rarer classes get higher weight)

| class | weight | effective % |
|---|---:|---:|
| empty | 0.0001 | 7.69% |
| white_pawn | 0.0005 | 7.69% |
| white_knight | 0.0025 | 7.69% |
| white_bishop | 0.0020 | 7.69% |
| white_rook | 0.0020 | 7.69% |
| white_queen | 0.0038 | 7.69% |
| white_king | 0.0030 | 7.69% |
| black_pawn | 0.0005 | 7.69% |
| black_knight | 0.0027 | 7.69% |
| black_bishop | 0.0022 | 7.69% |
| black_rook | 0.0020 | 7.69% |
| black_queen | 0.0038 | 7.69% |
| black_king | 0.0031 | 7.69% |

{: .info }
The training dataloader uses `WeightedRandomSampler` to balance class representation during training, giving rarer classes more frequent sampling.

### Visual summary

**Overview**

![](utils/light/dataset/overview.png)

**Class heatmap**

![](utils/light/dataset/class_heatmap.png)

**Top games by frames**

![](utils/light/dataset/games_top15.png)

**Dataloader weights**

![](utils/light/dataset/dataloader_weights.png)
