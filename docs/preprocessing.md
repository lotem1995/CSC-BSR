---
title: Building the Dataset and Pre-Processing
nav_order: 2
---


<!-- # Building the Dataset and Pre-Processing -->

We transform raw chess frames into **consistent 224×224 tile images** for both:
- **inference** (`predict_board.py`)
- **dataset building** (`preprocessing/build_dataset.py`)

{: .repro }
The *exact same* slicing + normalization is used during evaluation/inference to avoid train–test mismatch.

## Quick links
[Code: `preprocess_game_data.py`](https://github.com/lotem1995/CSC-BSR/blob/main/preprocessing/preprocess_game_data.py){: .btn .btn-outline .mr-2 }
[Code: `build_dataset.py`](https://github.com/lotem1995/CSC-BSR/blob/main/preprocessing/build_dataset.py){: .btn .btn-outline }
[Code: `splitting_images.py`](https://github.com/lotem1995/CSC-BSR/blob/main/preprocessing/splitting_images.py){: .btn .btn-outline }


## Pre-Processing frame into tile images when predicting a new board frame

The predict_board function in predict_board.py preprocesses chess frames for model predictions. It calls `slice_image_with_coordinates` to divide board images into individual tile images.


{: .decision }
**Why tiles (not full boards)?** Splitting the board image into tiles simplifies classification. Processing the whole board would require labeling every possible board state, which is unfeasible. By extracting individual tiles, we dramatically reduce label complexity and can augment training data efficiently using fewer board images.

# Pre-Processing and Creating the Dataset for training and evaluation

The dataset consists of chess games, each split into multiple frames (one per move). We preprocess frames via a script using a YAML configuration file to standardize the data.

We divided the dataset into training, validation, and test sets based on entire games, ensuring that all frames from the same game were included in a single set to prevent data leakage and reduce similarity between the sets.

Frames are converted into tile images and stored in the preprocessed_data directory. The splits directory includes a JSON manifest file and CSVs for the train, val, and test sets.

Next, we detail the specific preprocessing and dataset building procedures.

## Pre-processing

| Item | Value / Choice | Why it matters |
|---|---:|---|
| Tiles per frame | 64 | Reduce label complexity vs full-board labeling |
| Tile size | **224×224** | Stable model input |
| Overlap | **0.7** | Adds neighbor context; improves accuracy |
| Split strategy | **by game** | Prevent leakage across train/val/test |
| Missing tiles | **zero-padding** | Keeps shapes consistent |
| Augmentation | rotations / flips / color noise | Robustness, less overfit |
| OOD | tag + “unknown” label | Enable OOD detection |

Each chess frame is split into 64 overlapping tiles using `slice_image_with_coordinates`.

The overlap is achieved by expanding each tile’s bounding box slightly beyond its exact coordinates so each square includes some area from adjacent tiles. The reason for this overlap is to ensure that the model can see not just the contents of a single square, but also relevant context from neighboring tiles, such as the edges of pieces that may extend across squares. This additional context consistently improved classification accuracy in our tests. We chose a 0.7 (70%) overlap value after experimenting with different overlap levels, finding this value provided the best tradeoff between including helpful context and limiting redundancy.

In addition to slicing the chess frames, the preprocessing step also normalizes the tile images so that each tile image has a consistent format for the model. This includes resizing all tile images to the same dimensions, adjusting brightness and contrast for uniformity, and using data augmentation techniques, such as random rotations or flips, to make the model more robust to different visual conditions.

We also introduced a zero-padding technique. If tile images are partially occluded or missing in the original chess frame, we add a border of zeros (black pixels) around the tile image. This keeps all tile images the same size and shape, even when parts of them are not visible.


{: .decision }
**Things we tried:** In addition to the above preprocessing steps, we experimented with classical methods for tile image extraction, such as edge detection, but found that these methods detected more lines (edges) than just the chessboard lines, leading to inaccurate tile image extraction. Therefore, we opted for the coordinate-based slicing method described above.


#### In the code
The preprocessing script is `preprocessing/preprocess_game_data.py`, and for each frame, it calls `slice_image_with_coordinates` from `preprocessing/splitting_images.py` to create tile images with **0.7 overlap, a size of 224x224 pixels, and zero-padding for missing tiles.**

This function can adjust overlap, tile size, and zero-padding. The resulting images are saved with filenames that include row, column, and class information from the board.

### Creating the Dataset
The main code for building the dataset is in `preprocessing/build_dataset.py`.

This script processes all games in the data directory, applies the preprocessing steps, and structures tile images into the dataset.
The dataset is stored in a directory called `preprocessed_data`, containing tile images for each frame from all games in the `data` directory. There is also a splits directory with a manifest JSON file and three CSV files: `train.csv`, `val.csv`, and `test.csv`.

The manifest JSON provides dataset metadata, maps tile images to classes (such as piece type), and contains other information required for training and evaluation.

#### Splitting the data into datasets
The `build_dataset.py` script uses a YAML configuration file to define data splitting rules, such as proportions for the train, validation, and test sets, or to explicitly list which games go into each set.

{: .warning }
The split is done at the game level to avoid data leakage. By splitting data at the game level, all frames from a single game remain together in a single set, helping avoid data leakage and overfitting for reliable evaluation.

Users specify either percentages or game IDs in the YAML file, enabling flexible, reproducible dataset splits.

#### Data Augmentation
To balance learning, we equalize the distribution of tile classes in the train DataLoader using a random sampler. During each image request, augmentations include random rotations (90 degrees or small angles), mirroring, and color noise. These increase robustness and limit overfitting.

#### OOD
For out-of-distribution (OOD) images, we tag tiles from frames containing OOD elements (e.g., hands or foreign objects) and store them separately. These get a distinct dataset label, allowing the model to learn an 'unknown' class for OOD detection during inference.

<!-- Analize the dataset -->
## Dataset statistics
{% include preprocessing_stats.md %}