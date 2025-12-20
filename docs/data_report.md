# Data Directory Report

## 1. Directory Structure
The data is organized into 5 folders, one for each game. Each folder contains a CSV file with labels and a subdirectory with the corresponding images.

```text
data/
├── game2_per_frame/
│   ├── game2.csv          # Labels (FEN strings, frame numbers)
│   └── tagged_images/     # Directory containing .jpg images
├── game4_per_frame/ ...
├── game5_per_frame/ ...
├── game6_per_frame/ ...
└── game7_per_frame/ ...
```

## 2. Data Format

### CSV Files (`gameX.csv`)
Each CSV file contains the following columns:
*   **`from_frame`**: The starting frame number (corresponds to the image filename).
*   **`to_frame`**: The ending frame number (usually identical to `from_frame`).
*   **`fen`**: The [Forsyth–Edwards Notation](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) string representing the board state for that frame.

### Images (`tagged_images/`)
*   **Format**: JPEG image data (`.jpg`)
*   **Dimensions**: 480x480 pixels
*   **Color Space**: RGB (3 channels)
*   **Naming Convention**: `frame_XXXXXX.jpg` (where `XXXXXX` matches the `from_frame` column in the CSV, zero-padded to 6 digits).

## 3. Data Integrity Report

An automated check of the dataset was performed to verify that every entry in the CSV has a corresponding image and to identify any inconsistencies.

| Game | CSV Rows | Unique Frames | Actual Images | Status | Issues Found |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Game 2** | 77 | 77 | 77 | ✅ **Clean** | None |
| **Game 4** | 186 | 186 | 184 | ⚠️ **Issues** | **2 Missing Images**: `frame_027627.jpg`, `frame_035238.jpg` |
| **Game 5** | 110 | 109 | 109 | ⚠️ **Issues** | **1 Duplicate Row** in CSV. (Images are consistent with unique frames) |
| **Game 6** | 93 | 93 | 92 | ⚠️ **Issues** | **1 Missing Image**: `frame_022566.jpg` |
| **Game 7** | 57 | 56 | 55 | ⚠️ **Issues** | **1 Duplicate Row** in CSV. **1 Missing Image**: `frame_000769.jpg` |

**Summary:**
*   **Total CSV Entries:** 523
*   **Total Unique Frames:** 521
*   **Total Images Available:** 517
*   **Total Missing Images:** 4

## 4. Recommendations
*   **Handling Missing Images:** When loading the data, verify if the image file exists before attempting to load it. Rows with missing images should be skipped.
*   **Handling Duplicates:** For Games 5 and 7, drop duplicate rows based on the `from_frame` column during data loading.

## 5. Data Generation & External Resources

Research was conducted to identify methods for generating synthetic data and locating existing datasets to augment the current real-world data.

### A. Synthetic Data Generation
To generate photorealistic data that matches the "Real Data" requirement:
*   **Blender Python API:** The most robust method. Scripts can be written to load FEN strings, place 3D assets (from PolyHaven/Sketchfab), and render them with randomized lighting (HDRI maps) and camera angles.
*   **Existing Tools:**
    *   **[chess-generator](https://github.com/koryakinp/chess-generator):** Used for the popular Kaggle dataset.
    *   **Unity:** Can also be used for real-time generation if massive scale is needed.

### B. Existing Datasets
*   **[Kaggle: Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions):** ~100k synthetic images (400x400) labeled with FEN. Excellent for pre-training.
*   **Roboflow Universe:** Search for "Chess Pieces" for datasets with bounding box annotations (useful for object detection approaches like YOLO).
*   **Raspberry Turk Dataset:** A dataset of real images from a chess robot, which aligns well with the project domain.

### C. Recommended Strategy
1.  **Pre-train** the model on the large, synthetic **Kaggle Chess Positions** dataset.
2.  **Fine-tune** on the real `data/` directory.
3.  **Augmentation:** Apply **Perspective Warping** (to simulate camera angles) and **Lighting/Shadow Injection** (to simulate real-world conditions) during training.
