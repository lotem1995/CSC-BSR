---
title: Building the Dataset and Pre-Processing
nav_order: 2
---

<div class="bs-hero" markdown="block">

## Dataset Building & Preprocessing

We transform raw chess videos into **consistent 224×224 tile images** and CSV manifests that power both training and inference. The preprocessing pipeline pairs frame images with FEN strings, converts FEN to 8×8 board tensors, slices frames into tiles with 0.7 overlap, and splits data by game to prevent leakage.

{: .repro }
The *exact same* slicing and labeling conventions are used everywhere—a tile generated during preprocessing is identical to the tile produced at inference time for the same board position.

</div>

## Quick Links

[Code: `preprocess_game_data.py`](https://github.com/lotem1995/CSC-BSR/blob/main/preprocessing/preprocess_game_data.py){: .btn .btn-outline .mr-2 }
[Code: `build_dataset.py`](https://github.com/lotem1995/CSC-BSR/blob/main/preprocessing/build_dataset.py){: .btn .btn-outline .mr-2 }
[Code: `splitting_images.py`](https://github.com/lotem1995/CSC-BSR/blob/main/preprocessing/splitting_images.py){: .btn .btn-outline }

[Config: `dataset_config.yaml`](https://github.com/lotem1995/CSC-BSR/blob/main/preprocessing/dataset_config.example.yaml){: .btn .btn-outline }

---

## High-Level Pipeline

<div class="bs-cards" markdown="block">

<div class="bs-card" markdown="block">

### From Videos to Tiles

The preprocessing pipeline executes four critical steps:

1. **Pair frames with FENs** – `handle_game_CSV.py` reads each game's CSV and matches frame filenames to FEN strings
2. **Convert FEN → 8×8 board** – `handle_fen.py` parses FEN into canonical integer encoding
3. **Slice frames into tiles** – `splitting_images.py` crops frames into 8×8 grids with 0.7 overlap
4. **Build manifest + splits** – `build_dataset.py` generates tiles, marks OOD, splits by game, and writes CSVs

</div>

<div class="bs-card" markdown="block">

### Preprocessing Parameters

| Parameter | Value | Rationale |
|---|---:|---|
| **Tiles per frame** | 64 | Reduce label complexity vs full-board labeling |
| **Tile size** | 224×224 | Standard vision transformer input |
| **Overlap** | 0.7 | Adds neighbor context for border pieces |
| **Split strategy** | by game | Prevent leakage across train/val/test |
| **Zero-padding** | enabled | Maintains consistent tile dimensions |
| **Augmentation** | rotation/flip/jitter | Robustness against lighting/angle variations |

{: .decision }
**Why tiles instead of full boards?** A chessboard has a combinatorial number of legal positions. Tile-level classification reduces this to **17 classes** (12 pieces + empty + OOD), making training tractable while enabling board reconstruction.

</div>

</div>

---

## From Games to (Image, FEN) Pairs

### Robust CSV Handling

Games can arrive in two CSV formats:

<div markdown="block">

**Modern format (`gt.csv`)**:
```csv
image_name,fen,view
frame_000123.png,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,white
```

**Legacy format (ranges)**:
```csv
from_frame,to_frame,fen
1,50,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

</div>

`handle_game_CSV.py:pair_images_with_fens()` normalizes both formats into a unified `(image_path, fen)` stream:

```python
def pair_images_with_fens(csv_path: str, images_folder: str):
    df = pd.read_csv(csv_path)
    
    # Modern format: direct image_name → fen mapping
    if {"image_name", "fen"}.issubset(df.columns):
        for _, row in df.iterrows():
            image_path = images_folder / row["image_name"]
            if image_path.exists():
                yield (str(image_path), row["fen"])
    
    # Legacy format: map frame numbers to FEN ranges
    else:
        for image_path in sorted(images_folder.glob("*.png")):
            frame_num = extract_frame_number(image_path.name)
            matching = df[(df["from_frame"] <= frame_num) & 
                         (df["to_frame"] >= frame_num)]
            
            # Only use if all matches agree on single FEN
            if len(matching) > 0 and matching["fen"].nunique() == 1:
                yield (str(image_path), matching.iloc["fen"])
```

{: .decision }
**Why care about CSV robustness?** The preprocessing code handles both historical exports and newer per-frame layouts. This abstraction keeps downstream code model-agnostic—it never needs to special-case "gameX.csv vs gt.csv."

---

## FEN → Board Tensor

`handle_fen.py` converts FEN notation into an 8×8 integer grid using a canonical encoding:

```python
PIECE_MAP = {
    'P': 1,  'N': 2,  'B': 3,  'R': 4,  'Q': 5,  'K': 6,   # White pieces
    'p': 11, 'n': 12, 'b': 13, 'r': 14, 'q': 15, 'k': 16,  # Black pieces
}

def fen_to_board_int(fen: str) -> np.ndarray:
    board_str = fen.split(" ")
    ranks = board_str.split("/")
    board_rows = []
    
    for rank in ranks:
        row = []
        for ch in rank:
            if ch.isdigit():
                row.extend( * int(ch))  # Empty squares
            else:
                row.append(PIECE_MAP[ch])
        board_rows.append(row)
    
    return np.array(board_rows, dtype=np.int8)  # Shape: (8, 8)
```

This matrix drives tile labeling: each position `board[r, c]` encodes the piece class, which becomes part of the tile filename as `_class{board[r, c]}.png`.

---

## Slicing Frames into Tiles

### Core Function: `slice_image_with_coordinates`

<div markdown="block">

```python
def slice_image_with_coordinates(
    game_name: str,
    image_path: str,
    output_folder: str,
    board,                    # 8×8 np.array from fen_to_board_int
    overlap_percent: float = 0.7,
    final_size=(224, 224),
    zero_padding: bool = True,
):
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    cols = rows = 8
    stride_w = img_width / cols
    stride_h = img_height / rows
    
    # Enlarge crop box by overlap percentage
    crop_w = stride_w * (1 + overlap_percent)
    crop_h = stride_h * (1 + overlap_percent)
    
    for r in range(rows):
        for c in range(cols):
            # Center the crop on the cell
            center_x = (c * stride_w) + (stride_w / 2)
            center_y = (r * stride_h) + (stride_h / 2)
            
            left   = center_x - crop_w / 2
            upper  = center_y - crop_h / 2
            right  = center_x + crop_w / 2
            lower  = center_y + crop_h / 2
            
            # Zero-padding: allow crops outside image bounds
            if not zero_padding:
                left  = max(0, left)
                upper = max(0, upper)
                right = min(img_width, right)
                lower = min(img_height, lower)
            
            tile = img.crop((left, upper, right, lower))
            tile = tile.resize(final_size, Image.Resampling.LANCZOS)
            
            # Encode position and label in filename
            name_only = os.path.splitext(os.path.basename(image_path))
            tile_filename = (
                f"{game_name}_{name_only}"
                f"_tile_row{r}_column{c}_class{board[r, c]}.png"
            )
            tile.save(os.path.join(output_folder, tile_filename))
```

</div>

### Key Design Choices

<div class="bs-cards" markdown="block">

<div class="bs-card" markdown="block">

#### Overlap (0.7)

We enlarge each crop by **70%** relative to the cell stride. Each tile therefore contains:
- The full square it represents
- Context from neighboring squares (e.g., knights on borders, hands at edges)

This empirically improved classification vs strict non-overlapping crops.

</div>

<div class="bs-card" markdown="block">

#### Zero-Padding

With `zero_padding=True`, we *do not* clamp the crop box to image bounds. PIL automatically pads with black pixels outside the image. This:
- Ensures every tile has exactly 224×224 dimensions
- Preserves geometric layout consistency
- Prevents edge distortion from clamping

</div>

<div class="bs-card" markdown="block">

#### Encoded Filenames

Filenames encode complete metadata:
- `game_name` (canonical, e.g., `game4`)
- Original frame name
- Tile coordinates: `row{r}_column{c}` (0–7)
- **Label**: `_class{board[r, c]}`

Later, `build_dataset.py` uses regex `_class(\d+)` to recover labels and tile prefixes to derive `board_id`s.

</div>

</div>

{: .decision }
**Things we tried:** We experimented with edge detection for automatic board detection, but it detected spurious lines beyond chessboard boundaries. Coordinate-based slicing with known board dimensions proved more reliable.

---

## Dataset Construction: `build_dataset.py`

### Configuration via YAML

The script auto-detects a config file in priority order:

```python
CONFIG_CANDIDATES = [
    "preprocessing/dataset_config.yaml",
    "preprocessing/dataset_config.yml",
    "preprocessing/dataset_config.example.yaml",
]
```

<div markdown="block">

**Example config** (`dataset_config.example.yaml`):

```yaml
data_root: ./data
raw_tiles_dir: ./data/preprocessed_data

tile_size: 
tile_overlap: 0.7
zero_padding: true

embedding_dir: ./embeddings
embedding_ext: .npy

split:
  train: ["game5", "game6", "game7"]
  val:   ["game2", "game8", "game9", "game10"]
  test:  ["game4", "game11", "game12"]

seed: 42
```

</div>

{: .highlight }
> **Flexibility:** You can specify exact game-to-split mappings (as above) or use percentage splits. The builder respects your configuration while enforcing game-level splitting to prevent leakage.

### Game Discovery and Canonical IDs

```python
def discover_games(data_root: Path):
    games = []
    for gdir in sorted(data_root.iterdir()):
        if not gdir.is_dir():
            continue
        
        # Find images directory
        images = gdir / "images"
        if not images.exists():
            images = gdir / "tagged_images"
        
        # Find CSV files (prefer gt.csv)
        csvs = list(gdir.glob("gt.csv")) or sorted(gdir.glob("*.csv"))
        
        games.append((gdir.name, csvs, images))
```

Games may be stored as `game2_per_frame/`, `game11_per_frame/`, etc. To stabilize tile prefixes:

```python
def canonical_game_name(game_dir_name: str) -> str:
    if game_dir_name.endswith("_per_frame"):
        return game_dir_name[:-len("_per_frame")]
    return game_dir_name
```

This ensures tiles are named `game2_*` consistently.

### Generating Tiles

```python
def generate_tiles(games, tile_size, overlap, zero_padding):
    RAW_TILES_DIR.mkdir(parents=True, exist_ok=True)
    
    for game_dir_name, csv_path, img_dir in games:
        game = canonical_game_name(game_dir_name)
        pairs = pair_images_with_fens(str(csv_path), str(img_dir))
        
        for img, fen in pairs:
            board = fen_to_board_int(fen)
            slice_image_with_coordinates(
                game, img, str(RAW_TILES_DIR), board,
                overlap_percent=overlap,
                final_size=tile_size,
                zero_padding=zero_padding,
            )
```

### Marking OOD Tiles

OOD tiles (hands, foreign objects) are pre-rendered into `data/hands/`. We mark them by **filename equality**:

```python
def collect_hands() -> set:
    return {p.name for p in HANDS_DIR.rglob("*.png")}

def gather_tiles():
    hand_names = collect_hands()
    tiles = []
    
    for p in RAW_TILES_DIR.rglob("*.png"):
        label = extract_label(p)  # From _class(\d+) in filename
        
        if label not in PIECE_LABELS:
            continue
        
        is_ood = p.name in hand_names
        board_id = board_id_from_tile(p)  # Prefix before "_tile"
        
        tiles.append({
            "image": str(p),
            "label": int(label),
            "is_ood": bool(is_ood),
            "board_id": board_id,
        })
```

`PIECE_LABELS` defines the shared integer-to-name mapping:

```python
PIECE_LABELS = {
    0: "empty",
    1: "white_pawn",   11: "black_pawn",
    2: "white_knight", 12: "black_knight",
    3: "white_bishop", 13: "black_bishop",
    4: "white_rook",   14: "black_rook",
    5: "white_queen",  15: "black_queen",
    6: "white_king",   16: "black_king",
}
```

---

## Splitting into Train/Val/Test

Tiles are **grouped at the board level**, then split at the **game level** to avoid leakage:

```python
def split_tiles(tiles: List[Dict]) -> Dict[str, List[str]]:
    # Group tiles by board_id
    boards = defaultdict(list)
    for t in tiles:
        boards[t["board_id"]].append(t)
    
    # Map boards to games (board_id = "game4_frame_000856")
    game_to_board_ids = defaultdict(list)
    for bid in boards.keys():
        game_id = bid.split("_")
        game_to_board_ids[game_id].append(bid)
    
    # Assign boards by game membership
    splits = {"train": [], "val": [], "test": []}
    for game_id, bids in game_to_board_ids.items():
        # Determine destination split from config
        dest = determine_split(game_id)  # Uses YAML config
        
        for b in bids:
            splits[dest].extend(t["image"] for t in boards[b])
    
    return splits
```

{: .warning }
We always split **by game**, never by individual tiles. All tiles from the same board (and game) go into a single split, which is crucial for reliable generalization metrics.

---

## Writing Manifest and CSVs

Once tiles and splits are ready, we build a portable manifest:

```python
manifest = {
    "config": {
        "path_root": str(ROOT),
        "tile_size": tile_size,
        "overlap": overlap,
        "zero_padding": zero_padding,
    },
    "classes": PIECE_LABELS,
    "splits": {"train": [], "val": [], "test": []},
}

# Relativize paths for portability
def relativize(sample: Dict) -> Dict:
    out = dict(sample)
    out["image"] = os.path.relpath(out["image"], ROOT)
    return out

for split_name, imgs in splits.items():
    for img in imgs:
        manifest["splits"][split_name].append(relativize(by_path[img]))

MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
```

CSV files are written with columns matching `load_dataset.py` expectations:

```python
with (SPLITS_DIR / f"{split_name}.csv").open("w") as f:
    f.write("image,label,is_ood,board_id,embedding\n")
    for r in rows:
        emb = r.get("embedding") or ""
        f.write(f"{r['image']},{r['label']},{int(r['is_ood'])},"
                f"{r['board_id']},{emb}\n")
```

---

## Training Dataloader and Class Weighting

### WeightedRandomSampler Setup

For training, we apply **inverse frequency weighting** to balance the empty-dominant dataset:

```python
# Load labels from train.csv
labels = dataset.df['label'].copy()

# Optionally treat OOD as separate class 17
if consider_ood_as_class:
    is_ood_mask = dataset.df['is_ood'].astype(bool)
    labels[is_ood_mask] = 17

# Compute inverse frequency weights
class_counts = pd.Series(labels).value_counts()
class_weights = 1.0 / class_counts
class_weights_dict = class_weights.to_dict()

# Assign weight to each sample
sample_weights = torch.DoubleTensor([
    class_weights_dict.get(label, 0) for label in labels
])

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)
```

This produces an **almost uniform sampling distribution** across classes during training, despite the raw 66% empty tile imbalance.

### Data Augmentation

Training transforms apply aggressive augmentation:

```python
jittered_rotation = transforms.RandomChoice([
    transforms.RandomRotation((-5, 5)),
    transforms.RandomRotation((85, 95)),
    transforms.RandomRotation((175, 185)),
    transforms.RandomRotation((265, 275)),
])

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    jittered_rotation,
    transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                          saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
])
```

Combined with the sampler, this makes training robust to class imbalance, lighting variations, and board orientations.

---

## How to Reproduce

1. **Edit the config**

   ```bash
   cp preprocessing/dataset_config.example.yaml preprocessing/dataset_config.yaml
   # Edit splits, tile_size, overlap, etc.
   ```

2. **Run the builder**

   ```bash
   python preprocessing/build_dataset.py
   ```

   This regenerates:
   - `data/preprocessed_data/` (all tiles)
   - `data/splits/{train,val,test}.csv`
   - `data/dataset_manifest.json`

3. **Inspect stats**

   Run the stats script to refresh `stats.json` and plots, then reload this page to see updated distributions.

4. **Train/evaluate**

   Use `load_dataset.py` (`ChessTilesCSV` + `get_train_dataloader`) to feed splits into your model with consistent transforms and class weighting.

{% include preprocessing_stats.md %}
