# Modular Embedding Architecture - Complete Guide

## Overview

The embedding module has been refactored for **separation of concerns**, allowing seamless switching between different embedding models (Qwen, DINO-v2, and custom models).

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│           EmbeddingModel (Abstract Base)                │
│  (embedding_base.py)                                    │
│  - extract_embedding(image) → tensor                    │
│  - extract_batch_embeddings(images) → tensor            │
│  - get_embedding_dim() → int                            │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┬──────────────────┐
        │                     │                  │
   ┌────▼─────┐      ┌────────▼────────┐   ┌───▼─────┐
   │  Qwen    │      │   DINO-v2       │   │ Custom  │
   │(main.py) │      │  (dinov2.py)    │   │ Models  │
   │ 2048 dim │      │  384-1536 dim   │   │         │
   └────┬─────┘      └────────┬────────┘   └────┬────┘
        │                     │                  │
        └─────────────────────┼──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  FENClassifier    │
                    │ (classifier.py)   │
                    └────────┬──────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
           ┌────▼────┐  ┌────▼────┐  ┌──▼─────┐
           │   KNN   │  │Mahalano-│  │   OOD   │
           │         │  │  bis    │  │Detection│
           └─────────┘  └─────────┘  └─────────┘
                       │
                    ┌──▼──────────┐
                    │  FineTuner  │
                    │(fine_tune.py│
                    └─────────────┘
```

---

## Quick Start

### 1. Using QwenVisionEmbedding (Default)

```python
from classifier import FENClassifier

# Use default Qwen embedding
classifier = FENClassifier()

# Add FEN positions
classifier.add_fen_from_image(fen_string, board_image)
classifier.build_index()

# Predict
predicted_fen, confidence = classifier.predict_knn(tile_embeddings)
```

### 2. Using DINO-v2 Embedding

```python
from classifier import FENClassifier
from dinov2 import DINOv2Embedding

# Create DINO-v2 embedding model
dinov2 = DINOv2Embedding(model_size="base")

# Use with classifier
classifier = FENClassifier(embedding_extractor=dinov2)

# Rest is the same!
classifier.add_fen_from_image(fen_string, board_image)
classifier.build_index()
predicted_fen, confidence = classifier.predict_knn(tile_embeddings)
```

### 3. Fine-tuning with Different Models

```python
from fine_tune import FineTuner
from main import QwenVisionEmbedding
from dinov2 import DINOv2Embedding

# Fine-tune with Qwen
fine_tuner = FineTuner(embedding_model=QwenVisionEmbedding())
fine_tuner.train_batch(batch)

# Fine-tune with DINO-v2
dinov2 = DINOv2Embedding(model_size="base")
fine_tuner = FineTuner(embedding_model=dinov2)
fine_tuner.train_batch(batch)
```

---

## New Files

| File | Purpose |
|------|---------|
| `embedding_base.py` | Abstract `EmbeddingModel` interface |
| `dinov2.py` | DINO-v2 embedding implementation |
| Updated `main.py` | QwenVisionEmbedding now implements `EmbeddingModel` |
| Updated `classifier.py` | Works with any `EmbeddingModel` |
| Updated `fine_tune.py` | Works with any `EmbeddingModel` |

---

## Embedding Models

### QwenVisionEmbedding (main.py)

**Features:**
- Vision Language Model (VLM)
- 2048-dimensional embeddings
- 4-bit quantized (~2GB VRAM)
- Excellent for visual context understanding

**Usage:**
```python
from main import QwenVisionEmbedding

qwen = QwenVisionEmbedding(model_name="Qwen/Qwen3-VL-2B-Instruct")
embedding = qwen.extract_embedding(image)  # [2048]
embeddings = qwen.extract_batch_embeddings(images)  # [N, 2048]
dim = qwen.get_embedding_dim()  # 2048
```

### DINOv2Embedding (dinov2.py)

**Features:**
- Self-supervised vision model
- Multiple sizes: small (384), base (768), large (1024), giant (1536)
- Faster than Qwen
- Excellent for fine-grained discrimination

**Installation:**
```bash
pip install timm
```

**Usage:**
```python
from dinov2 import DINOv2Embedding

# Small (fast, efficient)
dinov2_small = DINOv2Embedding(model_size="small")      # 384 dims
# Base (balanced)
dinov2_base = DINOv2Embedding(model_size="base")        # 768 dims
# Large (powerful)
dinov2_large = DINOv2Embedding(model_size="large")      # 1024 dims
# Giant (maximum)
dinov2_giant = DINOv2Embedding(model_size="giant")      # 1536 dims

embedding = dinov2_base.extract_embedding(image)        # [768]
embeddings = dinov2_base.extract_batch_embeddings(imgs) # [N, 768]
dim = dinov2_base.get_embedding_dim()                   # 768
```

---

## FENClassifier

The classifier works with **any** embedding model implementing `EmbeddingModel`.

### Supported Methods

```python
classifier = FENClassifier(embedding_extractor=dinov2)

# Method 1: K-Nearest Neighbors
predicted_fen, confidence = classifier.predict_knn(tile_embeddings, k=3)

# Method 2: Mahalanobis Distance
predicted_fen, confidence = classifier.predict_mahalanobis(tile_embeddings, k=3)

# Method 3: OOD Detection
predicted_fen, confidence, is_ood = classifier.predict_with_ood(
    tile_embeddings, threshold=0.5
)

# Method 4: From image directly
predicted_fen, confidence = classifier.predict_from_image(
    board_image, method="knn", k=3
)
```

### Adaptive Classifier Head

Automatically adapts to any embedding dimension:

```python
# Qwen (2048) → 1024 → 13 classes
classifier1 = FENClassifier(embedding_extractor=QwenVisionEmbedding())

# DINO-v2 small (384) → 192 → 13 classes
classifier2 = FENClassifier(embedding_extractor=DINOv2Embedding("small"))

# DINO-v2 large (1024) → 512 → 13 classes
classifier3 = FENClassifier(embedding_extractor=DINOv2Embedding("large"))
```

---

## FineTuner

Fine-tune any embedding model on chess tiles.

### Basic Fine-tuning

```python
from fine_tune import FineTuner
from dinov2 import DINOv2Embedding

# Create fine-tuner with DINO-v2
dinov2 = DINOv2Embedding(model_size="base")
fine_tuner = FineTuner(embedding_model=dinov2)

# Train on batch
loss = fine_tuner.train_batch(batch)

# Evaluate
loss, balanced_acc, f1 = fine_tuner.evaluate_batch(batch)
```

### Full Training Loop

```python
from fine_tune import train_fine_tuning
from dinov2 import DINOv2Embedding

# Train with DINO-v2
dinov2 = DINOv2Embedding(model_size="base")
fine_tuner = train_fine_tuning(
    splits_dir="data/splits",
    embedding_model=dinov2,
    path_root="data",
    epochs=3,
    batch_size=4,
    use_val=True
)

# Model automatically saved to: chess_finetuned.pt
```

---

## Embedding Comparison

| Model | Dimensions | VRAM | Speed | Self-Supervised | Best For |
|-------|-----------|------|-------|-----------------|----------|
| Qwen 2B | 2048 | ~2GB | Medium | No | General visual understanding |
| DINO-v2 Small | 384 | ~1GB | Fast | Yes | Mobile/Edge, quick iteration |
| DINO-v2 Base | 768 | ~1GB | Medium | Yes | Balanced performance |
| DINO-v2 Large | 1024 | ~2GB | Medium | Yes | High-quality embeddings |
| DINO-v2 Giant | 1536 | ~3GB | Slow | Yes | Maximum accuracy |

---

## Custom Embedding Models

Implement the `EmbeddingModel` interface:

```python
from embedding_base import EmbeddingModel
import torch
from PIL import Image
from typing import List

class MyCustomEmbedding(EmbeddingModel):
    def __init__(self):
        self.model = ...
        self.device = torch.device("cuda")
    
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        # Return shape: [embedding_dim], on CPU
        return embedding
    
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        # Return shape: [batch_size, embedding_dim], on CPU
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return 768  # Your embedding dimension
```

Then use it like any other model:

```python
custom = MyCustomEmbedding()
classifier = FENClassifier(embedding_extractor=custom)
fine_tuner = FineTuner(embedding_model=custom)
```

---

## Examples

### Example 1: Compare Embeddings

```python
from main import QwenVisionEmbedding
from dinov2 import DINOv2Embedding
from PIL import Image

image = Image.open("chess_board.jpg")

qwen = QwenVisionEmbedding()
dinov2_base = DINOv2Embedding("base")
dinov2_small = DINOv2Embedding("small")

qwen_emb = qwen.extract_embedding(image)
dinov2_base_emb = dinov2_base.extract_embedding(image)
dinov2_small_emb = dinov2_small.extract_embedding(image)

print(f"Qwen: {qwen_emb.shape}")           # [2048]
print(f"DINO-v2 Base: {dinov2_base_emb.shape}")     # [768]
print(f"DINO-v2 Small: {dinov2_small_emb.shape}")   # [384]
```

### Example 2: Run Codacy Analysis

After making changes, run:

```bash
pip install timm
```

Then test both models:

```python
from integration_example import main
main()
```

### Example 3: Use Fine-tuned Model

```python
from fine_tune import FineTuner
from classifier import FENClassifier
from dinov2 import DINOv2Embedding

# Train
dinov2 = DINOv2Embedding(model_size="base")
fine_tuner = FineTuner(embedding_model=dinov2)
fine_tuner.train_batch(batch)
fine_tuner.save("model.pt")

# Use for classification
fine_tuner.load("model.pt")
classifier = FENClassifier(embedding_extractor=fine_tuner)
predicted_fen, confidence = classifier.predict_knn(tile_embeddings)
```

---

## Benefits of the New Architecture

✅ **Separation of Concerns** - Embedding models are independent  
✅ **Easy Switching** - Change models with one parameter  
✅ **Scalability** - Add new models without changing existing code  
✅ **Flexibility** - Different embedding dimensions supported  
✅ **Consistency** - All models implement the same interface  
✅ **Research-Ready** - Easy to benchmark different models  

---

## Migration from Old Code

**Before:**
```python
from classifier import FENClassifier
classifier = FENClassifier()  # Implicit Qwen
```

**After (Explicit):**
```python
from classifier import FENClassifier
from main import QwenVisionEmbedding
classifier = FENClassifier(embedding_extractor=QwenVisionEmbedding())
```

**Or use DINO-v2:**
```python
from classifier import FENClassifier
from dinov2 import DINOv2Embedding
classifier = FENClassifier(embedding_extractor=DINOv2Embedding("base"))
```

**Backward compatibility maintained:**
- `QwenFineTuner` is still available as an alias to `FineTuner`
- Default behavior unchanged (still uses Qwen)
