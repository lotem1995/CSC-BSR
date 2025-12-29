# Quick Reference: Modular Embedding Architecture

## File Overview

```
embadding/
├── embedding_base.py          ← NEW: Abstract interface
├── dinov2.py                  ← NEW: DINO-v2 implementation
├── main.py                    ← MODIFIED: Qwen + EmbeddingModel
├── classifier.py              ← MODIFIED: Any EmbeddingModel
├── fine_tune.py               ← MODIFIED: Any EmbeddingModel
├── integration_example.py      ← UPDATED: New examples
├── MODULAR_ARCHITECTURE.md    ← NEW: Complete guide
├── REFACTORING_SUMMARY.md     ← NEW: What changed
└── INTEGRATION_GUIDE.md       ← EXISTING: Integration notes
```

---

## API Reference

### EmbeddingModel Interface

```python
class EmbeddingModel(ABC):
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """Single image → embedding vector"""
        
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Batch of images → embedding vectors"""
        
    def get_embedding_dim(self) -> int:
        """Return embedding dimension"""
```

### QwenVisionEmbedding

```python
from main import QwenVisionEmbedding

model = QwenVisionEmbedding()
emb = model.extract_embedding(image)      # [2048]
embs = model.extract_batch_embeddings([image])  # [N, 2048]
dim = model.get_embedding_dim()           # 2048
```

### DINOv2Embedding

```python
from dinov2 import DINOv2Embedding

# Small: 384 dims, fastest
model = DINOv2Embedding(model_size="small")

# Base: 768 dims, balanced
model = DINOv2Embedding(model_size="base")

# Large: 1024 dims, powerful
model = DINOv2Embedding(model_size="large")

# Giant: 1536 dims, maximum
model = DINOv2Embedding(model_size="giant")

emb = model.extract_embedding(image)           # [384/768/1024/1536]
embs = model.extract_batch_embeddings([image]) # [N, embedding_dim]
dim = model.get_embedding_dim()                # 384/768/1024/1536
```

### FENClassifier

```python
from classifier import FENClassifier
from dinov2 import DINOv2Embedding

# With Qwen (default)
clf = FENClassifier()

# With DINO-v2
dinov2 = DINOv2Embedding(model_size="base")
clf = FENClassifier(embedding_extractor=dinov2)

# Methods (unchanged)
clf.add_fen_from_image(fen, board_image)
clf.build_index()
fen, conf = clf.predict_knn(embeddings, k=3)
fen, conf = clf.predict_mahalanobis(embeddings, k=3)
fen, conf, ood = clf.predict_with_ood(embeddings, threshold=0.5)
```

### FineTuner

```python
from fine_tune import FineTuner
from dinov2 import DINOv2Embedding

# With Qwen
tuner = FineTuner()

# With DINO-v2
dinov2 = DINOv2Embedding(model_size="base")
tuner = FineTuner(embedding_model=dinov2)

# Methods
loss = tuner.train_batch(batch)
loss, acc, f1 = tuner.evaluate_batch(batch)
tuner.save("model.pt")
tuner.load("model.pt")
```

---

## Common Tasks

### Compare Two Models

```python
from main import QwenVisionEmbedding
from dinov2 import DINOv2Embedding
from PIL import Image

image = Image.open("chess.jpg")

# Qwen
qwen = QwenVisionEmbedding()
qwen_emb = qwen.extract_embedding(image)

# DINO-v2
dinov2 = DINOv2Embedding(model_size="base")
dinov2_emb = dinov2.extract_embedding(image)

print(f"Qwen: {qwen_emb.shape}")          # [2048]
print(f"DINO-v2: {dinov2_emb.shape}")    # [768]
```

### Use DINO-v2 for Classification

```python
from classifier import FENClassifier
from dinov2 import DINOv2Embedding

dinov2 = DINOv2Embedding(model_size="base")
classifier = FENClassifier(embedding_extractor=dinov2)

# Now use classifier normally
classifier.add_fen_from_image(fen, board_image)
classifier.build_index()
predicted, confidence = classifier.predict_knn(tile_embeddings)
```

### Fine-tune DINO-v2

```python
from fine_tune import train_fine_tuning
from dinov2 import DINOv2Embedding

dinov2 = DINOv2Embedding(model_size="base")
fine_tuner = train_fine_tuning(
    embedding_model=dinov2,
    epochs=3,
    batch_size=4
)
```

### Batch Extract Embeddings

```python
from dinov2 import DINOv2Embedding
from PIL import Image

dinov2 = DINOv2Embedding(model_size="base")
images = [Image.new("RGB", (224, 224), color="red") for _ in range(64)]

embeddings = dinov2.extract_batch_embeddings(images)
print(embeddings.shape)  # [64, 768]
```

---

## Dimension Reference

| Model | Dimensions |
|-------|-----------|
| Qwen | 2048 |
| DINO-v2 Small | 384 |
| DINO-v2 Base | 768 |
| DINO-v2 Large | 1024 |
| DINO-v2 Giant | 1536 |

---

## Installation

```bash
# For DINO-v2
pip install timm

# For DINO-v2 with specific version
pip install timm==0.9.8
```

---

## Backward Compatibility

Old code still works:

```python
# These still work (unchanged behavior)
from classifier import FENClassifier
classifier = FENClassifier()  # Uses Qwen

from fine_tune import QwenFineTuner
tuner = QwenFineTuner()  # Alias to FineTuner
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'timm'" | `pip install timm` |
| "CUDA out of memory" | Use smaller DINO model (`small` instead of `large`) |
| "Embedding dimensions don't match" | Ensure same model for training & inference |
| "Model loading failed" | Check saved model has correct embedding model name |

---

## Testing

```python
# Test all examples
from integration_example import main
main()

# Test individual models
from dinov2 import DINOv2Embedding
dinov2 = DINOv2Embedding(model_size="base")
print(dinov2)  # Should print model info
```

---

## Documentation

- **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)** - Complete guide
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - What changed
- **[integration_example.py](integration_example.py)** - Working examples
