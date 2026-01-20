---
title: Embedding Models and Fine-Tuning
nav_order: 3
---

<div class="bs-hero" markdown="block">

## Tile Embedding

The next step in the architecture is a transformer-based encoding of tile images. Instead of creating an embedding from scratch, we opted to fine-tune an existing transformer for this task. We evaluated several models, including LoRA fine-tuning of Qwen/Qwen3-VL-2B-Instruct and DINOv2 fine-tuning, and ultimately selected a **fine-tuned DINOv2-small model** for tile embedding. This model takes a single chess tile image as input and outputs a **384-dimensional embedding vector** representing the visual content and piece identity.

For fine-tuning DINOv2, we tested two strategies: training only the classification head (linear probing) and fine-tuning the entire backbone. We found that **backbone fine-tuning** yielded superior results with **95.78% balanced accuracy**. While we initially used a classification head as a training objective to provide supervision, we discard it during inference and use only the learned backbone embeddings, which capture chess-specific visual patterns.

</div>

## Understanding the Selected Embedding: DINOv2

<div class="bs-cards" markdown="block">

<div class="bs-card" markdown="block">

### Why DINOv2 for Chess?

While generic vision models provide strong foundational representations, they lack chess-specific knowledge. DINOv2 (DINO Vision Transformer v2) emerged as the optimal choice for this task because:

1. **Self-Supervised Pre-Training**: DINOv2 uses self-supervised learning on a massive corpus of unlabeled images (1.2B images), learning robust visual features without requiring labeled data. This translates to exceptional feature quality even in specialized domains like chess.

2. **Fine-Grained Visual Discrimination**: Unlike models trained primarily for object classification, DINOv2 excels at distinguishing subtle visual differences. For chess tiles, this means it can discriminate between piece types, colors, board textures, and lighting conditions.

3. **Compact yet Expressive Embeddings**: The DINOv2-small variant we use produces 384-dimensional embeddings (base produces 768-dim and large 1024-dim), which is more computationally efficient than VLMs like Qwen/Qwen3-VL-2B-Instruct while remaining sufficiently expressive.

4. **Fast Inference**: DINOv2 is a pure vision model without language understanding overhead, making it suitable for real-time chess board analysis.

</div>

<div class="bs-card" markdown="block">

### Comparison with Alternatives

| Model | Strategy | Balanced Acc | Val Loss |
| --- | --- | --- | --- |
| **DINOv2** | Backbone | **95.78%** | **0.1720** |
| DINOv2 | Head-only | 75.24% | 0.7692 |
| Qwen/Qwen3-VL-2B-Instruct | LoRA | 77.82% | 0.7651 |
| Qwen/Qwen3-VL-2B-Instruct | Head-only | 68.89% | 0.9517 |

{: .highlight }
> **The decision**: DINOv2's self-supervised pre-training makes it uniquely suited for fine-grained visual discrimination in chess tile analysis. Once fine-tuned, it achieves near-perfect accuracy while remaining computationally efficient.

</div>

</div>

### Inference Pipeline

<div markdown="block">

The inference process transforms a chess tile image into a rich semantic embedding:

```
Input: Chess Tile Image (e.g., 224×224 RGB)
    ↓
[Preprocessing]
  - Resize to 224×224 (if needed)
  - Center crop
  - Normalize: (RGB - mean) / std
    ↓
[DINOv2 Vision Transformer]
  - Patch Embedding: Split image into 16×16 pixel patches (196 tokens)
  - Transformer Blocks: 12 layers of multi-head attention
  - Learned [CLS] token + Position Embeddings
  - Output: 1024-dimensional vector per image
    ↓
Output: Embedding (1024-dim)
  → Represents semantic content of the tile
  → Ready for downstream classification or similarity search
```

**Technical Details**:
- **Patch-based Architecture**: The Vision Transformer (ViT) divides the image into non-overlapping 16×16 patches, converting it into a sequence of patch embeddings. This allows the model to learn global context via self-attention.
- **No Classification Head During Inference**: The classification head (used only during training) is discarded. Instead, we use the learned feature representation as the tile embedding.
- **Normalization**: Embeddings are typically L2-normalized for use in cosine-similarity searches, though this is handled automatically in most frameworks.

</div>

## Fine-Tuning Vision Models

While generic vision models provide strong foundations, they aren't inherently "chess-aware." We bridge this gap by fine-tuning backbones like **DINOv2** and **Qwen/Qwen3-VL-2B-Instruct** to recognize specific chess pieces and tile states with high precision.

<div class="bs-cards" markdown="block">

<div class="bs-card" markdown="block">

### Fine-Tuning Strategies

1. **Head-Only (Linear Probing)**: Keep the vision backbone frozen and only train a 13-class MLP classifier on top of the extracted embeddings. Most efficient approach but limits adaptability.

2. **LoRA (Low-Rank Adaptation)**: Used specifically for **Qwen/Qwen3-VL-2B-Instruct**, this method adapts only 1%–2% of the model's parameters through low-rank decomposition. Uses **10x less VRAM** and is **10x faster** than full fine-tuning.

3. **Backbone Fine-Tuning**: Unfreeze the entire vision model and train it end-to-end with the classifier. Allows learning of chess-specific visual features but requires more GPU memory (~8-11GB).

</div>

<div class="bs-card" markdown="block">

### Model Comparison & Results

Results from training for 1 epoch with batch size 2 on our cluster:

| Model | Strategy | Balanced Acc | F1 Score | Val Loss |
| --- | --- | --- | --- | --- |
| **DinoV2 (Backbone)** | backbone | **95.78%** | **95.63%** | **0.1720** |
| DinoV2 (Head Only) | head-only | 75.24% | 74.09% | 0.7692 |
| Qwen/Qwen3-VL-2B-Instruct (LoRA) | lora | 77.82% | 78.16% | 0.7651 |
| Qwen/Qwen3-VL-2B-Instruct | head-only | 68.89% | 69.46% | 0.9517 |

{: .highlight }
> **Why DINOv2 Backbone Wins**: While Qwen/Qwen3-VL-2B-Instruct is powerful, DINOv2's self-supervised pre-training excels at fine-grained visual discrimination. The 20+ percentage point advantage validates this approach.

</div>

</div>

## Deep Dive: Implementation Details

<div markdown="block">

### Architecture Overview

The fine-tuning pipeline implements a unified, adapter-based design supporting multiple embedding models through a shared interface:

```
┌─────────────────────────────────────────────────┐
│ EmbeddingModel (Abstract Base Class)            │
│ ├─ extract_embedding()                          │
│ ├─ extract_batch_embeddings()                   │
│ └─ get_embedding_dim()                          │
└────────┬────────────────────────────────────────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
DINOv2Embedding              QwenVisionEmbedding
(dinov2.py)                  (qwen3.py)
    │                                 │
    └────────────────┬────────────────┘
                     │
                ┌────▼─────────────────────┐
                │ FineTuner Classes        │
                ├─ FineTuner               │
                ├─ QwenLoRAClassifierTrainer
                └─ DINOBackboneFineTuner
```

This abstraction allows seamless switching between embedding models while reusing fine-tuning logic.

</div>

### The EmbeddingModel Interface

All embedding models implement a common interface defined in `embedding_base.py`:

```python
class EmbeddingModel(ABC):
    @abstractmethod
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract embedding from a single image."""
        pass
    
    @abstractmethod
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract embeddings from multiple images efficiently."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        pass
```
{: .decision }
**Why this design?** The interface allows fine-tuning code to remain model-agnostic. A single `FineTuner` class can work with any embedding model without modification.

### DINOv2 Implementation (dinov2.py)

**Model Loading**:
- Uses Facebook's `timm` library to load pre-trained DINOv2 models
- Supports multiple sizes: small (384-dim), base (768-dim), large (1024-dim), giant (1536-dim)
- Removes the classification head (`num_classes=0`) to use the backbone output directly

**Preprocessing Pipeline**:
```python
transforms.Compose([
    transforms.Resize(image_size, interpolation=BICUBIC),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_values, std=std_values)
])
```

**Batch Extraction**:
- Processes images one-by-one via the transform pipeline
- Concatenates into a batch tensor
- Feeds through the model in a single forward pass
- Returns CPU-resident embeddings to free GPU memory

### Qwen/Qwen3-VL-2B-Instruct Implementation (qwen3.py)

**4-Bit Quantization**:
To fit Qwen/Qwen3-VL-2B-Instruct's 2B parameters into a single 8GB GPU, the implementation uses:
- `bitsandbytes` with NF4 (Normal Float 4-bit) quantization
- Double quantization for additional compression
- Compute dtype: bfloat16 (fast and low-precision)

This reduces VRAM from ~5GB (full precision) to ~2GB (4-bit).

**Direct Visual Tower Access**:
Instead of using the full chat/LLM pipeline (which is slow), the code directly calls the visual tower:

```python
# Fast path: skip LLM, go straight to visual embeddings
grid_thw = inputs.get('image_grid_thw')  # Grid layout info
visual_output = self.model.visual(inputs.pixel_values, grid_thw=grid_thw)
visual_features = visual_output[0]  # Patch embeddings
embedding = torch.mean(visual_features, dim=0)  # Global average pooling
```

This saves ~80% of inference time compared to the full LLM path.

### Fine-Tuning Process (fine_tune.py)

#### 1. Initialization

```python
class FineTuner:
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        # Load embedding model (default: Qwen)
        if embedding_model is None:
            embedding_model = QwenVisionEmbedding()
        
        self.embedding_model = embedding_model
        embedding_dim = embedding_model.get_embedding_dim()
        
        # Build adaptive classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 13)  # 13 piece classes
        )
        
        # Optimizer for the classifier
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
```
{: .decision }
**Key Design Choices**:
- The classification head is **adaptive**: it automatically scales to the embedding dimension
- **Dropout (0.2)** prevents overfitting on small chess datasets
- **AdamW** optimizer with weight decay provides stable, regularized training

#### 2. Batch Processing

**Training**:
```python
def train_batch(self, batch: Dict) -> float:
    self.classifier.train()
    
    # Get raw image tensors from batch
    image_tensors = batch["image"]  # Shape: [batch_size, 3, H, W]
    labels = batch["label"]  # Shape: [batch_size]
    
    # Convert tensors to PIL images for the embedding model
    images = []
    for img_tensor in image_tensors:
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(img_np))
    
    # Extract embeddings using the embedding model
    embeddings = self.embedding_model.extract_batch_embeddings(images)
    embeddings = embeddings.to(self.device)
    
    # Map raw piece labels (0-16 with gaps) to contiguous indices (0-12)
    labels = labels.to(self.device)
    labels = _remap_raw_piece_labels(labels)
    
    # Forward pass through classifier
    logits = self.classifier(embeddings)
    
    # Compute loss and backpropagate
    loss = self.criterion(logits, labels)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    return loss.item()
```

**Label Remapping**:
The raw dataset uses piece IDs with gaps (e.g., 0=empty, 1=white_pawn, 11=black_pawn). PyTorch's `CrossEntropyLoss` requires contiguous class indices (0-12), so we remap:

```python
_RAW_LABELS_IN_ORDER = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
_LABEL_LUT = torch.full((17,), -1, dtype=torch.long)
for _class_idx, _raw_id in enumerate(_RAW_LABELS_IN_ORDER):
    _LABEL_LUT[_raw_id] = _class_idx
```

**Evaluation**:
```python
@torch.no_grad()
def evaluate_batch(self, batch: Dict) -> Tuple[float, float, float]:
    self.classifier.eval()
    
    # ... extract embeddings and labels as in train_batch ...
    
    logits = self.classifier(embeddings)
    loss = self.criterion(logits, labels)
    preds = logits.argmax(dim=1)
    
    # Use sklearn metrics for balanced evaluation
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='weighted', zero_division=0)
    
    return loss.item(), balanced_acc, f1
```
{: .decision }
**Why balanced accuracy and F1?** Chess datasets are imbalanced (many more empty squares than rare pieces). These metrics provide meaningful evaluation beyond raw accuracy.

#### 3. Backbone Fine-Tuning (DINOBackboneFineTuner)

For DINOv2 backbone fine-tuning, we unfreeze the entire model and use **two different learning rates**:

```python
class DINOBackboneFineTuner:
    def __init__(self, dino: DINOv2Embedding):
        # ... setup classifier ...
        
        # Two learning rate groups: backbone gets smaller LR
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.parameters(), "lr": 5e-6},      # Backbone: conservative
            {"params": self.classifier.parameters(), "lr": 1e-4}  # Classifier: aggressive
        ], weight_decay=0.01)
```
{: .decision }
**Why two learning rates?**
- The backbone is already well-trained (ImageNet + self-supervised). Large updates would destroy learned features.
- The classifier head is randomly initialized and needs more aggressive updates.
- This discriminative fine-tuning technique is proven effective in transfer learning.

#### 4. LoRA Fine-Tuning (QwenLoRAClassifierTrainer)

LoRA adapts the model by injecting trainable low-rank decompositions:

```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=8,                              # Rank of the low-rank matrices
    lora_alpha=16,                    # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Adapt query and value projections
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(self.model, lora_config)
```

**What happens**: Every weight matrix $W$ is decomposed as:

$$W_{\text{new}} = W_{\text{original}} + \left(\mathbf{A} \times \mathbf{B}\right) \times \frac{\alpha}{r}$$

Where $\mathbf{A}$ and $\mathbf{B}$ are small trainable matrices (rank $r = 8$). This reduces trainable parameters from 2B to ~20M (1% of total).

### Training Loop (experiment_runner.py)

The experiment runner orchestrates multiple fine-tuning runs sequentially:

```python
def run(label, args_list):
    logger.info(f"RUN: {label}")
    proc = subprocess.run([
        sys.executable, "fine_tune.py",
        *args_list
    ], cwd=project_root)
    return proc.returncode

scenarios = [
    ("Qwen - head-only", [...]),
    ("DINO - head-only", [...]),
    ("Qwen - LoRA", [...]),
    ("DINO - backbone finetune", [...])
]

for label, cmd_args in scenarios:
    run(label, cmd_args)
```

Each scenario is independent and can run on different GPU configurations (e.g., Qwen LoRA on a 2GB GPU, DINO backbone on an 11GB GPU).

### Metrics and Logging

All experiments log structured metrics to JSON for reproducibility:

```python
metrics_file = f"metrics_{model}_{strategy}-{time}.json"

metrics = {
    "model": "dino-base",
    "strategy": "backbone",
    "epochs": 1,
    "batch_size": 2,
    "final_val_loss": 0.1720,
    "final_balanced_acc": 0.9578,
    "final_f1": 0.9563,
    "training_time_seconds": 1847
}
```

This allows automatic comparison and visualization across multiple runs.

## Quick Links

[Code: `fine_tune.py`](https://github.com/lotem1995/CSC-BSR/blob/main/embedding/fine_tune.py){: .btn .btn-outline .mr-2 }
[Code: `lorafinetune.py`](https://github.com/lotem1995/CSC-BSR/blob/main/embedding/lorafinetune.py){: .btn .btn-outline .mr-2 }
[Code: `qwen3.py`](https://github.com/lotem1995/CSC-BSR/blob/main/embedding/qwen3.py){: .btn .btn-outline .mr-2 }
[Code: `dinov2.py`](https://github.com/lotem1995/CSC-BSR/blob/main/embedding/dinov2.py){: .btn .btn-outline }

{: .repro }
**Reproducibility**: All training metrics, including validation loss and F1 scores, are automatically logged to JSON files (`metrics_{model}_{strategy}_{time}.json`) for comparison and visualization.

{% include finetuning_stats.md %}