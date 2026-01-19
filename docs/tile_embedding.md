## title: Fine-Tuning Vision Models nav_order: 3

# Fine-Tuning Vision Models

While generic vision models provide strong foundations, they aren't inherently "chess-aware." We bridge this gap by fine-tuning backbones like **DINOv2** and **Qwen3-VL** to recognize specific chess pieces and tile states with high precision.

{: .highlight }

> **Best Performer**: After testing multiple strategies, **DINOv2 (Backbone)** emerged as the champion, achieving **95.78%** balanced accuracy.

## Quick links

[Code: `fine_tune.py](https://www.google.com/search?q=%5Bhttps://github.com/lotem1995/CSC-BSR/blob/main/fine_tune.py%5D(https://github.com/lotem1995/CSC-BSR/blob/main/fine_tune.py))`{: .btn .btn-outline .mr-2 }
[Code: `lorafinetune.py](https://www.google.com/search?q=%5Bhttps://github.com/lotem1995/CSC-BSR/blob/main/lorafinetune.py%5D(https://github.com/lotem1995/CSC-BSR/blob/main/lorafinetune.py))`{: .btn .btn-outline .mr-2 }
[Code: `dinov2.py](https://www.google.com/search?q=%5Bhttps://github.com/lotem1995/CSC-BSR/blob/main/embedding/dinov2.py%5D(https://github.com/lotem1995/CSC-BSR/blob/main/embedding/dinov2.py))`{: .btn .btn-outline }

## Fine-Tuning Strategies

We implemented three distinct training strategies to balance computational cost and classification performance:

1. **Head-Only (Linear Probing):** We keep the vision backbone frozen and only train a 13-class MLP classifier on top of the extracted embeddings.
2. **LoRA (Low-Rank Adaptation):** Used specifically for **Qwen3-VL**, this method adapts only a tiny fraction (1%–2%) of the model’s parameters. It uses **10x less VRAM** and is **10x faster** than full fine-tuning while maintaining high quality.
3. **Backbone Fine-Tuning:** We unfreeze the entire vision model (e.g., DINOv2) and train it end-to-end with the classifier. This allows the model to learn chess-specific visual features but requires more GPU memory.

## Model Comparison & Results

The following results were obtained from training for 1 epoch with a batch size of 2 on our cluster:

| Model | Strategy | Balanced Acc | F1 Score | Val Loss |
| --- | --- | --- | --- | --- |
| **DinoV2 (Backbone)** | backbone | **95.78%** | **95.63%** | **0.1720** |
| DinoV2 (Head Only) | head-only | 75.24% | 74.09% | 0.7692 |
| Qwen2-VL (LoRA) | lora | 77.82% | 78.16% | 0.7651 |
| Qwen2-VL (Head Only) | head-only | 68.89% | 69.46% | 0.9517 |

{: .decision }
**Why DINOv2 Backbone?** While Qwen3-VL is a powerful VLM, the DINOv2 backbone's self-supervised pre-training is exceptionally good at fine-grained visual discrimination. When unfrozen, it adapts perfectly to the specific textures and shapes of individual chess pieces on a board.

## Implementation Details

### The Classification Head

Regardless of the backbone, we use a consistent MLP architecture for classification:

* **Input Layer**: Adapts to the embedding dimension (2048 for Qwen, 384-1024 for DINO).
* **Hidden Layer**: Linear (embedding_dim // 2) + ReLU activation.
* **Regularization**: Dropout (0.2).
* **Output Layer**: 13 units (representing 6 white pieces, 6 black pieces, and 1 empty square).

### Efficiency with LoRA

For large models like Qwen3-VL, we utilize 4-bit quantization (NF4) and double quantization via `bitsandbytes`. This reduces VRAM usage from ~5GB to approximately **2GB**, allowing us to run fine-tuning on consumer-grade GPUs or constrained cluster nodes.

{: .repro }
**Reproducibility**: All training metrics, including validation loss and F1 scores, are automatically logged to JSON files (`metrics_{model}_{strategy}_{time}.json`) for comparison and visualization.

{% include finetuning_stats.md %}