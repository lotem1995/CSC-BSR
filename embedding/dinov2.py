"""
DINO-v2 Vision Embedding Model

Implements the EmbeddingModel interface using Facebook's DINO-v2 model.
DINO-v2 is a self-supervised vision model that provides high-quality embeddings
without requiring fine-tuning.

Key advantages for chess:
- Self-supervised (no labeled data needed for pre-training)
- Excellent for fine-grained visual discrimination
- Lightweight and fast compared to VLMs
- Output embeddings are more compact (768-1024 dims)
"""

import torch
from typing import List
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from embedding_base import EmbeddingModel


SIZES = {
    "small": "vit_small_patch14_dinov2.lvd142m",
    "base":  "vit_base_patch14_dinov2.lvd142m",
    "large": "vit_large_patch14_dinov2.lvd142m",
}

class DINOv2Embedding(EmbeddingModel):
    """
    DINO-v2 embedding model for chess tile analysis.
    
    Available sizes:
    - "facebook/dinov2-small": 384 dimensions, ~21M params
    - "facebook/dinov2-base": 768 dimensions, ~86M params
    - "facebook/dinov2-large": 1024 dimensions, ~300M params
    - "facebook/dinov2-giant": 1536 dimensions, ~1.1B params
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize DINO-v2 embedding model.
        
        Args:
            model_size: One of "small", "base", "large", "giant"
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        
        if model_size not in SIZES:
            raise ValueError(f"Unknown model_size: {model_size}. Must be one of {list(SIZES.keys())}")
        
        model_name = SIZES[model_size]
        
        print(f"Loading DINO-v2 {model_size} model ({model_name})...")
        
        # Load DINO-v2 using timm
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for DINO-v2. "
                "Install with: pip install timm"
            )
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head, just use embeddings
        ).to(self.device)
        self.model.eval()
        
        # Get the data config for preprocessing
        self.data_config = timm.data.resolve_data_config(
            self.model.pretrained_cfg,
            model=self.model
        )
        
        # Create preprocessing pipeline
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.data_config["input_size"][-1], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.data_config["input_size"][-1]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(self.data_config["mean"]),
                std=torch.tensor(self.data_config["std"])
            )
        ])
        
        print(f"✓ DINO-v2 {model_size} loaded. Embedding dimension: {self.get_embedding_dim()}")
    
    def get_embedding_dim(self) -> int:
        """Return the dimension of DINO-v2 embeddings"""
        size_to_dim = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        return size_to_dim[self.model_size]
    
    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract embedding from a single image using DINO-v2.
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            torch.Tensor of shape [embedding_dim], on CPU
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Preprocess image
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract embedding (forward pass returns the token embeddings)
        embedding = self.model(x)
        
        return embedding.squeeze(0).cpu()
    
    @torch.no_grad()
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract embeddings from a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            torch.Tensor of shape [batch_size, embedding_dim], on CPU
        """
        batch_tensors = []
        
        for image in images:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess
            x = self.transform(image).unsqueeze(0)
            batch_tensors.append(x)
        
        # Stack batch
        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        
        # Extract embeddings
        embeddings = self.model(batch)
        
        return embeddings.cpu()


if __name__ == "__main__":
    try:
        # Test with base model
        embedder = DINOv2Embedding(model_size="base")
        print(f"Initialized: {embedder}")
        
        # Create a dummy image
        dummy_img = Image.new("RGB", (224, 224), color="red")
        embedding = embedder.extract_embedding(dummy_img)
        print(f"Single embedding shape: {embedding.shape}")
        
        # Test batch
        embeddings = embedder.extract_batch_embeddings([dummy_img, dummy_img])
        print(f"Batch embeddings shape: {embeddings.shape}")
        
        print("✓ DINO-v2 embedding test successful!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
