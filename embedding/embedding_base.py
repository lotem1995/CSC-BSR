"""
Abstract base class for embedding models.

This module provides a unified interface for different embedding models,
allowing seamless switching between Qwen, DINO-v2, and other vision models.
"""

from abc import ABC, abstractmethod
from typing import List
import torch
from PIL import Image


class EmbeddingModel(ABC):
    """
    Abstract base class for vision-based embedding models.
    
    All embedding models must implement:
    - extract_embedding: Single image → embedding vector
    - extract_batch_embeddings: Multiple images → embedding vectors
    - get_embedding_dim: Return the dimension of embeddings
    """
    
    @abstractmethod
    def extract_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract embedding from a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            torch.Tensor of shape [embedding_dim]
        """
        pass
    
    @abstractmethod
    def extract_batch_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract embeddings from a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            torch.Tensor of shape [batch_size, embedding_dim]
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Integer dimension
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.get_embedding_dim()})"
