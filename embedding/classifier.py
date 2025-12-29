"""
FEN Classification using Chess Tile Embeddings

Supports multiple embedding models:
1. QwenVisionEmbedding - Qwen3-VL vision model (2048-dim embeddings)
2. DINOv2Embedding - Facebook's self-supervised model (384-1536 dims)
3. Custom embedding models implementing EmbeddingModel interface

Classification methods:
1. KNN - Simple baseline (no training needed)
2. Mahalanobis Distance - Smarter KNN
3. Triplet Loss - Deep learning approach (training required)
4. OOD Detection - Know when uncertain
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import tempfile
import os
from PIL import Image
import sys

sys.path.insert(0, '/home/lotems/Documents/DL_Oren/CSC-BSR/preprocessing')
from splitting_images import slice_image_with_coordinates

sys.path.insert(0, '/home/lotems/Documents/DL_Oren/CSC-BSR/embadding')
from embedding_base import EmbeddingModel
from main import QwenVisionEmbedding
from dinov2 import DINOv2Embedding


class FENClassifier:
    """
    Classifies FEN positions using tile embeddings.
    Stores embeddings for each unique FEN and retrieves nearest match.
    
    Works with any embedding model that implements the EmbeddingModel interface.
    """
    
    def __init__(self, embedding_extractor: Optional[EmbeddingModel] = None):
        """
        Args:
            embedding_extractor: Any EmbeddingModel instance (QwenVisionEmbedding, DINOv2Embedding, etc).
                               If None, creates default QwenVisionEmbedding.
        """
        self.fen_database = {}  # fen -> list of N embeddings (N = 64 for full board)
        self.fen_labels = []    # List of unique FENs
        self.all_embeddings = []  # All embeddings (for KNN)
        self.all_fen_indices = []  # Map embeddings back to FEN
        self.scaler = StandardScaler()
        self.mahal_inv_cov = None
        
        # Embedding extractor (must implement EmbeddingModel interface)
        self.embedding_extractor = embedding_extractor
        if self.embedding_extractor is None:
            print("Initializing default QwenVisionEmbedding...")
            self.embedding_extractor = QwenVisionEmbedding()
        
        # Store embedding dimension for reference
        self.embedding_dim = self.embedding_extractor.get_embedding_dim()
        print(f"Using {self.embedding_extractor} for FEN classification")
        
    def extract_board_embeddings(self, board_image: Image.Image) -> torch.Tensor:
        """
        Extract embeddings for all 64 tiles from a chess board image.
        
        Args:
            board_image: PIL Image of full chess board
            
        Returns:
            Tensor of shape [64, 2048] with embeddings for each tile
        """
        # Save to temporary location for splitting
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_board:
            board_image.save(tmp_board.name)
            tmp_board_path = tmp_board.name
        
        # Get the base filename for tile naming
        base_filename = os.path.splitext(os.path.basename(tmp_board_path))[0]
        
        # Create temp directory for tiles
        with tempfile.TemporaryDirectory() as tmp_tiles_dir:
            # Create a dummy board array (8x8) with placeholder values
            # This is needed by slice_image_with_coordinates for filename generation
            import numpy as np
            dummy_board = np.zeros((8, 8), dtype=int)
            
            # Split board into 64 tiles
            slice_image_with_coordinates(
                image_path=tmp_board_path,
                output_folder=tmp_tiles_dir,
                board=dummy_board,  # Provide dummy board for filename generation
                overlap_percent=0.0,
                final_size=(224, 224)
            )
            
            # Load all 64 tiles in order (row by row)
            tile_images = []
            for row in range(8):
                for col in range(8):
                    # Filename format from slice_image_with_coordinates:
                    # {name_only}_tile_row{r}_column{c}_class{board[r, c]}.png
                    tile_filename = f"{base_filename}_tile_row{row}_column{col}_class{dummy_board[row, col]}.png"
                    tile_path = os.path.join(tmp_tiles_dir, tile_filename)
                    if os.path.exists(tile_path):
                        tile_images.append(Image.open(tile_path).copy())
                    else:
                        raise FileNotFoundError(f"Tile {tile_filename} not found")
            
            # Extract embeddings for all tiles
            tile_embeddings = self.embedding_extractor.extract_batch_embeddings(tile_images)
        
        # Clean up temp board image
        os.unlink(tmp_board_path)
        
        return tile_embeddings
    
    def add_fen_position(self, fen: str, tile_embeddings: torch.Tensor):
        """
        Add a FEN position and its 64 tile embeddings to the database.
        
        Args:
            fen: FEN string (e.g., "rnbqkbnr/pppppppp/...")
            tile_embeddings: Tensor of shape [64, 2048]
        """
        if fen not in self.fen_database:
            self.fen_labels.append(fen)
        
        # Store the full board embedding
        self.fen_database[fen] = tile_embeddings.float()
    
    def add_fen_from_image(self, fen: str, board_image: Image.Image):
        """
        Add a FEN position by extracting embeddings from a board image.
        
        Args:
            fen: FEN string
            board_image: PIL Image of full chess board
        """
        tile_embeddings = self.extract_board_embeddings(board_image)
        self.add_fen_position(fen, tile_embeddings)
        
    def build_index(self):
        """
        Build KNN index from all FEN positions.
        This allows fast nearest-neighbor search.
        """
        for fen in self.fen_labels:
            embeddings = self.fen_database[fen]
            # Average all 64 tile embeddings to get board-level embedding
            board_embedding = embeddings.mean(dim=0)
            self.all_embeddings.append(board_embedding)
            self.all_fen_indices.append(fen)
        
        if self.all_embeddings:
            self.all_embeddings = torch.stack(self.all_embeddings)
            
            # Normalize embeddings for KNN
            self.all_embeddings = torch.nn.functional.normalize(
                self.all_embeddings, p=2, dim=1
            )
            
            # Fit scaler for Mahalanobis distance
            embeddings_np = self.all_embeddings.cpu().numpy()
            self.scaler.fit(embeddings_np)
            
            # Compute covariance for Mahalanobis
            scaled = self.scaler.transform(embeddings_np)
            # Ledoit-Wolf shrinkage for numerical stability
            lw = LedoitWolf()
            lw.fit(scaled)
            self.mahal_inv_cov = np.linalg.inv(lw.covariance_)
    
    # ============ METHOD 1: KNN ============
    def predict_knn(self, tile_embeddings: torch.Tensor, k: int = 3) -> Tuple[str, float]:
        """
        Predict FEN using K-Nearest Neighbors with L2 normalized embeddings.
        
        Args:
            tile_embeddings: Tensor of shape [64, 2048]
            k: Number of neighbors to check (or None for adaptive)
            
        Returns:
            (predicted_fen, confidence)
        """
        if len(self.all_embeddings) == 0:
            raise ValueError("No FEN positions in database. Call add_fen_position() first.")
        
        # Validate and adapt k value
        k = self.validate_k_value(k)
        
        # Average tiles to get board embedding
        query_embedding = tile_embeddings.mean(dim=0)
        # L2 normalize for cosine similarity (research-backed improvement)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        
        # All embeddings are already normalized in build_index()
        # Compute cosine similarity to all stored embeddings
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.all_embeddings,
            dim=1
        )
        
        # Get top-k matches
        top_k_scores, top_k_indices = torch.topk(similarities, k=k)
        
        # Most common FEN among top-k
        top_k_fens = [self.all_fen_indices[idx] for idx in top_k_indices.tolist()]
        predicted_fen = max(set(top_k_fens), key=top_k_fens.count)
        
        # Confidence = average similarity of top-k
        confidence = top_k_scores.mean().item()
        
        return predicted_fen, confidence
    
    def validate_k_value(self, k: int) -> int:
        """Ensure k is not larger than number of stored embeddings
        
        Uses adaptive k based on dataset size if not specified.
        Research (Dasgupta et al.) recommends k ~ sqrt(n) for balanced accuracy.
        """
        n = len(self.all_embeddings)
        if n == 0:
            return 1
        
        # If k is None or 0, use adaptive k = sqrt(n)
        if k is None or k == 0:
            k = max(3, int(np.sqrt(n)))
        
        # Cap k to dataset size
        return max(1, min(n, k))
    
    # ============ METHOD 2: Mahalanobis Distance ============
    def predict_mahalanobis(self, tile_embeddings: torch.Tensor, k: int = 3) -> Tuple[str, float]:
        """
        Predict FEN using Mahalanobis distance (better than KNN for correlated features).
        
        Args:
            tile_embeddings: Tensor of shape [64, 2048]
            k: Number of neighbors
            
        Returns:
            (predicted_fen, confidence as inverse distance)
        """
        if self.mahal_inv_cov is None:
            raise ValueError("Must call build_index() first")
        
        # Get query embedding
        query_embedding = tile_embeddings.mean(dim=0).unsqueeze(0)
        query_scaled = self.scaler.transform(query_embedding.cpu().numpy())
        
        # Compute Mahalanobis distance to all embeddings
        embeddings_np = self.all_embeddings.cpu().numpy()
        embeddings_scaled = self.scaler.transform(embeddings_np)
        
        distances = []
        for emb in embeddings_scaled:
            diff = query_scaled[0] - emb
            mahal_dist = np.sqrt(diff @ self.mahal_inv_cov @ diff.T)
            distances.append(mahal_dist)
        
        distances = np.array(distances)
        
        # Get top-k (smallest distances = best matches)
        top_k_indices = np.argsort(distances)[:k]
        
        # Most common FEN among top-k
        top_k_fens = [self.all_fen_indices[idx] for idx in top_k_indices]
        predicted_fen = max(set(top_k_fens), key=top_k_fens.count)
        
        # Confidence = inverse of average distance
        avg_distance = distances[top_k_indices].mean()
        confidence = 1.0 / (1.0 + avg_distance)
        
        return predicted_fen, confidence
    
    def predict_from_image(self, board_image: Image.Image, method: str = "knn", 
                          k: int = 3) -> Tuple[str, float]:
        """
        Predict FEN directly from a board image.
        
        Args:
            board_image: PIL Image of full chess board
            method: "knn" or "mahalanobis"
            k: Number of neighbors
            
        Returns:
            (predicted_fen, confidence)
        """
        tile_embeddings = self.extract_board_embeddings(board_image)
        
        if method == "knn":
            return self.predict_knn(tile_embeddings, k)
        elif method == "mahalanobis":
            return self.predict_mahalanobis(tile_embeddings, k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # ============ METHOD 3: OOD Detection ============
    def predict_with_ood(self, tile_embeddings: torch.Tensor, 
                         threshold: float = 0.5) -> Tuple[str, float, bool]:
        """
        Predict FEN with Out-of-Distribution detection.
        Returns (predicted_fen, confidence, is_ood)
        
        OOD = True means "I'm not confident, this might be new"
        """
        if len(self.all_embeddings) == 0:
            raise ValueError("No FEN positions in database. Call add_fen_position() first.")
        query_embedding = tile_embeddings.mean(dim=0)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
        
        # Compute similarity to all stored embeddings
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.all_embeddings,
            dim=1
        )
        
        # Maximum similarity = how close to nearest FEN
        max_similarity = similarities.max().item()
        
        # Entropy-based OOD detection
        # High entropy = uncertain = OOD
        probs = torch.nn.functional.softmax(similarities * 10, dim=0)  # Temperature scaling
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # If entropy is high, model is uncertain
        is_ood = entropy > threshold
        
        # Get prediction anyway
        top_idx = similarities.argmax()
        predicted_fen = self.all_fen_indices[top_idx]
        
        return predicted_fen, max_similarity, is_ood
    
    def save(self, path: str):
        """Save classifier to disk"""
        torch.save({
            'fen_database': {k: v.cpu() for k, v in self.fen_database.items()},
            'fen_labels': self.fen_labels,
            'all_embeddings': self.all_embeddings,
            'all_fen_indices': self.all_fen_indices,
        }, path)
    
    def load(self, path: str):
        """Load classifier from disk"""
        data = torch.load(path)
        self.fen_database = data['fen_database']
        self.fen_labels = data['fen_labels']
        self.all_embeddings = data['all_embeddings']
        self.all_fen_indices = data['all_fen_indices']


class TripletLossTrainer:
    """
    Train a small neural network with Triplet Loss.
    
    Triplet Loss = learn to put similar FENs close, different FENs far
    """
    
    def __init__(self, embedding_dim: int = 2048, output_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
        # Small projection network: 2048 -> 256
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, output_dim),
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.projection.parameters(), lr=1e-3)
        self.criterion = torch.nn.TripletMarginLoss(margin=1.0)
    
    def train_on_batch(self, anchor: torch.Tensor, positive: torch.Tensor, 
                       negative: torch.Tensor):
        """
        Train on one triplet:
        - anchor: embedding of a FEN
        - positive: embedding of the SAME FEN
        - negative: embedding of a DIFFERENT FEN
        
        Goal: Make anchor close to positive, far from negative
        """
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        
        # Project embeddings
        anchor_proj = self.projection(anchor)
        positive_proj = self.projection(positive)
        negative_proj = self.projection(negative)
        
        # Compute loss
        loss = self.criterion(anchor_proj, positive_proj, negative_proj)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_projection(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project embedding using trained network"""
        with torch.no_grad():
            embedding = embedding.to(self.device)
            return self.projection(embedding)


# Example usage:
if __name__ == "__main__":
    print("FEN Classification Module Ready!")
    print("\nAvailable methods:")
    print("1. KNN - Fast, no training")
    print("2. Mahalanobis - Smarter than KNN")
    print("3. OOD Detection - Knows when uncertain")
    print("4. Triplet Loss - Best accuracy (requires training)")