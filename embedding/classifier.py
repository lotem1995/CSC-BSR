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
from preprocessing.splitting_images import slice_image_with_coordinates

sys.path.insert(0, '/home/lotems/Documents/DL_Oren/CSC-BSR/embadding')
from embedding_base import EmbeddingModel
from embedding.qwen3 import QwenVisionEmbedding
from dinov2 import DINOv2Embedding


class FENClassifier:
    """
    Per-tile classifier using KNN/Mahalanobis distance.
    
    Classifies each of 64 tiles independently:
    - For each tile position, maintains embeddings from all seen FENs
    - Predicts class and confidence for each tile separately
    - Outputs 64 class predictions (one per tile)
    
    Works with any embedding model that implements the EmbeddingModel interface.
    """
    
    def __init__(self, embedding_extractor: Optional[EmbeddingModel] = None):
        """
        Args:
            embedding_extractor: Any EmbeddingModel instance (QwenVisionEmbedding, DINOv2Embedding, etc).
                               If None, creates default QwenVisionEmbedding.
        """
        # Per-tile storage: tile_idx -> list of {embedding, label}
        self.tile_database = {}  # int (0-63) -> list of {embedding: Tensor, label: int}
        
        # Per-tile indices for KNN: tile_idx -> {embeddings: Tensor, labels: List[int]}
        self.tile_embeddings_index = {}  # int -> Tensor [n_samples, embedding_dim]
        self.tile_labels_index = {}      # int -> List[int]
        
        # Per-tile Mahalanobis: tile_idx -> {scaler, inv_cov, class_means}
        self.tile_scalers = {}  # int -> StandardScaler
        self.tile_mahal_inv_covs = {}  # int -> ndarray [embedding_dim, embedding_dim]
        self.tile_class_means = {}  # int -> dict mapping class_label -> mean embedding
        self.tile_ood_thresholds = {} # Format: { tile_idx: { 'knn': float, 'mahalanobis': float } }
        
        # Embedding extractor (must implement EmbeddingModel interface)
        self.embedding_extractor = embedding_extractor
        if self.embedding_extractor is None:
            print("Initializing default QwenVisionEmbedding...")
            self.embedding_extractor = QwenVisionEmbedding()
        
        # Store embedding dimension for reference
        self.embedding_dim = self.embedding_extractor.get_embedding_dim()
        print(f"Using {self.embedding_extractor} for per-tile FEN classification")
        
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
    
    def add_fen_position(self, fen: str, tile_embeddings: torch.Tensor, board_state: Optional[np.ndarray] = None):
        """
        Add a FEN position and its 64 tile embeddings to per-tile database.
        
        Args:
            fen: FEN string (e.g., "rnbqkbnr/pppppppp/...")
            tile_embeddings: Tensor of shape [64, embedding_dim]
            board_state: Optional [8, 8] array with class labels (0-16) for each square.
                        If None, assumes all zeros (empty squares).
        """
        if board_state is None:
            board_state = np.zeros((8, 8), dtype=int)
        
        # Flatten board state to 1D (row-major order)
        labels_1d = board_state.flatten()  # [64]
        
        # Store each tile separately in the database
        for tile_idx in range(64):
            if tile_idx not in self.tile_database:
                self.tile_database[tile_idx] = []
            
            self.tile_database[tile_idx].append({
                'embedding': tile_embeddings[tile_idx].float(),
                'label': int(labels_1d[tile_idx]),
            })
    
    def add_fen_from_image(self, fen: str, board_image: Image.Image, board_state: Optional[np.ndarray] = None):
        """
        Add a FEN position by extracting embeddings from a board image.
        
        Args:
            fen: FEN string
            board_image: PIL Image of full chess board
            board_state: Optional [8, 8] array with class labels for each square
        """
        tile_embeddings = self.extract_board_embeddings(board_image)
        self.add_fen_position(fen, tile_embeddings, board_state)
        
    def build_index(self):
        """
        Build per-tile KNN and Mahalanobis indices from the tile database.
        Must be called after adding FEN positions and before prediction.
        """
        self.tile_ood_thresholds = {}  # Reset cache because data changed
        for tile_idx in range(64):
            if tile_idx not in self.tile_database or len(self.tile_database[tile_idx]) == 0:
                continue
            
            # Extract embeddings and labels for this tile
            tile_data = self.tile_database[tile_idx]
            embeddings = torch.stack([item['embedding'] for item in tile_data])
            labels = [item['label'] for item in tile_data]
            
            # Normalize embeddings for KNN
            embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            self.tile_embeddings_index[tile_idx] = embeddings_norm
            self.tile_labels_index[tile_idx] = labels
            
            # Fit Mahalanobis: compute class means and shared covariance
            embeddings_np = embeddings.cpu().numpy()
            scaler = StandardScaler()
            scaler.fit(embeddings_np)
            
            # Compute scaled embeddings and shared covariance
            scaled = scaler.transform(embeddings_np)
            lw = LedoitWolf()
            lw.fit(scaled)
            
            # Compute per-class means in scaled space
            class_means = {}
            unique_labels = set(labels)
            for class_label in unique_labels:
                class_mask = np.array(labels) == class_label
                class_embeddings = scaled[class_mask]
                class_means[class_label] = class_embeddings.mean(axis=0)
            
            self.tile_scalers[tile_idx] = scaler
            self.tile_mahal_inv_covs[tile_idx] = np.linalg.inv(lw.covariance_)
            self.tile_class_means[tile_idx] = class_means
    
    # ============ METHOD 1: KNN ============
    def predict_knn(self, tile_embeddings: torch.Tensor, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class for each of 64 tiles using K-Nearest Neighbors.
        
        Args:
            tile_embeddings: Tensor of shape [64, embedding_dim]
            k: Number of neighbors to check
            
        Returns:
            (predictions, confidences)
                predictions: np.ndarray [64] with predicted class (0-16) for each tile
                confidences: np.ndarray [64] with confidence (0-1) for each tile
        """
        if len(self.tile_embeddings_index) == 0:
            raise ValueError("No tiles in database. Call add_fen_position() and build_index() first.")
        
        predictions = np.zeros(64, dtype=int)
        confidences = np.zeros(64, dtype=float)
        
        for tile_idx in range(64):
            if tile_idx not in self.tile_embeddings_index:
                # No training data for this tile, predict empty
                predictions[tile_idx] = 0
                confidences[tile_idx] = 0.0
                continue
            
            # Get query embedding for this tile
            query_emb = tile_embeddings[tile_idx].unsqueeze(0)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            
            # Get stored embeddings for this tile
            stored_embs = self.tile_embeddings_index[tile_idx]  # Already normalized
            labels = self.tile_labels_index[tile_idx]
            
            # Validate and adapt k value
            k_actual = self.validate_k_value(k, len(stored_embs))
            
            # Cosine similarity to all stored embeddings
            similarities = torch.nn.functional.cosine_similarity(query_emb, stored_embs, dim=1)
            
            # Get top-k matches
            top_k_scores, top_k_indices = torch.topk(similarities, k=k_actual)
            
            # Most common label among top-k
            top_k_labels = [labels[idx] for idx in top_k_indices.tolist()]
            predicted_label = max(set(top_k_labels), key=top_k_labels.count)
            
            predictions[tile_idx] = predicted_label
            confidences[tile_idx] = top_k_scores.mean().item()
        
        return predictions, confidences
    
    def validate_k_value(self, k: int, n: int) -> int:
        """Ensure k is not larger than number of stored embeddings
        
        Args:
            k: Requested k value
            n: Number of training samples available
        
        Uses adaptive k based on dataset size if k is None.
        Research (Dasgupta et al.) recommends k ~ sqrt(n) for balanced accuracy.
        """
        if n == 0:
            return 1
        
        # If k is None or 0, use adaptive k = sqrt(n)
        if k is None or k == 0:
            k = max(3, int(np.sqrt(n)))
        
        # Cap k to dataset size
        return max(1, min(n, k))
    
    # ============ METHOD 2: Mahalanobis Distance (Class-Conditional) ============
    def predict_mahalanobis(self, tile_embeddings: torch.Tensor, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class for each of 64 tiles using class-conditional Mahalanobis distance.
        Computes distance to each class mean: d_c(x) = (x - μ_c)^T Σ^-1 (x - μ_c)
        Predicts argmin over classes.
        
        Args:
            tile_embeddings: Tensor of shape [64, embedding_dim]
            k: Unused (kept for API compatibility). Mahalanobis uses class means, not k-NN.
            
        Returns:
            (predictions, confidences)
                predictions: np.ndarray [64] with predicted class (0-16) for each tile
                confidences: np.ndarray [64] with confidence as exp(-min_distance)
        """
        if len(self.tile_mahal_inv_covs) == 0:
            raise ValueError("Must call build_index() first")
        
        predictions = np.zeros(64, dtype=int)
        confidences = np.zeros(64, dtype=float)
        
        for tile_idx in range(64):
            if tile_idx not in self.tile_mahal_inv_covs:
                # No training data for this tile
                predictions[tile_idx] = 0
                confidences[tile_idx] = 0.0
                continue
            
            # Get query embedding for this tile
            query_emb = tile_embeddings[tile_idx].unsqueeze(0)
            query_np = query_emb.cpu().numpy()
            
            # Get class statistics for this tile
            scaler = self.tile_scalers[tile_idx]
            inv_cov = self.tile_mahal_inv_covs[tile_idx]
            class_means = self.tile_class_means[tile_idx]
            
            # Scale query
            query_scaled = scaler.transform(query_np)[0]
            
            # Compute Mahalanobis distance to each class mean
            class_distances = {}
            for class_label, class_mean in class_means.items():
                diff = query_scaled - class_mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
                class_distances[class_label] = mahal_dist
            
            # Predict class with minimum distance
            predicted_label = min(class_distances, key=class_distances.get)
            min_distance = class_distances[predicted_label]
            
            predictions[tile_idx] = predicted_label
            
            # Confidence: exponential decay of distance (clean, calibratable)
            confidences[tile_idx] = np.exp(-min_distance)
        
        return predictions, confidences
    
    def predict_from_image(self, board_image: Image.Image, method: str = "knn", 
                          k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class for all 64 tiles from a board image.
        
        Args:
            board_image: PIL Image of full chess board
            method: "knn" or "mahalanobis" or inner classifier
            k: Number of neighbors (used only for KNN)
            
        Returns:
            (predictions, confidences)
                predictions: np.ndarray [64] with predicted class (0-16) for each tile
                confidences: np.ndarray [64] with per-tile confidence
        """
        tile_embeddings = self.extract_board_embeddings(board_image)
        
        if method == "knn":
            return self.predict_knn(tile_embeddings, k)
        elif method == "mahalanobis":
            return self.predict_mahalanobis(tile_embeddings)
        ## TODO: Add inner classifier support
        # elif method=="inner":
        #     return self.embedding_extractor.classify_fen(board_image):
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # ============ METHOD 3: OOD Detection (Per-tile, Distance-based) ============
    def _calc_mahal_threshold(self, tile_idx: int) -> float:
        # 1. Verification: Does data exist?
        if tile_idx not in self.tile_database:
            print(f"[DEBUG ERR] Tile {tile_idx} not in database!")
            return 20.0  # Safer fallback

        tile_data = self.tile_database[tile_idx]
        if len(tile_data) == 0:
            print(f"[DEBUG ERR] Tile {tile_idx} database entry is empty!")
            return 20.0

            # 2. Calculation
        print("calculating mahal threshold")
        scaler = self.tile_scalers[tile_idx]
        inv_cov = self.tile_mahal_inv_covs[tile_idx]
        class_means = self.tile_class_means[tile_idx]

        raw_embs = torch.stack([item['embedding'] for item in tile_data]).cpu().numpy()
        labels = [item['label'] for item in tile_data]

        scaled_embs = scaler.transform(raw_embs)
        distances = []

        for i, emb in enumerate(scaled_embs):
            label = labels[i]
            mean = class_means[label]
            diff = emb - mean
            dist = np.sqrt(diff @ inv_cov @ diff.T)
            distances.append(dist)

        if not distances:
            print(f"[DEBUG ERR] Tile {tile_idx}: Loop finished but no distances calculated.")
            return 20.0  # Increased fallback from 3.0 to 20.0 (realistic for DINO)

        # 3. Percentile Calculation
        calc_threshold = float(np.percentile(distances, 95))

        # 4. SAFETY FLOOR (Crucial for high dimensions)
        # Prevent threshold from being impossibly low if validation data is too clean
        print(f"calc_threshold: {calc_threshold}")
        final_threshold = max(calc_threshold, 15.0)

        return calc_threshold

    def _calc_knn_threshold(self, tile_idx: int, k: int) -> float:
        # Get all stored embeddings
        stored_embs = self.tile_embeddings_index[tile_idx]

        # Compare every point to every other point (Self-Similarity Matrix)
        sim_matrix = torch.mm(stored_embs, stored_embs.t())

        # Find the score of the k-th nearest neighbor
        # We need k+1 because the closest match is always itself (score 1.0)
        k_adj = min(k + 1, len(stored_embs))
        top_k_scores, _ = torch.topk(sim_matrix, k=k_adj, dim=1)

        # The last column is the score of the neighbor we care about
        ith_neighbor_scores = top_k_scores[:, -1].cpu().numpy()

        if len(ith_neighbor_scores) == 0:
            return 0.7  # Fallback default

        # CHANGE 1: Use 1st percentile (More relaxed than 5th)
        calc_threshold = float(np.percentile(ith_neighbor_scores, 1))

        # CHANGE 2: Safety Ceiling (The Ultimate Fix)
        # "Even if the validation data is perfect (0.98),
        #  allow anything above 0.90 to pass."
        final_threshold = min(calc_threshold, 0.80)

        return final_threshold

    def _get_or_calculate_threshold(self, tile_idx: int, method: str, k: int = 3) -> float:
        # Check cache: Do we have it?
        if tile_idx in self.tile_ood_thresholds:
            if method in self.tile_ood_thresholds[tile_idx]:
                return self.tile_ood_thresholds[tile_idx][method]
        else:
            self.tile_ood_thresholds[tile_idx] = {}

        # Calculate if missing
        if method == "mahalanobis":
            threshold = self._calc_mahal_threshold(tile_idx)
        elif method == "knn":
            threshold = self._calc_knn_threshold(tile_idx, k)
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        # Save to cache and return
        self.tile_ood_thresholds[tile_idx][method] = threshold
        return threshold

    def predict_with_ood(self, tile_embeddings: torch.Tensor, 
                         method: str = "mahalanobis",
                         k: int = 3,
                         threshold: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict class for each of 64 tiles with Out-of-Distribution detection.
        Uses distance-based OOD scoring (Mahalanobis min distance or KNN distance).
        
        Args:
            tile_embeddings: Tensor of shape [64, embedding_dim]
            method: "mahalanobis" or "knn" for OOD scoring
            k: Number of neighbors (for KNN method)
            threshold: Distance threshold for OOD detection. If None, uses adaptive threshold:
                      - Mahalanobis: threshold = 3.0 (chi-squared approximation)
                      - KNN: threshold = 0.7 (cosine similarity)
            
        Returns:
            (predictions, confidences, is_ood)
                predictions: np.ndarray [64] with predicted class
                confidences: np.ndarray [64] with confidence (0-1)
                is_ood: np.ndarray [64] with bool, True = uncertain/unknown
        """
        if method == "mahalanobis":
            if len(self.tile_mahal_inv_covs) == 0:
                raise ValueError("Must call build_index() first")
            return self._predict_ood_mahalanobis(tile_embeddings, threshold)
        elif method == "knn":
            if len(self.tile_embeddings_index) == 0:
                raise ValueError("Must call build_index() first")
            return self._predict_ood_knn(tile_embeddings, k, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _predict_ood_mahalanobis(self, tile_embeddings: torch.Tensor, 
                                  threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """OOD detection using Mahalanobis min distance."""
        predictions = np.zeros(64, dtype=int)
        confidences = np.zeros(64, dtype=float)
        is_ood = np.zeros(64, dtype=bool)
        
        for tile_idx in range(64):
            if tile_idx not in self.tile_mahal_inv_covs:
                predictions[tile_idx] = 0
                confidences[tile_idx] = 0.0
                is_ood[tile_idx] = True
                continue

            current_threshold = threshold
            if current_threshold is None:
                # Calculate or fetch cached threshold automatically
                current_threshold = self._get_or_calculate_threshold(tile_idx, "mahalanobis")
            
            # Get query embedding
            query_emb = tile_embeddings[tile_idx].unsqueeze(0)
            query_np = query_emb.cpu().numpy()
            
            # Get class statistics
            scaler = self.tile_scalers[tile_idx]
            inv_cov = self.tile_mahal_inv_covs[tile_idx]
            class_means = self.tile_class_means[tile_idx]
            
            # Scale query
            query_scaled = scaler.transform(query_np)[0]
            
            # Compute distances to all class means
            class_distances = {}
            for class_label, class_mean in class_means.items():
                diff = query_scaled - class_mean
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
                class_distances[class_label] = mahal_dist
            
            # Get prediction and min distance
            predicted_label = min(class_distances, key=class_distances.get)
            min_distance = class_distances[predicted_label]
            
            predictions[tile_idx] = predicted_label
            confidences[tile_idx] = np.exp(-min_distance)
            is_ood[tile_idx] = min_distance > current_threshold
        
        return predictions, confidences, is_ood
    
    def _predict_ood_knn(self, tile_embeddings: torch.Tensor, k: int,
                         threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """OOD detection using KNN distance to nearest neighbor."""
        predictions = np.zeros(64, dtype=int)
        confidences = np.zeros(64, dtype=float)
        is_ood = np.zeros(64, dtype=bool)
        
        for tile_idx in range(64):
            if tile_idx not in self.tile_embeddings_index:
                predictions[tile_idx] = 0
                confidences[tile_idx] = 0.0
                is_ood[tile_idx] = True
                continue

            current_threshold = threshold
            if current_threshold is None:
                current_threshold = self._get_or_calculate_threshold(tile_idx, "knn", k)
            
            # Get query embedding (normalized)
            query_emb = tile_embeddings[tile_idx].unsqueeze(0)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            
            # Get stored embeddings
            stored_embs = self.tile_embeddings_index[tile_idx]
            labels = self.tile_labels_index[tile_idx]
            
            # Validate k
            k_actual = self.validate_k_value(k, len(stored_embs))
            
            # Compute cosine similarities
            similarities = torch.nn.functional.cosine_similarity(query_emb, stored_embs, dim=1)
            
            # Get top-k matches
            top_k_scores, top_k_indices = torch.topk(similarities, k=k_actual)
            
            # Prediction: majority vote
            top_k_labels = [labels[idx] for idx in top_k_indices.tolist()]
            predicted_label = max(set(top_k_labels), key=top_k_labels.count)
            
            # OOD: max similarity (nearest neighbor distance)
            max_similarity = top_k_scores[0].item()
            
            predictions[tile_idx] = predicted_label
            confidences[tile_idx] = top_k_scores.mean().item()
            is_ood[tile_idx] = max_similarity < current_threshold  # Low similarity = OOD

        return predictions, confidences, is_ood
    
    def save(self, path: str):
        """Save per-tile classifier to disk"""
        torch.save({
            'tile_database': self.tile_database,
            'tile_embeddings_index': self.tile_embeddings_index,
            'tile_labels_index': self.tile_labels_index,
            'tile_scalers': self.tile_scalers,
            'tile_mahal_inv_covs': self.tile_mahal_inv_covs,
            'tile_class_means': self.tile_class_means,
            'tile_ood_thresholds': self.tile_ood_thresholds
        }, path)
    
    def load(self, path: str):
        """Load per-tile classifier from disk"""
        data = torch.load(path)
        self.tile_database = data['tile_database']
        self.tile_embeddings_index = data['tile_embeddings_index']
        self.tile_labels_index = data['tile_labels_index']
        self.tile_scalers = data['tile_scalers']
        self.tile_mahal_inv_covs = data['tile_mahal_inv_covs']
        self.tile_class_means = data.get('tile_class_means', {})  # Backward compatibility
        self.tile_ood_thresholds = data.get('tile_ood_thresholds', {})


# Example usage:
if __name__ == "__main__":
    print("Per-Tile FEN Classification Module Ready!")
    print("\nArchitecture: Path B - Per-tile KNN/Mahalanobis")
    print("\nKey features:")
    print("- Classifies all 64 tiles independently")
    print("- Returns per-tile predictions (0-16) and per-tile confidences")
    print("- OOD detection identifies unknown or uncertain tiles")
    print("\nAvailable methods:")
    print("1. KNN - Fast, cosine similarity-based")
    print("2. Mahalanobis - Class-conditional distance (models class distributions)")
    print("3. OOD Detection - Distance-based uncertainty (calibratable on validation data)")
    print("\nUsage:")
    print("  classifier = FENClassifier(embedding_extractor)")
    print("  classifier.add_fen_position(fen, tile_embeddings, board_state)")
    print("  classifier.build_index()")
    print("  preds, confs = classifier.predict_knn(tile_embeddings, k=3)")
    print("  # Returns: predictions[64], confidences[64]")