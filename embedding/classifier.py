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
from typing import Tuple, Optional
import tempfile
import os
from PIL import Image
import sys
from collections import Counter

sys.path.insert(0, '/home/lotems/Documents/DL_Oren/CSC-BSR/preprocessing')
from preprocessing.splitting_images import slice_image_with_coordinates

sys.path.insert(0, '/home/lotems/Documents/DL_Oren/CSC-BSR/embadding')
from embedding_base import EmbeddingModel


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
        # GLOBAL Storage (No more per-tile dictionaries)


        self.global_embeddings = []  # Will hold tens of thousands of vectors
        self.global_labels = []  # Will hold the class label for each vector

        # The searchable Index (Built later)
        self.index_embeddings = None  # Tensor [Total_Samples, Dim]
        self.index_labels = None  # Tensor [Total_Samples]

        # OOD Threshold (One single value for the whole board)
        self.knn_k = 5

        self.knn_similarity_threshold = 0.60
        self.knn_ood_using_similarity = False

        self.knn_distance_threshold = 1.6
        self.knn_ood_using_distance = True

        self.knn_MIN_CONSENSUS = 0.7
        self.knn_ood_using_vote = False

        # Embedding extractor setup (Keep as is)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_extractor = embedding_extractor
        self.embedding_dim = self.embedding_extractor.get_embedding_dim()

        print(f"Using {self.embedding_extractor} for GLOBAL FEN classification")

        
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
        if board_state is None:
            board_state = np.zeros((8, 8), dtype=int)

        labels_1d = board_state.flatten()

        # Iterate over all 64 tiles in this new board
        for tile_idx in range(64):
            # ADD TO GLOBAL LIST
            # We treat every tile identically, regardless of its position (A1 vs E4)
            self.global_embeddings.append(tile_embeddings[tile_idx].float().cpu())
            self.global_labels.append(int(labels_1d[tile_idx]))
    
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

    def predict_with_ood(self, tile_embeddings: torch.Tensor,
                         method: str = "knn",  # Default to knn for global
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Check GLOBAL index, not per-tile index
        if self.index_embeddings is None:
            raise ValueError("Must call build_index() first")

        if method == "knn":
            return self._predict_ood_knn(tile_embeddings)
        elif method == "mahalanobis":
            # Mahalanobis is harder to implement globally; sticking to KNN is recommended
            print("Warning: Global Mahalanobis not implemented. Falling back to KNN.")
            return self._predict_ood_knn(tile_embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")

    def build_index(self):
        print(f"Building Global Index with {len(self.global_embeddings)} samples...")

        if len(self.global_embeddings) == 0:
            print("Warning: Database is empty!")
            return

        # 1. Stack everything into one giant tensor (N x Dim)
        # Move to GPU immediately for speed
        self.index_embeddings = torch.stack(self.global_embeddings).to(self.device)
        self.index_labels = torch.tensor(self.global_labels).to(self.device)
        self.index_embeddings = torch.nn.functional.normalize(self.index_embeddings, p=2, dim=1)

        print(f"Index built. Shape: {self.index_embeddings.shape}")

        # 3. AUTO-CALIBRATE THRESHOLD (The Self-Check)
        # We check how well the valid data matches itself.
        print("Auto-calibrating global threshold...")

        # To save memory/time, we take a random sample of 2000 images for calibration
        n_samples = self.index_embeddings.shape[0]
        sample_size = min(2000, n_samples)
        indices = torch.randperm(n_samples)[:sample_size]
        sample_embs = self.index_embeddings[indices]

        # All-vs-All comparison
        sim_matrix = torch.mm(sample_embs, sample_embs.t())

        # We need to find neighbors.
        # Since we are matching the database against itself, the "Best" match is always ITSELF (score 1.0).
        # We want the *next* k matches.
        # So we ask for k+1, and throw away the first column.
        k_calib = self.knn_k  # Must match the k used in prediction!
        top_k_vals, _ = torch.topk(sim_matrix, k=k_calib + 1, dim=1) # Get top k+1 scores
        neighbor_scores = top_k_vals[:, 1:] # Remove the first column (the self-match of 1.0)

        # --- CALIBRATION 1: Average Similarity ---
        avg_scores = neighbor_scores.mean(dim=1)
        calc_thresh_sim = float(np.percentile(avg_scores.cpu().numpy(), 1))
        # Safety ceiling for similarity (Average)
        self.knn_similarity_threshold = min(calc_thresh_sim, self.knn_similarity_threshold)

        print(f"Global OOD Threshold (Average-Based) set to: {self.knn_similarity_threshold:.4f} calc_thresh_sim:{calc_thresh_sim:.4f}")



        # --- CALIBRATION 2: 1/Distance to k-th Neighbor ---
        # Get the k-th score (the last column of neighbor_scores)
        kth_sims = neighbor_scores[:, -1]

        # Convert to Euclidean Distance: d = sqrt(2 * (1 - sim))
        kth_dists = torch.sqrt(torch.clamp(2 * (1 - kth_sims), min=0))
        kth_scores = 1.0 / (kth_dists + 1e-9) # Convert to Score: 1 / (d + epsilon)
        calc_thresh_dist = float(np.percentile(kth_scores.cpu().numpy(), 1))

        # Safety floor for distance score (Adjust as needed, 1.0 is roughly dist=1.0)
        self.knn_distance_threshold = max(calc_thresh_dist, self.knn_distance_threshold)
        print(
            f"knn_distance_threshold set to: {self.knn_similarity_threshold:.4f} calc_thresh_dist:{calc_thresh_dist:.4f}")

        print(
            f"Thresholds set | AvgSim: {self.knn_similarity_threshold:.4f} | 1/Dist(k): {self.knn_distance_threshold:.4f}")

    def save(self, path: str):
        """Save GLOBAL classifier to disk"""
        print(f"Saving global classifier with {len(self.global_embeddings)} embeddings...")
        torch.save({
            'global_embeddings': self.global_embeddings,
            'global_labels': self.global_labels,
            'knn_similarity_threshold': self.knn_similarity_threshold
        }, path)
        print("Saved successfully.")

    def load(self, path: str):
        """Load GLOBAL classifier from disk"""
        print(f"Loading classifier from {path}...")
        if not os.path.exists(path):
            print("Warning: Checkpoint not found. Database will be empty.")
            return

        data = torch.load(path)

        # Check if this is an old format file (migration check)
        if 'tile_database' in data:
            print("[WARNING] This is an OLD format file (Per-Tile). Ignoring it.")
            print("Please uncomment Step 4 in test_classifier.py to rebuild the database.")
            return

        self.global_embeddings = data['global_embeddings']
        self.global_labels = data['global_labels']

        # Restore threshold if it exists, otherwise keep default
        if 'knn_similarity_threshold' in data:
            self.knn_similarity_threshold = data['knn_similarity_threshold']

        print(f"Loaded {len(self.global_embeddings)} global embeddings.")
    
    # ============ METHOD 1: KNN ============
    def predict_knn(self, tile_embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.index_embeddings is None:
            raise ValueError("Call build_index() first")

            # FIX: Move to GPU
        device = self.index_embeddings.device
        tile_embeddings = tile_embeddings.to(device)

        # 1. Prepare Query (64 x Dim)
        query = torch.nn.functional.normalize(tile_embeddings, p=2, dim=1)

        # 2. Global Search
        sim_matrix = torch.mm(query, self.index_embeddings.t())

        # 3. Find Top K matches
        top_k_scores, top_k_indices = torch.topk(sim_matrix, k=self.knn_k, dim=1)

        predictions = np.zeros(64, dtype=int)
        confidences = np.zeros(64, dtype=float)

        # 4. Vote
        for i in range(64):
            # Get indices of neighbors
            indices = top_k_indices[i].cpu().tolist()
            # Look up their labels
            neighbor_labels = self.index_labels[indices].tolist()

            # Majority Vote
            predicted_label = max(set(neighbor_labels), key=neighbor_labels.count)

            predictions[i] = predicted_label
            # Confidence is the similarity score of the best match
            confidences[i] = top_k_scores[i, 0].item()

        return predictions, confidences

    def _predict_ood_knn(self, tile_embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.index_embeddings is None:
            raise ValueError("Call build_index() first")

        # FIX: Ensure the input is on the same device as the database (GPU)
        device = self.index_embeddings.device
        tile_embeddings = tile_embeddings.to(device)

        query = torch.nn.functional.normalize(tile_embeddings, p=2, dim=1)
        sim_matrix = torch.mm(query, self.index_embeddings.t())
        top_k_scores, top_k_indices = torch.topk(sim_matrix, k=self.knn_k, dim=1)

        predictions = np.zeros(64, dtype=int)
        confidences = np.zeros(64, dtype=float)
        is_ood = np.zeros(64, dtype=bool)

        # New: Define Consensus Threshold (e.g., 4 out of 5 must agree)
        # 0.6 = 3/5, 0.8 = 4/5, 1.0 = 5/5
        MIN_CONSENSUS = self.knn_MIN_CONSENSUS

        for i in range(64):
            # 1. Get Neighbors & Labels
            indices = top_k_indices[i].cpu().tolist()
            neighbor_labels = self.index_labels[indices].tolist()

            # 2. Vote Logic (Check 3)
            vote_counts = Counter(neighbor_labels)
            predicted_label, best_vote_count = vote_counts.most_common(1)[0]
            consensus_ratio = best_vote_count / self.knn_k

            # 3. Calculate Metrics
            # Metric A: Average Similarity
            avg_similarity = top_k_scores[i].mean().item()

            # Metric B: 1 / Distance to k-th Neighbor
            kth_similarity = top_k_scores[i, -1].item()
            kth_dist = np.sqrt(max(0, 2 * (1 - kth_similarity)))
            ood_score_kth = 1.0 / (kth_dist + 1e-9)

            predictions[i] = predicted_label
            confidences[i] = avg_similarity  # Keep using Avg Sim as the main confidence score

            # === TRIPLE OOD CHECK ===
            # Check 1: Average Similarity (Robustness)
            is_low_avg = avg_similarity < self.knn_similarity_threshold

            # Check 2: k-th Distance (Boundary Safety)
            is_far_kth = ood_score_kth < self.knn_distance_threshold

            # Check 3: Consensus (Ambiguity)
            is_ambiguous = consensus_ratio < self.knn_MIN_CONSENSUS

            # Combine Checks
            # If ANY enabled check fails, flag as OOD
            is_ood[i] = (
                    (self.knn_ood_using_similarity and is_low_avg) or
                    (self.knn_ood_using_distance and is_far_kth) or
                    (self.knn_ood_using_vote and is_ambiguous)
            )

        return predictions, confidences, is_ood

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


    # ============ METHOD 3: OOD Detection (Per-tile, Distance-based) ============



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