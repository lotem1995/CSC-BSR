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

from torch import nn
from unicodedata import is_normalized

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
        self.normalized_embeddings = None  # Tensor [Total_Samples, Dim]
        self.normalized_labels = None  # Tensor [Total_Samples]

        # --- KNN STORAGE ---
        self.knn_k = 5
        self.is_normalized = False

        self.knn_similarity_threshold = 0.60
        self.knn_ood_using_similarity = False

        self.knn_distance_threshold = 1.1
        self.knn_ood_using_distance = False

        self.knn_MIN_CONSENSUS = 0.7
        self.knn_ood_using_vote = True

        # --- MAHALANOBIS STORAGE ---
        self.global_means = None  # Tensor [13, Dim] (One mean per class)
        self.global_cov_inv = None  # Tensor [Dim, Dim] (Shared Inverse Covariance)
        self.mahal_threshold = 20.0  # OOD Threshold for distance

        # === NEW: Add Softmax Defaults here ===
        self.softmax_temperature = 3
        self.softmax_threshold = 0.15

        # Embedding extractor setup (Keep as is)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_extractor = embedding_extractor
        self.embedding_dim = self.embedding_extractor.get_embedding_dim()

        self.classifier_head: Optional[nn.Module] = None

        print(f"Using {self.embedding_extractor} for GLOBAL FEN classification")

    def set_classifier_head(self, head_model: nn.Module):
        """
        Attach a trained classifier head (torch.nn.Module) for Softmax predictions.
        """
        self.classifier_head = head_model.to(self.device)
        self.classifier_head.eval()  # Ensure it's in inference mode (no dropout)
        print("Classifier head attached successfully.")

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
                overlap_percent=0.7,
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
            if int(labels_1d[tile_idx]) == 17: # it is an ood
                continue
            self.global_embeddings.append(tile_embeddings[tile_idx].float().cpu())
            self.global_labels.append(int(labels_1d[tile_idx]))

    def predict_with_ood(self, tile_embeddings: torch.Tensor,
                         prediction_method: str = "knn",
                         ood_method: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flexible prediction pipeline allowing "Mix & Match".

        Args:
            tile_embeddings: The input embeddings [Batch, Dim]
            prediction_method: Method to use for Class and Confidence ("knn", "mahalanobis", "softmax")
            ood_method: Method to use for OOD Flag ("knn", "mahalanobis", "softmax")
                        If None, defaults to the same as prediction_method.
        """
        # If OOD method is not specified, use the prediction method (Standard behavior)
        if ood_method is None:
            ood_method = prediction_method

        # Optimization: If methods are the same, we are done!
        if prediction_method == ood_method:
            # We ignored the OOD return in Step 1, so we need to fetch it correctly now
            # (Or simpler: just re-run the specific function return)
            if prediction_method == "knn":
                return self.predict_knn(tile_embeddings)
            elif prediction_method == "mahalanobis":
                return self.predict_mahalanobis(tile_embeddings)
            elif prediction_method == "softmax":
                return self.predict_softmax(tile_embeddings)

        # --- STEP 1: Get Prediction (Class) and Confidence ---
        if prediction_method == "knn":
            preds, confs, _ = self.predict_knn(tile_embeddings)
        elif prediction_method == "mahalanobis":
            preds, confs, _ = self.predict_mahalanobis(tile_embeddings)
        elif prediction_method == "softmax":
            preds, confs, _ = self.predict_softmax(tile_embeddings)
        else:
            raise ValueError(f"Unknown prediction method: {prediction_method}")

        # --- STEP 2: Get OOD Flag (Safety Check) ---
        # We run the OOD method just to get the boolean flag
        if ood_method == "knn":
            _, _, is_ood = self.predict_knn(tile_embeddings)
        elif ood_method == "mahalanobis":
            _, _, is_ood = self.predict_mahalanobis(tile_embeddings)
        elif ood_method == "softmax":
            _, _, is_ood = self.predict_softmax(tile_embeddings)
        else:
            raise ValueError(f"Unknown OOD method: {ood_method}")

        # Combine results: Preds from Method A, OOD from Method B
        return preds, confs, is_ood

    def update_thresholds(self):
        print(f"Building Global Index with {len(self.global_embeddings)} samples...")

        if len(self.global_embeddings) == 0:
            print("Warning: Database is empty!")
            return

        # 1. Stack everything into one giant tensor (N x Dim)
        # Move to GPU immediately for speed
        self.normalized_embeddings = torch.stack(self.global_embeddings).to(self.device)
        self.normalized_labels = torch.tensor(self.global_labels).to(self.device)
        self.normalized_embeddings = torch.nn.functional.normalize(self.normalized_embeddings, p=2, dim=1)

        print(f"Index built. Shape: {self.normalized_embeddings.shape}")

        # 3. AUTO-CALIBRATE THRESHOLD (The Self-Check)
        # We check how well the valid data matches itself.
        print("Auto-calibrating global threshold...")

        # To save memory/time, we take a random sample of 2000 images for calibration
        n_samples = self.normalized_embeddings.shape[0]
        sample_size = min(2000, n_samples)
        indices = torch.randperm(n_samples)[:sample_size]
        sample_embs = self.normalized_embeddings[indices]

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
        self.knn_distance_threshold = min(calc_thresh_dist, self.knn_distance_threshold)
        print(
            f"knn_distance_threshold set to: {self.knn_similarity_threshold:.4f} calc_thresh_dist:{calc_thresh_dist:.4f}")

        print(
            f"Thresholds set | AvgSim: {self.knn_similarity_threshold:.4f} | 1/Dist(k): {self.knn_distance_threshold:.4f}")

        # ==========================================
        # 3. BUILD MAHALANOBIS STATISTICS
        # ==========================================
        print("Building Global Mahalanobis Statistics (Tied Covariance)...")

        # Use un-normalized embeddings for Mahalanobis (better for distribution modeling)
        # We need to restack because self.normalized_embeddings is normalized
        raw_embeddings = torch.stack(self.global_embeddings).to(self.device)
        dim = raw_embeddings.shape[1]

        unique_labels = torch.unique(self.normalized_labels)
        max_label = int(unique_labels.max().item())

        # Initialize storage
        self.global_means = torch.zeros((max_label + 1, dim), device=self.device)
        centered_data = []

        # A. Calculate Means
        valid_classes = []
        for label in unique_labels:
            label_idx = int(label.item())
            mask = (self.normalized_labels == label)
            class_samples = raw_embeddings[mask]

            class_mean = class_samples.mean(dim=0)
            self.global_means[label_idx] = class_mean

            # Center data (X - Mean) for covariance
            centered_data.append(class_samples - class_mean)
            valid_classes.append(label_idx)

        # B. Calculate Shared Covariance
        X_centered = torch.cat(centered_data, dim=0)
        N = X_centered.shape[0]

        # Covariance = (X.T @ X) / (N - 1)
        cov_matrix = torch.matmul(X_centered.t(), X_centered) / (N - 1)

        # C. Regularize (Add jitter to diagonal to allow inversion)
        epsilon = 1e-4
        cov_matrix.diagonal().add_(epsilon)

        # D. Invert
        try:
            self.global_cov_inv = torch.inverse(cov_matrix)
        except RuntimeError:
            print("Error inverting covariance. Falling back to Identity matrix.")
            self.global_cov_inv = torch.eye(dim, device=self.device)

        # E. Auto-Calibrate Mahalanobis Threshold
        # Check distances on a subset of training data
        sample_size = min(2000, N)
        indices = torch.randperm(N)[:sample_size]
        sample_subset = raw_embeddings[indices]
        sample_labels_subset = self.normalized_labels[indices]

        dists = []
        for i in range(sample_size):
            x = sample_subset[i]
            y_true = sample_labels_subset[i]
            mean = self.global_means[y_true]

            # Mahalanobis Distance Formula
            delta = (x - mean).unsqueeze(0)
            d = torch.sqrt(torch.matmul(torch.matmul(delta, self.global_cov_inv), delta.t()))
            dists.append(d.item())

        # Set threshold at 99th percentile (reject extreme outliers)
        self.mahal_threshold = float(np.percentile(dists, 99))
        print(f"Mahalanobis Threshold set to: {self.mahal_threshold:.4f}")

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
    def predict_knn(self, tile_embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.normalized_embeddings is None:
            raise ValueError("Call build_index() first")

        # FIX: Ensure the input is on the same device as the database (GPU)
        device = self.normalized_embeddings.device
        tile_embeddings = tile_embeddings.to(device)

        # --- BRANCH: NORMALIZED VS UNNORMALIZED ---
        if self.is_normalized:
            # OPTION A: Cosine Similarity (Normalized)
            # High Score = Good Match (1.0 is perfect)
            query = torch.nn.functional.normalize(tile_embeddings, p=2, dim=1)

            # Dot Product
            sim_matrix = torch.mm(query, self.normalized_embeddings.t())

            # Top-K (Largest)
            top_k_scores, top_k_indices = torch.topk(sim_matrix, k=self.knn_k, dim=1, largest=True)

        else:
            # OPTION B: Euclidean Distance (Unnormalized)
            # Low Score = Good Match (0.0 is perfect)
            # Note: We assume self.normalized_embeddings contains raw vectors if is_normalized=False
            query = tile_embeddings

            # Euclidean Distance
            dists = torch.cdist(query, self.normalized_embeddings, p=2)

            # Top-K (Smallest)
            raw_dists, top_k_indices = torch.topk(dists, k=self.knn_k, dim=1, largest=False)

            # CONVERSION: Turn Distance into Similarity Score (0-1) for compatibility
            # sim = 1 / (1 + dist)
            top_k_scores = 1.0 / (1.0 + raw_dists + 1e-9)

        predictions = np.zeros(64, dtype=int)
        confidences = np.zeros(64, dtype=float)
        is_ood = np.zeros(64, dtype=bool)

        for i in range(64):
            # 1. Get Neighbors & Labels
            indices = top_k_indices[i].cpu().tolist()
            neighbor_labels = self.normalized_labels[indices].tolist()

            # 2. Vote Logic (Check 3)
            vote_counts = Counter(neighbor_labels)
            predicted_label, best_vote_count = vote_counts.most_common(1)[0]
            consensus_ratio = best_vote_count / self.knn_k

            # 3. Calculate Metrics
            # Metric A: Average Similarity (Works for both now)
            avg_similarity = top_k_scores[i].mean().item()

            # Metric B: 1 / Distance to k-th Neighbor
            # For Unnormalized, top_k_scores is already inverted distance
            kth_similarity = top_k_scores[i, -1].item()

            if self.is_normalized:
                # Convert Cosine Sim back to approximate Distance for OOD check
                kth_dist = np.sqrt(max(0, 2 * (1 - kth_similarity)))
            else:
                # Reverse the conversion: dist = (1/sim) - 1
                kth_dist = (1.0 / (kth_similarity + 1e-9)) - 1.0

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
    def predict_mahalanobis(self, tile_embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Global Mahalanobis prediction using Tied Covariance.
        Returns: predictions, confidences, is_ood
        """
        if self.global_means is None:
            raise ValueError("Call update_thresholds() first to build statistics")

        device = self.global_means.device
        # Use raw embeddings (do not normalize) for Mahalanobis
        tile_embeddings = tile_embeddings.to(device)

        batch_size = tile_embeddings.shape[0]
        n_classes = self.global_means.shape[0]

        predictions = np.zeros(batch_size, dtype=int)
        confidences = np.zeros(batch_size, dtype=float)
        is_ood = np.zeros(batch_size, dtype=bool)

        # Calculate distance to EVERY class for each tile
        # We process one class at a time to save memory
        dists_per_class = torch.full((batch_size, n_classes), float('inf'), device=device)

        # We iterate only over classes that actually exist in training
        # (Assuming we stored them or check for zero-means,
        # but here checking all is safer if means are initialized to 0)

        for c in range(n_classes):
            mean = self.global_means[c]

            # Skip empty classes (if initialized with zeros and never updated)
            if mean.abs().sum() == 0:
                continue

            delta = tile_embeddings - mean

            # Efficient Mahalanobis Distance:
            # dist = sqrt( diag( delta @ Inv @ delta.T ) )

            # 1. temp = delta @ Inv
            temp = torch.matmul(delta, self.global_cov_inv)

            # 2. dot product row-wise
            dist_sq = (temp * delta).sum(dim=1)

            # 3. Sqrt
            dist = torch.sqrt(torch.clamp(dist_sq, min=0))

            dists_per_class[:, c] = dist

        # Find closest class
        min_dists, best_classes = torch.min(dists_per_class, dim=1)

        min_dists_np = min_dists.cpu().numpy()
        best_classes_np = best_classes.cpu().numpy()

        for i in range(batch_size):
            predictions[i] = best_classes_np[i]

            # Confidence: exp(-distance)
            confidences[i] = np.exp(-min_dists_np[i])

            # OOD Check
            if min_dists_np[i] > self.mahal_threshold:
                is_ood[i] = True

        return predictions, confidences, is_ood

    # ============ METHOD 3: temperature ============
    def predict_softmax(self, tile_embeddings: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict using the trained classifier head with temperature scaling.
        Uses self.softmax_temperature and self.softmax_threshold.
        """
        # === FIX: Put the actual checks back here ===
        if self.classifier_head is None:
            raise ValueError("Classifier head not set. Call set_classifier_head() first.")

        device = self.embedding_extractor.device if self.embedding_extractor else torch.device('cuda')

        tile_embeddings = tile_embeddings.to(device)
        target_dtype = next(self.classifier_head.parameters()).dtype
        if tile_embeddings.dtype != target_dtype:
            tile_embeddings = tile_embeddings.to(dtype=target_dtype)
        # ============================================

        # 1. Get Logits
        with torch.no_grad():
            logits = self.classifier_head(tile_embeddings)

        # 2. Apply Temperature Scaling
        scaled_logits = logits / self.softmax_temperature

        # 3. Softmax
        probs = torch.softmax(scaled_logits, dim=1)

        # 4. Get Prediction and Confidence
        confidences, predictions = torch.max(probs, dim=1)

        # 5. OOD Detection
        is_ood = confidences < self.softmax_threshold

        return predictions.cpu().numpy(), confidences.cpu().numpy(), is_ood.cpu().numpy()


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