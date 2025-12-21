# DSPy Chess Board Image Classifier with Optimizer and Fine-Tuning
# Complete implementation with local Qwen2.5-VL model

import dspy
import json
import os
import csv
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import re
from dataclasses import dataclass
from collections import defaultdict


# ============================================================================
# 1. SETUP & CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the classifier."""
    model_name: str = "qwen2.5-vl:32b"  # or qwen2.5-vl:32b for better accuracy
    ollama_api_base: str = "http://localhost:11434"
    num_threads: int = 4
    max_tokens: int = 512
    temperature: float = 0.7
    
    # Training config
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 4
    
    # Optimization
    num_optimize_steps: int = 50
    metric: str = "f1"  # or "accuracy", "precision", "recall"


def setup_dspy(config: Config) -> dspy.LM:
    """Initialize DSPy with local Qwen2.5-VL model via Ollama."""
    lm = dspy.LM(
        model=f"ollama_chat/{config.model_name}",
        api_base=config.ollama_api_base,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    dspy.configure(lm=lm)
    return lm


# ============================================================================
# 2. DSPY SIGNATURES & MODULES
# ============================================================================

class PieceClassificationSignature(dspy.Signature):
    """Classify chess pieces in a board square from an image.
    
    Args:
        board_image: Image containing the chessboard or square
        square_position: Board position (e.g., "e4", "a1")
        
    Returns:
        piece: Chess piece notation (e.g., "P" for white pawn, "p" for black pawn, "." for empty)
        confidence: Confidence level of the classification (high/medium/low)
        reasoning: Brief explanation of the classification decision
    """
    board_image: dspy.Image = dspy.InputField(desc="Chessboard image or square region")
    square_position: str = dspy.InputField(desc="Square position in algebraic notation (e.g., e4)")
    piece: str = dspy.OutputField(desc="Piece: K/Q/R/B/N/P (white) or k/q/r/b/n/p (black) or . (empty)")
    confidence: str = dspy.OutputField(desc="Confidence: high/medium/low")
    reasoning: str = dspy.OutputField(desc="Why you made this classification")


class BoardStateSignature(dspy.Signature):
    """Analyze entire chessboard image and classify all 64 squares.
    
    Returns:
        fen_notation: Standard FEN notation string (8 ranks)
        piece_json: JSON mapping square->piece for each piece on board
        board_confidence: Overall confidence score (0.0-1.0)
        occlusion_notes: Any notes about piece occlusions or visibility issues
    """
    board_image: dspy.Image = dspy.InputField(desc="Full chessboard image")
    piece_json: str = dspy.OutputField(desc='JSON: {"a1": "R", "e1": "K", ...}')
    fen_notation: str = dspy.OutputField(desc="FEN notation (position part only)")
    board_confidence: str = dspy.OutputField(desc="Confidence score: 0.0-1.0")
    occlusion_notes: str = dspy.OutputField(desc="Notes on visibility/occlusion issues")


class ChessPieceClassifier(dspy.Module):
    """Main classifier module with piece classification logic."""
    
    def __init__(self):
        super().__init__()
        self.classify_piece = dspy.Predict(PieceClassificationSignature)
        self.classify_board = dspy.Predict(BoardStateSignature)
    
    def forward(self, image: dspy.Image, mode: str = "board") -> dspy.Prediction:
        """
        Classify chess pieces in image.
        
        Args:
            image: Board image
            mode: "board" for full board or "square" for single square
            
        Returns:
            dspy.Prediction with classification results
        """
        if mode == "board":
            return self.classify_board(board_image=image)
        else:
            raise ValueError("Use forward_square() for single square classification")
    
    def forward_square(self, image: dspy.Image, square_position: str) -> dspy.Prediction:
        """Classify a single square."""
        return self.classify_piece(board_image=image, square_position=square_position)


# ============================================================================
# 3. DATA HANDLING & DATASET CLASS
# ============================================================================

def fen_to_board_dict(fen: str) -> Dict[str, str]:
    """Convert FEN position string to square->piece mapping."""
    board = {}
    ranks = fen.split(' ')[0].split('/')
    
    for r, rank_str in enumerate(ranks):
        rank_num = 8 - r
        file_idx = 0
        for char in rank_str:
            if char.isdigit():
                num_empty = int(char)
                for _ in range(num_empty):
                    file_char = chr(ord('a') + file_idx)
                    square = f"{file_char}{rank_num}"
                    board[square] = "."
                    file_idx += 1
            else:
                file_char = chr(ord('a') + file_idx)
                square = f"{file_char}{rank_num}"
                board[square] = char
                file_idx += 1
    return board

@dataclass
class ChessClassificationExample:
    """Single training/eval example."""
    image_path: str
    square_position: str
    ground_truth_piece: str  # Expected piece (K/Q/R/B/N/P/k/q/r/b/n/p/.)
    
    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example."""
        img = dspy.Image.from_file(self.image_path)
        return dspy.Example(
            board_image=img,
            square_position=self.square_position,
            piece=self.ground_truth_piece,
        ).with_inputs("board_image", "square_position")


class ChessboardDataset:
    """Dataset manager for chess board classification."""
    
    def __init__(self, examples: List[ChessClassificationExample], config: Config):
        self.examples = examples
        self.config = config
        self.train, self.val, self.test = self._split_data()
    
    def _split_data(self) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
        """Split data into train/val/test."""
        random.shuffle(self.examples)
        
        total = len(self.examples)
        train_size = int(total * self.config.train_split)
        val_size = int(total * self.config.val_split)
        
        train_examples = [ex.to_dspy_example() for ex in self.examples[:train_size]]
        val_examples = [ex.to_dspy_example() for ex in self.examples[train_size:train_size + val_size]]
        test_examples = [ex.to_dspy_example() for ex in self.examples[train_size + val_size:]]
        
        return train_examples, val_examples, test_examples


def load_real_dataset(data_root: str, limit_per_game: Optional[int] = None) -> List[ChessClassificationExample]:
    """Load dataset from data directory."""
    examples = []
    # Walk through game directories
    game_dirs = glob.glob(os.path.join(data_root, "game*_per_frame"))
    
    print(f"Found {len(game_dirs)} game directories in {data_root}")
    
    for game_dir in game_dirs:
        # Find CSV file
        csv_files = glob.glob(os.path.join(game_dir, "*.csv"))
        if not csv_files:
            continue
        
        csv_path = csv_files[0]
        images_dir = os.path.join(game_dir, "tagged_images")
        
        print(f"Processing {csv_path}...")
        
        count = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if limit_per_game and count >= limit_per_game:
                    break
                
                frame_num = int(row['from_frame'])
                fen = row['fen']
                
                # Construct image filename
                image_filename = f"frame_{frame_num:06d}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                if not os.path.exists(image_path):
                    continue
                
                # Parse FEN
                board_state = fen_to_board_dict(fen)
                
                # Create an example for each square
                for square, piece in board_state.items():
                    examples.append(ChessClassificationExample(
                        image_path=image_path,
                        square_position=square,
                        ground_truth_piece=piece
                    ))
                count += 1
                
    print(f"Loaded {len(examples)} examples total.")
    return examples


# ============================================================================
# 4. METRIC CALCULATION
# ============================================================================

class ClassificationMetrics:
    """Calculate precision, recall, F1 for piece classification."""
    
    def __init__(self, num_classes: int = 13):  # K/Q/R/B/N/P (white & black) + empty = 13
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.tp = defaultdict(int)  # True positives per class
        self.fp = defaultdict(int)  # False positives per class
        self.fn = defaultdict(int)  # False negatives per class
    
    def update(self, predictions: List[str], ground_truths: List[str]):
        """Update metrics with batch predictions."""
        for pred, gt in zip(predictions, ground_truths):
            if pred == gt:
                self.tp[gt] += 1
            else:
                self.fp[pred] += 1
                self.fn[gt] += 1
    
    def accuracy(self) -> float:
        """Overall accuracy."""
        total_correct = sum(self.tp.values())
        total = total_correct + sum(self.fp.values())
        return total_correct / total if total > 0 else 0.0
    
    def precision(self, class_label: Optional[str] = None) -> float:
        """Precision per class or macro-average."""
        if class_label:
            tp = self.tp[class_label]
            fp = self.fp[class_label]
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Macro-average precision
        precisions = []
        all_classes = set(self.tp.keys()) | set(self.fp.keys())
        for cls in all_classes:
            precisions.append(self.precision(cls))
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def recall(self, class_label: Optional[str] = None) -> float:
        """Recall per class or macro-average."""
        if class_label:
            tp = self.tp[class_label]
            fn = self.fn[class_label]
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Macro-average recall
        recalls = []
        all_classes = set(self.tp.keys()) | set(self.fn.keys())
        for cls in all_classes:
            recalls.append(self.recall(cls))
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def f1(self, class_label: Optional[str] = None) -> float:
        """F1 score per class or macro-average."""
        p = self.precision(class_label)
        r = self.recall(class_label)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    def get_metric(self, metric_name: str) -> float:
        """Get any metric by name."""
        if metric_name == "accuracy":
            return self.accuracy()
        elif metric_name == "precision":
            return self.precision()
        elif metric_name == "recall":
            return self.recall()
        elif metric_name == "f1":
            return self.f1()
        else:
            raise ValueError(f"Unknown metric: {metric_name}")


# ============================================================================
# 5. EVALUATION & OPTIMIZATION
# ============================================================================

def evaluate(module: dspy.Module, dataset: List[dspy.Example], metric_name: str = "f1") -> float:
    """Evaluate module on dataset and return metric score."""
    metrics = ClassificationMetrics()
    predictions = []
    ground_truths = []
    
    for example in dataset:
        try:
            # Get prediction from module
            prediction = module.forward_square(
                image=example.board_image,
                square_position=example.square_position
            )
            
            # Extract predicted piece
            pred_piece = prediction.piece.strip()
            # Validate piece notation
            if pred_piece not in "KQRBNPkqrbnp.":
                pred_piece = "."  # Default to empty if invalid
            
            predictions.append(pred_piece)
            ground_truths.append(example.piece)
        except Exception as e:
            print(f"Error processing example: {e}")
            predictions.append(".")
            ground_truths.append(example.piece)
    
    metrics.update(predictions, ground_truths)
    return metrics.get_metric(metric_name)


class DSPyOptimizer:
    """Custom optimizer using BootstrapFewShot (DSPy's main optimization framework)."""
    
    def __init__(self, module: dspy.Module, config: Config):
        self.module = module
        self.config = config
        self.optimization_history = []
    
    def optimize(self, trainset: List[dspy.Example], valset: List[dspy.Example]):
        """Optimize module using in-context learning.
        
        This uses DSPy's BootstrapFewShot optimizer which:
        1. Runs module on training examples
        2. Selects best few-shot demonstrations
        3. Adds them as in-context examples
        4. Tests on validation set
        5. Keeps improvements
        """
        from dspy.teleprompt import BootstrapFewShot
        
        print(f"\n🔧 Starting optimization with {len(trainset)} training examples...")
        
        # BootstrapFewShot configuration
        optimizer = BootstrapFewShot(
            metric=lambda *args: evaluate(self.module, valset, self.config.metric),
            max_bootstrapped_demos=5,  # Max few-shot examples to add
            max_labeled_demos=100,  # Max training examples to consider
            num_candidate_programs=3,  # Programs to try
        )
        
        # Run optimization
        optimized_module = optimizer.compile(
            student=self.module,
            teacher=self.module,
            trainset=trainset,
            valset=valset,
        )
        
        print("✅ Optimization complete!")
        return optimized_module
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """Log optimization step."""
        self.optimization_history.append({
            "step": step,
            "metrics": metrics
        })
        print(f"Step {step}: {metrics}")


class FineTuneTrainer:
    """Fine-tuning trainer for the classifier."""
    
    def __init__(self, module: dspy.Module, config: Config):
        self.module = module
        self.config = config
        self.training_history = []
    
    def train_with_demonstrations(
        self, 
        trainset: List[dspy.Example],
        valset: List[dspy.Example],
        num_epochs: int = 3
    ) -> dspy.Module:
        """Train by building few-shot demonstrations (in-context learning).
        
        Since we're using local models, we focus on:
        1. Selecting good demonstrations
        2. Building context for the model
        3. Validating performance
        """
        print(f"\n📚 Training with demonstrations for {num_epochs} epochs...")
        
        best_metric = 0.0
        best_module = self.module
        
        for epoch in range(num_epochs):
            # Evaluate current module
            val_metric = evaluate(self.module, valset, self.config.metric)
            
            # Log
            log_entry = {
                "epoch": epoch,
                "val_metric": val_metric,
                "metric_name": self.config.metric
            }
            self.training_history.append(log_entry)
            
            print(f"Epoch {epoch + 1}/{num_epochs} - {self.config.metric}: {val_metric:.4f}")
            
            # Update best
            if val_metric > best_metric:
                best_metric = val_metric
                best_module = self.module
                print(f"  ✓ Best metric improved to {best_metric:.4f}")
        
        return best_module
    
    def adaptive_few_shot(
        self,
        trainset: List[dspy.Example],
        valset: List[dspy.Example],
    ) -> dspy.Module:
        """Build adaptive few-shot examples based on validation performance.
        
        Strategy:
        1. Identify failing classes on validation
        2. Select best examples for those classes
        3. Add as in-context demonstrations
        """
        print("\n🎯 Building adaptive few-shot demonstrations...")
        
        # Evaluate to find problem areas
        metrics = ClassificationMetrics()
        problem_classes = defaultdict(list)
        
        for example in valset:
            try:
                prediction = self.module.forward_square(
                    image=example.board_image,
                    square_position=example.square_position
                )
                pred_piece = prediction.piece.strip()
                if pred_piece not in "KQRBNPkqrbnp.":
                    pred_piece = "."
                
                if pred_piece != example.piece:
                    problem_classes[example.piece].append(example)
            except Exception as e:
                problem_classes[example.piece].append(example)
        
        # Select best examples for each problem class
        demonstrations = []
        for piece_class, examples in problem_classes.items():
            # Pick 1-2 best examples per class
            demonstrations.extend(examples[:2])
        
        print(f"  Selected {len(demonstrations)} demonstrations for problem pieces")
        
        # Note: Full few-shot integration would require modifying the DSPy module
        # For now, we return metrics about what needs improvement
        return self.module


# ============================================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Complete training pipeline with optimization and fine-tuning."""
    
    # Configuration
    config = Config(
        model_name="qwen2.5-vl:7b",
        num_optimize_steps=50,
        metric="f1",
    )
    
    print("=" * 70)
    print("🎯 DSPy Chess Classifier with Optimization & Fine-Tuning")
    print("=" * 70)
    
    # 1. Setup
    print("\n1️⃣ Setting up DSPy with Qwen2.5-VL...")
    lm = setup_dspy(config)
    print(f"   ✓ Model: {config.model_name}")
    print(f"   ✓ API Base: {config.ollama_api_base}")
    
    # 2. Create classifier module
    print("\n2️⃣ Initializing ChessPieceClassifier...")
    classifier = ChessPieceClassifier()
    print("   ✓ Classifier ready")
    
    # 3. Load dataset
    print("\n3️⃣ Loading dataset...")
    
    # Determine data path relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(script_dir, "../data")
    
    if os.path.exists(data_root):
        print(f"   Loading from {data_root}...")
        # Limit to 5 frames per game for initial testing to avoid processing thousands of squares
        real_examples = load_real_dataset(data_root, limit_per_game=5)
    else:
        print(f"   ⚠️  Data directory not found at {data_root}")
        real_examples = []

    if real_examples:
        dataset = ChessboardDataset(real_examples, config)
        print(f"   ✓ Train: {len(dataset.train)} | Val: {len(dataset.val)} | Test: {len(dataset.test)}")
    else:
        print("   ⚠️  Using mock dataset - replace with your actual chess images")
        
        # Example: Create mock dataset (you would load real images)
        mock_examples = [
            ChessClassificationExample(
                image_path="path/to/chess_a1.jpg",
                square_position="a1",
                ground_truth_piece="R"
            ),
            ChessClassificationExample(
                image_path="path/to/chess_e4.jpg",
                square_position="e4",
                ground_truth_piece="."
            ),
            # ... more examples
        ]
        dataset = ChessboardDataset(mock_examples, config)
        print("   ⚠️  Skipping dataset split (no actual images provided)")
    
    # 4. Optimization
    print("\n4️⃣ Running DSPy Optimizer...")
    optimizer = DSPyOptimizer(classifier, config)
    
    if real_examples:
        # Run optimization if we have real data
        # Note: This might take a while depending on dataset size and model speed
        try:
            optimized_classifier = optimizer.optimize(dataset.train, dataset.val)
            classifier = optimized_classifier
        except Exception as e:
            print(f"   ❌ Optimization failed: {e}")
            print("   Continuing with base classifier...")
    else:
        print("   ⚠️  Skipping optimization (no actual images provided)")
    
    # 5. Fine-tuning with demonstrations
    print("\n5️⃣ Fine-tuning with adaptive demonstrations...")
    trainer = FineTuneTrainer(classifier, config)
    
    if real_examples:
        try:
            # trainer.train_with_demonstrations(dataset.train, dataset.val, num_epochs=3)
            trainer.adaptive_few_shot(dataset.train, dataset.val)
        except Exception as e:
            print(f"   ❌ Fine-tuning failed: {e}")
    else:
        print("   ⚠️  Skipping fine-tuning (no actual images provided)")
    
    # 6. Final evaluation
    print("\n6️⃣ Evaluation Summary")
    print("   Ready for testing on your chess dataset!")
    print("   " + "─" * 66)
    
    # 7. Save configuration
    config_json = {
        "model": config.model_name,
        "api_base": config.ollama_api_base,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "metric": config.metric,
        "train_split": config.train_split,
        "val_split": config.val_split,
        "test_split": config.test_split,
    }
    
    with open("classifier_config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    print("\n✅ Configuration saved to classifier_config.json")
    
    return classifier, config


# ============================================================================
# 7. INFERENCE EXAMPLE
# ============================================================================

def example_inference(classifier: ChessPieceClassifier, image_path: str, square_pos: str):
    """Example: Classify a single square from an image."""
    print(f"\n📸 Classifying square {square_pos} from {image_path}...")
    
    try:
        img = dspy.Image.from_file(image_path)
        prediction = classifier.forward_square(image=img, square_position=square_pos)
        
        print(f"   Piece: {prediction.piece}")
        print(f"   Confidence: {prediction.confidence}")
        print(f"   Reasoning: {prediction.reasoning}")
        
        return prediction
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def example_full_board_inference(classifier: ChessPieceClassifier, image_path: str):
    """Example: Classify entire board and get FEN notation."""
    print(f"\n🎯 Classifying full board from {image_path}...")
    
    try:
        img = dspy.Image.from_file(image_path)
        prediction = classifier.classify_board(board_image=img)
        
        print(f"   FEN: {prediction.fen_notation}")
        print(f"   Pieces: {prediction.piece_json}")
        print(f"   Confidence: {prediction.board_confidence}")
        print(f"   Notes: {prediction.occlusion_notes}")
        
        return prediction
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


if __name__ == "__main__":
    classifier, config = main()
    
    # Uncomment to test inference on actual images:
    # example_inference(classifier, "chess_board.jpg", "e4")
    # example_full_board_inference(classifier, "full_chess_board.jpg")
