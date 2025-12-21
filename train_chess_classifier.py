#!/usr/bin/env python3
"""
DSPy Chess Classifier Training Script - BGU Cluster Optimized

This script handles:
- SLURM environment variables and GPU setup
- Data loading from cluster storage
- Checkpointing for long-running jobs
- Progress logging to cluster storage
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import base64
from datetime import datetime

import dspy
import torch
import importlib.util
# Add the DSPy classifier to path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
current_dir = os.path.dirname(os.path.abspath(__file__))
classifier_path = os.path.join(current_dir, "DSPy-Classifier", "dspy-chess-classifier.py")
if not os.path.exists(classifier_path):
    # Fallback: try looking in current directory
    classifier_path = os.path.join(current_dir, "dspy-chess-classifier.py")

if os.path.exists(classifier_path):
    spec = importlib.util.spec_from_file_location("dspy_chess_classifier", classifier_path)
    dspy_chess_classifier = importlib.util.module_from_spec(spec)
    sys.modules["dspy_chess_classifier"] = dspy_chess_classifier
    spec.loader.exec_module(dspy_chess_classifier)
else:
    print(f"CRITICAL ERROR: Could not find classifier file at {classifier_path}")
    sys.exit(1)

from dspy_chess_classifier import (
    Config,
    setup_dspy,
    ChessPieceClassifier,
    ChessboardDataset,
    load_real_dataset,
    evaluate,
    DSPyOptimizer,
    FineTuneTrainer,
)
def file_to_b64(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    # Validate locally (catches “path string” etc.)
    base64.b64decode(b64, validate=True)
    return b64
# ============================================================================
# Setup Logging for Cluster
# ============================================================================

def setup_logging(log_file: str):
    """Setup logging to both console and file."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# SLURM Environment Detection
# ============================================================================

class SLURMEnvironment:
    """Detect and handle SLURM environment variables."""
    
    @staticmethod
    def is_slurm_job():
        """Check if running under SLURM."""
        return 'SLURM_JOB_ID' in os.environ
    
    @staticmethod
    def get_info():
        """Get SLURM job information."""
        return {
            'job_id': os.environ.get('SLURM_JOB_ID', 'N/A'),
            'node_list': os.environ.get('SLURM_JOB_NODELIST', 'N/A'),
            'gpus': os.environ.get('SLURM_GPUS', 'N/A'),
            'cpus': os.environ.get('SLURM_CPUS_PER_TASK', 'N/A'),
            'memory': os.environ.get('SLURM_MEM_PER_NODE', 'N/A'),
            'time_limit': os.environ.get('SLURM_TIME_LIMIT', 'N/A'),
            'scratch_dir': os.environ.get('SLURM_SCRATCH_DIR', 'N/A'),
        }
    
    @staticmethod
    def setup_gpu():
        """Setup GPU environment for SLURM."""
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        logger.info(f"GPU ID: {gpu_id}")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(int(gpu_id.split(',')[0]))
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available! Running on CPU (will be very slow)")


# ============================================================================
# Checkpoint Management
# ============================================================================

class CheckpointManager:
    """Manage checkpoints for resumable training."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: Config, config_file: str = "config.json"):
        """Save configuration."""
        config_path = self.checkpoint_dir / config_file
        config_dict = {
            'model_name': config.model_name,
            'ollama_api_base': config.ollama_api_base,
            'max_tokens': config.max_tokens,
            'temperature': config.temperature,
            'train_split': config.train_split,
            'val_split': config.val_split,
            'test_split': config.test_split,
            'metric': config.metric,
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config saved to {config_path}")
    
    def save_metrics(self, metrics: dict, metrics_file: str = "metrics.json"):
        """Save training metrics."""
        metrics_path = self.checkpoint_dir / metrics_file
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    def get_latest_checkpoint(self):
        """Get latest checkpoint file if it exists."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if checkpoint_files:
            return max(checkpoint_files, key=os.path.getctime)
        return None


# ============================================================================
# Main Training Function
# ============================================================================

def train_classifier(args):
    """Main training function."""
    
    logger.info("=" * 80)
    logger.info("🎯 DSPy Chess Classifier - BGU Cluster Training")
    logger.info("=" * 80)
    
    # Log SLURM environment
    if SLURMEnvironment.is_slurm_job():
        logger.info("Running on BGU Cluster (SLURM)")
        slurm_info = SLURMEnvironment.get_info()
        for key, value in slurm_info.items():
            logger.info(f"  {key}: {value}")
        SLURMEnvironment.setup_gpu()
    else:
        logger.info("Running locally (not on SLURM)")
    
    # ========================================================================
    # 1. Configuration
    # ========================================================================
    
    logger.info("\n1️⃣ Setting up configuration...")
    
    config = Config(
        model_name=args.model,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        train_split=float(args.train_split),
        val_split=float(args.val_split),
        test_split=float(args.test_split),
        batch_size=int(args.batch_size),
        metric=args.metric,
    )
    
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Max tokens: {config.max_tokens}")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Metric: {config.metric}")
    
    # ========================================================================
    # 2. Setup DSPy
    # ========================================================================
    
    logger.info("\n2️⃣ Setting up DSPy...")
    
    try:
        lm = setup_dspy(config)
        logger.info("  ✅ DSPy configured successfully")
    except Exception as e:
        logger.error(f"  ❌ Failed to setup DSPy: {e}")
        return 1
    
    # ========================================================================
    # 3. Create Classifier
    # ========================================================================
    
    logger.info("\n3️⃣ Creating classifier module...")
    
    classifier = ChessPieceClassifier()
    logger.info("  ✅ Classifier created")
    
    # ========================================================================
    # 4. Load Dataset
    # ========================================================================
    
    logger.info("\n4️⃣ Loading dataset...")
    
    data_root = args.data_root
    
    if not os.path.exists(data_root):
        logger.warning(f"  ⚠️ Data directory not found: {data_root}")
        logger.info("  Creating mock dataset for testing...")
        from dspy_chess_classifier import ChessClassificationExample
        real_examples = []  # Empty for now, would need real data
    else:
        logger.info(f"  Loading from {data_root}...")
        
        try:
            # Limit per game to avoid processing millions of squares
            limit = int(args.limit_per_game) if args.limit_per_game else None
            real_examples = load_real_dataset(data_root, limit_per_game=limit)
            logger.info(f"  ✅ Loaded {len(real_examples)} examples")
        except Exception as e:
            logger.error(f"  ❌ Failed to load dataset: {e}")
            real_examples = []
    
    if real_examples:
        try:
            dataset = ChessboardDataset(real_examples, config)
            logger.info(f"  Train: {len(dataset.train)}")
            logger.info(f"  Val: {len(dataset.val)}")
            logger.info(f"  Test: {len(dataset.test)}")
        except Exception as e:
            logger.error(f"  ❌ Failed to create dataset: {e}")
            return 1
    else:
        logger.warning("  ⚠️ No dataset available - skipping training")
        dataset = None
    
    # ========================================================================
    # 5. Optimization
    # ========================================================================
    
    if dataset and len(dataset.train) > 0:
        logger.info("\n5️⃣ Running DSPy Optimization...")
        
        optimizer = DSPyOptimizer(classifier, config)
        
        try:
            classifier = optimizer.optimize(dataset.train, dataset.val)
            logger.info("  ✅ Optimization complete")
        except Exception as e:
            logger.error(f"  ❌ Optimization failed: {e}")
            logger.info("  Continuing with base classifier...")
    
    # ========================================================================
    # 6. Fine-tuning
    # ========================================================================
    
    if dataset and len(dataset.train) > 0:
        logger.info("\n6️⃣ Fine-tuning with adaptive demonstrations...")
        
        trainer = FineTuneTrainer(classifier, config)
        
        try:
            num_epochs = int(args.epochs)
            classifier = trainer.train_with_demonstrations(
                dataset.train,
                dataset.val,
                num_epochs=num_epochs
            )
            
            trainer.adaptive_few_shot(dataset.train, dataset.val)
            logger.info("  ✅ Fine-tuning complete")
        except Exception as e:
            logger.error(f"  ❌ Fine-tuning failed: {e}")
    
    # ========================================================================
    # 7. Evaluation
    # ========================================================================
    
    if dataset and len(dataset.test) > 0:
        logger.info("\n7️⃣ Final Evaluation...")
        
        try:
            test_metric = evaluate(classifier, dataset.test, metric_name=config.metric)
            logger.info(f"  Test {config.metric.upper()}: {test_metric:.4f}")
        except Exception as e:
            logger.error(f"  ❌ Evaluation failed: {e}")
    
    # ========================================================================
    # 8. Save Results
    # ========================================================================
    
    logger.info("\n8️⃣ Saving results...")
    
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)
    
    try:
        # Save config
        checkpoint_manager.save_config(config)
        
        # Save classifier
        import pickle
        classifier_path = Path(args.checkpoint_dir) / "classifier_final.pkl"
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        logger.info(f"  ✅ Classifier saved to {classifier_path}")
        
        # Save metrics
        metrics = {
            'model': config.model_name,
            'timestamp': datetime.now().isoformat(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'N/A'),
        }
        checkpoint_manager.save_metrics(metrics)
        
    except Exception as e:
        logger.error(f"  ❌ Failed to save results: {e}")
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training completed successfully!")
    logger.info("=" * 80)
    
    return 0


# ============================================================================
# CLI Argument Parser
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DSPy Chess Classifier - BGU Cluster Training'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default=os.path.join(os.path.expanduser('~'), 'chess_classifier/data'),
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Checkpoint directory'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='training.log',
        help='Log file path'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen3-vl:7b',
        choices=['qwen3-vl:7b', 'qwen3-vl:32b', 'llama3.2-vision:11b'],
        help='Model name'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='f1',
        choices=['f1', 'accuracy', 'precision', 'recall'],
        help='Evaluation metric'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Max tokens for model output'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Model temperature (0.0-1.0, lower=more deterministic)'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training data split ratio'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation data split ratio'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.1,
        help='Test data split ratio'
    )
    
    parser.add_argument(
        '--limit-per-game',
        type=int,
        default=5,
        help='Limit examples per game (to avoid processing millions)'
    )
    
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID to use'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Run training
    exit_code = train_classifier(args)
    
    # Exit
    sys.exit(exit_code)
