#!/usr/bin/env python3
"""
Unified Training Script for CSC-BSR (Chessboard Square Classification and Board-State Reconstruction)

This script provides a modular training pipeline for the complete model training workflow:
1. Dataset Preprocessing (build_dataset.py)
2. Embedding Model Fine-tuning (fine_tune.py with DINO backbone)
3. Binary OOD Guard Training (train_binary_ood.py, auto-selects best epoch)

Usage Examples:
    # Train complete pipeline
    python train.py --stage all

    # Train individual stages
    python train.py --stage preprocess --config preprocessing/dataset_config.example.yaml
    python train.py --stage embedding --epochs 10 --batch-size 32
    python train.py --stage ood --epochs 5 --batch-size 8

    # Custom configuration
    python train.py --stage all --epochs 10 --batch-size 16 --dino-size small

For data placement instructions, see README.md
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


class TrainingPipeline:
    """Orchestrates the complete training pipeline"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = PROJECT_ROOT
        
        # Define paths
        self.data_dir = self.project_root / "data"
        self.splits_dir = self.data_dir / "splits"
        self.preprocessed_dir = self.data_dir / "preprocessed_data"
        self.embedding_dir = self.project_root / "embedding"
        
    def validate_raw_data(self) -> bool:
        """Validate that raw data exists for preprocessing"""
        logger.info("Validating raw data presence...")
        
        # Check for game folders
        game_folders = list(self.data_dir.glob("game*_per_frame"))
        if not game_folders:
            logger.error(f"No game folders found in {self.data_dir}")
            logger.error("Expected folders like: game2_per_frame, game4_per_frame, etc.")
            return False
        
        # Check for hands folder (OOD data)
        hands_dir = self.data_dir / "hands"
        if not hands_dir.exists():
            logger.warning(f"Hands folder not found at {hands_dir}")
            logger.warning("OOD detection may not work properly without hands data")
        
        logger.info(f"Found {len(game_folders)} game folders")
        return True
    
    def check_preprocessing_done(self) -> bool:
        """Check if preprocessing has been completed"""
        train_split = self.splits_dir / "train.csv"
        val_split = self.splits_dir / "val.csv"
        test_split = self.splits_dir / "test.csv"
        
        all_exist = train_split.exists() and val_split.exists() and test_split.exists()
        
        if all_exist:
            logger.info("✓ Preprocessing already completed (splits exist)")
            if self.preprocessed_dir.exists():
                tile_count = len(list(self.preprocessed_dir.glob("*.png")))
                logger.info(f"  Found {tile_count} preprocessed tiles")
        
        return all_exist
    
    def run_preprocessing(self) -> bool:
        """Run dataset preprocessing"""
        logger.info("="*80)
        logger.info("STAGE 1: Dataset Preprocessing")
        logger.info("="*80)
        
        # Check if already done
        if self.check_preprocessing_done() and not self.args.force_rebuild:
            logger.info("Skipping preprocessing (already done). Use --force-rebuild to rerun.")
            return True
        
        # Validate raw data
        if not self.validate_raw_data():
            logger.error("Raw data validation failed. Please check data placement.")
            return False
        
        # Import and run build_dataset
        try:
            from preprocessing.build_dataset import main as build_dataset_main
            
            # Override sys.argv for build_dataset
            original_argv = sys.argv.copy()
            sys.argv = ["build_dataset.py"]
            if self.args.config:
                sys.argv.extend(["--config", self.args.config])
            
            logger.info(f"Running: preprocessing/build_dataset.py {' '.join(sys.argv[1:])}")
            build_dataset_main()
            
            # Restore original argv
            sys.argv = original_argv
            
            logger.info("✓ Preprocessing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_embedding_model_exists(self) -> Optional[Path]:
        """Check if embedding model already exists"""
        model_name = f"chess_encoder_finetuned_dino-{self.args.dino_size}_backbone.pt"
        model_path = self.embedding_dir / model_name
        
        if model_path.exists():
            logger.info(f"✓ Embedding model found: {model_path}")
            return model_path
        return None
    
    def run_embedding_training(self) -> bool:
        """Run embedding model fine-tuning (DINO backbone strategy)"""
        logger.info("="*80)
        logger.info("STAGE 2: Embedding Model Fine-tuning (DINO Backbone)")
        logger.info("="*80)
        
        # Check if preprocessing is done
        if not self.check_preprocessing_done():
            logger.error("Preprocessing not completed. Run preprocessing stage first.")
            return False
        
        # Check if model already exists
        existing_model = self.check_embedding_model_exists()
        if existing_model and not self.args.force_retrain:
            logger.info("Skipping embedding training (model exists). Use --force-retrain to rerun.")
            return True
        
        try:
            from embedding.fine_tune import train_fine_tuning
            from embedding.dinov2 import DINOv2Embedding
            
            logger.info(f"Training DINO-{self.args.dino_size} with backbone strategy")
            logger.info(f"  Epochs: {self.args.epochs}")
            logger.info(f"  Batch size: {self.args.batch_size}")
            logger.info(f"  Num workers: {self.args.num_workers}")
            
            # Initialize DINO model
            embedding_model = DINOv2Embedding(model_size=self.args.dino_size)
            
            # Train
            train_fine_tuning(
                splits_dir=str(self.splits_dir),
                embedding_model=embedding_model,
                path_root=str(self.project_root),  # Changed from "data" to "."
                epochs=self.args.epochs,
                batch_size=self.args.batch_size,
                use_val=True,
                num_workers=self.args.num_workers,
                strategy="backbone",
                embedding_model_name=f"dino-{self.args.dino_size}"
            )
            
            logger.info("✓ Embedding training completed successfully")
            
            # Copy model to root directory with canonical name
            source = self.embedding_dir / f"chess_encoder_finetuned_dino-{self.args.dino_size}_backbone.pt"
            dest = self.project_root / f"chess_encoder_finetuned_dino-{self.args.dino_size}_backbone.pt"
            if source.exists() and not dest.exists():
                shutil.copy2(source, dest)
                logger.info(f"✓ Copied model to root: {dest.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def select_best_ood_checkpoint(self, dino_size: str, total_epochs: int) -> Optional[Path]:
        """
        Select best OOD checkpoint based on validation performance.
        As per README.md: epoch 3 is the best (validation got worse after that)
        """
        # As documented in README, epoch 3 is the best checkpoint
        best_epoch = 3
        
        if total_epochs < best_epoch:
            logger.warning(f"Training only {total_epochs} epochs, but best is epoch {best_epoch}")
            best_epoch = total_epochs
        
        checkpoint_path = self.embedding_dir / f"binary_ood_dino_{dino_size}_epoch{best_epoch}.pt"
        
        if checkpoint_path.exists():
            logger.info(f"✓ Selected best checkpoint: epoch {best_epoch} (as per README.md)")
            return checkpoint_path
        else:
            logger.warning(f"Best checkpoint not found: {checkpoint_path}")
            # Fallback to last epoch
            for epoch in range(total_epochs, 0, -1):
                fallback = self.embedding_dir / f"binary_ood_dino_{dino_size}_epoch{epoch}.pt"
                if fallback.exists():
                    logger.info(f"Using fallback checkpoint: epoch {epoch}")
                    return fallback
        
        return None
    
    def check_ood_model_exists(self) -> Optional[Path]:
        """Check if OOD model already exists in root"""
        model_name = f"binary_ood_dino_{self.args.dino_size}_epoch3.pt"
        model_path = self.project_root / model_name
        
        if model_path.exists():
            logger.info(f"✓ OOD model found: {model_path}")
            return model_path
        return None
    
    def run_ood_training(self) -> bool:
        """Run binary OOD guard training"""
        logger.info("="*80)
        logger.info("STAGE 3: Binary OOD Guard Training")
        logger.info("="*80)
        
        # Check if preprocessing is done
        if not self.check_preprocessing_done():
            logger.error("Preprocessing not completed. Run preprocessing stage first.")
            return False
        
        # Check if model already exists
        existing_model = self.check_ood_model_exists()
        if existing_model and not self.args.force_retrain:
            logger.info("Skipping OOD training (model exists). Use --force-retrain to rerun.")
            return True
        
        try:
            from embedding.train_binary_ood import train_binary_ood
            
            logger.info(f"Training Binary OOD with DINO-{self.args.dino_size}")
            logger.info(f"  Epochs: {self.args.ood_epochs}")
            logger.info(f"  Batch size: {self.args.batch_size}")
            logger.info(f"  Best epoch will be auto-selected (epoch 3 as per README)")
            
            # Train
            train_binary_ood(
                epochs=self.args.ood_epochs,
                batch_size=self.args.batch_size,
                dino_size=self.args.dino_size,
                num_workers=self.args.num_workers
            )
            
            logger.info("✓ OOD training completed successfully")
            
            # Select and copy best checkpoint to root
            best_checkpoint = self.select_best_ood_checkpoint(
                self.args.dino_size,
                self.args.ood_epochs
            )
            
            if best_checkpoint:
                dest = self.project_root / best_checkpoint.name
                if not dest.exists() or self.args.force_retrain:
                    shutil.copy2(best_checkpoint, dest)
                    logger.info(f"✓ Copied best OOD model to root: {dest.name}")
            else:
                logger.warning("Could not find best checkpoint to copy")
            
            return True
            
        except Exception as e:
            logger.error(f"OOD training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_comparison(self) -> bool:
        """Run strategy comparison experiment (optional)"""
        logger.info("="*80)
        logger.info("OPTIONAL: Strategy Comparison Experiment")
        logger.info("="*80)
        
        # Check if preprocessing is done
        if not self.check_preprocessing_done():
            logger.error("Preprocessing not completed. Run preprocessing stage first.")
            return False
        
        try:
            from embedding.experiment_runner import main as experiment_main
            
            # Override sys.argv for experiment_runner
            original_argv = sys.argv.copy()
            sys.argv = [
                "experiment_runner.py",
                "--splits-dir", str(self.splits_dir),
                "--path-root", str(self.project_root),
                "--epochs", str(self.args.epochs),
                "--batch-size", str(self.args.batch_size),
                "--num-workers", str(self.args.num_workers),
                "--dino-size", self.args.dino_size
            ]
            
            logger.info(f"Running: embedding/experiment_runner.py with {self.args.epochs} epochs")
            logger.info("This will train 4 different strategies for comparison")
            
            experiment_main()
            
            # Restore original argv
            sys.argv = original_argv
            
            logger.info("✓ Strategy comparison completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """Print training summary and next steps"""
        logger.info("="*80)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("="*80)
        
        logger.info("\nGenerated Models:")
        
        # Check embedding model
        emb_model = self.project_root / f"chess_encoder_finetuned_dino-{self.args.dino_size}_backbone.pt"
        if emb_model.exists():
            logger.info(f"  ✓ {emb_model.name}")
        else:
            logger.warning(f"  ✗ {emb_model.name} (not found)")
        
        # Check OOD model
        ood_model = self.project_root / f"binary_ood_dino_{self.args.dino_size}_epoch3.pt"
        if ood_model.exists():
            logger.info(f"  ✓ {ood_model.name}")
        else:
            logger.warning(f"  ✗ {ood_model.name} (not found)")
        
        # Check classifier database (if it exists from KNN training)
        classifier_db = self.project_root / f"classifier_dino_{self.args.dino_size}.pt"
        if classifier_db.exists():
            logger.info(f"  ✓ {classifier_db.name}")
        else:
            logger.info(f"  ℹ {classifier_db.name} (optional, created during KNN setup)")
        
        logger.info("\nNext Steps:")
        logger.info("  1. The models are ready for inference")
        logger.info("  2. Run predictions: python predict_board.py")
        logger.info("  3. See README.md for more information")
    
    def _execute_stage(self, stage: str) -> bool:
        """Execute a single training stage"""
        stage_map = {
            "preprocess": self.run_preprocessing,
            "embedding": self.run_embedding_training,
            "ood": self.run_ood_training,
            "compare": self.run_comparison
        }
        
        handler = stage_map.get(stage)
        if handler:
            return handler()
        return False
    
    def run(self) -> bool:
        """Execute the training pipeline based on selected stage"""
        stages = self._get_stages_to_run()
        
        logger.info(f"Starting training pipeline: {' → '.join(stages)}")
        logger.info(f"Project root: {self.project_root}")
        
        success = self._run_all_stages(stages)
        
        if success and self.args.stage in ["all", "ood"]:
            self.print_summary()
        
        return success
    
    def _get_stages_to_run(self):
        """Determine which stages to run based on args"""
        if self.args.stage == "all":
            return ["preprocess", "embedding", "ood"]
        elif self.args.stage == "compare":
            return ["preprocess", "compare"]
        else:
            return [self.args.stage]
    
    def _run_all_stages(self, stages) -> bool:
        """Run all stages in sequence, stop on first failure"""
        for stage in stages:
            if not self._execute_stage(stage):
                return False
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified training script for CSC-BSR chess classification project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train complete pipeline
  python train.py --stage all

  # Train only preprocessing
  python train.py --stage preprocess --config preprocessing/dataset_config.example.yaml

  # Train only embedding model
  python train.py --stage embedding --epochs 10 --batch-size 32

  # Train only OOD guard
  python train.py --stage ood --epochs 5 --batch-size 8

  # Run strategy comparison
  python train.py --stage compare --epochs 3

For data placement instructions, see README.md
        """
    )
    
    # Stage selection
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "embedding", "ood", "compare"],
        default="all",
        help="Training stage to run (default: all)"
    )
    
    # Preprocessing arguments
    parser.add_argument(
        "--config",
        default="preprocessing/dataset_config.example.yaml",
        help="Config file for preprocessing (default: preprocessing/dataset_config.example.yaml)"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild dataset even if splits exist"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for embedding training (default: 10)"
    )
    parser.add_argument(
        "--ood-epochs",
        type=int,
        default=5,
        help="Number of epochs for OOD training (default: 5, best=epoch 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)"
    )
    parser.add_argument(
        "--dino-size",
        choices=["small", "base"],
        default="small",
        help="DINO model size (default: small)"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain models even if they exist"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrainingPipeline(args)
    success = pipeline.run()
    
    if success:
        logger.info("✓ Training pipeline completed successfully!")
        return 0
    else:
        logger.error("✗ Training pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
