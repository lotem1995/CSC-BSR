#!/bin/bash
## Chess Embedding/Classifier job for single-GPU (11GB) nodes
## Target GPUs: RTX 2080 Ti / GTX 1080 Ti
## Adjust partition/constraint to match your cluster naming.

#SBATCH --job-name=chess_embed
#SBATCH --qos=course
#SBATCH --partition=rtx2080ti
#SBATCH --output=logs/chess_embed-%j.out
#SBATCH --error=logs/chess_embed-%j.err
#SBATCH --mail-user=lotemsak@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=0-02:00:00
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --gpus=rtx_2080ti
## Example partition/constraint (commented—adapt to your cluster)
## #SBATCH --partition=rtx2080
## #SBATCH --constraint=rtx2080ti

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints results

# Conda/venv activation (edit as needed)
module load cuda/12.1
# source ~/miniconda3/bin/activate chess_env
echo "Working directory: $(pwd)"
## Make sure you're not in conda environment when submitting!
conda deactivate 2>/dev/null || true
# Use the small Qwen variant by default; override with export QWEN_MODEL_NAME=...
export QWEN_MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
# Force single-GPU placement even if multiple are visible
export QWEN_DEVICE_MAP="cuda:0"

source activate chess_embedding
# Tunables for quick, low-VRAM runs
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-2}
EPOCHS=${EPOCHS:-1}
SPLITS_DIR=${SPLITS_DIR:-data/splits}
DATA_ROOT=${DATA_ROOT:-data}
EMBEDDING_BACKBONE=${EMBEDDING_BACKBONE:-qwen}  # qwen|dino-small|dino-base

# Run all scenarios sequentially (Qwen/DINO head-only, Qwen LoRA, DINO backbone)
python embedding/experiment_runner.py \
  --splits-dir "$SPLITS_DIR" \
  --path-root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

EXIT_CODE=$?
echo "Job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE