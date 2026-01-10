#!/bin/bash
## Chess Embedding/Classifier job for single-GPU (11GB) nodes
## Target GPUs: RTX 2080 Ti / GTX 1080 Ti
## Adjust partition/constraint to match your cluster naming.

#SBATCH --job-name=chess_embed
#SBATCH --account=brafman
#SBATCH --qos=normal
#SBATCH --partition=rtx4090
#SBATCH --output=logs/chess_embed-%j.out
#SBATCH --error=logs/chess_embed-%j.err
#SBATCH --mail-user=lotemsak@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=0-04:00:00
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=10
#SBATCH --gpus=rtx_4090:1
## Example partition/constraint (commented—adapt to your cluster)
## #SBATCH --partition=rtx2080
## #SBATCH --constraint=rtx2080ti

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints results

# Conda/venv activation (edit as needed)
module load anaconda
module load cuda/13
# source ~/miniconda3/bin/activate chess_env
echo "Working directory: $(pwd)"
## Make sure you're not in conda environment when submitting!
conda deactivate 2>/dev/null || true
# Use the small Qwen variant by default; override with export QWEN_MODEL_NAME=...
export QWEN_MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
# Force single-GPU placement even if multiple are visible
export QWEN_DEVICE_MAP="cuda:0"

source activate chess_embedding

#build dataset (if not already built)
python preprocessing/build_dataset.py --config preprocessing/dataset_config.example.yaml --output data/dataset_manifest.json --split-indices-dir data/splits

# fine-tune embedding model
python embedding/fine_tune.py --splits-dir data/splits --path-root . --epochs 1 --batch-size 2 --num-workers 2 --embedding-model dino-small --strategy backbone

#test clasiifer
python embedding/test_classifier.py
EXIT_CODE=$?
echo "Job finished with exit code: $EXIT_CODE"
exit $EXIT_CODE