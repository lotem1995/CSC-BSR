#!/bin/bash
## DSPy Chess Classifier - BGU Cluster Training Script
## Optimized for BGU's SLURM cluster with Qwen2.5-VL

#SBATCH --account=brafman
#SBATCH --qos=normal
#SBATCH --partition=rtx4090
#SBATCH --time=0-4:00:00
#SBATCH --job-name=chess_classifier
#SBATCH --output=chess_classifier-%J.log
#SBATCH --error=chess_classifier-%J.err
#SBATCH --mail-user=lotemsak@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
## ============================================================================
## BGU Cluster Environment Setup
## ============================================================================

echo "Starting BGU Cluster Chess Classifier Job"
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_GPUS=$SLURM_GPUS"

## Set working directory to cluster storage
cd $HOME/chess_classifier || exit 1
echo "Working directory: $(pwd)"
## Make sure you're not in conda environment when submitting!
conda deactivate 2>/dev/null || true

## ============================================================================
## Load Required Modules
## ============================================================================

echo "Loading modules..."
module load anaconda
module load cuda/13.1              ## Latest CUDA version (check available: module avail cuda)

## ============================================================================
## Conda Environment Setup
## ============================================================================

echo "Setting up conda environment..."

## Create environment if it doesn't exist (run this once before submitting)
## conda create -n chess_dspy python=3.11 -y
##conda init
## Activate environment
source activate chess_dspy

## Install required packages (run these manually first: pip install dspy-ai torch torchvision)
## Or uncomment to auto-install on first run:
## pip install dspy-ai torch torchvision ollama pillow

## ============================================================================
## Ollama Setup (Local Model Server)
## ============================================================================

echo "Starting Ollama server..."

## Option 1: Ollama from pre-installed location on cluster
## module load ollama  (if available)
OLLAMA_DIR="$HOME/ollama_install"
OLLAMA_BINARY="$OLLAMA_DIR/bin/ollama"
## Option 2: Pre-download Qwen2.5-VL model locally (RECOMMENDED)
## Run on login node once:
## ollama pull qwen2.5-vl:32b  #(or qwen2.5-vl:32b for better accuracy)

## Start Ollama server in background
export OLLAMA_NUM_PARALLEL=1           ## Limit parallel requests
export OLLAMA_NUM_GPU=1                ## Use single GPU
export OLLAMA_MODELS=$HOME/.ollama/models  ## Store models in storage

## Check if Ollama is available
if command -v ollama &> /dev/null; then
    echo "Starting Ollama server..."
    ollama serve > ollama_server.log 2>&1 &
    OLLAMA_PID=$!
    echo "Ollama PID: $OLLAMA_PID"
    sleep 10  ## Wait for server to start
else
    echo "⚠️ Ollama not found. Installing..."
    pip install ollama
    ollama serve > ollama_server.log 2>&1 &
    OLLAMA_PID=$!
    sleep 15
fi

## Verify Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama server failed to start!"
    tail -20 ollama_server.log
    exit 1
fi

echo "✅ Ollama server running"

## ============================================================================
## Data Setup (Handle Cluster Storage Paths)
## ============================================================================

echo "Setting up data paths..."

## Create necessary directories
mkdir -p results
mkdir -p checkpoints
mkdir -p logs

## Data is stored at: /storage/users/$USER/chess_classifier/data/
## OR copy from shared storage if available
DATA_ROOT="$HOME/chess_classifier/data"

if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️ Data directory not found at $DATA_ROOT"
    echo "Make sure you have uploaded your data with:"
    echo "  scp -r local_data/* $USER@slurm.bgu.ac.il:~/chess_classifier/data/"
fi

## Optional: Use /scratch for faster I/O on local node SSD
if [ -n "$SLURM_SCRATCH_DIR" ]; then
    echo "Using local SSD scratch: $SLURM_SCRATCH_DIR"
    mkdir -p $SLURM_SCRATCH_DIR/data
    if [ -d "$DATA_ROOT" ]; then
        cp -r $DATA_ROOT/* $SLURM_SCRATCH_DIR/data/
        DATA_ROOT=$SLURM_SCRATCH_DIR/data
    fi
fi

## ============================================================================
## Run Training Script
## ============================================================================

echo "Starting training script..."

python train_chess_classifier.py \
    --data-root "$DATA_ROOT" \
    --output-dir "./results" \
    --checkpoint-dir "./checkpoints" \
    --model "qwen3-vl:32b" \
    --epochs 3 \
    --batch-size 4 \
    --gpu-id 0 \
    --log-file "logs/training_${SLURM_JOBID}.log" \
    --num-workers 4

TRAINING_EXIT_CODE=$?

echo "Training completed with exit code: $TRAINING_EXIT_CODE"

## ============================================================================
## Cleanup
## ============================================================================

echo "Cleaning up..."

## Kill Ollama server
if [ ! -z "$OLLAMA_PID" ]; then
    kill $OLLAMA_PID 2>/dev/null || true
    echo "Ollama server stopped"
fi

## Copy results back from scratch if used
if [ -n "$SLURM_SCRATCH_DIR" ] && [ -d "$SLURM_SCRATCH_DIR/results" ]; then
    echo "Copying results back to storage..."
    cp -r $SLURM_SCRATCH_DIR/results/* ./results/ 2>/dev/null || true
fi

## ============================================================================
## Release Job Resources (Optional)
## ============================================================================

## Uncomment if you want to auto-release resources when done:
#import os
#job_cancel_str="scancel " + os.environ['SLURM_JOBID']
## os.system(job_cancel_str)

echo "Job finished at $(date)"
exit $TRAINING_EXIT_CODE
