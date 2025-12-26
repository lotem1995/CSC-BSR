# DSPy Chess Classifier on BGU Cluster - Complete Setup Guide

## Quick Start

### 1. One-Time Setup (On Login Node)

```bash
# SSH to BGU cluster
ssh your_user@slurm.bgu.ac.il

# Go to your storage directory
cd /storage/users/$USER

# Clone or download the classifier
git clone <your_repo> chess_classifier
cd chess_classifier

# Create conda environment
conda create -n chess_dspy python=3.11 -y
conda activate chess_dspy

# Install dependencies
pip install dspy-ai torch torchvision ollama pillow

# Download the vision model locally (do this ONCE, takes ~30 mins)
# This avoids downloading during job execution
ollama pull qwen2.5-vl:7b
# OR for better accuracy (if you have GPU with 24GB+):
# ollama pull qwen2.5-vl:32b

# Upload your data
scp -r local_data/* $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/data/

# Deactivate environment
conda deactivate
```

### 2. Submit Training Job

```bash
# Make sure you're NOT in any conda environment
conda deactivate

# Submit job
sbatch train_bgu_cluster.sbatch

# Check job status
squeue --me

# Monitor job output (while running)
tail -f chess_classifier-JOBID.log

# Check GPU usage
sinfo -Nel
```

---

## File Organization

Your cluster directory should look like:

```
/storage/users/$USER/chess_classifier/
├── dspy-chess-classifier.py          # Original classifier code
├── train_chess_classifier.py          # Main training script (cluster-optimized)
├── train_bgu_cluster.sbatch           # SLURM job submission script
├── data/
│   ├── game2_per_frame/
│   │   ├── game2.csv
│   │   └── tagged_images/
│   ├── game4_per_frame/
│   │   ├── game4.csv
│   │   └── tagged_images/
│   └── ...
├── results/                           # Output directory (auto-created)
├── checkpoints/                       # Model checkpoints (auto-created)
└── logs/                              # Training logs (auto-created)
```

---

## Step-by-Step Cluster Usage

### Step 1: Prepare Your Code

Copy all necessary files to cluster:

```bash
# From your local machine
scp dspy-chess-classifier.py $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/
scp train_chess_classifier.py $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/
scp train_bgu_cluster.sbatch $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/
```

### Step 2: Configure SBATCH Script

Edit `train_bgu_cluster.sbatch`:

```bash
# Change email
#SBATCH --mail-user=your_email@post.bgu.ac.il

# Adjust time if needed (default: 24 hours)
#SBATCH --time 0-24:00:00

# Choose GPU type (optional):
# #SBATCH --constraint=rtx_3090  # RTX 3090 (24GB, best for 32B model)
# #SBATCH --constraint=rtx_2080  # RTX 2080 (11GB)
# #SBATCH --constraint=gtx_1080  # GTX 1080 (11GB, slowest)
```

### Step 3: Submit Job

```bash
# Navigate to your cluster directory
ssh your_user@slurm.bgu.ac.il
cd /storage/users/$USER/chess_classifier

# Make sbatch executable
chmod +x train_bgu_cluster.sbatch

# Submit job
sbatch train_bgu_cluster.sbatch
```

This will output something like:
```
Submitted batch job 12345
```

### Step 4: Monitor Job

```bash
# Check all your running jobs
squeue --me

# Get detailed info about your job
sstat -j 12345 --format=MaxRSS,MaxVMSize,AveVMSize,AveRSS

# Monitor GPU usage
sinfo -Nel  # Shows all nodes and GPUs

# View job output in real-time
tail -f chess_classifier-12345.log

# View error log (if job fails)
cat chess_classifier-12345.err
```

### Step 5: Retrieve Results

```bash
# Check results while job is running or after
ls -lh /storage/users/$USER/chess_classifier/results/

# Download results to local machine
scp -r $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/results/* ./local_results/
```

---

## Advanced Features

### Using Faster GPU (RTX 3090)

Edit `train_bgu_cluster.sbatch`:

```bash
#SBATCH --constraint=rtx_3090  # RTX 3090 has 24GB VRAM
#SBATCH --mem=32G              # Allocate enough RAM
```

Then use larger model:

```bash
python train_chess_classifier.py \
    --model "qwen2.5-vl:32b" \  # Better accuracy
    ...
```

### Using SSD for Faster I/O

Edit `train_bgu_cluster.sbatch` to use local node SSD:

```bash
#SBATCH --tmp=100G              # Allocate 100GB on /scratch

# In script:
export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
cp -r $DATA_ROOT $SLURM_SCRATCH_DIR/data/
```

### Job Arrays for Multiple Runs

Create `train_array.sbatch`:

```bash
#!/bin/bash
#SBATCH --array=1-5%2           # Run 5 jobs, max 2 simultaneously

# Different seeds or hyperparameters per task
SEED=$SLURM_ARRAY_TASK_ID

python train_chess_classifier.py \
    --seed $SEED \
    --output-dir "./results/run_$SLURM_ARRAY_TASK_ID" \
    ...
```

Submit:
```bash
sbatch train_array.sbatch
```

### Interactive Session (for debugging)

```bash
# Start interactive job
sinteractive --time 0-2:00:00 --gpu 1

# On compute node:
module load anaconda
source activate chess_dspy
python train_chess_classifier.py --data-root ...

# Don't forget to cancel when done
scancel <job_id>
```

---

## Common Issues & Solutions

### Issue 1: "Ollama server failed to start"

**Solution:**
```bash
# Pre-download model on login node
ollama pull qwen2.5-vl:7b

# Then add to sbatch:
export OLLAMA_MODELS=/storage/users/$USER/.ollama/models
```

### Issue 2: "Out of Memory"

**Solution:**
```bash
# Reduce batch size in train_chess_classifier.py:
parser.add_argument('--batch-size', type=int, default=2)  # was 4

# Or limit examples per game:
--limit-per-game 2  # was 5
```

### Issue 3: "CUDA out of memory"

**Solution:**
```bash
# Use smaller model (7B instead of 32B)
--model "qwen2.5-vl:7b"

# Or allocate GPU with more VRAM
#SBATCH --constraint=rtx_3090
```

### Issue 4: Data files not found

**Solution:**
```bash
# Verify data structure
ssh $USER@slurm.bgu.ac.il
ls -R /storage/users/$USER/chess_classifier/data/

# If missing, copy:
scp -r local_data/* $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/data/
```

### Issue 5: Job stays in PENDING state

**Solution:**
```bash
# Check why job is pending
sinfo -l
squeue -j <job_id> -l

# Common causes:
# - Not enough GPUs available → wait or reduce --time
# - Wrong partition → use #SBATCH --partition main
# - Insufficient memory → reduce --mem
```

---

## Performance Tuning

### Model Selection

| Model | VRAM | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|---|
| qwen2.5-vl:7b | 10-12GB | Fast | Good | Start here |
| qwen2.5-vl:32b | 20-24GB | Medium | Very Good | RTX 3090 only |
| llama3.2-vision:11b | 10-12GB | Medium | Good | Alternative |

### Temperature Tuning

- **0.1-0.3**: Deterministic (good for classification) ✅
- **0.5-0.7**: Balanced (default)
- **0.8-1.0**: Creative (not for classification)

Use `--temperature 0.3` for best results.

### Resource Allocation

```bash
# For 7B model:
#SBATCH --gpus=1                   # Single GPU
#SBATCH --mem=32G                  # 32GB RAM
#SBATCH --cpus-per-task=4          # 4 CPUs
#SBATCH --time 0-12:00:00          # 12 hours

# For 32B model (RTX 3090 only):
#SBATCH --gpus=1
#SBATCH --constraint=rtx_3090
#SBATCH --mem=48G                  # More RAM
#SBATCH --cpus-per-task=6          # More CPUs
#SBATCH --time 0-24:00:00          # Full day
```

---

## Monitoring & Logging

### Check Job Status

```bash
# List your jobs
squeue --me

# Get detailed info
sacct -j <job_id> --format=JobName,MaxRSS,AllocTRES,State,Elapsed,Start,ExitCode

# Monitor resource usage (while running)
sstat -j <job_id> --format=MaxRSS,MaxVMSize
```

### View Logs

```bash
# Main log
tail -100 chess_classifier-12345.log

# Error log
cat chess_classifier-12345.err

# Ollama server log (if needed)
tail -50 ollama_server.log
```

### Email Notifications

Edit `train_bgu_cluster.sbatch`:

```bash
#SBATCH --mail-user=your_email@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
```

---

## Checkpointing & Resuming

The training script automatically saves:
- `checkpoints/config.json` - Training configuration
- `checkpoints/classifier_final.pkl` - Final model
- `checkpoints/metrics.json` - Training metrics

To resume:

```bash
# Load previous checkpoint
python train_chess_classifier.py \
    --checkpoint-dir ./checkpoints \
    ...
```

---

## Post-Training Analysis

### Check Results

```bash
ls -lh results/
cat results/metrics.json

# Download for analysis
scp -r $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/results/* ./analysis/
```

### Analyze Metrics

```python
import json

with open('results/metrics.json') as f:
    metrics = json.load(f)
    
print(f"Model: {metrics['model']}")
print(f"F1 Score: {metrics.get('test_f1', 'N/A')}")
print(f"Accuracy: {metrics.get('test_accuracy', 'N/A')}")
```

---

## Getting Help

### Useful Commands

```bash
# List available GPU types
sinfo -Nel

# Show cluster status
sres

# Show partition info
sinfo -l

# Cancel job
scancel <job_id>

# View finished job info
sacct -j <job_id> -l
```

### BGU Cluster Documentation

- Official Guide: `/storage/ISE_CS_DT_2024ClusterUserGuide.pdf`
- Moodle Course: https://moodle.bgu.ac.il/moodle/course/view.php?id=60163
- Password: `cluster20252`

### Support

Contact IT team:
- Email: support@bgu.ac.il
- Include: Username, Job ID, sbatch file, error message

---

## Example Workflow

```bash
# 1. Prepare code (local machine)
git clone <your_repo>
cd chess_classifier

# 2. Copy to cluster
scp -r . $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/

# 3. SSH to cluster
ssh $USER@slurm.bgu.ac.il

# 4. One-time setup
cd /storage/users/$USER/chess_classifier
conda create -n chess_dspy python=3.11 -y
conda activate chess_dspy
pip install dspy-ai torch torchvision ollama pillow
ollama pull qwen2.5-vl:7b
conda deactivate

# 5. Upload data
exit  # Back to local machine
scp -r data/* $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/data/

# 6. SSH back and submit job
ssh $USER@slurm.bgu.ac.il
cd /storage/users/$USER/chess_classifier
sbatch train_bgu_cluster.sbatch

# 7. Monitor
squeue --me
tail -f chess_classifier-*.log

# 8. Download results
exit  # Back to local machine
scp -r $USER@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/results/* ./
```

---

Good luck with your chess classifier training! 🎯
