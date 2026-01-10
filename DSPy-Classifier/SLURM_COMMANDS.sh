#!/bin/bash
# BGU Cluster Quick Reference - Common Commands & Workflows

# ============================================================================
# SLURM JOB SUBMISSION
# ============================================================================

# Submit a job
sbatch train_bgu_cluster.sbatch

# Submit job with custom parameters
sbatch --time 0-12:00:00 --mem=48G train_bgu_cluster.sbatch

# Submit with environment variables
sbatch --export=ALL,MODEL='qwen2.5-vl:32b',EPOCHS='5' train_bgu_cluster.sbatch

# Submit to specific GPU type
sbatch --constraint=rtx_3090 train_bgu_cluster.sbatch

# ============================================================================
# MONITORING JOBS
# ============================================================================

# List all your jobs
squeue --me

# List all jobs with details
squeue --me -l

# Get detailed job info
scontrol show job <job_id>

# Check job status in full
sacct -j <job_id> -l

# Get resource usage of running job
sstat -j <job_id> --format=MaxRSS,MaxVMSize,AveCPU

# Watch job progress in real-time
watch -n 5 'squeue --me'

# ============================================================================
# VIEWING OUTPUT
# ============================================================================

# View last 50 lines of job output
tail -50 chess_classifier-12345.log

# Follow log in real-time
tail -f chess_classifier-12345.log

# View entire log file
cat chess_classifier-12345.log

# View error output
cat chess_classifier-12345.err

# View both stdout and stderr
tail -f chess_classifier-12345.log chess_classifier-12345.err

# Search for errors in log
grep -i "error\|failed\|exception" chess_classifier-12345.log

# ============================================================================
# CANCELING JOBS
# ============================================================================

# Cancel single job
scancel <job_id>

# Cancel by job name
scancel --name chess_classifier

# Cancel all pending jobs
scancel -t PENDING -u $USER

# Cancel all jobs for a user
scancel -u $USER

# ============================================================================
# CLUSTER INFORMATION
# ============================================================================

# Show cluster node information
sinfo -Nel

# Show partition info
sinfo -l

# Show cluster resource status
sres

# List available GPU types
sinfo -Nel | grep gpu

# Show node list with GPUs
sinfo -N -O NodeList,Gres,GresUsed,CPUsState,Memory

# Check specific node info
sinfo -n <node_name> -l

# ============================================================================
# RESOURCE ALLOCATION
# ============================================================================

# Check GPU availability
sinfo -Nel | grep -E "gpu|NODELIST"

# Show available GPUs
sinfo -l --Format=NodeList,Gres,FreeMem

# Check memory on specific node
srun --nodelist=<node_name> free -h

# Show current cluster utilization
sres

# ============================================================================
# DATA TRANSFER
# ============================================================================

# Copy files FROM local TO cluster
scp file.txt user@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/

# Copy directory FROM local TO cluster
scp -r local_data/ user@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/

# Copy files FROM cluster TO local
scp user@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/results.zip ./

# Copy recursively FROM cluster TO local
scp -r user@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/results/* ./

# Sync directories (faster for large transfers)
rsync -avz local_data/ user@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/data/

# ============================================================================
# INTERACTIVE SESSION
# ============================================================================

# Start interactive session with GPU
sinteractive --gpu 1 --time 0-2:00:00

# Interactive session with specific GPU type
sinteractive --gpu 1 --constraint=rtx_3090 --time 0-4:00:00

# Interactive session with Jupyter
sjupyter --gpu 1 --time 0-4:00:00

# Exit interactive session
exit

# Don't forget to cancel when done
scancel <job_id>

# ============================================================================
# CONDA ENVIRONMENT (ON LOGIN NODE)
# ============================================================================

# Create environment
conda create -n chess_dspy python=3.11 -y

# Activate environment
conda activate chess_dspy

# List environments
conda env list

# Install packages
pip install dspy-ai torch torchvision

# Remove environment
conda env remove -n chess_dspy

# Deactivate environment
conda deactivate

# ============================================================================
# ADVANCED SBATCH FEATURES
# ============================================================================

# Job dependencies (start after another job finishes)
sbatch --dependency=afterok:<job_id> train_bgu_cluster.sbatch

# Job array (run multiple jobs)
sbatch --array=1-10 train_bgu_cluster.sbatch

# Job array with limit (run max 2 simultaneously)
sbatch --array=1-10%2 train_bgu_cluster.sbatch

# Set job priority
scontrol update JobId=<job_id> Nice=500

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Check why job is pending
squeue -j <job_id> -l

# Get detailed job error info
scontrol show job <job_id>

# Check node status
sinfo -R

# Test GPU access in job
sbatch -C rtx_3090 --wrap "nvidia-smi"

# Run command on compute node
srun --pty bash  # Open shell on allocated node

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test model download
ollama pull qwen2.5-vl:7b

# ============================================================================
# TYPICAL WORKFLOW
# ============================================================================

# 1. Setup (one time)
# cd /storage/users/$USER/chess_classifier
# conda create -n chess_dspy python=3.11 -y
# conda activate chess_dspy
# pip install dspy-ai torch torchvision ollama
# ollama pull qwen2.5-vl:7b
# conda deactivate

# 2. Submit job
# sbatch train_bgu_cluster.sbatch

# 3. Monitor job
# squeue --me
# tail -f chess_classifier-*.log

# 4. Check results
# ls -lh results/
# cat results/metrics.json

# 5. Download results
# scp -r user@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/results/* ./

# ============================================================================
# USEFUL ONE-LINERS
# ============================================================================

# Get total job runtime
sacct -j <job_id> --format=Elapsed

# Get peak memory usage
sacct -j <job_id> --format=MaxRSS

# List all completed jobs
sacct --format=JobName,JobID,State,ExitCode

# Show running job's CPU/Memory
sstat -j <job_id> --format=AveCPU,AveRSS

# Copy entire results directory and delete remote
rsync -avz --remove-source-files user@slurm.bgu.ac.il:/storage/users/$USER/chess_classifier/results/* ./results/

# Count files in dataset
find /storage/users/$USER/chess_classifier/data -name "*.jpg" | wc -l

# Show disk usage
du -sh /storage/users/$USER/chess_classifier/*

# Find recent log files
find . -name "*.log" -type f -mtime -1  # Last 24 hours

# ============================================================================
# GPU-SPECIFIC COMMANDS
# ============================================================================

# Show all GPU info
sinfo -Nel --Format=NodeList,Gres,GresUsed

# Show available RTX 3090s
sinfo -Nel --Format=NodeList,Gres | grep "rtx_3090"

# Count GPUs by type
sinfo -Nel | grep -o "gpu:[^,]*" | sort | uniq -c

# Check CUDA version available
module avail cuda

# Load CUDA 12.4
module load cuda/12.4

# Check GPU memory on running job
srun -j <job_id> nvidia-smi

# ============================================================================
# BATCH PROCESSING MULTIPLE GAMES
# ============================================================================

# Process each game separately (create array job)
cat > process_games.sbatch << 'EOF'
#!/bin/bash
#SBATCH --array=2,4,5,6,7

GAME_ID=$SLURM_ARRAY_TASK_ID
python train_chess_classifier.py \
    --data-root "/storage/users/$USER/chess_classifier/data/game${GAME_ID}_per_frame" \
    --output-dir "./results/game_${GAME_ID}" \
    --epochs 3
EOF

sbatch process_games.sbatch

# ============================================================================
# MAIL NOTIFICATIONS
# ============================================================================

# Add to sbatch to get email when job:
# - Begins
#SBATCH --mail-type=BEGIN

# - Ends successfully
#SBATCH --mail-type=END

# - Fails
#SBATCH --mail-type=FAIL

# - Any event
#SBATCH --mail-type=ALL

# - Array job events
#SBATCH --mail-type=ARRAY_TASKS

# Set email address
#SBATCH --mail-user=your_email@post.bgu.ac.il

# ============================================================================
# USEFUL ALIAS (ADD TO ~/.bashrc)
# ============================================================================

# alias jobs='squeue --me'
# alias jobinfo='scontrol show job'
# alias killjob='scancel'
# alias myspace='du -sh /storage/users/$USER'
# alias cluster_status='sres'
# alias gpu_info='sinfo -Nel'
# alias tail_job='tail -f chess_classifier-*.log'

# ============================================================================
# HELP & DOCUMENTATION
# ============================================================================

# Get help for any slurm command
man sbatch
man squeue
man srun
man scancel

# Show sbatch help
sbatch --help

# Show sinteractive help
sinteractive --help

# BGU Cluster guide location
# /storage/ISE_CS_DT_2024ClusterUserGuide.pdf

# Moodle video tutorials
# https://moodle.bgu.ac.il/moodle/course/view.php?id=60163
# Password: cluster20252

# BGU IT Support
# Email: support@bgu.ac.il
