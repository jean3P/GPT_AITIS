#!/bin/bash
#SBATCH --job-name=phi4_rag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40_48gb:1
#SBATCH --error=logs/phi4_rag_%j-err.log
# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

cleanup() {
    echo "=== CLEANUP STARTED at $(date) ==="

    # Kill any remaining Python processes from this job
    if [ ! -z "$MAIN_PID" ]; then
        echo "Killing main process $MAIN_PID"
        kill -TERM $MAIN_PID 2>/dev/null || true
        sleep 5
        kill -KILL $MAIN_PID 2>/dev/null || true
    fi

    # Clear GPU memory (fix the nvidia-smi command)
    echo "Clearing GPU memory..."
    # nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true  # This option doesn't exist

    # Clear Python cache
    echo "Clearing Python cache..."
    find $PROJECT_DIR -name "*.pyc" -delete 2>/dev/null || true
    find $PROJECT_DIR -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    # Deactivate virtual environment
    if [ ! -z "$VIRTUAL_ENV" ]; then
        echo "Deactivating virtual environment"
        deactivate 2>/dev/null || true
    fi

    echo "=== CLEANUP COMPLETED at $(date) ==="
}

# Set up signal handlers for cleanup
trap cleanup EXIT
trap cleanup SIGTERM
trap cleanup SIGINT
trap cleanup SIGQUIT

# =============================================================================
# JOB SETUP
# =============================================================================

# Print job information
echo "=== JOB STARTED at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Running on host: $(hostname)"

# Set up HuggingFace cache environment variables
export HF_HUB_CACHE="/cluster/scratch/cache/huggingface/hub"
export HF_ASSETS_CACHE="/cluster/scratch/cache/huggingface/assets"

# Path to your project and model
PROJECT_DIR="$HOME/repos/GPT_AITIS"
MODEL_DIR="/cluster/scratch/$USER/models/qwen2.5-7b"  # Fixed path
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# =============================================================================
# VALIDATION
# =============================================================================

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️  Warning: Qwen model not found at $MODEL_DIR"
    echo "Will attempt to download from HuggingFace"
else
    echo "✅ Found local model at: $MODEL_DIR"
fi

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Python not found at $VENV_PYTHON"
    echo "Make sure your virtual environment is set up"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "❌ Error: No GPU available or nvidia-smi not working"
    exit 1
fi

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Change to project directory
cd $PROJECT_DIR || exit 1

# Activate virtual environment
source .venv/bin/activate

# Environment information
echo "=== ENVIRONMENT INFO ==="
echo "Python version: $($VENV_PYTHON --version)"
echo "Virtual environment: $VIRTUAL_ENV"
echo "Working directory: $(pwd)"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "=== GPU INFORMATION ==="
nvidia-smi

echo "=== DISK SPACE ==="
df -h $PROJECT_DIR
df -h $HF_HUB_CACHE

# =============================================================================
# MAIN EXECUTION
# =============================================================================

echo "=== STARTING RAG PIPELINE at $(date) ==="

# Fix the model name - remove extra quotes!
MODEL_NAME="Qwen/Qwen2.5-7B"
QUESTIONS="1,2,3,4,5,6,7,8,12,13,16,17,18,10,19,26,9,11,14,39"
#QUESTIONS="7,16"
POLICY_ID="20"


# Store the command for easier modification
RAG_COMMAND="$VENV_PYTHON src/main.py --model hf --model-name $MODEL_NAME --prompt precise_v2_qwen --k 3 --log-level DEBUG --questions $QUESTIONS --policy-id $POLICY_ID --batch"
#RAG_COMMAND="$VENV_PYTHON src/main.py --model hf --model-name $MODEL_NAME --prompt precise_v2_qwen --k 3 --log-level DEBUG --questions $QUESTIONS --batch"

echo "Executing: $RAG_COMMAND"

# Run the command and capture PID, with better error handling
$RAG_COMMAND &
MAIN_PID=$!

# Wait for the process to complete
wait $MAIN_PID
EXIT_CODE=$?

echo "=== RAG PIPELINE COMPLETED with exit code: $EXIT_CODE ==="

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

echo "=== JOB SUMMARY ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Job completed successfully"

    # Show results if they exist
    if [ -d "$PROJECT_DIR/resources/results/json_output" ]; then
        echo "=== OUTPUT FILES ==="
        ls -la "$PROJECT_DIR/resources/results/json_output/"*.json 2>/dev/null || echo "No JSON output files found"
    fi
else
    echo "❌ Job failed with exit code: $EXIT_CODE"

    # Show recent log entries
    echo "=== CHECKING FOR LOG FILES ==="
    LOG_DIR="$PROJECT_DIR/resources/results/logs"
    if [ -d "$LOG_DIR" ]; then
        LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
        if [ -f "$LATEST_LOG" ]; then
            echo "=== RECENT LOG ENTRIES FROM: $LATEST_LOG ==="
            tail -50 "$LATEST_LOG"
        else
            echo "No log files found in $LOG_DIR"
        fi
    else
        echo "Log directory not found: $LOG_DIR"
    fi

    # Also check SLURM error file
    echo "=== SLURM ERROR FILE ==="
    if [ -f "logs/phi4_rag_${SLURM_JOB_ID}-err.log" ]; then
        cat "logs/phi4_rag_${SLURM_JOB_ID}-err.log"
    else
        echo "SLURM error file not found"
    fi
fi

# The cleanup function will be called automatically due to the trap
exit $EXIT_CODE
