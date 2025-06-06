#!/bin/bash
#SBATCH --job-name=phi4_rag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40_48gb:1
#SBATCH --error=logs/phi4_rag_%j-err.log


# Print job information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"

# Set up HuggingFace cache environment variables
export HF_HUB_CACHE="/cluster/scratch/cache/huggingface/hub"
export HF_ASSETS_CACHE="/cluster/scratch/cache/huggingface/assets"

# Path to your project and model
PROJECT_DIR="$HOME/repos/GPT_AITIS"
MODEL_DIR="/cluster/scratch/$USER/models/phi-4"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Check if model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Phi-4 model not found at $MODEL_DIR"
    echo "Please run the download_phi4.sh script first"
    exit 1
fi

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python not found at $VENV_PYTHON"
    echo "Make sure your virtual environment is set up"
    exit 1
fi

# Activate virtual environment (this is optional when using full path to python)
cd $PROJECT_DIR
source .venv/bin/activate

# Environment information
echo "Python version: $($VENV_PYTHON --version)"
echo "Virtual environment: $VIRTUAL_ENV"
echo "Working directory: $(pwd)"
echo "GPU information:"
nvidia-smi

# Run the RAG pipeline
echo "Starting RAG pipeline at $(date)"
$VENV_PYTHON src/main.py --model hf --model-name microsoft/phi-4 --prompt precise_v2_2 --k 3 --log-level DEBUG --filter-irrelevant --prompt-relevant relevance_filter_v2 --batch
#$VENV_PYTHON src/main.py --model hf --model-name microsoft/phi-4 --prompt precise_v4 --questions 1,2,3,4,5,6,7,8,12,13,16,17,18,10,19,26,9,11,14,39 --k 3 --log-level DEBUG --batch
#$VENV_PYTHON src/main.py --model hf --model-name microsoft/phi-4 --num-questions 3 --policy-id 20 --prompt standard --log-level DEBUG --batch
#$VENV_PYTHON src/main.py --model hf --model-name microsoft/phi-4 --questions 1,2,3,4,5,6,7,8,12,13,16,17,18,10,19,26,9,11,14,39  --policy-id 20 --prompt standard --log-level DEBUG --persona --batch
#$VENV_PYTHON src/main.py --model hf --model-name microsoft/phi-4 --prompt precise_v2 --k 3 --log-level DEBUG --batch

# Print completion time
echo "Job finished at $(date)"
