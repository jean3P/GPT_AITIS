#!/bin/bash
# download_phi4.sh - Script to download Microsoft Phi-4 model to scratch space

# Create directory structure in scratch space
SCRATCH_DIR="/cluster/scratch/$USER"
MODEL_DIR="$SCRATCH_DIR/models/phi-4"
mkdir -p $MODEL_DIR

# Make sure HuggingFace cache points to shared cache
export HF_HUB_CACHE="/cluster/scratch/cache/huggingface/hub"
export HF_ASSETS_CACHE="/cluster/scratch/cache/huggingface/assets"

# Path to your project's virtual environment
PROJECT_DIR="$HOME/repos/GPT_AITIS"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python not found at $VENV_PYTHON"
    echo "Make sure your virtual environment is set up in $PROJECT_DIR/.venv"
    echo "You can create it with: cd $PROJECT_DIR && uv venv"
    exit 1
fi

# Create Python script for downloading
cat > $SCRATCH_DIR/download_phi4.py << 'EOL'
import os
import logging
from huggingface_hub import login, snapshot_download

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load token from environment if available
token = os.environ.get("HUGGINGFACE_TOKEN")
if token:
    logger.info("Logging in to Hugging Face with token")
    login(token=token)
else:
    logger.warning("No Hugging Face token found in environment. Some models might not be accessible.")

# Define the model ID
model_id = "microsoft/phi-4"
scratch_dir = "/cluster/scratch/" + os.environ.get("USER")
model_dir = os.path.join(scratch_dir, "models", "phi-4")

logger.info(f"Downloading model {model_id} to {model_dir}")
try:
    # Force download with appropriate flags for large models
    path = snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,  # Actual files, not symlinks
        token=token,
        ignore_patterns=["*.msgpack", "*.h5"],  # Skip unnecessary files
        local_files_only=False,  # Force download
        resume_download=True,  # Resume if interrupted
    )
    logger.info(f"Model downloaded successfully to {path}")
except Exception as e:
    logger.error(f"Error downloading model: {e}")
    raise
EOL

echo "Making sure huggingface_hub is installed in your virtual environment..."
$VENV_PYTHON -m pip install huggingface_hub

# Submit the download as an interactive job using the virtual environment's Python
echo "Submitting download job for Phi-4 model..."
srun --gres=gpu:a40_48gb:1 --pty $VENV_PYTHON $SCRATCH_DIR/download_phi4.py

# Check if download completed
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
    echo "Model saved to: $MODEL_DIR"
    echo ""
    echo "To use this model in your Python code:"
    echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
    echo "model = AutoModelForCausalLM.from_pretrained('$MODEL_DIR')"
    echo ""
    echo "Make sure to set these environment variables in your job scripts:"
    echo "export HF_HUB_CACHE=/cluster/scratch/cache/huggingface/hub"
    echo "export HF_ASSETS_CACHE=/cluster/scratch/cache/huggingface/assets"
else
    echo "Download failed. Check error messages above."
fi
