#!/bin/bash
# download_embedding_model.sh - Script to download a standard embedding model to scratch space

# Create directory structure in scratch space
USER_NAME="$USER"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"  # This is a reliable, small embedding model
SAFE_MODEL_NAME=$(echo $MODEL_NAME | tr '/' '_')

echo "Using username: $USER_NAME"
echo "Model to download: $MODEL_NAME"

# Create directory structure in scratch space
SCRATCH_DIR="/cluster/scratch/$USER_NAME"
MODEL_DIR="$SCRATCH_DIR/models/embeddings"
mkdir -p $MODEL_DIR

# Make sure HuggingFace cache points to shared cache
export HF_HUB_CACHE="/cluster/scratch/cache/huggingface/hub"
export HF_ASSETS_CACHE="/cluster/scratch/cache/huggingface/assets"

# Path to your project's virtual environment
PROJECT_DIR="$HOME/repos/GPT_AITIS"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Load HuggingFace token from .env file if available
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading HuggingFace token from .env file..."
    export $(grep -v '^#' $PROJECT_DIR/.env | grep HUGGINGFACE_TOKEN | xargs)
    echo "Token loaded: ${HUGGINGFACE_TOKEN:0:3}...${HUGGINGFACE_TOKEN: -3}"
else
    echo "No .env file found. Proceeding without token."
fi

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python not found at $VENV_PYTHON"
    echo "Make sure your virtual environment is set up in $PROJECT_DIR/.venv"
    echo "You can create it with: cd $PROJECT_DIR && uv venv"
    exit 1
fi

# Create Python script for downloading with explicit username
cat > $SCRATCH_DIR/download_embedding.py << EOL
import os
import logging
import sys
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Login to Hugging Face with token
token = os.environ.get('HUGGINGFACE_TOKEN')
if token:
    logger.info("Logging in to Hugging Face with token")
    login(token=token)
else:
    logger.warning("No Hugging Face token found in environment")

# Define the model ID and paths - use explicit username instead of environment variable
model_id = 'sentence-transformers/all-MiniLM-L6-v2'
user_name = "$USER_NAME"  # Use the bash variable directly
logger.info(f"Using username: {user_name}")

scratch_dir = f"/cluster/scratch/{user_name}"
logger.info(f"Scratch directory: {scratch_dir}")

safe_model_name = model_id.replace('/', '_')
model_dir = os.path.join(scratch_dir, "models", "embeddings")
model_save_path = os.path.join(model_dir, safe_model_name)

logger.info(f"Downloading embedding model {model_id} to {model_save_path}")

try:
    # Explicitly verify and create directories
    logger.info(f"Ensuring directory exists: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)

    # Download model using sentence_transformers
    logger.info("Downloading model...")
    model = SentenceTransformer(model_id)

    # Save the model
    logger.info(f"Saving model to {model_save_path}")
    model.save(model_save_path)

    logger.info(f"Model downloaded and saved successfully to {model_save_path}")
    print(f"\\nTo use this model in your code:\\nEMBEDDING_MODEL_PATH = '{model_save_path}'")
except Exception as e:
    logger.error(f"Error downloading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOL

# Install required packages
echo "Installing required packages in your virtual environment..."
$VENV_PYTHON -m pip install sentence-transformers huggingface_hub

# Submit the download as an interactive job using the virtual environment's Python
echo "Submitting download job for embedding model '$MODEL_NAME'..."
srun --pty --export=HUGGINGFACE_TOKEN $VENV_PYTHON $SCRATCH_DIR/download_embedding.py

# Check if download completed
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
    echo "Models saved to: $MODEL_DIR/$SAFE_MODEL_NAME"
    echo ""
    echo "To update your config.py with this model, add:"
    echo "EMBEDDING_MODEL_PATH = '$MODEL_DIR/$SAFE_MODEL_NAME'"
else
    echo "Download failed. Check error messages above."
fi
