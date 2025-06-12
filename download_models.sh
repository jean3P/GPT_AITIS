#!/bin/bash
# download_models.sh - Generic script to download HuggingFace models to scratch space

# Configuration
SCRATCH_DIR="/cluster/scratch/$USER"
MODELS_BASE_DIR="$SCRATCH_DIR/models"
PROJECT_DIR="$HOME/repos/GPT_AITIS"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Make sure HuggingFace cache points to shared cache
export HF_HUB_CACHE="/cluster/scratch/cache/huggingface/hub"
export HF_ASSETS_CACHE="/cluster/scratch/cache/huggingface/assets"

# Available models configuration
declare -A MODELS
MODELS["phi-4"]="microsoft/phi-4"
MODELS["qwen2.5-32b"]="Qwen/Qwen2.5-32B"
MODELS["qwen2.5-7b"]="Qwen/Qwen2.5-7B"
#MODELS["qwen2.5-14b"]="Qwen/Qwen2.5-14B"
# Add more models here as needed
# MODELS["llama3.1-8b"]="meta-llama/Meta-Llama-3.1-8B"
# MODELS["mistral-7b"]="mistralai/Mistral-7B-v0.1"

# Function to display usage
usage() {
    echo "Usage: $0 [model_name] [options]"
    echo ""
    echo "Available models:"
    for local_name in "${!MODELS[@]}"; do
        echo "  $local_name -> ${MODELS[$local_name]}"
    done
    echo ""
    echo "Options:"
    echo "  --force     Force download even if model exists"
    echo "  --all       Download all available models"
    echo "  --list      List available models"
    echo "  --help|-h   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 phi-4                    # Download Phi-4 model"
    echo "  $0 qwen2.5-32b             # Download Qwen 2.5 32B model"
    echo "  $0 --all                   # Download all models"
    echo "  $0 phi-4 --force           # Force re-download Phi-4"
}

# Function to check if model exists
model_exists() {
    local model_dir="$1"
    if [[ -d "$model_dir" ]] && [[ -n "$(ls -A "$model_dir" 2>/dev/null)" ]]; then
        # Check for essential files
        if [[ -f "$model_dir/config.json" ]] && [[ -f "$model_dir/tokenizer.json" || -f "$model_dir/tokenizer_config.json" ]]; then
            return 0  # Model exists and appears complete
        fi
    fi
    return 1  # Model doesn't exist or is incomplete
}

# Function to check if model name is valid
is_valid_model() {
    local model_name="$1"
    local key
    for key in "${!MODELS[@]}"; do
        if [[ "$key" == "$model_name" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to download a single model with flexible resource allocation
download_model() {
    local local_name="$1"
    local hf_model_id="$2"
    local force_download="$3"

    local model_dir="$MODELS_BASE_DIR/$local_name"

    echo "=================================================="
    echo "Processing: $local_name ($hf_model_id)"
    echo "Target directory: $model_dir"
    echo "=================================================="

    # Check if model already exists
    if model_exists "$model_dir" && [[ "$force_download" != "true" ]]; then
        echo "✓ Model $local_name already exists at $model_dir"
        echo "  Use --force to re-download"
        return 0
    fi

    # Create directory
    mkdir -p "$model_dir"

    # Create Python download script
    cat > "$SCRATCH_DIR/download_${local_name}.py" << EOL
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

# Model configuration
model_id = "$hf_model_id"
model_dir = "$model_dir"
local_name = "$local_name"

logger.info(f"Downloading model {model_id} to {model_dir}")

try:
    # Download with appropriate settings
    path = snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,  # Actual files, not symlinks
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin"],  # Skip unnecessary files, prefer safetensors
        local_files_only=False,  # Force download
        resume_download=True,  # Resume if interrupted
    )
    logger.info(f"Model {local_name} downloaded successfully to {path}")

    # Verify essential files exist
    essential_files = ["config.json"]
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]

    for file in essential_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✓ Found {file}")
        else:
            logger.warning(f"⚠ Missing {file}")

    # Check for at least one tokenizer file
    tokenizer_found = False
    for file in tokenizer_files:
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            logger.info(f"✓ Found {file}")
            tokenizer_found = True
            break

    if not tokenizer_found:
        logger.warning("⚠ No tokenizer files found")

    logger.info(f"Download of {local_name} completed successfully!")

except Exception as e:
    logger.error(f"Error downloading model {local_name}: {e}")
    raise
EOL

    echo "Starting download for $local_name..."

    local success=false

    # Try multiple resource allocation strategies
    echo "Attempting download with flexible resource allocation..."

    # Strategy 1: Try any available GPU
    if srun --gres=gpu:1 --time=60 --pty "$VENV_PYTHON" "$SCRATCH_DIR/download_${local_name}.py" 2>/dev/null; then
        success=true
        echo "✓ Downloaded using GPU resources"
    else
        echo "GPU allocation failed, trying CPU with more memory..."

        # Strategy 2: CPU with high memory (suitable for downloads)
        if srun --cpus-per-task=4 --mem=16G --time=120 --pty "$VENV_PYTHON" "$SCRATCH_DIR/download_${local_name}.py" 2>/dev/null; then
            success=true
            echo "✓ Downloaded using CPU resources"
        else
            echo "Slurm allocation failed, trying direct execution..."

            # Strategy 3: Direct execution (works if you're on a compute node or login node allows it)
            if "$VENV_PYTHON" "$SCRATCH_DIR/download_${local_name}.py"; then
                success=true
                echo "✓ Downloaded using direct execution"
            fi
        fi
    fi

    # Clean up temporary script
    rm -f "$SCRATCH_DIR/download_${local_name}.py"

    if [[ "$success" == "true" ]]; then
        echo "✓ Successfully downloaded $local_name"
        return 0
    else
        echo "✗ Failed to download $local_name with all methods"
        return 1
    fi
}


# Parse command line arguments
FORCE_DOWNLOAD=false
DOWNLOAD_ALL=false
MODEL_TO_DOWNLOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --list)
            echo "Available models:"
            for local_name in "${!MODELS[@]}"; do
                model_dir="$MODELS_BASE_DIR/$local_name"
                if model_exists "$model_dir"; then
                    status="✓ Downloaded"
                else
                    status="⚬ Not downloaded"
                fi
                echo "  $local_name -> ${MODELS[$local_name]} ($status)"
            done
            exit 0
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            if [[ -z "$MODEL_TO_DOWNLOAD" ]]; then
                MODEL_TO_DOWNLOAD="$1"
            else
                echo "Error: Multiple model names specified"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Install required packages
echo "Ensuring huggingface_hub is installed in virtual environment..."
"$VENV_PYTHON" -m pip install huggingface_hub

# Create base models directory
mkdir -p "$MODELS_BASE_DIR"

# Main logic
if [[ "$DOWNLOAD_ALL" == "true" ]]; then
    echo "Downloading all available models..."
    failed_models=()

    for local_name in "${!MODELS[@]}"; do
        hf_model_id="${MODELS[$local_name]}"
        if ! download_model "$local_name" "$hf_model_id" "$FORCE_DOWNLOAD"; then
            failed_models+=("$local_name")
        fi
        echo ""
    done

    # Summary
    echo "=================================================="
    echo "DOWNLOAD SUMMARY"
    echo "=================================================="

    if [[ ${#failed_models[@]} -eq 0 ]]; then
        echo "✓ All models downloaded successfully!"
    else
        echo "⚠ Some models failed to download:"
        for model in "${failed_models[@]}"; do
            echo "  ✗ $model"
        done
    fi

elif [[ -n "$MODEL_TO_DOWNLOAD" ]]; then
    # Check if model is available using the fixed function
    if ! is_valid_model "$MODEL_TO_DOWNLOAD"; then
        echo "Error: Model '$MODEL_TO_DOWNLOAD' not found in available models."
        echo ""
        echo "Available models:"
        for local_name in "${!MODELS[@]}"; do
            echo "  $local_name -> ${MODELS[$local_name]}"
        done
        exit 1
    fi

    hf_model_id="${MODELS[$MODEL_TO_DOWNLOAD]}"
    download_model "$MODEL_TO_DOWNLOAD" "$hf_model_id" "$FORCE_DOWNLOAD"

else
    echo "Error: No model specified."
    echo ""
    usage
    exit 1
fi

# Final instructions
echo ""
echo "=================================================="
echo "USAGE INSTRUCTIONS"
echo "=================================================="
echo "Downloaded models are stored in: $MODELS_BASE_DIR"
echo ""
echo "To use these models in your Python code:"
echo ""
echo "# For Phi-4:"
echo "model = AutoModelForCausalLM.from_pretrained('$MODELS_BASE_DIR/phi-4')"
echo ""
echo "# For Qwen 2.5 32B:"
echo "model = AutoModelForCausalLM.from_pretrained('$MODELS_BASE_DIR/qwen2.5-32b')"
echo ""
echo "Make sure to set these environment variables in your job scripts:"
echo "echo 'export HF_HUB_CACHE=/cluster/scratch/cache/huggingface/hub'"
echo "echo 'export HF_ASSETS_CACHE=/cluster/scratch/cache/huggingface/assets'"
