# Installation Guide

This guide will walk you through installing GPT_AITIS and all its dependencies. The installation process typically takes 10-15 minutes.

## 📋 System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS 12+, or Windows 10+ with WSL2
- **Python**: 3.12 or higher
- **RAM**: 16GB minimum
- **Storage**: 50GB free space (for models and data)
- **Internet**: Required for downloading models and packages

## 🚀 Quick Installation

For users who want to get started quickly:

```bash
# Clone the repository
git clone https://github.com/jean3P/GPT_AITIS.git
cd GPT_AITIS

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## 📦 Detailed Installation Steps

### Step 1: Install System Dependencies

=== "Ubuntu/Debian"

    ```bash
    # Update package list
    sudo apt update
    
    # Install Python 3.12 and development tools
    sudo apt install -y python3.12 python3.12-dev python3.12-venv
    
    # Install build essentials
    sudo apt install -y build-essential git curl wget
    
    # Install PDF processing libraries
    sudo apt install -y poppler-utils tesseract-ocr
    ```

=== "macOS"

    ```bash
    # Install Homebrew if not already installed
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install Python 3.12
    brew install python@3.12
    
    # Install PDF processing tools
    brew install poppler tesseract
    ```

=== "Windows (WSL2)"

    ```bash
    # Inside WSL2 Ubuntu
    sudo apt update
    sudo apt install -y python3.12 python3.12-dev python3.12-venv
    sudo apt install -y build-essential git curl wget
    sudo apt install -y poppler-utils tesseract-ocr
    ```

### Step 2: Install UV Package Manager

UV is a fast Python package installer that we use for dependency management:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to your PATH (add to ~/.bashrc or ~/.zshrc for persistence)
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installation
uv --version
```

### Step 3: Clone the Repository

```bash
# Clone via HTTPS
git clone https://github.com/jean3P/GPT_AITIS.git

# Or clone via SSH (if you have SSH keys set up)
git clone git@github.com:jean3P/GPT_AITIS.git

# Navigate to project directory
cd GPT_AITIS
```

### Step 4: Create Virtual Environment

```bash
# Create a virtual environment with Python 3.12
uv venv --python 3.12

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Step 5: Install Project Dependencies

```bash
# Install all dependencies including development tools
uv pip install -e .

# This installs all packages from pyproject.toml including:
# - PyTorch with CUDA support
# - Transformers and model libraries
# - PDF processing tools
# - Evaluation and visualization libraries
```

### Step 6: Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

Add your API keys to the `.env` file:

```bash
# OpenAI API Key (required for GPT models)
OPENAI_API_KEY=sk-your-openai-api-key-here

# HuggingFace Token (required for model downloads)
HUGGINGFACE_TOKEN=hf_your-huggingface-token-here

# OpenRouter API Key (required for OpenRouter models)
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here

# Optional: OpenRouter configuration
OPENROUTER_SITE_URL=http://localhost:3000
OPENROUTER_SITE_NAME=GPT_AITIS

# Optional: Custom cache directories (defaults are fine for most users)
# HF_HUB_CACHE=/path/to/huggingface/cache
# HF_ASSETS_CACHE=/path/to/huggingface/assets
```

## 🤖 Installing Models

### Local Models

For local model inference, you'll need to download the models:

```bash
# Download Phi-4 model (recommended for getting started)
./download_models.sh phi-4

# Download Qwen models (requires more disk space)
./download_models.sh qwen2.5-7b
./download_models.sh qwen2.5-32b  # Requires 64GB+ disk space

# List available models
./download_models.sh --list

# Download all available models
./download_models.sh --all
```

### Cloud Models

For cloud-based models (OpenAI, OpenRouter), ensure your API keys are properly configured in the `.env` file.

## 🔧 Post-Installation Setup

### 1. Download Embedding Model

The system uses sentence embeddings for RAG. Download the default model:

```bash
# Download the embedding model
./download_embedding_model.sh

# This downloads sentence-transformers/all-MiniLM-L6-v2
# Location: /cluster/scratch/$USER/models/embeddings/ (on HPC)
# Or: ./models/embeddings/ (local)
```

### 3. Test Installation

Run a simple test to ensure everything works:

```bash
# Create a test question file
echo "Id,Questions" > resources/questions/test.xlsx
echo "1,Is baggage loss covered?" >> resources/questions/test.xlsx

# Run with a small test (you'll need at least one PDF in policies folder)
python src/main.py \
  --model hf \
  --model-name microsoft/phi-4 \
  --questions "1" \
  --log-level DEBUG
```

## 🛠️ Troubleshooting

### Common Installation Issues

??? failure "UV Installation Fails"
    ```bash
    # Alternative installation via pip
    pip install uv
    
    # Or manual installation
    wget -qO- https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
    ```

??? failure "Python 3.12 Not Found"
    ```bash
    # Ubuntu: Add deadsnakes PPA
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.12 python3.12-venv
    
    # macOS: Use pyenv
    brew install pyenv
    pyenv install 3.12
    pyenv global 3.12
    ```

??? failure "CUDA/GPU Not Detected"
    ```bash
    # Check NVIDIA driver
    nvidia-smi
    
    # Install CUDA toolkit
    # Visit: https://developer.nvidia.com/cuda-downloads
    
    # Install PyTorch with CUDA
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

??? failure "Out of Memory During Model Download"
    ```bash
    # Use environment variables to limit memory usage
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Download models one at a time
    python scripts/download_models.py phi-4
    # Wait for completion before downloading next model
    ```

??? failure "Permission Denied Errors"
    ```bash
    # Fix permissions
    sudo chown -R $USER:$USER GPT_AITIS/
    chmod -R 755 GPT_AITIS/
    ```

## 🔄 Updating the Installation

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
uv pip install -e . --upgrade

# Download any new models if needed
python scripts/download_models.py --list
```

## 📚 Next Steps

Once installation is complete:

1. **[Quick Start Guide](quickstart.md)** - Run your first analysis
2. **[Configuration Guide](configuration.md)** - Fine-tune system settings
3. **[Model Selection](../user-guide/model-selection.md)** - Choose the right model
4. **[Running Analysis](../user-guide/running-analysis.md)** - Detailed usage guide

---

!!! success "Installation Complete!"
    If you've followed all the steps above, GPT_AITIS should now be installed and ready to use. Head over to the [Quick Start Guide](quickstart.md) to run your first insurance policy analysis!

!!! question "Need Help?"
    If you encounter any issues during installation, please:
    
    - Check the [Troubleshooting Guide](../troubleshooting/common.md)
    - Search existing [GitHub Issues](https://github.com/jean3P/GPT_AITIS/issues)
    - Create a new issue with detailed error messages and system information