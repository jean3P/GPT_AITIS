# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment and configuration variables
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME: str = "gpt-4o"
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Keep for backward compatibility

# Configure embedding model path for the downloaded sentence transformer model
EMBEDDING_MODEL_PATH = "/cluster/scratch/jeanpool.pereyrap/models/embeddings/sentence-transformers_all-MiniLM-L6-v2"

# Verify the model exists, otherwise fall back to remote
if os.path.exists(EMBEDDING_MODEL_PATH):
    print(f"Using local embedding model from: {EMBEDDING_MODEL_PATH}")
else:
    # Fall back to HuggingFace model if local copy not found
    print(f"Local model not found at {EMBEDDING_MODEL_PATH}")
    print(f"Falling back to remote model")
    EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# HuggingFace cache settings
# Set environment variables if not already set
if "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = "/cluster/scratch/cache/huggingface/hub"
if "HF_ASSETS_CACHE" not in os.environ:
    os.environ["HF_ASSETS_CACHE"] = "/cluster/scratch/cache/huggingface/assets"

# Constants for reference
HF_HUB_CACHE: str = os.environ["HF_HUB_CACHE"]
HF_ASSETS_CACHE: str = os.environ["HF_ASSETS_CACHE"]

# =============================================================================
# MODEL CONFIGURATION AND PATH MANAGEMENT
# =============================================================================

# Base directory for locally downloaded models
MODELS_BASE_DIR = f"/cluster/scratch/{os.environ.get('USER', 'jeanpool.pereyrap')}/models"

# Model path mappings - maps HuggingFace model names to local directories
MODEL_PATHS = {
    # Phi models
    "microsoft/phi-4": os.path.join(MODELS_BASE_DIR, "phi-4"),

    # Qwen models
    "Qwen/Qwen2.5-32B": os.path.join(MODELS_BASE_DIR, "qwen2.5-32b"),
    "Qwen/Qwen2.5-7B": os.path.join(MODELS_BASE_DIR, "qwen2.5-7b"),
    # "Qwen/Qwen2.5-14B": os.path.join(MODELS_BASE_DIR, "qwen2.5-14b"),

    # Local name aliases for convenience
    "phi-4": os.path.join(MODELS_BASE_DIR, "phi-4"),
    "qwen2.5-32b": os.path.join(MODELS_BASE_DIR, "qwen2.5-32b"),
    "qwen2.5-7b": os.path.join(MODELS_BASE_DIR, "qwen2.5-7b"),
    # "qwen2.5-14b": os.path.join(MODELS_BASE_DIR, "qwen2.5-14b"),
}

# Model-specific generation configurations
MODEL_CONFIGS = {
    "microsoft/phi-4": {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "max_new_tokens": 2048,
        "temperature": 0.1,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "pad_token_id": None  # Will be set to eos_token_id
    },
    "Qwen/Qwen2.5-32B": {
         # Model loading parameters (required)
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,

        # Generation parameters (essential for speed)
        "max_new_tokens": 1024,
        # "temperature": 0.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "pad_token_id": None
    },
    "Qwen/Qwen2.5-7B": {
        # Same config as above
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": None,
        "eos_token_id": None,
        "top_p": 0.9,
        "num_return_sequences": 1,
        "output_scores": False,
        "use_cache": True,
        "no_repeat_ngram_size": 3,
    },
}

# Copy configs for local name aliases
MODEL_CONFIGS["phi-4"] = MODEL_CONFIGS["microsoft/phi-4"]
MODEL_CONFIGS["qwen2.5-32b"] = MODEL_CONFIGS["Qwen/Qwen2.5-32B"]
MODEL_CONFIGS["qwen2.5-7b"] = MODEL_CONFIGS["Qwen/Qwen2.5-7B"]
# MODEL_CONFIGS["qwen2.5-14b"] = MODEL_CONFIGS["Qwen/Qwen2.5-14B"]


def get_local_model_path(model_name: str) -> str:
    """
    Get local path for a model, falling back to original name if not found locally.

    Args:
        model_name: HuggingFace model name or local identifier

    Returns:
        Local path if model exists locally, otherwise original model name
    """
    # Check if it's already a local path
    if os.path.exists(model_name) and os.path.isdir(model_name):
        print(f"Using provided local path: {model_name}")
        return model_name

    # Check our model mappings
    local_path = MODEL_PATHS.get(model_name)
    if local_path and os.path.exists(local_path) and os.path.isdir(local_path):
        # Verify it's a valid model directory
        if os.path.exists(os.path.join(local_path, "config.json")):
            print(f"Using local model from: {local_path}")
            return local_path
        else:
            print(f"Local model directory found but missing config.json: {local_path}")

    # Check for case-insensitive matches
    model_name_lower = model_name.lower()
    for key, path in MODEL_PATHS.items():
        if key.lower() == model_name_lower and os.path.exists(path) and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "config.json")):
                print(f"Using local model from case-insensitive match: {path}")
                return path

    # Fall back to original name (will download from HuggingFace)
    print(f"Local model not found for '{model_name}', using remote download")
    return model_name


def get_model_config(model_name: str) -> dict:
    """
    Get model-specific configuration.

    Args:
        model_name: Model name or path

    Returns:
        Dictionary with model configuration parameters
    """
    # First try to get config by exact name
    config = MODEL_CONFIGS.get(model_name)
    if config:
        return config.copy()

    # Try case-insensitive match
    model_name_lower = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if key.lower() == model_name_lower:
            return config.copy()

    # Check if it's a local path and try to infer from directory name
    if os.path.exists(model_name):
        dir_name = os.path.basename(model_name.rstrip('/'))
        config = MODEL_CONFIGS.get(dir_name)
        if config:
            return config.copy()

    # Default configuration for unknown models
    print(f"Using default configuration for unknown model: {model_name}")
    return {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "max_new_tokens": 2048,
        "temperature": 0.1,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "pad_token_id": None
    }


def is_qwen_model(model_name: str) -> bool:
    """
    Check if a model is a Qwen model.

    Args:
        model_name: Model name or path

    Returns:
        True if it's a Qwen model, False otherwise
    """
    model_name_lower = model_name.lower()
    qwen_indicators = ["qwen", "qwen2", "qwen2.5"]

    # Check direct indicators
    for indicator in qwen_indicators:
        if indicator in model_name_lower:
            return True

    # Check if it's a local path pointing to a Qwen model
    if os.path.exists(model_name):
        dir_name = os.path.basename(model_name.rstrip('/')).lower()
        for indicator in qwen_indicators:
            if indicator in dir_name:
                return True

    return False


def is_phi_model(model_name: str) -> bool:
    """
    Check if a model is a Phi model.

    Args:
        model_name: Model name or path

    Returns:
        True if it's a Phi model, False otherwise
    """
    model_name_lower = model_name.lower()
    phi_indicators = ["phi", "microsoft/phi"]

    # Check direct indicators
    for indicator in phi_indicators:
        if indicator in model_name_lower:
            return True

    # Check if it's a local path pointing to a Phi model
    if os.path.exists(model_name):
        dir_name = os.path.basename(model_name.rstrip('/')).lower()
        if "phi" in dir_name:
            return True

    return False


# =============================================================================
# LEGACY COMPATIBILITY (keeping your existing variables)
# =============================================================================

# Keep your existing Qwen variables for backward compatibility
QWEN_MODEL_PATH = MODEL_PATHS.get("Qwen/Qwen2.5-32B", "/cluster/scratch/jeanpool.pereyrap/models/qwen2.5-32b")
QWEN_MODEL_NAME = "Qwen/Qwen2.5-32B"
QWEN_CONFIG = MODEL_CONFIGS["Qwen/Qwen2.5-32B"]

# =============================================================================
# APPLICATION PATHS AND SETTINGS
# =============================================================================

MAX_QUESTIONS = 1
DATASET_PATH: str = os.path.join(base_dir, "resources/questions/questions.xlsx")
DOCUMENT_DIR: str = os.path.join(base_dir, "resources/documents/policies/")
RESULT_PATH: str = os.path.join(base_dir, f"resources/results/run_output_{MAX_QUESTIONS}.tsv")
RESPONSE_FORMAT_PATH: str = os.path.join(base_dir, "resources/response_formats/travel_insurance_agent.json")
LOG_DIR: str = os.path.join(base_dir, "resources/results/logs")
JSON_PATH: str = os.path.join(base_dir, "resources/results/json_output")

RAW_GT_PATH: str = os.path.join(base_dir, "resources/raw_ground_truth/")
GT_PATH: str = os.path.join(base_dir, "resources/ground_truth/")

VECTOR_STORE_EXPIRATION_DAYS: int = 30
VECTOR_NAME_PREFIX: str = "AITIS_"
EVALUATION_RESULTS_PATH: str = os.path.join(base_dir, "resources/results/")
EVALUATION_RESULTS_FILES_PATH: str = os.path.join(base_dir, "resources/results/evaluation_results/")
DASHBOARD_PATH: str = os.path.join(base_dir, "resources/results/dashboard.html")

# New constants for embedding caching
EMBEDDINGS_DIR: str = os.path.join(base_dir, "resources/embeddings")
CACHE_EMBEDDINGS: bool = True  # Can be set to False to force re-embedding


# =============================================================================
# UTILITY FUNCTIONS FOR MODEL MANAGEMENT
# =============================================================================

def list_available_local_models() -> dict:
    """
    List all locally available models.

    Returns:
        Dictionary mapping model names to their local paths
    """
    available_models = {}

    for model_name, path in MODEL_PATHS.items():
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it looks like a valid model directory
            if os.path.exists(os.path.join(path, "config.json")):
                available_models[model_name] = path

    return available_models


def get_model_info(model_name: str) -> dict:
    """
    Get comprehensive information about a model.

    Args:
        model_name: Model name or path

    Returns:
        Dictionary with model information
    """
    local_path = get_local_model_path(model_name)
    config = get_model_config(model_name)

    info = {
        "model_name": model_name,
        "local_path": local_path,
        "is_local": local_path != model_name and os.path.exists(local_path),
        "is_qwen": is_qwen_model(model_name),
        "is_phi": is_phi_model(model_name),
        "config": config
    }

    # Add size information if local
    if info["is_local"]:
        try:
            # Removed redundant 'import os' - os is already imported at module level
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(local_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            info["size_gb"] = round(total_size / (1024 ** 3), 2)
        except Exception as e:
            print(f"Warning: Could not calculate model size for {model_name}: {e}")
            info["size_gb"] = "unknown"

    return info


