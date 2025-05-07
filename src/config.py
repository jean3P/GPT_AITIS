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
