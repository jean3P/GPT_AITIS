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
EMBEDDING_MODEL: str = "text-embedding-ada-002"

MAX_QUESTIONS = 1
DATASET_PATH: str = os.path.join(base_dir, "resources/questions/questions.xlsx")
DOCUMENT_DIR: str = os.path.join(base_dir, "resources/documents/policies/")
RESULT_PATH: str = os.path.join(base_dir, f"resources/results/run_output_{MAX_QUESTIONS}.tsv")
RESPONSE_FORMAT_PATH: str = os.path.join(base_dir, "resources/response_formats/travel_insurance_agent.json")
LOG_DIR: str = os.path.join(base_dir, "resources/results/logs")

VECTOR_STORE_EXPIRATION_DAYS: int = 15
VECTOR_NAME_PREFIX: str = "AITIS_"
