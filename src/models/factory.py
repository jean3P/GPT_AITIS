# src/models/factory.py

from config import DOCUMENT_DIR
from models.hf_model import HuggingFaceModelClient
from models.openai_model import OpenAIModelClient
from utils import list_pdf_paths

def get_model_client(provider: str, model_name: str, sys_prompt: str):
    file_paths = list_pdf_paths(DOCUMENT_DIR)
    if provider == "openai":
        return OpenAIModelClient(model_name, sys_prompt, file_paths)
    elif provider == "hf":
        return HuggingFaceModelClient(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

