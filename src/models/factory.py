# src/models/factory.py

from config import DOCUMENT_DIR
from models.hf_model import HuggingFaceModelClient
from models.openai_model import OpenAIModelClient
from models.shared_client import SharedModelClient
from utils import list_pdf_paths


def get_model_client(provider: str, model_name: str, sys_prompt: str):
    file_paths = list_pdf_paths(DOCUMENT_DIR)
    if provider == "openai":
        return OpenAIModelClient(model_name, sys_prompt, file_paths)
    elif provider == "hf":
        return HuggingFaceModelClient(model_name, sys_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_shared_relevance_client(base_client, relevance_prompt: str):
    """
    Create a shared model client for relevance filtering.

    Args:
        base_client: The main model client to share
        relevance_prompt: The prompt to use for relevance filtering

    Returns:
        SharedModelClient instance
    """
    return SharedModelClient(base_client, relevance_prompt)

