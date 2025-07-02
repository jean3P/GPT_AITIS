# src/models/factory.py

from config import DOCUMENT_DIR, is_qwen_model, is_phi_model, is_openrouter_model
from models.hf_model import HuggingFaceModelClient
from models.openai_model import OpenAIModelClient
from models.qwen_model import QwenModelClient
from models.openrouter_model import OpenRouterModelClient
from models.shared_client import SharedModelClient
from utils import list_policy_paths


def get_model_client(provider: str, model_name: str, sys_prompt: str):
    """
    Create appropriate model client based on provider and model name.

    Args:
        provider: Model provider ("openai", "hf", "qwen", "openrouter")
        model_name: Name of the model to use
        sys_prompt: System prompt for the model

    Returns:
        Appropriate model client instance
    """
    file_paths = list_policy_paths(DOCUMENT_DIR)

    if provider == "openai":
        return OpenAIModelClient(model_name, sys_prompt, file_paths)
    elif provider == "openrouter":
        return OpenRouterModelClient(model_name, sys_prompt)
    elif provider in ["hf", "qwen"]:
        # Auto-detect model type and use appropriate client
        if is_openrouter_model(model_name):
            # If user specified hf/qwen but model looks like OpenRouter, suggest correction
            print(f"Warning: Model '{model_name}' appears to be an OpenRouter model. Consider using --model openrouter")
            return OpenRouterModelClient(model_name, sys_prompt)
        elif is_qwen_model(model_name):
            return QwenModelClient(model_name, sys_prompt)
        elif is_phi_model(model_name):
            return HuggingFaceModelClient(model_name, sys_prompt)
        else:
            return HuggingFaceModelClient(model_name, sys_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, hf, qwen, openrouter")


def get_shared_relevance_client(base_client, relevance_prompt: str):
    """Create a shared model client for relevance filtering."""
    return SharedModelClient(base_client, relevance_prompt)

