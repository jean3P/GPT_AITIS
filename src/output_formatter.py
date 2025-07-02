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


# =============================================================================
# FILE 4: src/output_formatter.py (REPLACE EXISTING CONTENT)
# =============================================================================

import os
import re
import json
import logging
import datetime
from typing import Dict, List, Any

from config import get_clean_model_name

logger = logging.getLogger(__name__)


def extract_policy_id(file_path: str) -> str:
    """
    Extract policy ID from PDF or TXT filename.
    Examples:
    - "10_nobis_policy.pdf" -> "10"
    - "18_medical_coverage.txt" -> "18"
    - "10-1_AXA 20220316 DIP AGGIUNTIVO ALI@TOP.pdf" -> "10-1"
    """
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Extract policy ID from the filename using regex
    match = re.match(r'^([\d-]+)_', name_without_ext)
    if match:
        return match.group(1)
    else:
        logger.warning(f"Could not extract policy ID from filename: {filename}")
        return name_without_ext


def format_results_as_json(policy_path: str, question_results: List[List[str]]) -> Dict[str, Any]:
    """
    Format question results for a single policy as a JSON object.

    Args:
        policy_path: Path to the policy PDF
        question_results: List of question results in the format:
                         [model_name, q_id, question, eligibility, eligibility_policy, amount_policy]

    Returns:
        A JSON-serializable dictionary for the policy
    """
    policy_id = extract_policy_id(policy_path)

    # Initialize the policy JSON structure
    policy_json = {
        "policy_id": policy_id,
        "questions": []
    }

    # Add each question result to the policy JSON
    for result in question_results:
        _, q_id, question, eligibility, eligibility_policy, amount_policy = result

        question_json = {
            "request_id": q_id,
            "question": question,
            "outcome": eligibility,
            "outcome_justification": eligibility_policy,
            "payment_justification": amount_policy,
        }

        policy_json["questions"].append(question_json)

    return policy_json


def create_model_specific_output_dir(base_output_dir: str, model_name: str) -> str:
    """
    Create a model-specific output directory.

    Args:
        base_output_dir: Base output directory
        model_name: Full model name (e.g., "microsoft/phi-4", "qwen/qwen-2.5-72b-instruct")

    Returns:
        Path to the model-specific directory
    """
    clean_model_name = get_clean_model_name(model_name)
    model_output_dir = os.path.join(base_output_dir, clean_model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    logger.info(f"Created model-specific output directory: {model_output_dir}")
    return model_output_dir


def save_policy_json(policy_json: Dict[str, Any], output_dir: str, model_name: str) -> str:
    """
    Save the policy JSON to a model-specific file.

    Args:
        policy_json: JSON-serializable dictionary
        output_dir: Base output directory
        model_name: Model name for creating subdirectory

    Returns:
        Path to the saved file
    """
    # Create model-specific output directory
    model_output_dir = create_model_specific_output_dir(output_dir, model_name)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    policy_id = policy_json["policy_id"]

    # Include clean model name in filename for clarity
    clean_model_name = get_clean_model_name(model_name)
    output_filename = f"policy_id-{policy_id}__{clean_model_name}__{timestamp}.json"
    output_path = os.path.join(model_output_dir, output_filename)

    # Save JSON to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(policy_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved policy JSON to {output_path}")
    return output_path


def process_policy_results(policy_paths: List[str], all_results: List[List[str]],
                           output_dir: str, model_name: str) -> List[str]:
    """
    Process all results and create a JSON file for each policy in model-specific directory.

    Args:
        policy_paths: List of paths to policy PDFs
        all_results: List of all question results
        output_dir: Base output directory
        model_name: Model name for organizing outputs

    Returns:
        List of paths to the saved JSON files
    """
    saved_files = []

    # Convert to absolute path if needed
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    # Group results by policy ID
    for policy_path in policy_paths:
        policy_id = extract_policy_id(policy_path)
        logger.info(f"Processing results for policy ID: {policy_id}")

        # Filter results for this policy (you may need to modify this based on your data structure)
        policy_results = all_results

        # Format and save policy JSON
        policy_json = format_results_as_json(policy_path, policy_results)
        saved_path = save_policy_json(policy_json, output_dir, model_name)
        saved_files.append(saved_path)

    return saved_files
