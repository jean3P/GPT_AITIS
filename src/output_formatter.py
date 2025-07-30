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


def get_timestamp_dir() -> str:
    """
    Generate a timestamp-based directory name in the format DD-MM-YY--HH-MM-SS.

    Returns:
        Timestamp string for directory naming
    """
    now = datetime.datetime.now()
    return now.strftime("%d-%m-%y--%H-%M-%S")

# Global variable to store the current run's timestamp
_current_run_timestamp = None


def get_or_create_run_timestamp() -> str:
    """
    Get the current run's timestamp, creating it if it doesn't exist.
    This ensures all outputs from a single run use the same timestamp.

    Returns:
        The timestamp for the current run
    """
    global _current_run_timestamp
    if _current_run_timestamp is None:
        _current_run_timestamp = get_timestamp_dir()
    return _current_run_timestamp


def reset_run_timestamp() -> None:
    """Reset the run timestamp (useful for testing or new runs)."""
    global _current_run_timestamp
    _current_run_timestamp = None


def create_model_specific_output_dir(base_output_dir: str, model_name: str, k: int = None,
                                     use_timestamp: bool = True, complete_policy: bool = False) -> str:
    """
    Create a directory structure based on model name, k parameter (or complete-policy), and timestamp.

    Args:
        base_output_dir: Base directory for outputs
        model_name: Name of the model (e.g., "microsoft/phi-4", "qwen/qwen-2.5-72b-instruct")
        k: Number of chunks parameter (optional, ignored if complete_policy=True)
        use_timestamp: Whether to include timestamp in directory structure (default: True)
        complete_policy: Whether using complete policy mode (default: False)

    Returns:
        Path to the created directory

    Directory structure:
        For RAG mode:
            base_output_dir/model_name/k=3/DD-MM-YY--HH-MM-SS/
        For complete policy mode:
            base_output_dir/model_name/complete-policy/DD-MM-YY--HH-MM-SS/
    """
    # Clean the model name for directory naming
    if "/" in model_name:
        # For patterns like "microsoft/phi-4" or "qwen/qwen-2.5-72b"
        clean_model_name = model_name.replace("/", "_").replace(":", "_")
    else:
        clean_model_name = model_name

    # Build the directory path step by step
    path_components = [base_output_dir, clean_model_name]

    # Add mode-specific subdirectory
    if complete_policy:
        path_components.append("complete-policy")
    elif k is not None:
        path_components.append(f"k={k}")

    # Add timestamp directory if requested
    if use_timestamp:
        timestamp = get_or_create_run_timestamp()
        path_components.append(timestamp)

    # Create the full path
    output_dir = os.path.join(*path_components)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    return output_dir


def save_policy_json(policy_data: Dict[str, Any], output_dir: str, model_name: str = None,
                     k: int = None, use_timestamp: bool = True, complete_policy: bool = False) -> str:
    """
    Save policy results to a JSON file in the appropriate directory.

    Args:
        policy_data: Dictionary containing policy results
        output_dir: Base output directory
        model_name: Model name for directory organization
        k: Number of chunks parameter for subdirectory (ignored if complete_policy=True)
        use_timestamp: Whether to use timestamp in directory structure
        complete_policy: Whether using complete policy mode

    Returns:
        Path to the saved JSON file
    """
    # Create the appropriate output directory
    if model_name:
        final_output_dir = create_model_specific_output_dir(
            output_dir, model_name, k, use_timestamp=use_timestamp,
            complete_policy=complete_policy
        )
    else:
        final_output_dir = output_dir
        os.makedirs(final_output_dir, exist_ok=True)

    # Create filename based on policy ID
    policy_id = policy_data.get("policy_id", "unknown")
    filename = f"policy_{policy_id}_results.json"
    filepath = os.path.join(final_output_dir, filename)

    # Save the JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(policy_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved policy results to: {filepath}")

    return filepath


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
