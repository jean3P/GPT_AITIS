# src/output_formatter.py

import os
import re
import json
import logging
import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def extract_policy_id(pdf_path: str) -> str:
    """
    Extract policy ID from the PDF filename.
    Examples:
    - "10_nobis_policy.pdf" -> "10"
    - "10-1_AXA 20220316 DIP AGGIUNTIVO ALI@TOP.pdf" -> "10-1"

    Args:
        pdf_path: Path to the PDF file
    Returns:
        The policy ID as a string
    """
    filename = os.path.basename(pdf_path)
    # Extract policy ID from the filename using regex
    # Updated to handle hyphens in policy IDs
    match = re.match(r'^([\d-]+)_', filename)
    if match:
        return match.group(1)
    else:
        logger.warning(f"Could not extract policy ID from filename: {filename}")
        # Return the filename without extension as fallback
        return os.path.splitext(filename)[0]


def format_results_as_json(policy_path: str, question_results: List[List[str]]) -> Dict[str, Any]:
    """
    Format question results for a single policy as a JSON object.

    Args:
        policy_path: Path to the policy PDF
        question_results: List of question results in the format:
                         [model_name, q_id, question, eligibility, eligibility_policy, amount_policy, amount_policy_line]

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
        _, q_id, question, eligibility, eligibility_policy, amount_policy, amount_policy_line = result

        question_json = {
            "request_id": q_id,
            "question": question,
            "outcome": eligibility,
            "outcome_justification": eligibility_policy,
            "payment_justification": amount_policy_line if amount_policy else None
        }

        policy_json["questions"].append(question_json)

    return policy_json


def save_policy_json(policy_json: Dict[str, Any], output_dir: str) -> str:
    """
    Save the policy JSON to a file.

    Args:
        policy_json: JSON-serializable dictionary
        output_dir: Directory where to save the JSON file

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # Define output file path
    policy_id = policy_json["policy_id"]
    output_path = os.path.join(output_dir, f"policy_id-{policy_id}__{timestamp}.json")

    # Save JSON to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(policy_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved policy JSON to {output_path}")
    return output_path


def process_policy_results(policy_paths: List[str], all_results: List[List[str]], output_dir: str) -> List[str]:
    """
    Process all results and create a JSON file for each policy.

    Args:
        policy_paths: List of paths to policy PDFs
        all_results: List of all question results
        output_dir: Directory where to save the JSON files

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

        # Filter results for this policy
        # This is a placeholder - you'll need to modify this based on how you track which result belongs to which policy
        # For now, we're assuming all questions are processed for each policy
        policy_results = all_results

        # Format and save policy JSON
        policy_json = format_results_as_json(policy_path, policy_results)
        saved_path = save_policy_json(policy_json, output_dir)
        saved_files.append(saved_path)

    return saved_files

