# src/utils.py
import json

import pandas as pd
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def read_questions(path: str) -> pd.DataFrame:
    """
    Load an Excel file containing insurance-related questions.

    Args:
        path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame of questions.
    """
    logger.info(f"Read question path: {path}")
    df = pd.read_excel(path)
    return df

# def list_pdf_paths(directory: str) -> List[str]:
#     """
#     List all PDF file paths within the specified directory.
#
#     Args:
#         directory (str): Path to the directory containing PDFs.
#
#     Returns:
#         list[str]: List of absolute file paths.
#     """
#     logger.info(f"List pdf paths: {directory}")
#     return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]

def load_response_schema(path: str) -> Dict[str, Any]:
    """
    Load the JSON schema used to enforce assistant response structure.

    Args:
        path (str): Path to the JSON schema file.

    Returns:
        dict: JSON schema dictionary.
    """
    logger.info(f"Load response schema: {path}")
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def list_policy_paths(directory: str) -> List[str]:
    """
    List all PDF and TXT file paths within the specified directory.

    Args:
        directory (str): Path to the directory containing policy files.

    Returns:
        list[str]: List of absolute file paths.
    """
    logger.info(f"List policy paths: {directory}")
    policy_files = []

    for f in os.listdir(directory):
        if f.endswith((".pdf", ".txt")):
            policy_files.append(os.path.join(directory, f))

    logger.info(f"Found {len(policy_files)} policy files (PDF and TXT)")
    return policy_files




