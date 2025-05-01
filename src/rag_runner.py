# src/rag_runner.py

import logging
import os
from typing import Optional

from config import *
from models.factory import get_model_client
from utils import read_questions, list_pdf_paths
from models.vector_store import LocalVectorStore
from output_formatter import extract_policy_id, format_results_as_json, save_policy_json
from prompts.insurance_prompts import InsurancePrompts

logger = logging.getLogger(__name__)


def run_rag(
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        max_questions: Optional[int] = None,
        output_dir: Optional[str] = None,
        prompt_name: str = "standard"
) -> None:
    """
    Executes the RAG pipeline using a modular model client, either OpenAI or HuggingFace.
    Now generates a JSON file for each policy with question results.

    Args:
        model_provider (str): One of "openai" or "hf" (Hugging Face).
        model_name (str): Model name (e.g., "gpt-4o" or "microsoft/phi-4").
        max_questions (Optional[int]): Maximum number of questions to process (None = all questions).
        output_dir (Optional[str]): Directory to save JSON output files.
        prompt_name (str): Name of the prompt template to use.
    """
    # Select the prompt template
    try:
        # Get the prompt by name from our InsurancePrompts class
        sys_prompt = InsurancePrompts.get_prompt(prompt_name)
        logger.info(f"Using prompt template: {prompt_name}")
    except ValueError as e:
        # If prompt not found, fall back to standard prompt
        logger.warning(f"Prompt selection error: {str(e)}. Falling back to standard prompt.")
        sys_prompt = InsurancePrompts.standard_coverage()

    # Create output directory for JSON files
    if output_dir is None:
        output_dir = os.path.join(base_dir, JSON_PATH)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"JSON output will be saved to: {output_dir}")

    # List all policy PDFs
    pdf_paths = list_pdf_paths(DOCUMENT_DIR)
    if not pdf_paths:
        logger.error("No PDF policies found in directory")
        return

    # Read questions (use max_questions if provided, otherwise use all questions)
    questions_df = read_questions(DATASET_PATH)
    if max_questions is None:
        # Use all questions
        questions = questions_df[["Id", "Questions"]].to_numpy()
        logger.info(f"Processing all {len(questions)} questions")
    else:
        # Use limited number of questions
        questions = questions_df[["Id", "Questions"]].to_numpy()[:max_questions]
        logger.info(f"Processing {len(questions)} out of {len(questions_df)} questions")

    # Process each policy
    for pdf_path in pdf_paths:
        policy_id = extract_policy_id(pdf_path)
        logger.info(f"Processing policy ID: {policy_id} from file: {os.path.basename(pdf_path)}")

        # Initialize the model client
        model_client = get_model_client(model_provider, model_name, sys_prompt)

        # Initialize vector store with just this policy file
        try:
            logger.info(f"Initializing vector store for policy: {policy_id}")
            context_provider = LocalVectorStore(model_name=EMBEDDING_MODEL_PATH)
            context_provider.index_documents([pdf_path])
            logger.info(f"Vector store initialized successfully for policy: {policy_id}")
        except Exception as e:
            logger.error(f"Error initializing vector store for policy {policy_id}: {e}")
            logger.info("Continuing without vector store")
            context_provider = None

        # Process all questions for this policy
        policy_results = []

        for q_id, question in questions:
            try:
                logger.info(f"→ Querying policy {policy_id} with question {q_id}: {question}")

                # Get relevant context if vector store is available
                context_texts = []
                if context_provider:
                    try:
                        context_texts = context_provider.retrieve(question, k=3)
                        logger.info(f"Retrieved {len(context_texts)} context chunks")
                    except Exception as e:
                        logger.error(f"Error retrieving context: {e}")

                # Query the model with the question and context
                response = model_client.query(question, context_files=context_texts)

                result_row = [
                    model_name,
                    str(q_id),
                    question,
                    response.get("answer", {}).get("eligibility", ""),
                    response.get("answer", {}).get("eligibility_policy", ""),
                    response.get("answer", {}).get("amount_policy", ""),
                    response.get("answer", {}).get("amount_policy_line", "")
                ]
                policy_results.append(result_row)
                logger.info(f"✓ Processed question {q_id} for policy {policy_id}")
            except Exception as e:
                logger.error(f"✗ Error processing question {q_id} for policy {policy_id}: {e}")
                policy_results.append([model_name, str(q_id), question, "Error", str(e), "", ""])

        # Format and save policy JSON
        policy_json = format_results_as_json(pdf_path, policy_results)
        save_policy_json(policy_json, output_dir)

    logger.info("✅ RAG run completed successfully.")


def run_batch_rag(
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        max_questions: Optional[int] = None,
        output_dir: Optional[str] = None,
        prompt_name: str = "standard"
) -> None:
    """
    Alternative implementation that processes all policies together and then
    organizes results into separate JSON files.

    Args:
        model_provider (str): One of "openai" or "hf" (Hugging Face).
        model_name (str): Model name (e.g., "gpt-4o" or "microsoft/phi-4").
        max_questions (Optional[int]): Maximum number of questions to process (None = all questions).
        output_dir (Optional[str]): Directory to save JSON output files.
        prompt_name (str): Name of the prompt template to use.
    """
    # Select the prompt template
    try:
        # Get the prompt by name from our InsurancePrompts class
        sys_prompt = InsurancePrompts.get_prompt(prompt_name)
        logger.info(f"Using prompt template: {prompt_name}")
    except ValueError as e:
        # If prompt not found, fall back to standard prompt
        logger.warning(f"Prompt selection error: {str(e)}. Falling back to standard prompt.")
        sys_prompt = InsurancePrompts.standard_coverage()

    # Initialize the model client
    model_client = get_model_client(model_provider, model_name, sys_prompt)

    # List all policy PDFs
    pdf_paths = list_pdf_paths(DOCUMENT_DIR)

    # Structure to store results by policy
    policy_results = {}

    # Create output directory for JSON files
    if output_dir is None:
        output_dir = os.path.join(base_dir, "resources/results/json_output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"JSON output will be saved to: {output_dir}")

    # Initialize and populate the vector store with all PDFs
    context_provider = None
    if pdf_paths:
        logger.info(f"Initializing vector store with {len(pdf_paths)} PDF documents")
        try:
            context_provider = LocalVectorStore(model_name=EMBEDDING_MODEL_PATH)
            context_provider.index_documents(pdf_paths)
            logger.info(f"Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            logger.info("Continuing without vector store")

    # Read questions (use max_questions if provided, otherwise use all questions)
    questions_df = read_questions(DATASET_PATH)
    if max_questions is None:
        # Use all questions
        questions = questions_df[["Id", "Questions"]].to_numpy()
        logger.info(f"Processing all {len(questions)} questions")
    else:
        # Use limited number of questions
        questions = questions_df[["Id", "Questions"]].to_numpy()[:max_questions]
        logger.info(f"Processing {len(questions)} out of {len(questions_df)} questions")

    # Initialize policy result dictionaries
    for pdf_path in pdf_paths:
        policy_id = extract_policy_id(pdf_path)
        policy_results[policy_id] = {
            "policy_path": pdf_path,
            "results": []
        }

    # Process all questions for all policies
    for q_id, question in questions:
        try:
            logger.info(f"→ Querying: {question}")

            # Process each policy separately for this question
            for pdf_path in pdf_paths:
                policy_id = extract_policy_id(pdf_path)
                logger.info(f"  Processing policy {policy_id}")

                # Get relevant context for this policy
                context_texts = []
                if context_provider:
                    try:
                        # Try to retrieve context specifically for this policy
                        context_texts = context_provider.retrieve(query=question, k=3, policy_id=policy_id)
                        logger.info(f"Retrieved {len(context_texts)} context chunks for policy {policy_id}")
                    except Exception as e:
                        logger.error(f"Error retrieving context: {e}")

                # Query the model with the question and context
                response = model_client.query(question, context_files=context_texts)

                result_row = [
                    model_name,
                    str(q_id),
                    question,
                    response.get("answer", {}).get("eligibility", ""),
                    response.get("answer", {}).get("eligibility_policy", ""),
                    response.get("answer", {}).get("amount_policy", ""),
                    response.get("answer", {}).get("amount_policy_line", "")
                ]
                policy_results[policy_id]["results"].append(result_row)
                logger.info(f"✓ Processed question {q_id} for policy {policy_id}")
        except Exception as e:
            logger.error(f"✗ Error processing question {q_id}: {e}")
            # Add error result to all policies
            for policy_id in policy_results:
                policy_results[policy_id]["results"].append([model_name, str(q_id), question, "Error", str(e), "", ""])

    # Format and save results for each policy
    for policy_id, data in policy_results.items():
        policy_json = format_results_as_json(data["policy_path"], data["results"])
        save_policy_json(policy_json, output_dir)

    logger.info("✅ RAG run completed successfully.")
