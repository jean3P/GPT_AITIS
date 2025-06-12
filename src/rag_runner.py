# src/rag_runner.py

import logging
from typing import Optional, Tuple, List

from config import *
from models.factory import get_model_client, get_shared_relevance_client
from utils import read_questions, list_policy_paths
from models.vector_store import LocalVectorStore
from output_formatter import extract_policy_id, format_results_as_json, save_policy_json
from prompts.insurance_prompts import InsurancePrompts

logger = logging.getLogger(__name__)


def check_query_relevance(
        question: str,
        context_chunks: List[str],
        relevance_client
) -> Tuple[bool, str]:
    """
    Check if a query is relevant to the policy content.

    Args:
        question: The user query
        context_chunks: Retrieved context chunks from the policy
        relevance_client: Model client initialized with the relevance filter prompt

    Returns:
        Tuple of (is_relevant, reason)
    """
    try:
        # Use the relevance client to query
        response = relevance_client.query(question, context_files=context_chunks)

        # Extract the relevance information from the response
        if isinstance(response, dict):
            is_relevant = response.get("is_relevant", True)  # Default to True if key missing
            reason = response.get("reason", "No reason provided")
            return bool(is_relevant), str(reason)  # Ensure proper types
        else:
            logger.warning(f"Unexpected response format from relevance check: {response}")
            return True, "Unexpected response format, assuming relevant"

    except Exception as e:
        logger.warning(f"Error in relevance check: {e}, assuming relevant")
        return True, "Error in relevance check, assuming relevant"

def run_rag(
        model_provider: str = "openai",
        model_name: str = "gpt-4o",
        max_questions: Optional[int] = None,
        output_dir: Optional[str] = None,
        prompt_name: str = "standard",
        use_persona: bool = False,
        question_ids: Optional[list] = None,
        policy_id: Optional[str] = None,
        k: int = 3,
        filter_irrelevant: bool = False,
        relevance_prompt_name: str = "relevance_filter_v1",
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
        use_persona (bool): Whether to use persona extraction for the queries (default: False).
        question_ids (Optional[list]): List of question IDs to process (None = all questions).
        policy_id (Optional[str]): Filter to only process a specific policy ID (None = all policies).
        k (int): Number of context chunks to retrieve for each question (default: 3).
        filter_irrelevant (bool): Whether to filter out irrelevant context chunks (default: False).
        relevance_prompt_name (str): Name of the prompt template to use for relevance filtering.
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
    pdf_paths = list_policy_paths(DOCUMENT_DIR)
    if not pdf_paths:
        logger.error("No PDF policies found in directory")
        return
    # Filter policies by ID if specified
    if policy_id:
        filtered_paths = []
        for path in pdf_paths:
            if extract_policy_id(path) == policy_id:
                filtered_paths.append(path)
                logger.info(f"Found policy with ID {policy_id}: {os.path.basename(path)}")

        if not filtered_paths:
            logger.warning(f"No policy found with ID {policy_id}. Check if the policy exists and ID is correct.")
            return

        pdf_paths = filtered_paths
        logger.info(f"Filtered to {len(pdf_paths)} policies matching ID {policy_id}")

    # Read questions (use max_questions if provided, otherwise use all questions)
    questions_df = read_questions(DATASET_PATH)

    # First filter by question_ids if provided
    if question_ids:
        questions_df = questions_df[questions_df["Id"].astype(str).isin(question_ids)]
        logger.info(f"Filtered to {len(questions_df)} questions by ID: {', '.join(question_ids)}")

    # Then apply max_questions limit
    if max_questions is not None:
        # Limit to specified number of questions
        questions = questions_df[["Id", "Questions"]].to_numpy()[:max_questions]
        logger.info(f"Processing {len(questions)} out of {len(questions_df)} questions (limited by --num-questions)")
    else:
        # Use all questions (or all filtered by ID)
        questions = questions_df[["Id", "Questions"]].to_numpy()
        logger.info(f"Processing all {len(questions)} questions")

    # Process each policy
    for pdf_path in pdf_paths:
        policy_id = extract_policy_id(pdf_path)
        logger.info(f"Processing policy ID: {policy_id} from file: {os.path.basename(pdf_path)}")

        # Initialize the model client
        model_client = get_model_client(model_provider, model_name, sys_prompt)

        # Initialize relevance checking client if filtering is enabled
        # Initialize relevance checking client if filtering is enabled
        relevance_client = None
        if filter_irrelevant:
            try:
                relevance_prompt = InsurancePrompts.get_prompt(relevance_prompt_name)
                # Use shared model client to avoid loading a second model instance
                relevance_client = get_shared_relevance_client(model_client, relevance_prompt)
                logger.info(f"Relevance filtering enabled - using shared model with prompt: {relevance_prompt_name}")
            except ValueError as e:
                logger.warning(f"Relevance prompt selection error: {str(e)}. Falling back to relevance_filter_v1.")
                relevance_prompt = InsurancePrompts.relevance_filter_v1()
                relevance_client = get_shared_relevance_client(model_client, relevance_prompt)

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
                        context_texts = context_provider.retrieve(question, k=k)
                        logger.info(f"Retrieved {len(context_texts)} context chunks")
                    except Exception as e:
                        logger.error(f"Error retrieving context: {e}")

                # Only check relevance if filtering is enabled and we have context chunks
                if filter_irrelevant and relevance_client and context_texts:
                    # First assess query relevance using the lightweight filter
                    is_relevant, reason = check_query_relevance(question, context_texts, relevance_client)

                    if not is_relevant:
                        logger.info(f"✓ Question {q_id} marked as IRRELEVANT: {reason}")
                        # Create standardized "irrelevant" response and skip main processing
                        result_row = [
                            model_name,
                            str(q_id),
                            question,
                            "No - Unrelated event",
                            "",
                            None,
                        ]
                        policy_results.append(result_row)
                        continue  # Skip to next question
                    else:
                        logger.info(
                            f"✓ Question {q_id} IS RELEVANT: {reason} - proceeding with detailed analysis")

                # Query the model with the question and context
                response = model_client.query(question, context_files=context_texts, use_persona=use_persona)

                result_row = [
                    model_name,
                    str(q_id),
                    question,
                    response.get("answer", {}).get("eligibility", ""),
                    response.get("answer", {}).get("eligibility_policy", ""),
                    response.get("answer", {}).get("amount_policy", ""),
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
        prompt_name: str = "standard",
        use_persona: bool = False,
        question_ids: Optional[list] = None,
        policy_id: Optional[str] = None,
        k: int = 3,
        filter_irrelevant: bool = False,
        relevance_prompt_name: str = "relevance_filter_v1",
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
        use_persona (bool): Whether to use persona extraction for the queries (default: False).
        question_ids (Optional[list]): List of question IDs to process (None = all questions).
        policy_id (Optional[str]): Filter to only process a specific policy ID (None = all policies).
        k (int): Number of context chunks to retrieve for each question (default: 3).
        filter_irrelevant (bool): Whether to filter out irrelevant context chunks (default: False).
        relevance_prompt_name (str): Name of the prompt template to use for relevance filtering.
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

    # Initialize relevance checking client if filtering is enabled
    # Initialize relevance checking client if filtering is enabled
    relevance_client = None
    if filter_irrelevant:
        try:
            relevance_prompt = InsurancePrompts.get_prompt(relevance_prompt_name)
            # Use shared model client to avoid loading a second model instance
            relevance_client = get_shared_relevance_client(model_client, relevance_prompt)
            logger.info(f"Relevance filtering enabled - using shared model with prompt: {relevance_prompt_name}")
        except ValueError as e:
            logger.warning(f"Relevance prompt selection error: {str(e)}. Falling back to relevance_filter_v1.")
            relevance_prompt = InsurancePrompts.relevance_filter_v1()
            relevance_client = get_shared_relevance_client(model_client, relevance_prompt)

    # List all policy PDFs
    pdf_paths = list_policy_paths(DOCUMENT_DIR)

    # Filter policies by ID if specified
    if policy_id:
        filtered_paths = []
        for path in pdf_paths:
            if extract_policy_id(path) == policy_id:
                filtered_paths.append(path)
                logger.info(f"Found policy with ID {policy_id}: {os.path.basename(path)}")

        if not filtered_paths:
            logger.warning(f"No policy found with ID {policy_id}. Check if the policy exists and ID is correct.")
            return

        pdf_paths = filtered_paths
        logger.info(f"Filtered to {len(pdf_paths)} policies matching ID {policy_id}")

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

    questions_df = read_questions(DATASET_PATH)
    # First filter by question_ids if provided
    if question_ids:
        questions_df = questions_df[questions_df["Id"].astype(str).isin(question_ids)]
        logger.info(f"Filtered to {len(questions_df)} questions by ID: {', '.join(question_ids)}")

    # Then apply max_questions limit
    if max_questions is not None:
        # Limit to specified number of questions
        questions = questions_df[["Id", "Questions"]].to_numpy()[:max_questions]
        logger.info(f"Processing {len(questions)} out of {len(questions_df)} questions (limited by --num-questions)")
    else:
        # Use all questions (or all filtered by ID)
        questions = questions_df[["Id", "Questions"]].to_numpy()
        logger.info(f"Processing all {len(questions)} questions")

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
                        context_texts = context_provider.retrieve(query=question, k=k, policy_id=policy_id)
                        logger.info(f"Retrieved {len(context_texts)} context chunks for policy {policy_id}")
                    except Exception as e:
                        logger.error(f"Error retrieving context: {e}")

                # Only check relevance if filtering is enabled and we have context chunks
                if filter_irrelevant and relevance_client and context_texts:
                    # First assess query relevance using the lightweight filter
                    is_relevant, reason = check_query_relevance(question, context_texts, relevance_client)

                    if not is_relevant:
                        logger.info(f"✓ Question {q_id} marked as IRRELEVANT for policy {policy_id}: {reason}")
                        # Create standardized "irrelevant" response and skip main processing
                        result_row = [
                            model_name,
                            str(q_id),
                            question,
                            "No - Unrelated event",
                            "",
                            None,
                        ]
                        policy_results[policy_id]["results"].append(result_row)
                        continue  # Skip to next policy for this question
                    else:
                        logger.info(
                            f"✓ Question {q_id} IS RELEVANT for policy {policy_id}: {reason} - proceeding with detailed analysis")

                # Query the model with the question and context
                response = model_client.query(question, context_files=context_texts, use_persona=use_persona)
                result_row = [
                    model_name,
                    str(q_id),
                    question,
                    response.get("answer", {}).get("eligibility", ""),
                    response.get("answer", {}).get("eligibility_policy", ""),
                    response.get("answer", {}).get("amount_policy", "")
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
