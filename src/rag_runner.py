# src/rag_runner.py

import logging
from typing import Optional, Tuple, List

from config import *
from models.factory import get_model_client, get_shared_relevance_client
from utils import read_questions, list_policy_paths
from models.vector_store import LocalVectorStore, EnhancedLocalVectorStore
from output_formatter import extract_policy_id, format_results_as_json, save_policy_json, \
    create_model_specific_output_dir
from prompts.insurance_prompts import InsurancePrompts

logger = logging.getLogger(__name__)


def load_complete_policy(pdf_path: str) -> str:
    """
    Load the complete text content of a policy PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Complete text content of the policy
    """
    try:
        import PyPDF2

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = []

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    full_text.append(text)

            complete_text = "\n".join(full_text)
            logger.info(f"Loaded complete policy: {len(complete_text)} characters")
            return complete_text

    except Exception as e:
        logger.error(f"Error loading complete policy from {pdf_path}: {e}")
        # Try alternative method with pdfplumber
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                full_text = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text.append(text)

                complete_text = "\n".join(full_text)
                logger.info(f"Loaded complete policy with pdfplumber: {len(complete_text)} characters")
                return complete_text

        except Exception as e2:
            logger.error(f"Error with pdfplumber: {e2}")
            raise


def check_policy_size_for_model(policy_text: str, model_name: str) -> None:
    """
    Check if policy size might exceed model token limits and warn user.
    """
    # Rough estimation: 1 token ≈ 4 characters
    estimated_tokens = len(policy_text) / 4

    # Model context windows (approximate)
    model_limits = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4o": 128000,
        "gpt-3.5": 4096,
        "phi-4": 100000,
        "qwen": 62768,
        "qwen-2.5-72b": 32768,
    }

    for model_key, limit in model_limits.items():
        if model_key in model_name.lower():
            if estimated_tokens > limit * 0.8:  # 80% threshold
                logger.warning(
                    f"Complete policy (~{int(estimated_tokens)} tokens) may exceed "
                    f"{model_name} context limit ({limit} tokens). "
                    f"Consider using RAG mode or a model with larger context."
                )
            break

def get_vector_store(strategy: str, model_name: str):
    """Factory function to get appropriate vector store based on strategy."""
    if strategy == "simple":
        return LocalVectorStore(model_name=model_name)
    elif strategy == "section":
        return EnhancedLocalVectorStore(
            model_name=model_name,
            chunking_strategy="section",
            chunking_config={
                "max_section_length": 2000,
                "preserve_subsections": True,
                "include_front_matter": False,
                "sentence_window_size": 5
            }
        )
    elif strategy == "smart_size":
        return EnhancedLocalVectorStore(
            model_name=model_name,
            chunking_strategy="smart_size",
            chunking_config={
                "base_chunk_words": 105,
                "min_chunk_words": 50,
                "max_chunk_words": 280,
                "importance_multiplier": 1.6,
                "preserve_complete_clauses": True,
                "overlap_words": 5
            }
        )
    elif strategy == "semantic":
        return EnhancedLocalVectorStore(
            model_name=model_name,
            chunking_strategy="semantic",
            chunking_config={
                "embedding_model": "all-MiniLM-L6-v2",
                "breakpoint_threshold_type": "percentile",
                "breakpoint_threshold_value": 75,
                "min_chunk_sentences": 2,
                "max_chunk_sentences": 15,
                "preserve_paragraph_boundaries": True,
                "device": "cpu"
            }
        )
    elif strategy == "graph":
        # NEW: Graph-based strategy for PankRAG
        return EnhancedLocalVectorStore(
            model_name=model_name,
            chunking_strategy="graph",
            chunking_config={
                "max_chunk_size": 512,
                "community_size": 50,
                "enable_hierarchical": True
            }
        )
    elif strategy == "semantic_graph":
        # Semantic graph-based strategy - combines embeddings with graph structure
        return EnhancedLocalVectorStore(
            model_name=model_name,
            chunking_strategy="semantic_graph",
            chunking_config={
                "max_chunk_size": 512,
                "similarity_threshold": 0.75,
                "min_community_size": 3,
                "enable_hierarchical": False,
                "embedding_model": "all-MiniLM-L6-v2",
                "semantic_window": 5,
                "entity_weight": 1.5,
                "sequential_weight": 0.8
            }
        )
    else:
        raise ValueError(f"Unknown RAG strategy: {strategy}")


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
        rag_strategy: str = "simple",
        complete_policy: bool = False,
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
        rag_strategy (str): Name of the approach strategy
        complete_policy (bool): Whether to pass the complete policy document instead of using RAG (default: False)
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

    model_output_dir = create_model_specific_output_dir(
        output_dir, model_name, k, complete_policy=complete_policy
    )
    logger.info(f"JSON output will be saved to: {model_output_dir}")

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
        current_policy_id = extract_policy_id(pdf_path)
        logger.info(f"Processing policy ID: {current_policy_id} from file: {os.path.basename(pdf_path)}")

        # Initialize the model client
        model_client = get_model_client(model_provider, model_name, sys_prompt)

        # Initialize relevance checking client if filtering is enabled and not in complete policy mode
        relevance_client = None
        if filter_irrelevant and not complete_policy:
            try:
                relevance_prompt = InsurancePrompts.get_prompt(relevance_prompt_name)
                # Use shared model client to avoid loading a second model instance
                relevance_client = get_shared_relevance_client(model_client, relevance_prompt)
                logger.info(f"Relevance filtering enabled - using shared model with prompt: {relevance_prompt_name}")
            except ValueError as e:
                logger.warning(f"Relevance prompt selection error: {str(e)}. Falling back to relevance_filter_v1.")
                relevance_prompt = InsurancePrompts.relevance_filter_v1()
                relevance_client = get_shared_relevance_client(model_client, relevance_prompt)

        # Handle complete policy mode vs RAG mode
        complete_policy_text = None
        context_provider = None

        if complete_policy:
            # Load the complete policy text
            logger.info(f"Loading complete policy document for policy: {current_policy_id}")
            try:
                complete_policy_text = load_complete_policy(pdf_path)
                logger.info(f"Complete policy loaded: {len(complete_policy_text)} characters")

                # Check if policy size might exceed model limits
                check_policy_size_for_model(complete_policy_text, model_name)
            except Exception as e:
                logger.error(f"Failed to load complete policy: {e}")
                continue
        else:
            # Initialize vector store with just this policy file
            try:
                logger.info(f"Initializing vector store for policy: {current_policy_id}")
                context_provider = get_vector_store(rag_strategy, EMBEDDING_MODEL_PATH)
                context_provider.index_documents([pdf_path])
                logger.info(f"Vector store initialized successfully for policy: {current_policy_id}")
            except Exception as e:
                logger.error(f"Error initializing vector store for policy {current_policy_id}: {e}")
                logger.info("Continuing without vector store")
                context_provider = None

        # Process all questions for this policy
        policy_results = []

        for q_id, question in questions:
            try:
                logger.info(f"→ Querying policy {current_policy_id} with question {q_id}: {question}")

                # Get context based on mode
                context_texts = []

                if complete_policy:
                    # Use complete policy as context
                    context_texts = [complete_policy_text]
                    logger.info(f"Using complete policy as context ({len(complete_policy_text)} characters)")
                else:
                    # Get relevant context if vector store is available
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
                            logger.info(f"Added irrelevant result: {result_row}")
                            continue  # Skip to next question
                        else:
                            logger.info(
                                f"✓ Question {q_id} IS RELEVANT: {reason} - proceeding with detailed analysis")

                # Query the model with the question and context
                response = model_client.query(question, context_files=context_texts, use_persona=use_persona)

                # COMPREHENSIVE LOGGING OF MODEL RESPONSE
                logger.info(f"=== RAG RESPONSE LOGGING Q{q_id} ===")
                logger.info(f"Raw model response: {response}")
                logger.info(f"Response type: {type(response)}")

                if isinstance(response, dict) and "answer" in response:
                    answer = response["answer"]
                    logger.info(f"Answer section: {answer}")
                    logger.info(f"Answer keys: {list(answer.keys())}")

                    # Log each field extraction
                    eligibility = answer.get('eligibility', 'MISSING')
                    outcome_just = answer.get('outcome_justification', 'MISSING')
                    payment_just = answer.get('payment_justification', 'MISSING')

                    logger.info(f"Eligibility: '{eligibility}'")
                    logger.info(f"Outcome justification: '{outcome_just}'")
                    logger.info(f"Payment justification: '{payment_just}'")

                    # Also check for old field names (debugging)
                    old_eligibility_policy = answer.get('eligibility_policy', 'NOT_FOUND')
                    old_amount_policy = answer.get('amount_policy', 'NOT_FOUND')
                    logger.info(f"OLD eligibility_policy (should be NOT_FOUND): '{old_eligibility_policy}'")
                    logger.info(f"OLD amount_policy (should be NOT_FOUND): '{old_amount_policy}'")
                else:
                    logger.warning(f"Unexpected response format: {response}")

                result_row = [
                    model_name,
                    str(q_id),
                    question,
                    response.get("answer", {}).get("eligibility", ""),
                    response.get("answer", {}).get("outcome_justification", ""),  # NEW FIELD NAME
                    response.get("answer", {}).get("payment_justification", ""),  # NEW FIELD NAME
                ]

                logger.info(f"Created result_row: {result_row}")
                logger.info(f"Result row length: {len(result_row)}")
                logger.info(f"=== END RAG RESPONSE LOGGING ===")

                policy_results.append(result_row)
                logger.info(f"✓ Processed question {q_id} for policy {current_policy_id}")

            except Exception as e:
                logger.error(f"✗ Error processing question {q_id} for policy {current_policy_id}: {e}")
                error_result = [model_name, str(q_id), question, "Error", str(e), "", ""]
                policy_results.append(error_result)
                logger.info(f"Added error result: {error_result}")

        # COMPREHENSIVE LOGGING BEFORE JSON FORMATTING
        logger.info(f"=== FINAL POLICY RESULTS FOR {current_policy_id} ===")
        logger.info(f"Number of results: {len(policy_results)}")
        for i, result in enumerate(policy_results):
            logger.info(f"Result {i}: {result}")
        logger.info(f"=== END POLICY RESULTS ===")

        # Format and save policy JSON
        policy_json = format_results_as_json(pdf_path, policy_results)

        # COMPREHENSIVE LOGGING OF FINAL JSON
        logger.info(f"=== GENERATED JSON FOR {current_policy_id} ===")
        logger.info(f"JSON keys: {list(policy_json.keys())}")
        logger.info(f"Policy ID in JSON: {policy_json.get('policy_id', 'MISSING')}")

        if "questions" in policy_json:
            logger.info(f"JSON questions count: {len(policy_json['questions'])}")
            # Log first 3 questions in detail
            for i, q in enumerate(policy_json["questions"][:3]):
                logger.info(f"JSON Question {i}: {q}")
            if len(policy_json["questions"]) > 3:
                logger.info(f"... and {len(policy_json['questions']) - 3} more questions")

        logger.info(f"Complete JSON structure: {policy_json}")
        logger.info(f"=== END GENERATED JSON ===")

        save_policy_json(
            policy_json, output_dir, model_name, k,
            use_timestamp=True, complete_policy=complete_policy
        )
        logger.info(f"Saved JSON for policy {current_policy_id}")

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
        rag_strategy: str = "simple",
        complete_policy: bool = False,
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
        rag_strategy (str): Name of the approach strategy
        complete_policy (bool): Whether to pass the complete policy document instead of using RAG (default: False)
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

    # Initialize relevance checking client if filtering is enabled and not in complete policy mode
    relevance_client = None
    if filter_irrelevant and not complete_policy:
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

    model_output_dir = create_model_specific_output_dir(
        output_dir, model_name, k, complete_policy=complete_policy
    )
    logger.info(f"JSON output will be saved to: {model_output_dir}")

    # Structure to store complete policies if needed
    complete_policies = {}

    # Initialize and populate based on mode
    context_provider = None
    if complete_policy:
        # Load all complete policies
        logger.info(f"Loading complete text for {len(pdf_paths)} policies")
        for pdf_path in pdf_paths:
            current_policy_id = extract_policy_id(pdf_path)
            try:
                policy_text = load_complete_policy(pdf_path)
                complete_policies[current_policy_id] = policy_text
                logger.info(f"Loaded complete policy {current_policy_id}: {len(policy_text)} characters")

                # Check if policy size might exceed model limits
                check_policy_size_for_model(policy_text, model_name)
            except Exception as e:
                logger.error(f"Failed to load policy {current_policy_id}: {e}")
    else:
        # Initialize and populate the vector store with all PDFs
        if pdf_paths:
            logger.info(f"Initializing vector store with {len(pdf_paths)} PDF documents")
            try:
                context_provider = get_vector_store(rag_strategy, EMBEDDING_MODEL_PATH)
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
        current_policy_id = extract_policy_id(pdf_path)
        policy_results[current_policy_id] = {
            "policy_path": pdf_path,
            "results": []
        }

    # Process all questions for all policies
    for q_id, question in questions:
        try:
            logger.info(f"→ Querying: {question}")

            # Process each policy separately for this question
            for pdf_path in pdf_paths:
                current_policy_id = extract_policy_id(pdf_path)
                logger.info(f"  Processing policy {current_policy_id}")

                # Get context based on mode
                context_texts = []

                if complete_policy:
                    # Use complete policy text
                    if current_policy_id in complete_policies:
                        context_texts = [complete_policies[current_policy_id]]
                        logger.info(
                            f"Using complete policy for {current_policy_id} ({len(context_texts[0])} characters)")
                    else:
                        logger.error(f"Complete policy not loaded for {current_policy_id}")
                        continue
                else:
                    # Get relevant context for this policy
                    if context_provider:
                        try:
                            # Try to retrieve context specifically for this policy
                            context_texts = context_provider.retrieve(query=question, k=k, policy_id=current_policy_id)
                            logger.info(f"Retrieved {len(context_texts)} context chunks for policy {current_policy_id}")
                        except Exception as e:
                            logger.error(f"Error retrieving context: {e}")

                    # Only check relevance in RAG mode
                    if filter_irrelevant and relevance_client and context_texts:
                        # First assess query relevance using the lightweight filter
                        is_relevant, reason = check_query_relevance(question, context_texts, relevance_client)

                        if not is_relevant:
                            logger.info(
                                f"✓ Question {q_id} marked as IRRELEVANT for policy {current_policy_id}: {reason}")
                            # Create standardized "irrelevant" response and skip main processing
                            result_row = [
                                model_name,
                                str(q_id),
                                question,
                                "No - Unrelated event",
                                "",
                                None,
                            ]
                            policy_results[current_policy_id]["results"].append(result_row)
                            logger.info(f"Added irrelevant result for policy {current_policy_id}: {result_row}")
                            continue  # Skip to next policy for this question
                        else:
                            logger.info(
                                f"✓ Question {q_id} IS RELEVANT for policy {current_policy_id}: {reason} - proceeding with detailed analysis")

                # Query the model with the question and context
                response = model_client.query(question, context_files=context_texts, use_persona=use_persona)

                # COMPREHENSIVE LOGGING OF MODEL RESPONSE (BATCH VERSION)
                logger.info(f"=== BATCH RAG RESPONSE Q{q_id} P{current_policy_id} ===")
                logger.info(f"Raw model response: {response}")
                logger.info(f"Response type: {type(response)}")

                if isinstance(response, dict) and "answer" in response:
                    answer = response["answer"]
                    logger.info(f"Answer section: {answer}")
                    logger.info(f"Answer keys: {list(answer.keys())}")

                    # Log each field extraction
                    eligibility = answer.get('eligibility', 'MISSING')
                    outcome_just = answer.get('outcome_justification', 'MISSING')
                    payment_just = answer.get('payment_justification', 'MISSING')

                    logger.info(f"Eligibility: '{eligibility}'")
                    logger.info(f"Outcome justification: '{outcome_just}'")
                    logger.info(f"Payment justification: '{payment_just}'")

                    # Also check for old field names (debugging)
                    old_eligibility_policy = answer.get('eligibility_policy', 'NOT_FOUND')
                    old_amount_policy = answer.get('amount_policy', 'NOT_FOUND')
                    logger.info(f"OLD eligibility_policy (should be NOT_FOUND): '{old_eligibility_policy}'")
                    logger.info(f"OLD amount_policy (should be NOT_FOUND): '{old_amount_policy}'")
                else:
                    logger.warning(f"Unexpected response format: {response}")

                result_row = [
                    model_name,
                    str(q_id),
                    question,
                    response.get("answer", {}).get("eligibility", ""),
                    response.get("answer", {}).get("outcome_justification", ""),  # NEW FIELD NAME
                    response.get("answer", {}).get("payment_justification", ""),  # NEW FIELD NAME
                ]

                logger.info(f"Batch result_row for policy {current_policy_id}: {result_row}")
                logger.info(f"=== END BATCH RAG RESPONSE ===")

                policy_results[current_policy_id]["results"].append(result_row)
                logger.info(f"✓ Processed question {q_id} for policy {current_policy_id}")

        except Exception as e:
            logger.error(f"✗ Error processing question {q_id}: {e}")
            # Add error result to all policies
            for current_policy_id in policy_results:
                error_result = [model_name, str(q_id), question, "Error", str(e), "", ""]
                policy_results[current_policy_id]["results"].append(error_result)
                logger.info(f"Added error result for policy {current_policy_id}: {error_result}")

    # Format and save results for each policy
    for current_policy_id, data in policy_results.items():
        # COMPREHENSIVE LOGGING BEFORE JSON FORMATTING (BATCH VERSION)
        logger.info(f"=== BATCH FINAL POLICY RESULTS FOR {current_policy_id} ===")
        logger.info(f"Number of results: {len(data['results'])}")
        for i, result in enumerate(data["results"]):
            logger.info(f"Result {i}: {result}")
        logger.info(f"=== END BATCH POLICY RESULTS ===")

        policy_json = format_results_as_json(data["policy_path"], data["results"])

        # COMPREHENSIVE LOGGING OF FINAL JSON (BATCH VERSION)
        logger.info(f"=== BATCH GENERATED JSON FOR {current_policy_id} ===")
        logger.info(f"JSON keys: {list(policy_json.keys())}")
        logger.info(f"Policy ID in JSON: {policy_json.get('policy_id', 'MISSING')}")

        if "questions" in policy_json:
            logger.info(f"JSON questions count: {len(policy_json['questions'])}")
            # Log first 3 questions in detail
            for i, q in enumerate(policy_json["questions"][:3]):
                logger.info(f"JSON Question {i}: {q}")
            if len(policy_json["questions"]) > 3:
                logger.info(f"... and {len(policy_json['questions']) - 3} more questions")

        logger.info(f"Complete JSON structure: {policy_json}")
        logger.info(f"=== END BATCH GENERATED JSON ===")

        save_policy_json(
            policy_json, output_dir, model_name, k,
            use_timestamp=True, complete_policy=complete_policy
        )
        logger.info(f"Saved JSON for policy {current_policy_id}")

    logger.info("✅ RAG run completed successfully.")

