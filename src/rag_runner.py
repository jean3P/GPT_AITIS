# src/rag_runner.py

import numpy as np
import logging
from config import *
from models.factory import get_model_client
from utils import read_questions


logger = logging.getLogger(__name__)


def run_rag(model_provider: str = "openai", model_name: str = "gpt-4o") -> None:
    """
    Executes the RAG pipeline using a modular model client, either OpenAI or HuggingFace.

    Args:
        model_provider (str): One of "openai" or "hf" (Hugging Face).
        model_name (str): Model name (e.g., "gpt-4o" or "microsoft/phi-4").
    """
    sys_prompt = """
            You are an expert assistant helping users understand their insurance coverage.
            Given a question and access to a policy document, follow these instructions:
        
            1. Determine if the case is covered:
               - Answer with one of the following:
                 - "Yes - it's covered"
                 - "No - not relevant"
                 - "No - not covered"
            2. If the answer is "Yes - it's covered" or "No - not covered":
               - Quote the exact sentence(s) from the policy that support your decision.
            3. If the answer is "Yes - it's covered":
               - State the **maximum amount** the policy will cover in this case.
               - Quote the exact sentence from the policy that specifies this amount.
        
            Return the answer in JSON format with the following fields:
            {
              "answer": {
                "eligibility": "Yes - it's covered" | "No - not relevant" | "No - not covered",
                "eligibility_policy": "Quoted text from policy",
                "amount_policy": "1000 CHF",
                "amount_policy_line": "Quoted line about the amount"
              }
            }
        """

    model_client = get_model_client(model_provider, model_name, sys_prompt)

    questions_df = read_questions(DATASET_PATH)
    questions = questions_df[["Id", "Questions"]].to_numpy()[:MAX_QUESTIONS]
    results = []

    logger.info(f"Processing {len(questions)} questions with model '{model_name}' from provider '{model_provider}'...")

    for q_id, question in questions:
        try:
            logger.info(f"→ Querying: {question}")
            response = model_client.query(question, context_files=[])  # context_files can be enhanced if needed
            result_row = [
                model_name,
                str(q_id),
                question,
                response.get("answer", {}).get("eligibility", ""),
                response.get("answer", {}).get("eligibility_policy", ""),
                response.get("answer", {}).get("amount_policy", ""),
                response.get("answer", {}).get("amount_policy_line", "")
            ]
            results.append(result_row)
            logger.info(f"✓ Processed: {q_id}")
        except Exception as e:
            logger.error(f"✗ Error processing question {q_id}: {e}")
            results.append([model_name, str(q_id), question, "Error", str(e), "", ""])

    logger.info(f"Saving results to: {RESULT_PATH}")
    np.savetxt(RESULT_PATH, results, delimiter="\t", fmt='%s')
    logger.info("✅ RAG run completed successfully.")



