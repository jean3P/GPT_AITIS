# src/models/shared_client.py

"""
Shared model client for memory-efficient relevance filtering.
Uses the same model pipeline with different prompts to avoid loading multiple models.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class SharedModelClient:
    """
    A wrapper that uses the same model pipeline with different prompts.
    This avoids loading multiple model instances and saves GPU memory.
    """

    def __init__(self, base_client, relevance_prompt: str):
        """
        Initialize the shared model client.

        Args:
            base_client: The main model client (HuggingFaceModelClient or OpenAIModelClient)
            relevance_prompt: The prompt to use for relevance filtering
        """
        self.base_client = base_client
        self.relevance_prompt = relevance_prompt
        self.original_prompt = base_client.base_prompt
        self.original_json_format = getattr(base_client, 'json_format', None)
        logger.info(f"Initialized SharedModelClient for relevance filtering")

    def query(self, question: str, context_files: List[str]) -> Dict[str, Any]:
        """
        Query the model using the relevance prompt.
        Temporarily switches the base client's prompt, runs the query, then restores.

        Args:
            question: The question to analyze for relevance
            context_files: List of relevant policy files

        Returns:
            Dictionary containing relevance analysis result
        """
        # Store original values
        original_prompt = self.base_client.base_prompt
        original_json_format = getattr(self.base_client, 'json_format', None)

        try:
            # Temporarily switch to relevance prompt
            self.base_client.base_prompt = self.relevance_prompt.strip()

            # Also temporarily switch JSON format if the base client has one
            if hasattr(self.base_client, 'json_format'):
                self.base_client.json_format = """
                    Return exactly this JSON:
                    {
                      "is_relevant": true/false,
                      "reason": "Brief explanation (≤ 25 words)"
                    }
                    """

            logger.debug(f"Switched to relevance prompt for query: {question}")

            # Query with relevance prompt (persona extraction not needed for relevance checks)
            result = self.base_client.query(question, context_files, use_persona=False)

            logger.debug(f"Relevance filtering result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in shared model relevance query: {str(e)}")
            # Return a safe default that assumes relevance
            return {
                "is_relevant": True,
                "reason": f"Error in relevance check: {str(e)}"
            }
        finally:
            # Always restore original values
            self.base_client.base_prompt = original_prompt
            if hasattr(self.base_client, 'json_format') and original_json_format is not None:
                self.base_client.json_format = original_json_format
            logger.debug("Restored original prompt and JSON format after relevance check")
