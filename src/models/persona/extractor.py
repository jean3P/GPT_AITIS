# src/models/persona/extractor.py

"""
Main persona extraction functionality.
Coordinates rule-based and LLM-based extraction methods.
"""
import logging
from typing import Dict, Any

from models.persona.rule_based import RuleBasedExtractor
from models.persona.llm_based import LLMBasedExtractor
from models.persona.formatters import format_persona_text

logger = logging.getLogger(__name__)


class PersonaExtractor:
    """
    Extracts persona information from insurance queries.
    Combines rule-based and LLM-based approaches.
    """

    def __init__(self, pipe):
        """
        Initialize persona extractors.

        Args:
            pipe: HuggingFace pipeline for LLM-based extraction
        """
        self.rule_based_extractor = RuleBasedExtractor()
        self.llm_based_extractor = LLMBasedExtractor(pipe)

    def extract_personas(self, question: str) -> Dict[str, Any]:
        """
        Extract persona information from a question.

        Args:
            question: The insurance query to analyze

        Returns:
            Dictionary with persona information
        """
        logger.info(f"Extracting personas from question: {question}")

        # First try rule-based extraction
        rule_based_result = self.rule_based_extractor.extract(question)

        # Try LLM-based approach as a backup
        llm_result = self.llm_based_extractor.extract(question)

        # Use LLM result if available, otherwise use rule-based
        if llm_result:
            logger.info("Using LLM-extracted persona information")
            return llm_result
        else:
            logger.info("Using rule-based persona extraction")
            return rule_based_result

    def format_persona_text(self, personas_info: Dict[str, Any]) -> str:
        """
        Format persona information for inclusion in a prompt.

        Args:
            personas_info: Dictionary with persona information

        Returns:
            Formatted persona text
        """
        return format_persona_text(personas_info)