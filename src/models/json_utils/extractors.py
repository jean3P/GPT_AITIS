# src/models/json_utils/extractors.py

"""
JSON extraction utilities.
Functions for extracting valid JSON from model outputs.
"""
import json
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class JSONExtractor:
    """
    Extracts and validates JSON from model outputs.
    Supports multiple extraction methods and validation.
    """

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from a text string using multiple methods.

        Args:
            text: The text to extract JSON from

        Returns:
            Extracted JSON dictionary or None if extraction failed
        """
        # Try multiple extraction methods
        return self._try_all_json_extraction_methods(text)

    def _try_all_json_extraction_methods(self, generated: str) -> Optional[Dict[str, Any]]:
        """
        Try multiple JSON extraction methods in sequence.

        Args:
            generated: The generated text to extract JSON from

        Returns:
            Extracted JSON or None if all methods failed
        """
        # Try solution block extraction (specific to Phi pattern)
        solution_json = self._extract_json_from_solution_block(generated)
        if solution_json:
            logger.info("Successfully parsed JSON from solution block format")
            return solution_json

        # Try code block extraction
        code_block_json = self._extract_json_from_code_block(generated)
        if code_block_json:
            logger.info("Successfully parsed JSON from code block")
            return code_block_json

        # Try generic JSON extraction
        generic_json = self._extract_json_from_text(generated)
        if generic_json:
            logger.info("Successfully parsed JSON using generic extraction")
            return generic_json

        return None

    def _extract_json_from_solution_block(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from Phi's "Solution 1" block format.

        Args:
            text: Generated text containing the solution block

        Returns:
            Parsed JSON dictionary or None if not found
        """
        solution_pattern = r'## Solution 1:.*?```json\s*(.*?)\s*```'
        match = re.search(solution_pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                logger.debug(f"Successfully extracted JSON from solution block: {json_str}")
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(f"Found solution block but JSON is invalid: {e}")

        return None

    def _extract_json_from_code_block(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from markdown code blocks.

        Args:
            text: Generated text that might contain code blocks

        Returns:
            Parsed JSON dictionary or None if not found
        """
        code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        for block in matches:
            try:
                parsed = json.loads(block.strip())
                if self._is_valid_answer_json(parsed):
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract valid JSON from text that might contain additional content.

        Args:
            text: The generated text that should contain JSON

        Returns:
            The parsed JSON dictionary or None if not found
        """
        json_candidates = self._extract_json_candidate_strings(text)

        if not json_candidates:
            logger.warning("No JSON-like patterns found in the text")
            return None

        for json_candidate in json_candidates:
            json_obj = self._parse_json_candidate(json_candidate)
            if json_obj:
                return json_obj

        logger.warning("No valid JSON with expected structure found")
        return None

    def _extract_json_candidate_strings(self, text: str) -> List[str]:
        """
        Extract potential JSON strings from text.

        Args:
            text: The text to extract JSON candidates from

        Returns:
            List of potential JSON strings, sorted by length (descending)
        """
        if not text:
            logger.warning("Empty text provided for JSON extraction")
            return []

        # Look for complex JSON pattern
        pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        matches = re.findall(pattern, text)

        # If no matches found, try a simpler pattern
        if not matches:
            logger.debug("No complex JSON patterns found, trying simpler pattern")
            simple_pattern = r'\{[^{}]*\}'
            matches = re.findall(simple_pattern, text)

        # Sort matches from longest to shortest
        matches.sort(key=len, reverse=True)
        return matches

    def _parse_json_candidate(self, json_candidate: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse a JSON candidate string.

        Args:
            json_candidate: The string to parse as JSON

        Returns:
            Parsed JSON object or None if parsing failed
        """
        try:
            # Attempt to parse the candidate
            json_obj = json.loads(json_candidate)
            if self._is_valid_answer_json(json_obj):
                return json_obj
        except json.JSONDecodeError:
            # Try to fix common JSON issues and try again
            try:
                # Replace single quotes with double quotes
                fixed_json = json_candidate.replace("'", '"')
                json_obj = json.loads(fixed_json)
                if self._is_valid_answer_json(json_obj):
                    logger.info("Successfully parsed JSON after fixing quotes")
                    return json_obj
            except json.JSONDecodeError:
                pass

        return None

    def _is_valid_answer_json(self, json_obj: Dict[str, Any]) -> bool:
        """
        Check if the JSON object has the expected structure.

        Args:
            json_obj: The JSON object to check

        Returns:
            True if the JSON has the expected structure, False otherwise
        """
        # Check for the answer structure for insurance queries
        if "answer" in json_obj and isinstance(json_obj["answer"], dict) and "eligibility" in json_obj["answer"]:
            return True

        # Check for the personas structure for persona extraction
        if "personas" in json_obj and isinstance(json_obj["personas"], dict):
            return True

        # Check for relevance filtering structure
        if "is_relevant" in json_obj and "reason" in json_obj:
            return True

        return False
