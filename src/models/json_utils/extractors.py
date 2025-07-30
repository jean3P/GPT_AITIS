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
        Extract JSON from text, handling Qwen thinking tags and malformed JSON.
        """
        # Remove thinking tags that interfere with JSON extraction
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove any text before the first { and after the last }
        # This helps with models that add explanatory text
        first_brace = text.find('{')
        last_brace = text.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]

        # QUICK FIX: Replace escaped quotes in policy text with single quotes
        text = re.sub(r'\\"([^"]*)\\"', r"'\1'", text)

        # Try multiple extraction methods
        extracted = self._try_all_json_extraction_methods(text)
        if extracted:
            normalized = self._normalize_json_fields(extracted)
            logger.info(f"Successfully extracted and normalized JSON: {normalized}")
            return normalized
        return None

    def _normalize_json_fields(self, json_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize model output JSON to the standard structure:
        {
          "answer": {
            "eligibility": <str>,
            "outcome_justification": <str>,
            "payment_justification": <str|null>
          }
        }
        """
        if "answer" in json_obj and isinstance(json_obj["answer"], dict):
            answer = json_obj["answer"]
            normalized_answer = {}

            # 1. Normalize eligibility
            eligibility = ""
            for key in ["eligibility", "elgibility", "eligiblity", "eligible"]:
                if key in answer and isinstance(answer[key], str):
                    eligibility = answer[key].strip()
                    break
            normalized_answer["eligibility"] = eligibility

            # 2. Normalize outcome_justification
            outcome = ""
            for key in ["outcome_justification", "eligibility_policy", "justification", "text", "description"]:
                if key in answer:
                    val = answer[key]
                    if isinstance(val, str):
                        outcome = val.strip()
                    elif isinstance(val, dict):
                        # flatten if nested keys contain single string values
                        outcome = "; ".join(str(v) for v in val.values() if isinstance(v, str))
                    elif isinstance(val, list):
                        outcome = "; ".join(str(v) for v in val if isinstance(v, str))
                    break
            normalized_answer["outcome_justification"] = outcome

            # 3. Normalize payment_justification
            payment = None
            for key in ["payment_justification", "amount_policy", "amount", "coverage_amount", "payment"]:
                if key in answer:
                    val = answer[key]
                    if isinstance(val, str) and val.strip():
                        payment = val.strip()
                    else:
                        payment = None
                    break
            normalized_answer["payment_justification"] = payment

            json_obj["answer"] = normalized_answer

        return json_obj

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
                if self._is_valid_answer_json_flexible(parsed):
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
        Enhanced JSON candidate extraction.
        """
        if not text:
            logger.warning("Empty text provided for JSON extraction")
            return []

        # First, try to find JSON objects with flexible key matching
        patterns = [
            # Standard JSON pattern
            r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}',
            # More flexible pattern for malformed JSON
            r'\{[^{}]*["\'](?:answer|eligibility|eligible)["\'][^{}]*\}',
        ]

        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            all_matches.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for match in all_matches:
            if match not in seen:
                seen.add(match)
                unique_matches.append(match)

        # Sort by length (longest first)
        unique_matches.sort(key=len, reverse=True)
        return unique_matches

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
            if self._is_valid_answer_json_flexible(json_obj):
                return json_obj
        except json.JSONDecodeError:
            # Try to fix common JSON issues and try again
            try:
                # Replace single quotes with double quotes
                fixed_json = json_candidate.replace("'", '"')
                json_obj = json.loads(fixed_json)
                if self._is_valid_answer_json_flexible(json_obj):
                    logger.info("Successfully parsed JSON after fixing quotes")
                    return json_obj
            except json.JSONDecodeError:
                pass
        return None

    def _is_valid_answer_json_flexible(self, json_obj: Dict[str, Any]) -> bool:
        """
        Very flexible validation - accepts almost any JSON with recognizable structure.
        Args:
            json_obj: The JSON object to check
        Returns:
            True if the JSON has some recognizable structure, False otherwise
        """
        # Check for the answer structure (very flexible)
        if "answer" in json_obj and isinstance(json_obj["answer"], dict):
            return True

        # Check for the personas structure for persona extraction
        if "personas" in json_obj and isinstance(json_obj["personas"], dict):
            return True

        # Check for relevance filtering structure
        if "is_relevant" in json_obj and "reason" in json_obj:
            return True

        # If it has at least one of the key fields we care about
        if isinstance(json_obj, dict):
            answer_keys = ["eligibility", "elgibility", "eligiblity", "eligibilty"]
            if any(key in json_obj for key in answer_keys):
                return True

        return False

    def _is_valid_answer_json(self, json_obj: Dict[str, Any]) -> bool:
        """
        Legacy method - kept for compatibility but now calls the flexible version
        """
        return self._is_valid_answer_json_flexible(json_obj)
