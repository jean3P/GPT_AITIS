import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import HUGGINGFACE_TOKEN
from models.base import BaseModelClient

# Login with your token
login(token=HUGGINGFACE_TOKEN)

logger = logging.getLogger(__name__)


class HuggingFaceModelClient(BaseModelClient):
    def __init__(self, model_name: str, sys_prompt: str):
        logger.info(f"Loading HuggingFace model: {model_name}")
        self._log_cache_locations()
        model_name = self._check_model_in_scratch(model_name)

        self.pipe = self._initialize_pipeline(model_name)
        self.base_prompt = sys_prompt.strip()
        self.json_format = self._get_json_format_template()

    def _log_cache_locations(self) -> None:
        """Log HuggingFace cache locations to verify they're set."""
        logger.info(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', 'Not set')}")
        logger.info(f"HF_ASSETS_CACHE: {os.environ.get('HF_ASSETS_CACHE', 'Not set')}")

    def _check_model_in_scratch(self, model_name: str) -> str:
        """Check if the model exists in scratch directory and return the appropriate path."""
        if model_name == "microsoft/phi-4":
            scratch_model_path = os.path.join("/cluster/scratch", os.environ.get("USER", ""), "models", "phi-4")

            if os.path.exists(scratch_model_path):
                logger.info(f"Found phi-4 model in scratch: {scratch_model_path}")
                return scratch_model_path
            else:
                logger.warning(f"Phi-4 not found in scratch directory. Looking for it in HuggingFace cache.")

        return model_name

    def _initialize_pipeline(self, model_name: str):
        """Initialize the HuggingFace pipeline with the specified model."""
        try:
            logger.info(f"Initializing pipeline with model: {model_name}")

            tokenizer = self._load_tokenizer(model_name)
            model = self._load_model(model_name)

            # Create pipeline with loaded model and tokenizer
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )

            logger.info("Pipeline successfully initialized")
            return pipe

        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

    def _load_tokenizer(self, model_name: str):
        """Load the tokenizer for the specified model."""
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def _load_model(self, model_name: str):
        """Load the model with appropriate configuration."""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

    def _get_json_format_template(self) -> str:
        """Return the JSON format template for model output."""
        return """
            Format the output EXACTLY in the following JSON schema WITHOUT ANY ADDITIONAL TEXT BEFORE OR AFTER:
            {
              "answer": {
                "eligibility": "Yes - it's covered | No - not relevant | No - not covered | No - condition(s) not met | Maybe",
                "eligibility_policy": "Quoted text from policy",
                "amount_policy": "Amount like '1000 CHF' or null",
                "amount_policy_line": "Quoted policy text or null"
              }
            }
            """

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

    def _is_valid_answer_json(self, json_obj: Dict[str, Any]) -> bool:
        """
        Check if the JSON object has the expected structure.

        Args:
            json_obj: The JSON object to check

        Returns:
            True if the JSON has the expected structure, False otherwise
        """
        return (
                "answer" in json_obj and
                isinstance(json_obj["answer"], dict) and
                "eligibility" in json_obj["answer"]
        )

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

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create a formatted error response.

        Args:
            error_message: The error message to include

        Returns:
            Formatted error response dictionary
        """
        return {
            "answer": {
                "eligibility": "Error",
                "eligibility_policy": error_message,
                "amount_policy": None,
                "amount_policy_line": None
            }
        }

    def _extract_family_relationships(self, question: str) -> Tuple[List[str], List[str]]:
        """
        Extract family relationship terms from the question.

        Args:
            question: The question to analyze

        Returns:
            Tuple of (found_relationships, mentioned_people)
        """
        found_relationships = []
        mentioned_people = []

        # Family relationship terms
        family_terms = {
            'daughter': 'daughter',
            'son': 'son',
            'child': 'child',
            'children': 'children',
            'wife': 'wife',
            'husband': 'husband',
            'spouse': 'spouse',
            'partner': 'partner',
            'mother': 'mother',
            'father': 'father',
            'parent': 'parent',
            'parents': 'parents',
            'grandparent': 'grandparent',
            'grandmother': 'grandmother',
            'grandfather': 'grandfather',
            'sister': 'sister',
            'brother': 'brother',
            'sibling': 'sibling',
            'aunt': 'aunt',
            'uncle': 'uncle',
            'cousin': 'cousin',
            'niece': 'niece',
            'nephew': 'nephew',
            'family': 'family member'
        }

        # Patterns for indirect references
        indirect_references = [
            r'my\s+(?:business\s+)?partner\s+had',
            r'my\s+(?:family\s+)?member\s+(?:had|was|got)',
            r'my\s+(?:colleague|coworker)\s+(?:had|was|got)',
            r'(?:death|illness|injury)\s+of\s+(?:my|the)',
        ]

        # Check for family members and relationships
        for term, relationship_type in family_terms.items():
            if re.search(r'\b' + term + r'\b', question, re.IGNORECASE):
                found_relationships.append(relationship_type)

                # Check if this is mentioned as a non-claimant
                is_indirect = any(
                    re.search(pattern + r'.*\b' + term + r'\b', question, re.IGNORECASE)
                    for pattern in indirect_references
                )

                # Add to mentioned people
                if is_indirect:
                    mentioned_people.append(f"{relationship_type} (not a claimant)")
                else:
                    mentioned_people.append(relationship_type)

        return found_relationships, mentioned_people

    def _extract_non_family_relationships(self, question: str) -> Tuple[List[str], List[str]]:
        """
        Extract non-family relationship terms from the question.

        Args:
            question: The question to analyze

        Returns:
            Tuple of (found_relationships, mentioned_people)
        """
        found_relationships = []
        mentioned_people = []

        # Non-family relationship terms
        other_terms = {
            'friend': 'friend',
            'friends': 'friends',
            'colleague': 'colleague',
            'coworker': 'coworker',
            'business partner': 'business partner',
            'neighbor': 'neighbor',
            'guest': 'guest',
            'traveler': 'fellow traveler'
        }

        # Patterns for indirect references
        indirect_references = [
            r'my\s+(?:business\s+)?partner\s+had',
            r'my\s+(?:family\s+)?member\s+(?:had|was|got)',
            r'my\s+(?:colleague|coworker)\s+(?:had|was|got)',
            r'(?:death|illness|injury)\s+of\s+(?:my|the)',
        ]

        # Check for non-family relationships
        for term, relationship_type in other_terms.items():
            if re.search(r'\b' + term + r'\b', question, re.IGNORECASE):
                found_relationships.append(relationship_type)

                # Check if this is mentioned as a non-claimant
                is_indirect = any(
                    re.search(pattern + r'.*\b' + term + r'\b', question, re.IGNORECASE)
                    for pattern in indirect_references
                )

                # Add to mentioned people
                if is_indirect:
                    mentioned_people.append(f"{relationship_type} (not a claimant)")
                else:
                    mentioned_people.append(relationship_type)

        return found_relationships, mentioned_people

    def _count_people_from_pronouns(self, question: str) -> int:
        """
        Count distinct people based on pronoun usage, with improved contextual understanding.

        Args:
            question: The question to analyze

        Returns:
            Estimated count of distinct people
        """
        # Check if "our" is being used in institutional context rather than indicating multiple people
        institutional_our_pattern = r'\b(our|at our) (hotel|resort|company|office|facility|premises|building|property|organization|institution)\b'
        has_institutional_our = bool(re.search(institutional_our_pattern, question, re.IGNORECASE))

        # Check for different types of pronouns
        has_first_person_singular = bool(re.search(r'\b(I|me|my)\b', question, re.IGNORECASE))

        # Modified first person plural check to exclude institutional "our"
        has_first_person_plural = bool(re.search(r'\b(we|us)\b', question, re.IGNORECASE))
        if not has_first_person_plural and re.search(r'\bour\b', question, re.IGNORECASE) and not has_institutional_our:
            has_first_person_plural = True

        has_second_person = bool(re.search(r'\b(you|your)\b', question, re.IGNORECASE))
        has_third_person_singular_male = bool(re.search(r'\b(he|him|his)\b', question, re.IGNORECASE))
        has_third_person_singular_female = bool(re.search(r'\b(she|her)\b', question, re.IGNORECASE))
        has_third_person_plural = bool(re.search(r'\b(they|them|their)\b', question, re.IGNORECASE))

        # Count distinct people based on pronoun types
        distinct_people_count = 0

        if has_first_person_singular:
            distinct_people_count += 1  # The speaker/policy holder

        if has_first_person_plural and not has_institutional_our:
            # 'We' implies at least 2 people, but only if not institutional
            distinct_people_count = max(distinct_people_count, 2)

        # Only count 'you' as another person when it's clearly referring to a different individual
        # and not the customer service or entity being addressed
        if has_second_person and re.search(r'\b(you|your) (?:and|with|also|too)\b', question, re.IGNORECASE):
            distinct_people_count += 1

        # More contextual checks for second person
        # Don't count 'you' in service requests like "can you help"
        if has_second_person and not re.search(r'(?:can|could|would|will|please)\s+you', question, re.IGNORECASE):
            distinct_people_count += 1

        if has_third_person_singular_male:
            distinct_people_count += 1

        if has_third_person_singular_female:
            distinct_people_count += 1

        if has_third_person_plural:
            # 'They' implies at least 2 more people
            if distinct_people_count == 0:
                distinct_people_count = 2  # Minimum for 'they'
            else:
                distinct_people_count += 1  # Add at least one more person

        # If no pronouns were found, default to 1 person (the claimant)
        if distinct_people_count == 0:
            distinct_people_count = 1

        return distinct_people_count

    def _determine_default_persona_info(
            self,
            question: str,
            found_relationships: List[str]
    ) -> Tuple[str, str, int, int, str]:
        """
        Determine default persona information based on pronouns and context.

        Args:
            question: The question to analyze
            found_relationships: List of relationships found in the question

        Returns:
            Tuple of (policy_user, relationship, total_count, claimant_count, relationship)
        """
        # Default based on pronouns
        if re.search(r'\b(I|me|my)\b', question, re.IGNORECASE) and not found_relationships:
            policy_user = "policyholder (individual)"
            relationship = "none mentioned"
            claimant_count = 1

        elif re.search(r'\b(we|us|our)\b', question, re.IGNORECASE) and not found_relationships:
            policy_user = "policyholder (group)"
            relationship = "group (unspecified)"
            claimant_count = 2  # At least 2 people

        elif found_relationships:
            # If relationships found but no clear policy user, assume relationship is claimant
            policy_user = found_relationships[0]
            relationship = "family member of policyholder"
            claimant_count = 1

        else:
            # Default if nothing could be determined
            policy_user = "undetermined"
            relationship = "Not clearly specified"
            claimant_count = 1

        # Get people count from pronouns
        total_count = self._count_people_from_pronouns(question)

        return policy_user, relationship, total_count, claimant_count, relationship

    def _rule_based_persona_extraction(self, question: str) -> Dict[str, Any]:
        """
        Extract persona information using rule-based approach, now including
        who experienced the event.

        Args:
            question: The insurance query to analyze

        Returns:
            Dictionary with persona information
        """
        # Initialize defaults
        policy_user = None
        affected_person = None  # New field for who experienced the event
        mentioned_people = []
        relationship = None
        total_count = 1
        claimant_count = 1

        # Extract family relationships
        found_relationships, family_mentioned = self._extract_family_relationships(question)
        mentioned_people.extend(family_mentioned)

        # Extract non-family relationships
        other_relationships, other_mentioned = self._extract_non_family_relationships(question)
        found_relationships.extend(other_relationships)
        mentioned_people.extend(other_mentioned)

        # Try to identify who experienced the event
        affected_person = self._identify_affected_person(question)

        # Handle special cases
        (special_policy_user,
         special_mentioned,
         special_relationship,
         special_total_count,
         special_claimant_count,
         special_affected_person) = self._handle_special_case_personas(question, found_relationships)

        if special_policy_user:
            policy_user = special_policy_user
        if special_mentioned:
            mentioned_people = special_mentioned
        if special_relationship:
            relationship = special_relationship
        if special_total_count:
            total_count = special_total_count
        if special_claimant_count:
            claimant_count = special_claimant_count
        if special_affected_person:
            affected_person = special_affected_person

        # If not a special case, determine defaults
        if not policy_user:
            (policy_user,
             relationship,
             default_total,
             default_claimant,
             relationship) = self._determine_default_persona_info(
                question,
                found_relationships
            )

            # Only use defaults if not set by special cases
            if not special_total_count:
                total_count = default_total
            if not special_claimant_count:
                claimant_count = default_claimant

        # If there are specific people mentioned, update total count
        if mentioned_people:
            mentioned_count = len(mentioned_people)
            total_count = max(total_count, mentioned_count)

        # If affected_person is still None, default to policy_user
        if not affected_person:
            affected_person = f"{policy_user} (inferred)"

        # Format the result
        return {
            "personas": {
                "policy_user": policy_user,
                "affected_person": affected_person,
                "mentioned_people": ", ".join(mentioned_people) if mentioned_people else "None specifically mentioned",
                "total_count": total_count,
                "claimant_count": claimant_count,
                "relationship": relationship if relationship else "Not clearly specified"
            }
        }

    def _identify_affected_person(self, question: str) -> Optional[str]:
        """
        Identify who experienced the event/accident/illness in the query.

        Args:
            question: The query to analyze

        Returns:
            String describing who experienced the event or None if unclear
        """
        # Patterns for first-person event experiencing
        first_person_patterns = [
            r'I\s+(?:had|have|got|experienced|suffered|am suffering|was diagnosed with|developed)\s+(?:a|an)?\s*(?:illness|sickness|disease|condition|injury|accident|problem)',
            r'I\s+(?:broke|injured|hurt|damaged|lost)\s+my',
            r'I\s+(?:am|was|feel|felt)\s+(?:sick|ill|unwell|not well|injured)',
            r'my\s+(?:illness|sickness|disease|condition|injury|accident|problem)',
            r'I\s+need\s+(?:a|to see)\s+(?:doctor|medical|physician|hospital)',
            r'I\s+(?:had|have)\s+(?:pain|discomfort|symptoms)',
        ]

        # Patterns for family members experiencing events
        family_patterns = {
            'child': r'(?:my|our)\s+(?:child|kid|son|daughter)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'spouse': r'(?:my|our)\s+(?:spouse|husband|wife|partner)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'parent': r'(?:my|our)\s+(?:parent|father|mother|dad|mom)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'family': r'(?:my|our)\s+(?:family member|relative|brother|sister|sibling|uncle|aunt|cousin|nephew|niece)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
        }

        # Check for first person as affected
        for pattern in first_person_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "policyholder (self)"

        # Check for family members as affected
        for family_type, pattern in family_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                return f"{family_type} of policyholder"

        # Check for others as affected
        other_patterns = {
            'friend': r'(?:my|our)\s+friend\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'colleague': r'(?:my|our)\s+(?:colleague|coworker|co-worker)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'travel companion': r'(?:my|our)\s+(?:travel companion|fellow traveler|traveling partner)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
        }

        for other_type, pattern in other_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                return f"{other_type} of policyholder"

        # Default to None if no clear matches
        return None

    def _handle_special_case_personas(
            self,
            question: str,
            found_relationships: List[str]
    ) -> Tuple[Optional[str], Optional[List[str]], Optional[str], Optional[int], Optional[int], Optional[str]]:
        """
        Handle special case persona scenarios, now including affected person.

        Args:
            question: The question to analyze
            found_relationships: List of relationships found in the question

        Returns:
            Tuple of (policy_user, mentioned_people, relationship, total_count, claimant_count, affected_person)
            Returns None for any values that couldn't be determined by special cases
        """
        policy_user = None
        mentioned_people = None
        relationship = None
        total_count = None
        claimant_count = None
        affected_person = None

        # Special case: business partner scenario
        if re.search(r'my\s+(?:business\s+)?partner\s+had', question, re.IGNORECASE):
            if re.search(r'I\s+am\s+traveling\s+alone', question, re.IGNORECASE):
                policy_user = "policyholder (individual traveler)"
                mentioned_people = ["business partner (not traveling/insured)"]
                relationship = "business relationship (but not traveling together)"
                total_count = 2
                claimant_count = 1
                affected_person = "business partner (not a claimant)"
            else:
                policy_user = "policyholder (individual)"
                mentioned_people = ["business partner"]
                relationship = "business relationship"
                total_count = 2
                claimant_count = 1
                affected_person = "business partner"

        # Special case: family member scenarios
        elif re.search(r'\bmy\s+(daughter|son|child|children|wife|husband|family)\b', question, re.IGNORECASE):
            policy_user = "policyholder (parent/spouse)"
            relationship = "family - " + ", ".join(found_relationships)

            # Check if family member is experiencing the event
            family_terms = ['daughter', 'son', 'child', 'children', 'wife', 'husband', 'family']
            for term in family_terms:
                if re.search(r'my\s+' + term + r'\s+(?:is|was|had|got|needs|has been)', question, re.IGNORECASE):
                    affected_person = term
                    break

            # Check if family member is a claimant
            if any(re.search(r'my\s+' + term + r'\s+(?:is|was|had|got|needs)', question, re.IGNORECASE)
                   for term in family_terms):
                claimant_count = 2  # Both policyholder and family member
            else:
                claimant_count = 1  # Just the policyholder

        # Special case: our family scenario
        elif re.search(r'\bour\s+(daughter|son|child|children|family)\b', question, re.IGNORECASE):
            policy_user = "policyholder (joint policy with family)"
            relationship = "family - " + ", ".join(found_relationships)
            claimant_count = 2  # At least two people covered

            # Determine the affected person
            family_terms = ['daughter', 'son', 'child', 'children', 'family']
            for term in family_terms:
                if re.search(r'our\s+' + term + r'\s+(?:is|was|had|got|needs|has been)', question, re.IGNORECASE):
                    affected_person = term
                    break

        # Special case: daughter/we scenario
        elif 'daughter' in question.lower() and 'we' in question.lower():
            policy_user = "parent (policyholder)"
            mentioned_people = ["daughter"]
            relationship = "parent-child"
            total_count = 2
            claimant_count = 2  # Both parent and child are covered

            # Check if daughter is affected
            if re.search(r'(?:my|our)\s+daughter\s+(?:is|was|had|got|needs|has been)', question, re.IGNORECASE):
                affected_person = "daughter of policyholder"

        # Special case: traveling alone but mentioning business partner
        elif re.search(r'I\s+am\s+traveling\s+alone', question, re.IGNORECASE) and 'partner' in question.lower():
            policy_user = "policyholder (individual traveler)"
            mentioned_people = ["business partner (not traveling/insured)"]
            relationship = "business relationship"
            total_count = 2
            claimant_count = 1
            affected_person = "policyholder (self)"

        # Special case: personal item theft or loss
        elif re.search(r'(my|I).*(handbag|purse|bag|phone|document|wallet|passport|luggage)', question, re.IGNORECASE):
            policy_user = "policyholder (individual)"
            mentioned_people = []
            relationship = "none mentioned"
            total_count = 1
            claimant_count = 1
            affected_person = "policyholder (self)"

        # Special case: food poisoning or illness
        elif re.search(r'(I had|I got).*?(sick|ill|poisoning|diarrhea|vomiting|stomach|pain)', question, re.IGNORECASE):
            policy_user = "policyholder (individual)"
            mentioned_people = []
            relationship = "none mentioned"
            total_count = 1
            claimant_count = 1
            affected_person = "policyholder (self)"

        return policy_user, mentioned_people, relationship, total_count, claimant_count, affected_person

    def _llm_persona_extraction(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Extract persona information using LLM-based approach, including who was affected.

        Args:
            question: The insurance query to analyze

        Returns:
            Dictionary with persona information or None if extraction failed
        """
        try:
            # Construct a persona-focused prompt
            persona_prompt = self._create_persona_prompt(question)

            # Use a shorter context window
            outputs = self.pipe(
                persona_prompt,
                max_new_tokens=200,
                do_sample=False,
                num_return_sequences=1,
                return_full_text=False
            )

            # Get the generated text
            generated = outputs[0]["generated_text"]
            logger.debug(f"Persona extraction output: {generated}")

            # Try multiple JSON extraction methods
            llm_result = self._try_all_json_extraction_methods(generated)

            # Return the result if valid
            if llm_result and "personas" in llm_result:
                # Validate and fix values if needed
                llm_result = self._validate_llm_persona_result(llm_result)
                logger.info("Using LLM-extracted persona information")
                return llm_result

            return None

        except Exception as e:
            logger.warning(f"LLM persona extraction failed: {str(e)}")
            return None

    def _create_persona_prompt(self, question: str) -> str:
        """
        Create a prompt for persona extraction, including who experienced the event.

        Args:
            question: The insurance query to analyze

        Returns:
            Formatted prompt for the LLM
        """
        return f"""
        Analyze this insurance query and extract ONLY information about the people involved:

        Query: "{question}"

        1. Who is making the insurance claim (the primary policy user/policyholder)?
        2. Who actually experienced the event, accident, or health issue that is the subject of the claim?
        3. Who else is mentioned but NOT making the claim?
        4. Total number of people in the scenario?
        5. Number of people actually covered by the insurance or making claims?
        6. What is their relationship?

        IMPORTANT: Distinguish between:
        - The POLICYHOLDER (who owns the policy and is making the claim)
        - The AFFECTED PERSON (who actually experienced the event, accident, or health issue)
        - OTHER PEOPLE who are merely mentioned but not directly involved

        IMPORTANT: The answer must ONLY be valid JSON in this EXACT format:
        {{
          "personas": {{
            "policy_user": "Who is making the claim/policyholder",
            "affected_person": "Who experienced the event/accident/illness",
            "mentioned_people": "Who else is mentioned but not a claimant",
            "total_count": number of ALL people mentioned,
            "claimant_count": number of people actually claiming/using the insurance,
            "relationship": "Relationship between people"
          }}
        }}
        """

    def _validate_llm_persona_result(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix persona extraction results.

        Args:
            llm_result: The LLM extraction result to validate

        Returns:
            Validated and corrected result
        """
        # Ensure the personas key exists
        if "personas" not in llm_result:
            llm_result["personas"] = {}

        # VALIDATION: Ensure the counts make sense
        if "total_count" in llm_result["personas"]:
            # Get the count from LLM
            llm_count = llm_result["personas"]["total_count"]
            # Ensure it's reasonable (1-5) for most queries
            if not isinstance(llm_count, int) or llm_count < 1 or llm_count > 5:
                # If unreasonable, use a default count
                llm_result["personas"]["total_count"] = 1
                logger.warning(f"Adjusted implausible LLM total_count from {llm_count} to 1")

        # Make sure claimant_count is not greater than total_count
        if "claimant_count" in llm_result["personas"] and "total_count" in llm_result["personas"]:
            if llm_result["personas"]["claimant_count"] > llm_result["personas"]["total_count"]:
                llm_result["personas"]["claimant_count"] = llm_result["personas"]["total_count"]
                logger.warning("Adjusted claimant_count to not exceed total_count")

        # Ensure claimant_count exists
        if "claimant_count" not in llm_result["personas"]:
            llm_result["personas"]["claimant_count"] = 1
            logger.warning(f"Added missing claimant_count: 1")

        # If affected_person is missing, use policy_user as a fallback
        if "affected_person" not in llm_result["personas"] and "policy_user" in llm_result["personas"]:
            llm_result["personas"]["affected_person"] = f"{llm_result['personas']['policy_user']} (inferred)"
            logger.warning("Added missing affected_person, inferred from policy_user")

        # Ensure policy_user exists
        if "policy_user" not in llm_result["personas"]:
            llm_result["personas"]["policy_user"] = "policyholder (individual)"
            logger.warning("Added missing policy_user with default value")

        return llm_result

    def extract_personas(self, question: str) -> Dict[str, Any]:
        """
        Extracts information about personas involved in the query.
        Distinguishes between the actual policy user and other people mentioned.

        Args:
            question: The insurance query to analyze

        Returns:
            Dictionary containing persona information
        """
        logger.info(f"Extracting personas from question: {question}")

        # First try rule-based extraction
        rule_based_result = self._rule_based_persona_extraction(question)

        # Try LLM-based approach as a backup
        llm_result = self._llm_persona_extraction(question)

        # Use LLM result if available, otherwise use rule-based
        if llm_result:
            return llm_result
        else:
            logger.info("Using rule-based persona extraction")
            return rule_based_result

    def _format_persona_text(self, personas_info: Dict[str, Any]) -> str:
        """
        Format persona information for inclusion in the prompt, including the affected person.

        Args:
            personas_info: Dictionary with persona information

        Returns:
            Formatted persona text
        """
        persona_text = "IMPORTANT INFORMATION FROM THE QUESTION:\n"

        try:
            # Extract persona details
            policy_user = personas_info["personas"]["policy_user"]
            affected_person = personas_info["personas"].get("affected_person", f"{policy_user} (inferred)")
            mentioned_people = personas_info["personas"]["mentioned_people"]
            total_count = personas_info["personas"]["total_count"]
            claimant_count = personas_info["personas"]["claimant_count"]
            relationship = personas_info["personas"]["relationship"]

            # Build detailed persona text
            persona_text += f"- Primary policy user/claimant: {policy_user}\n"
            persona_text += f"- Person who experienced the event/accident: {affected_person}\n"
            persona_text += f"- Other people mentioned (not policy users): {mentioned_people}\n"
            persona_text += f"- Total number of people mentioned: {total_count}\n"
            persona_text += f"- Number of people actually claiming/covered: {claimant_count}\n"
            persona_text += f"- Relationships: {relationship}\n\n"
            persona_text += "When determining coverage, focus primarily on who experienced the event and their relationship to the policy user.\n"
            persona_text += "Take into account whether the policy covers just the policyholder or additional people based on their relationship.\n"
        except (KeyError, TypeError):
            persona_text += "Unable to extract detailed persona information. Consider who is actually making the claim vs. who experienced the event vs. who is just mentioned.\n"
        return persona_text

    def _format_context_text(self, context_files: List[str]) -> str:
        """
        Format context information from policy files.

        Args:
            context_files: List of context file contents

        Returns:
            Formatted context text
        """
        if not context_files or len(context_files) == 0:
            return ""

        context_text = "\n\nRelevant policy information:\n\n"
        context_text += ("""
                        WARNING: Some of the following context may contain "SECTION DEFINITIONS" of terms or general "
                        "information that does not directly indicate coverage. Please carefully distinguish "
                        "between definitions and actual coverage provisions.\n\n
                        """)

        for i, ctx in enumerate(context_files, 1):
            context_text += f"Policy context {i}:\n{ctx}\n\n"

        return context_text

    def _build_prompt(self, question: str, context_text: str, persona_text: str) -> str:
        """
        Build the full prompt for the model.

        Args:
            question: The question to answer
            context_text: Formatted context information
            persona_text: Formatted persona information

        Returns:
            Complete prompt for the model
        """
        json_solution = "Then the json Solution is:\n\n"
        full_prompt = f"{self.base_prompt}\n\n{context_text}\n\nQuestion: {question}\n\n{persona_text}\n\n{self.json_format}\n\n{json_solution}"
        logger.debug(f"Full prompt: {full_prompt}")
        return full_prompt

    def _generate_model_output(self, prompt: str) -> Dict[str, Any]:
        """
        Generate output from the model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Parsed JSON response or error message
        """
        try:
            # Generate text with appropriate parameters for JSON output
            outputs = self.pipe(
                prompt,
                max_new_tokens=1024,
                do_sample=False,
                num_return_sequences=1,
                return_full_text=False
            )

            # Get the generated text
            generated = outputs[0]["generated_text"]
            logger.debug(f"Model output: {generated}")

            # Try to extract JSON from the output
            extracted_json = self._try_all_json_extraction_methods(generated)

            if extracted_json:
                return extracted_json

            # If all extraction methods fail, return a formatted error
            logger.error("All JSON extraction methods failed")
            return self._create_error_response("Failed to extract valid JSON from model output")

        except Exception as e:
            error_msg = f"Error during model inference: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)

    def query(self, question: str, context_files: List[str]) -> Dict[str, Any]:
        """
        Process a query and return the model's response.

        Args:
            question: The question to answer
            context_files: List of relevant policy files

        Returns:
            Dictionary containing the answer
        """
        logger.info(f"Querying model with question: {question}")

        # First, extract personas from the question
        personas_info = self.extract_personas(question)
        logger.info(f"Extracted personas: {personas_info}")

        # Format the persona information
        persona_text = self._format_persona_text(personas_info)

        # Format context information
        context_text = self._format_context_text(context_files)

        # Build the full prompt
        full_prompt = self._build_prompt(question, context_text, persona_text)

        # Generate and process the model output
        return self._generate_model_output(full_prompt)

