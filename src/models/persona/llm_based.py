# src/models/persona/llm_based.py

"""
LLM-based persona extraction.
Uses large language models to extract persona information.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMBasedExtractor:
    """
    LLM-based extractor for persona information.
    Uses language models to extract information.
    """

    def __init__(self, pipe):
        """
        Initialize the LLM-based extractor.

        Args:
            pipe: HuggingFace pipeline for inference
        """
        self.pipe = pipe

    def extract(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Extract persona information using LLM-based approach.

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

            # Try to extract JSON from the output
            from models.json_utils.extractors import JSONExtractor
            json_extractor = JSONExtractor()
            llm_result = json_extractor.extract_json(generated)

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
        Create a prompt for persona extraction with enhanced focus on affected person
        and their location during the event.

        Args:
            question: The insurance query to analyze

        Returns:
            Formatted prompt for the LLM
        """
        return f"""
        Analyze this insurance query and extract ONLY information about the people involved and their locations:

        Query: "{question}"

        1. Who is making the insurance claim (the primary policy user/policyholder)?
        2. Who actually experienced the event, accident, health issue, loss, or damage that is the subject of the claim?
        3. WHERE was the affected person when the event occurred? (e.g., at home, abroad, at the airport, in a hotel)
        4. What is the relationship between the policyholder and the affected person?
        5. Is the affected person covered by the policy? (Usually yes if they are the policyholder, spouse, or dependent)
        6. Who else is mentioned but NOT making the claim or experiencing the event?
        7. Total number of people in the scenario?
        8. Number of people actually covered by the insurance or making claims?

        IMPORTANT: Carefully distinguish between these roles:
        - The POLICYHOLDER (who owns the policy and usually makes the claim)
        - The AFFECTED PERSON (who actually experienced the event, accident, illness, loss, or damage)
        - OTHER PEOPLE who are merely mentioned but not directly involved

        Pay special attention to LOCATION information, which is often critical for insurance claims:
        - Was the event domestic or international?
        - Was the person in transit (airport, train station, etc.)?
        - Was the person at a specific venue (hotel, resort, hospital)?
        - Was the person in their home country or abroad?

        Examples to consider:
        - "At the airport my baggage was lost" → Location: airport
        - "During our vacation in Spain my daughter got sick" → Location: abroad (Spain)
        - "My house was damaged by a storm" → Location: home/domestic
        - "While staying at the hotel, my wallet was stolen" → Location: hotel

        IMPORTANT: The answer must ONLY be valid JSON in this EXACT format:
        {{
          "personas": {{
            "policy_user": "Who is making the claim/policyholder",
            "affected_person": "Who experienced the event/accident/illness/loss/damage",
            "location": "Where the affected person was when the event occurred",
            "is_abroad": true or false (whether the event occurred outside home country),
            "relationship_to_policyholder": "Relationship between affected person and policyholder (self, spouse, child, etc.)",
            "is_affected_covered": true or false (whether the affected person is likely covered),
            "mentioned_people": "Who else is mentioned but not a claimant or affected",
            "total_count": number of ALL people mentioned,
            "claimant_count": number of people actually claiming/using the insurance,
            "relationship": "Relationships between all mentioned people"
          }}
        }}
        """

    def _validate_llm_persona_result(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix persona extraction results, including location information.

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

        # Ensure policy_user exists
        if "policy_user" not in llm_result["personas"]:
            llm_result["personas"]["policy_user"] = "policyholder (individual)"
            logger.warning("Added missing policy_user with default value")

        # If affected_person is missing, use policy_user as a fallback
        if "affected_person" not in llm_result["personas"] and "policy_user" in llm_result["personas"]:
            llm_result["personas"]["affected_person"] = f"{llm_result['personas']['policy_user']} (inferred)"
            logger.warning("Added missing affected_person, inferred from policy_user")

        # If location is missing, infer from context or set to unknown
        if "location" not in llm_result["personas"]:
            # Try to infer from other fields
            affected_person_desc = llm_result["personas"].get("affected_person", "").lower()
            context = affected_person_desc + " " + llm_result["personas"].get("relationship", "").lower()

            if "airport" in context:
                llm_result["personas"]["location"] = "airport"
            elif "hotel" in context or "resort" in context:
                llm_result["personas"]["location"] = "hotel/resort"
            elif "hospital" in context or "clinic" in context:
                llm_result["personas"]["location"] = "hospital/medical facility"
            elif "home" in context or "house" in context:
                llm_result["personas"]["location"] = "home"
            elif any(place in context for place in ["abroad", "foreign", "overseas", "international"]):
                llm_result["personas"]["location"] = "abroad (unspecified location)"
            else:
                llm_result["personas"]["location"] = "unspecified location"

            logger.warning(f"Added missing location: {llm_result['personas']['location']}")

        # If is_abroad is missing, infer from location
        if "is_abroad" not in llm_result["personas"]:
            location = llm_result["personas"].get("location", "").lower()
            is_abroad = any(term in location for term in ["abroad", "foreign", "overseas", "international"])

            # Also check for specific country or region names that would indicate abroad
            country_indicators = ["europe", "asia", "america", "africa", "australia"]
            if any(country in location for country in country_indicators):
                is_abroad = True

            llm_result["personas"]["is_abroad"] = is_abroad
            logger.warning(f"Added missing is_abroad: {is_abroad}")

        # Add relationship_to_policyholder if missing
        if "relationship_to_policyholder" not in llm_result["personas"]:
            # If affected person is the same as policy user, relationship is "self"
            if (llm_result["personas"].get("affected_person", "").lower() ==
                    llm_result["personas"].get("policy_user", "").lower() or
                    "self" in llm_result["personas"].get("affected_person", "").lower()):
                llm_result["personas"]["relationship_to_policyholder"] = "self"
            else:
                # Otherwise infer from context
                affected = llm_result["personas"].get("affected_person", "").lower()
                if any(term in affected for term in ["spouse", "wife", "husband", "partner"]):
                    llm_result["personas"]["relationship_to_policyholder"] = "spouse/partner"
                elif any(term in affected for term in ["child", "son", "daughter"]):
                    llm_result["personas"]["relationship_to_policyholder"] = "child/dependent"
                elif any(term in affected for term in ["parent", "father", "mother"]):
                    llm_result["personas"]["relationship_to_policyholder"] = "parent"
                else:
                    llm_result["personas"]["relationship_to_policyholder"] = "unknown (inferred)"
            logger.warning(
                f"Added missing relationship_to_policyholder: {llm_result['personas']['relationship_to_policyholder']}")

        # Add is_affected_covered if missing
        if "is_affected_covered" not in llm_result["personas"]:
            # Default assumption based on relationship
            rel = llm_result["personas"].get("relationship_to_policyholder", "").lower()
            if any(r in rel for r in ["self", "spouse", "partner", "child", "dependent"]):
                llm_result["personas"]["is_affected_covered"] = True
            else:
                llm_result["personas"]["is_affected_covered"] = False
            logger.warning(f"Added missing is_affected_covered: {llm_result['personas']['is_affected_covered']}")

        return llm_result
