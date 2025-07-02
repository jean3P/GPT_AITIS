# =============================================================================
# Enhanced OpenRouter Model with Complete Prompt and Output Logging
# =============================================================================

"""
OpenRouter model client implementation for insurance policy analysis.
Uses OpenRouter's API to access Qwen and other models through a unified endpoint.
Enhanced with comprehensive logging to show prompts and outputs.
"""
import logging
import os
import json
from typing import List, Dict, Any

from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME
from models.base import BaseModelClient
from models.json_utils.extractors import JSONExtractor

logger = logging.getLogger(__name__)


class OpenRouterModelClient(BaseModelClient):
    """
    Client for using OpenRouter API to access various models including Qwen.
    Handles model initialization, inference, and result formatting.
    Enhanced with comprehensive logging.
    """

    def __init__(self, model_name: str, sys_prompt: str):
        """
        Initialize an OpenRouter model client.

        Args:
            model_name: Name of the model to use (e.g., "qwen/qwen-2.5-72b-instruct")
            sys_prompt: System prompt for the model
        """
        logger.info(f"Loading OpenRouter model: {model_name}")

        # Log the system prompt being used
        logger.info(f"System prompt: {sys_prompt}")

        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        self.model_name = model_name
        self.base_prompt = sys_prompt.strip()

        # Initialize JSON extractor
        self.json_extractor = JSONExtractor()

        logger.info(f"OpenRouter client initialized successfully for model: {model_name}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a formatted error response."""
        return {
            "answer": {
                "eligibility": "Error",
                "eligibility_policy": error_message,
                "amount_policy": None
            }
        }

    def _format_context_text(self, context_files: List[str]) -> str:
        """Format context information from policy files."""
        if not context_files or len(context_files) == 0:
            return ""

        context_text = "\n\nRelevant policy information:\n\n"
        context_text += """
                        WARNING: Some of the following context may contain "SECTION DEFINITIONS" of terms or general 
                        information that does not directly indicate coverage. Please carefully distinguish 
                        between definitions, exclusions and actual coverage provisions.\n\n
                        """

        for i, ctx in enumerate(context_files, 1):
            context_text += f"Policy context {i}:\n{ctx.strip()}\n\n"

        # Log the formatted context
        logger.debug(f"Formatted context text (length: {len(context_text)} chars)")
        logger.debug(f"Context preview: {context_text[:500]}...")

        return context_text

    def _extract_persona_via_api(self, question: str) -> str:
        """Extract persona information using the API model."""
        try:
            persona_prompt = f"""
            Analyze this insurance query and extract ONLY information about the people involved and their locations:

            Query: "{question}"

            1. Who is making the insurance claim (the primary policy user/policyholder)?
            2. Who actually experienced the event, accident, health issue, loss, or damage?
            3. WHERE was the affected person when the event occurred?
            4. What is the relationship between the policyholder and the affected person?
            5. Is the affected person covered by the policy?
            6. Who else is mentioned but NOT making the claim?
            7. Total number of people in the scenario?
            8. Number of people actually covered by the insurance?

            Return JSON format:
            {{
              "personas": {{
                "policy_user": "Who is making the claim/policyholder",
                "affected_person": "Who experienced the event",
                "location": "Where the event occurred",
                "is_abroad": true or false,
                "relationship_to_policyholder": "Relationship",
                "is_affected_covered": true or false,
                "mentioned_people": "Others mentioned",
                "total_count": number,
                "claimant_count": number,
                "relationship": "Relationships between people"
              }}
            }}
            """

            # Log the persona extraction prompt
            logger.debug("=== PERSONA EXTRACTION PROMPT ===")
            logger.debug(persona_prompt)

            # Create extra headers for OpenRouter
            extra_headers = {}
            if OPENROUTER_SITE_URL:
                extra_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
            if OPENROUTER_SITE_NAME:
                extra_headers["X-Title"] = OPENROUTER_SITE_NAME

            response = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You extract persona information from insurance queries."},
                    {"role": "user", "content": persona_prompt}
                ],
                max_tokens=512,
                temperature=0.1
            )

            persona_response = response.choices[0].message.content
            logger.debug(f"=== PERSONA EXTRACTION RESPONSE ===")
            logger.debug(f"Raw persona response: {persona_response}")

            extracted_json = self.json_extractor.extract_json(persona_response)

            if extracted_json and "personas" in extracted_json:
                from models.persona.formatters import format_persona_text
                formatted_persona = format_persona_text(extracted_json)
                logger.info(f"Successfully extracted persona: {formatted_persona}")
                return formatted_persona
            else:
                logger.warning("Failed to extract valid persona JSON from API response")
                return ""

        except Exception as e:
            logger.warning(f"API persona extraction failed: {str(e)}")
            return ""

    def _build_messages(self, question: str, context_text: str, persona_text: str) -> List[Dict[str, str]]:
        """Build the message array for the OpenRouter API call."""
        if persona_text:
            user_content = f"{context_text}\n\nQuestion: {question}\n\n{persona_text}"
        else:
            user_content = f"{context_text}\n\nQuestion: {question}"

        messages = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": user_content}
        ]

        # Log the complete messages being sent to the API
        logger.info("=== MAIN QUERY MESSAGES ===")
        logger.info(f"System message: {messages[0]['content']}")
        logger.info(f"User message length: {len(messages[1]['content'])} characters")
        logger.debug(f"Full user message: {messages[1]['content']}")

        return messages

    def _generate_model_output(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate output from the OpenRouter API."""
        try:
            # Create extra headers for OpenRouter
            extra_headers = {}
            if OPENROUTER_SITE_URL:
                extra_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
            if OPENROUTER_SITE_NAME:
                extra_headers["X-Title"] = OPENROUTER_SITE_NAME

            logger.info(f"Making API call to model: {self.model_name}")
            logger.debug(f"Extra headers: {extra_headers}")

            response = self.client.chat.completions.create(
                extra_headers=extra_headers,
                model=self.model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.1,
                top_p=0.9
            )

            generated = response.choices[0].message.content

            # Log the complete raw output
            logger.info("=== MODEL OUTPUT ===")
            logger.info(f"Raw model response: {generated}")

            # Log token usage if available
            if hasattr(response, 'usage'):
                logger.info(f"Token usage: {response.usage}")

            extracted_json = self.json_extractor.extract_json(generated)

            if extracted_json:
                logger.info("=== EXTRACTED JSON ===")
                logger.info(f"Parsed JSON: {json.dumps(extracted_json, indent=2)}")
                return extracted_json

            logger.error("JSON extraction failed - returning default response")
            logger.error(f"Failed to extract JSON from: {generated}")
            return self._create_default_no_conditions_response()

        except Exception as e:
            error_msg = f"Error during OpenRouter API call: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)

    def _create_default_no_conditions_response(self) -> Dict[str, Any]:
        """Create a default response when JSON extraction fails."""
        default_response = {
            "answer": {
                "eligibility": "",
                "eligibility_policy": "",
                "amount_policy": ""
            }
        }
        logger.warning(f"Using default response: {default_response}")
        return default_response

    def query(self, question: str, context_files: List[str], use_persona: bool = False) -> Dict[str, Any]:
        """Process a query and return the model's response."""
        logger.info("=" * 60)
        logger.info(f"NEW QUERY STARTED")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Question: {question}")
        logger.info(f"Context files count: {len(context_files) if context_files else 0}")
        logger.info(f"Use persona: {use_persona}")
        logger.info("=" * 60)

        persona_text = ""
        if use_persona:
            logger.info("Starting persona extraction...")
            persona_text = self._extract_persona_via_api(question)
            if persona_text:
                logger.info("Successfully extracted persona information")
                logger.debug(f"Persona text: {persona_text}")
            else:
                logger.info("No persona information extracted")
        else:
            logger.info("Skipping persona extraction")

        context_text = self._format_context_text(context_files)
        messages = self._build_messages(question, context_text, persona_text)

        result = self._generate_model_output(messages)

        logger.info("=== FINAL RESULT ===")
        logger.info(f"Final result: {json.dumps(result, indent=2)}")
        logger.info("=" * 60)

        return result


# =============================================================================
# LOGGING CONFIGURATION HELPER
# =============================================================================

def setup_detailed_logging():
    """
    Setup logging configuration to see all the prompt and output details.
    Call this at the start of your application.
    """
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to see all logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('openrouter_model.log')  # File output
        ]
    )

    # You can also set specific loggers to different levels
    logging.getLogger('openai').setLevel(logging.WARNING)  # Reduce OpenAI client noise
    logging.getLogger('httpx').setLevel(logging.WARNING)  # Reduce HTTP client noise


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Setup logging first
    setup_detailed_logging()

    # Example usage
    client = OpenRouterModelClient(
        model_name="qwen/qwen-2.5-72b-instruct",
        sys_prompt="You are an insurance policy analyzer..."
    )

    result = client.query(
        question="Is my trip to Spain covered?",
        context_files=["Policy section about international travel..."],
        use_persona=True
    )
