# =============================================================================
# Enhanced OpenRouter Model with Rate Limiting and Complete Prompt/Output Logging
# =============================================================================

"""
OpenRouter model client implementation for insurance policy analysis.
Uses OpenRouter's API to access Qwen and other models through a unified endpoint.
Enhanced with comprehensive logging and intelligent rate limiting.
"""
import logging
import json
import time
from typing import List, Dict, Any
from datetime import datetime

from openai import OpenAI
import httpx

from config import OPENROUTER_API_KEY, OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME
from models.base import BaseModelClient
from models.json_utils.extractors import JSONExtractor

logger = logging.getLogger(__name__)


class OpenRouterModelClient(BaseModelClient):
    """
    Client for using OpenRouter API to access various models including Qwen.
    Handles model initialization, inference, and result formatting.
    Enhanced with comprehensive logging and intelligent rate limiting.
    """

    def __init__(self, model_name: str, sys_prompt: str, requests_per_minute: int = 12):
        """
        Initialize an OpenRouter model client.

        Args:
            model_name: Name of the model to use (e.g., "qwen/qwen-2.5-72b-instruct")
            sys_prompt: System prompt for the model
            requests_per_minute: Maximum requests per minute (default: 12, conservative)
        """
        logger.info(f"Loading OpenRouter model: {model_name}")
        logger.info(f"Rate limit: {requests_per_minute} requests per minute")

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

        # Rate limiting
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute  # Minimum delay between requests
        self.last_request_time = 0
        self.rate_limit_reset_time = None
        self.remaining_requests = None

        logger.info(f"OpenRouter client initialized successfully for model: {model_name}")
        logger.info(f"Rate limiting: {self.min_delay:.2f}s minimum delay between requests")

    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()

        # Check if we need to wait for rate limit reset
        if self.rate_limit_reset_time and current_time < self.rate_limit_reset_time:
            wait_time = self.rate_limit_reset_time - current_time
            logger.warning(f"Rate limit exceeded. Waiting {wait_time:.1f}s for reset...")
            time.sleep(wait_time)
            # Reset tracking after waiting
            self.rate_limit_reset_time = None
            self.remaining_requests = None

        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            logger.info(f"Rate limiting: waiting {sleep_time:.2f}s before next request")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _handle_rate_limit_headers(self, response_headers):
        """Extract and handle rate limit information from response headers."""
        try:
            if 'x-ratelimit-remaining' in response_headers:
                self.remaining_requests = int(response_headers['x-ratelimit-remaining'])
                logger.debug(f"Remaining requests: {self.remaining_requests}")

            if 'x-ratelimit-reset' in response_headers:
                reset_timestamp = int(response_headers['x-ratelimit-reset']) / 1000  # Convert from ms
                self.rate_limit_reset_time = reset_timestamp
                reset_time = datetime.fromtimestamp(reset_timestamp)
                logger.debug(f"Rate limit resets at: {reset_time}")

            # If we're running low on requests, slow down
            if self.remaining_requests is not None and self.remaining_requests < 3:
                logger.warning(f"Low on requests ({self.remaining_requests} remaining). Increasing delay.")
                self.min_delay = max(self.min_delay * 2, 5.0)  # At least 5s delay

        except (ValueError, TypeError) as e:
            logger.debug(f"Error parsing rate limit headers: {e}")

    def _make_api_call_with_retry(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """
        Make API call with intelligent retry logic and rate limiting.
        """
        for attempt in range(max_retries):
            try:
                # Wait if necessary for rate limiting
                self._wait_for_rate_limit()

                # Create extra headers for OpenRouter
                extra_headers = {}
                if OPENROUTER_SITE_URL:
                    extra_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
                if OPENROUTER_SITE_NAME:
                    extra_headers["X-Title"] = OPENROUTER_SITE_NAME

                logger.info(f"Making API call to model: {self.model_name} (attempt {attempt + 1})")
                logger.debug(f"Extra headers: {extra_headers}")

                response = self.client.chat.completions.create(
                    extra_headers=extra_headers,
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.1,
                    top_p=0.9
                )

                # Success! Handle rate limit headers from response
                if hasattr(response, 'response') and hasattr(response.response, 'headers'):
                    self._handle_rate_limit_headers(response.response.headers)

                generated = response.choices[0].message.content
                logger.info("API call successful")

                # Log token usage if available
                if hasattr(response, 'usage'):
                    logger.info(f"Token usage: {response.usage}")

                return generated

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded on attempt {attempt + 1}")

                    # Parse rate limit headers from error response
                    self._handle_rate_limit_headers(e.response.headers)

                    if attempt < max_retries - 1:
                        # Calculate wait time with exponential backoff
                        base_wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds

                        # If we have reset time, use it
                        if self.rate_limit_reset_time:
                            wait_time = max(base_wait, self.rate_limit_reset_time - time.time())
                        else:
                            wait_time = base_wait

                        logger.info(f"Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Max retries reached for rate limit")
                        raise

                else:
                    # Other HTTP error
                    logger.error(f"HTTP error {e.response.status_code}: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        raise Exception("Max retries reached")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a formatted error response."""
        return {
            "answer": {
                "eligibility": "Error",
                "outcome_justification": error_message,  # NEW field name
                "payment_justification": None  # NEW field name
            }
        }

    def _create_default_no_conditions_response(self) -> Dict[str, Any]:
        """Create a default response when JSON extraction fails."""
        default_response = {
            "answer": {
                "eligibility": "",
                "outcome_justification": "",  # NEW field name
                "payment_justification": ""  # NEW field name
            }
        }
        logger.warning(f"Using default response: {default_response}")
        return default_response

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
        logger.debug(f"Context preview: {context_text}...")

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

            messages = [
                {"role": "system", "content": "You extract persona information from insurance queries."},
                {"role": "user", "content": persona_prompt}
            ]

            persona_response = self._make_api_call_with_retry(messages)

            logger.debug(f"=== PERSONA EXTRACTION RESPONSE ===")
            logger.debug(f"Raw persona response: {persona_response}")

            extracted_json = self.json_extractor.extract_json(persona_response)

            if extracted_json and "personas" in extracted_json:
                try:
                    from models.persona.formatters import format_persona_text
                    formatted_persona = format_persona_text(extracted_json)
                    logger.info(f"Successfully extracted persona: {formatted_persona}")
                    return formatted_persona
                except ImportError:
                    logger.warning("Persona formatters not available, using raw JSON")
                    return json.dumps(extracted_json, indent=2)
            else:
                logger.warning("Failed to extract valid persona JSON from API response")
                return ""

        except Exception as e:
            logger.warning(f"API persona extraction failed: {str(e)}")
            return ""

    def _build_messages(self, question: str, context_text: str, persona_text: str) -> List[Dict[str, str]]:
        """Build the message array for the OpenRouter API call."""

        # Format the system prompt with actual values
        formatted_system_prompt = self.base_prompt.replace(
            "{RETRIEVED_POLICY_TEXT}", context_text
        ).replace(
            "{USER_QUESTION}", question
        )

        # If using persona, include it in the user message
        if persona_text:
            user_content = f"Additional context:\n{persona_text}"
        else:
            user_content = "Please analyze the policy context and question provided above."

        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": user_content}
        ]

        return messages

    def _generate_model_output(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate output from the OpenRouter API with rate limiting."""
        try:
            generated = self._make_api_call_with_retry(messages)

            # Log the complete raw output
            logger.info("=== MODEL OUTPUT ===")
            logger.info(f"Raw model response: {generated}")

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

        logger.info("=== PROMPT FINAL ===")
        logger.info(f"Final prompt: {json.dumps(messages, indent=2)}")
        result = self._generate_model_output(messages)


        logger.info("=== FINAL RESULT ===")
        logger.info(f"Final result: {json.dumps(result, indent=2)}")
        logger.info("=" * 60)

        return result

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status information."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        status = {
            "requests_per_minute": self.requests_per_minute,
            "min_delay": self.min_delay,
            "last_request_time": self.last_request_time,
            "time_since_last_request": time_since_last,
            "remaining_requests": self.remaining_requests,
            "rate_limit_reset_time": self.rate_limit_reset_time,
            "ready_for_next_request": time_since_last >= self.min_delay
        }

        return status


# =============================================================================
# LOGGING CONFIGURATION HELPER
# =============================================================================

def setup_detailed_logging(log_level: str = "INFO", log_file: str = "openrouter_model.log"):
    """
    Setup logging configuration to see all the prompt and output details.
    Call this at the start of your application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Configure logging
    handlers = [logging.StreamHandler()]  # Console output

    if log_file:
        handlers.append(logging.FileHandler(log_file))  # File output

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s',
        handlers=handlers
    )

    # Reduce noise from external libraries
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    logger.info(f"Logging configured with level: {log_level}")
    if log_file:
        logger.info(f"Log file: {log_file}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_wait_time_from_reset(reset_timestamp_ms: int) -> float:
    """
    Calculate how long to wait based on OpenRouter's rate limit reset timestamp.

    Args:
        reset_timestamp_ms: Reset timestamp in milliseconds

    Returns:
        Wait time in seconds
    """
    current_time = time.time()
    reset_time = reset_timestamp_ms / 1000.0  # Convert to seconds
    wait_time = max(0, reset_time - current_time)
    return wait_time


def get_recommended_delay(model_name: str) -> float:
    """
    Get recommended delay between requests based on model type.

    Args:
        model_name: OpenRouter model name

    Returns:
        Recommended delay in seconds
    """
    if "free" in model_name.lower():
        return 6.0  # 10 requests per minute for free models
    elif "claude" in model_name.lower():
        return 3.0  # 20 requests per minute for Claude models
    elif "gpt" in model_name.lower():
        return 2.0  # 30 requests per minute for GPT models
    else:
        return 4.0  # 15 requests per minute as default

