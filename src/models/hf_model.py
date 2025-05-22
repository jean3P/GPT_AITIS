# src/models/hf_model.py

"""
HuggingFace model client implementation for insurance policy analysis.
"""
import logging
import os
from typing import List, Dict, Any

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import HUGGINGFACE_TOKEN
from models.base import BaseModelClient
from models.persona.extractor import PersonaExtractor
from models.json_utils.extractors import JSONExtractor

# Login with your token
login(token=HUGGINGFACE_TOKEN)

logger = logging.getLogger(__name__)


class HuggingFaceModelClient(BaseModelClient):
    """
    Client for using HuggingFace models to analyze insurance policies.
    Handles model initialization, inference, and result formatting.
    """

    def __init__(self, model_name: str, sys_prompt: str):
        """
        Initialize a HuggingFace model client.

        Args:
            model_name: Name of the model to use
            sys_prompt: System prompt for the model
        """
        logger.info(f"Loading HuggingFace model: {model_name}")
        self._log_cache_locations()
        model_name = self._check_model_in_scratch(model_name)

        self.pipe = self._initialize_pipeline(model_name)
        self.base_prompt = sys_prompt.strip()
        self.json_format = self._get_json_format_template()

        # Initialize extractors
        self.persona_extractor = PersonaExtractor(self.pipe)
        self.json_extractor = JSONExtractor()

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
                "eligibility": "Yes | No - Unrelated event | No - condition(s) not met",
                "eligibility_policy": "Quoted text from policy",
                "amount_policy": "Amount like '1000 CHF' or null"
              }
            }
            """

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
                        "between definitions, exclusions and actual coverage provisions.\n\n
                        """)

        for i, ctx in enumerate(context_files, 1):
            context_text += f"Policy context {i}:\n{ctx.strip()}\n\n"

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
        prompt_final = "\nThen the json Solution is:\n\n"

        # Determine JSON format based on the type of prompt being used
        json_format = self._get_appropriate_json_format()

        if persona_text:
            full_prompt = f"{self.base_prompt}\n\n{context_text}\n\nQuestion: {question}\n\n{persona_text}\n\n{json_format}\n\n{prompt_final}"
            logger.debug(f"Full prompt: {full_prompt}")
        else:
            full_prompt = f"{self.base_prompt}\n\n{context_text}\n\nQuestion: {question}\n\n{json_format}\n\n{prompt_final}"
            logger.debug(f"Full prompt: {full_prompt}")
        return full_prompt

    def _get_appropriate_json_format(self) -> str:
        """
        Get the appropriate JSON format based on the current prompt type.

        Returns:
            The appropriate JSON format string
        """
        # Check if this is a relevance filtering prompt
        if "INSURANCE-POLICY RELEVANCE FILTER" in self.base_prompt:
            return """
                Return exactly this JSON:
                {
                  "is_relevant": true/false,
                  "reason": "Brief explanation (≤ 25 words)"
                }
                """
        else:
            # Default to insurance analysis format
            return self.json_format


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
                max_new_tokens=2048,
                do_sample=False,
                num_return_sequences=1,
                return_full_text=False
            )

            # Get the generated text
            generated = outputs[0]["generated_text"]
            logger.debug(f"Model output: {generated}")

            # Try to extract JSON from the output
            extracted_json = self.json_extractor.extract_json(generated)

            if extracted_json:
                return extracted_json

            # If all extraction methods fail, return a formatted error
            logger.error("All JSON extraction methods failed")
            return self._create_error_response("Failed to extract valid JSON from model output")

        except Exception as e:
            error_msg = f"Error during model inference: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)

    def query(self, question: str, context_files: List[str], use_persona: bool = False) -> Dict[str, Any]:
        """
        Process a query and return the model's response.

        Args:
            question: The question to answer
            context_files: List of relevant policy files
            use_persona: Whether to extract and use persona information (default: False)

        Returns:
            Dictionary containing the answer
        """
        logger.info(f"Querying model with question: {question}")

        # Extract personas from the question if use_persona is True
        persona_text = ""
        if use_persona:
            personas_info = self.persona_extractor.extract_personas(question)
            logger.info(f"Extracted personas: {personas_info}")
            # Format the persona information
            persona_text = self.persona_extractor.format_persona_text(personas_info)
            logger.info("Using persona information in prompt")
        else:
            logger.info("Skipping persona extraction (--persona flag not used)")

        # Format context information
        context_text = self._format_context_text(context_files)

        # Build the full prompt
        full_prompt = self._build_prompt(question, context_text, persona_text)

        # Generate and process the model output
        return self._generate_model_output(full_prompt)
