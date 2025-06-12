# src/models/hf_model.py

"""
HuggingFace model client implementation for insurance policy analysis.
"""
import logging
import os
from typing import List, Dict, Any

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import HUGGINGFACE_TOKEN, get_local_model_path, get_model_config
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
        """Check if the model exists locally and return appropriate path."""
        return get_local_model_path(model_name)

    def _initialize_pipeline(self, model_name: str):
        """Initialize the HuggingFace pipeline with the specified model."""
        try:
            logger.info(f"Initializing pipeline with model: {model_name}")
            model_config = get_model_config(model_name)

            tokenizer = self._load_tokenizer(model_name)
            model = self._load_model(model_name, model_config)

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

    def _load_model(self, model_name: str, model_config: dict):
        """Load the model with configuration from config.py."""
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_config["torch_dtype"],
            device_map=model_config["device_map"],
            trust_remote_code=model_config["trust_remote_code"],
            low_cpu_mem_usage=model_config["low_cpu_mem_usage"]
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
        Build the full prompt for Qwen model using native chat template.
        """
        if persona_text:
            user_content = f"{context_text}\n\nQuestion: {question}\n\n{persona_text}"
        else:
            user_content = f"{context_text}\n\nQuestion: {question}"

        # Use Qwen chat template format - DO NOT repeat JSON schema in user message
        system_message = f"<|im_start|>system\n{self.base_prompt}<|im_end|>\n"
        user_message = f"<|im_start|>user\n{user_content}<|im_end|>\n"
        assistant_start = "<|im_start|>assistant\n"  # This is what add_generation_prompt=True does

        full_prompt = system_message + user_message + assistant_start

        logger.debug(f"Qwen chat template prompt: {full_prompt}")
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
            model_config = get_model_config(self.pipe.model.name_or_path)
            # Generate text with appropriate parameters for JSON output
            outputs = self.pipe(
                prompt,
                max_new_tokens=model_config["max_new_tokens"],
                temperature=model_config["temperature"],
                do_sample=model_config["do_sample"],
                repetition_penalty=model_config["repetition_penalty"],
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
