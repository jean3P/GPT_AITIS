# src/models/qwen_model.py

"""
Qwen model client implementation for insurance policy analysis.
Based on the HuggingFace model client but optimized for Qwen models.
"""
import logging
import os
import re
from typing import Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import get_local_model_path, get_model_config
from models.hf_model import HuggingFaceModelClient
from models.json_utils.extractors import JSONExtractor
from models.persona.extractor import PersonaExtractor

logger = logging.getLogger(__name__)


class QwenModelClient(HuggingFaceModelClient):
    """
    Specialized client for Qwen models.
    Inherits from HuggingFaceModelClient but with Qwen-specific optimizations.
    """

    def __init__(self, model_name: str, sys_prompt: str):
        """
        Initialize a Qwen model client.

        Args:
            model_name: Name of the Qwen model to use
            sys_prompt: System prompt for the model
        """
        logger.info(f"Loading Qwen model: {model_name}")
        self._log_cache_locations()

        # Check for local model first
        model_path = get_local_model_path(model_name)
        self.model_config = get_model_config(model_name)

        self.pipe = self._initialize_qwen_pipeline(model_path)
        self.base_prompt = sys_prompt.strip()

        self.persona_extractor = PersonaExtractor(self.pipe)
        self.json_extractor = JSONExtractor()

    def _check_qwen_model_in_scratch(self, model_name: str) -> str:
        """Check if Qwen model exists in scratch directory."""
        # Support both 32B and 7B models
        model_mappings = {
            "Qwen/Qwen2.5-32B": "qwen2.5-32b",
            "qwen2.5-32b": "qwen2.5-32b",
            "Qwen/Qwen2.5-7B": "qwen2.5-7b",  # Add 7B support
            "qwen2.5-7b": "qwen2.5-7b"  # Add 7B support
        }

        if model_name in model_mappings:
            scratch_model_path = os.path.join(
                "/cluster/scratch",
                os.environ.get("USER", ""),
                "models",
                model_mappings[model_name]
            )

            if os.path.exists(scratch_model_path):
                logger.info(f"Found Qwen model in scratch: {scratch_model_path}")
                return scratch_model_path
            else:
                logger.warning(f"Qwen model not found in scratch directory.")

        return model_name

    def _initialize_qwen_pipeline(self, model_path: str):
        """Initialize the Qwen pipeline with optimized settings."""
        try:
            logger.info(f"Initializing Qwen pipeline with model: {model_path}")

            # Load tokenizer with Qwen-specific settings
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"  # Qwen often works better with left padding
            )

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with Qwen-optimized settings
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.model_config["torch_dtype"],
                device_map=self.model_config["device_map"],
                trust_remote_code=self.model_config["trust_remote_code"],
                low_cpu_mem_usage=self.model_config["low_cpu_mem_usage"]
            )

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )

            logger.info("Qwen pipeline successfully initialized")
            return pipe

        except Exception as e:
            logger.error(f"Error initializing Qwen pipeline: {e}")
            raise

    def _generate_model_output(self, prompt: str) -> Dict[str, Any]:
        """
        Generate output from Qwen model with optimized parameters.
        """
        try:
            # Qwen-specific generation parameters
            generation_params = {
                "max_new_tokens": self.model_config.get("max_new_tokens", 512),
                "temperature": None,
                "do_sample": self.model_config.get("do_sample", False),
                "repetition_penalty": self.model_config.get("repetition_penalty", 1.1),
                "num_return_sequences": 1,
                "return_full_text": False,
                "pad_token_id": self.pipe.tokenizer.eos_token_id,
                "eos_token_id": self.pipe.tokenizer.eos_token_id,
                "top_p": None,
                "top_k": None,
                "no_repeat_ngram_size": self.model_config.get("no_repeat_ngram_size", 3),
            }

            # Generate text
            outputs = self.pipe(prompt, **generation_params)

            # Get the generated text
            generated = outputs[0]["generated_text"]

            generated = self._clean_qwen_output(generated)
            logger.debug(f"Qwen model output: {generated}")

            # Try to extract JSON from the output
            extracted_json = self.json_extractor.extract_json(generated)

            if extracted_json:
                return extracted_json

            # If extraction fails, return formatted error
            logger.error("Failed to extract valid JSON from Qwen model output")
            return self._create_error_response("Failed to extract valid JSON from model output")

        except Exception as e:
            error_msg = f"Error during Qwen model inference: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)

    def _clean_qwen_output(self, text: str) -> str:
        """
        Clean Qwen output to improve JSON extraction.
        """
        # Remove thinking tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Remove any explanatory text before JSON
        lines = text.split('\n')
        json_started = False
        cleaned_lines = []

        for line in lines:
            if '{' in line and not json_started:
                json_started = True
            if json_started:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines) if cleaned_lines else text

    def _build_prompt(self, question: str, context_text: str, persona_text: str) -> str:
        """
        Build the full prompt for Qwen model using proper chat template.
        """
        # Build user content - NO JSON schema
        if persona_text:
            user_content = f"{context_text}\n\nQuestion: {question}\n\n{persona_text}"
        else:
            user_content = f"{context_text}\n\nQuestion: {question}"

        # Use Qwen's apply_chat_template if available
        if hasattr(self.pipe.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    # {"role": "system", "content": },
                    {"role": "user", "content": (self.base_prompt + user_content)}
                ]

                formatted_prompt = self.pipe.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                logger.debug(f"Qwen chat template prompt: {formatted_prompt}")
                return formatted_prompt

            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}, falling back to manual formatting")

        return ""

