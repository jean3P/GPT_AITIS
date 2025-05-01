# src/models/hf_model.py
import json
import logging
import os
from typing import List, Dict, Any
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

        # Log HuggingFace cache locations to verify they're set
        logger.info(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', 'Not set')}")
        logger.info(f"HF_ASSETS_CACHE: {os.environ.get('HF_ASSETS_CACHE', 'Not set')}")

        # For phi-4, check if it's in the scratch directory
        if model_name == "microsoft/phi-4":
            scratch_model_path = os.path.join("/cluster/scratch", os.environ.get("USER", ""), "models", "phi-4")

            if os.path.exists(scratch_model_path):
                logger.info(f"Found phi-4 model in scratch: {scratch_model_path}")
                model_name = scratch_model_path
            else:
                logger.warning(f"Phi-4 not found in scratch directory. Looking for it in HuggingFace cache.")

        try:
            logger.info(f"Initializing pipeline with model: {model_name}")

            # Load the tokenizer and model separately for better configuration
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Create pipeline with loaded model and tokenizer
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )

            logger.info("Pipeline successfully initialized")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise

        # Store the base prompt template
        self.base_prompt = sys_prompt.strip()
        self.json_format = """
            Format the output exactly in the following JSON schema:
            {
              "answer": {
                "eligibility": "Yes - it's covered | No - not relevant | No - not covered",
                "eligibility_policy": "Quoted text from policy",
                "amount_policy": "Amount like '1000 CHF' or null",
                "amount_policy_line": "Quoted policy text or null"
              }
            }
            """

    def query(self, question: str, context_files: List[str]) -> Dict[str, Any]:
        logger.info(f"Querying model with question: {question}")

        # Include context information if available
        context_text = ""
        if context_files and len(context_files) > 0:
            context_text = "\n\nRelevant policy information:\n"
            for i, ctx in enumerate(context_files, 1):
                context_text += f"Policy context {i}:\n{ctx}\n\n"

        # Format prompt as plain text (not using the message dict format)
        full_prompt = f"{self.base_prompt}\n\n{context_text}\n\nQuestion: {question}\n\n{self.json_format}"

        try:
            # Generate text with appropriate parameters for JSON output
            outputs = self.pipe(
                full_prompt,  # Use text prompt instead of the dictionary format
                max_new_tokens=512,
                do_sample=False,  # Deterministic output for reliable JSON
                num_return_sequences=1,
                return_full_text=False  # Don't include the prompt in output
            )

            try:
                # Get the generated text
                generated = outputs[0]["generated_text"]
                logger.debug(f"Model output: {generated}")

                # Find and extract JSON
                json_start = generated.find("{")
                if json_start == -1:
                    # Try looking for JSON in the entire text if not found in the generated portion
                    error_msg = "No JSON found in response"
                    logger.error(error_msg)
                    return {
                        "answer": {
                            "eligibility": "Error",
                            "eligibility_policy": error_msg,
                            "amount_policy": None,
                            "amount_policy_line": None
                        }
                    }

                json_str = generated[json_start:]

                # Try to clean up the JSON string if needed
                # Find the position of the last closing brace
                json_end = json_str.rfind("}")
                if json_end != -1:
                    json_str = json_str[:json_end + 1]

                # Parse the JSON string into a Python dictionary
                parsed = json.loads(json_str)

                if "answer" in parsed:
                    logger.info("Successfully parsed JSON response")
                    return parsed
                else:
                    error_msg = "Missing 'answer' field in model output"
                    logger.error(error_msg)
                    return {
                        "answer": {
                            "eligibility": "Error",
                            "eligibility_policy": error_msg,
                            "amount_policy": None,
                            "amount_policy_line": None
                        }
                    }
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse JSON: {str(e)}"
                logger.error(error_msg)
                # Log the generated text to help debug
                logger.error(f"Generated text: {generated}")
                return {
                    "answer": {
                        "eligibility": "Error",
                        "eligibility_policy": error_msg,
                        "amount_policy": None,
                        "amount_policy_line": None
                    }
                }
        except Exception as e:
            error_msg = f"Error during model inference: {str(e)}"
            logger.error(error_msg)
            return {
                "answer": {
                    "eligibility": "Error",
                    "eligibility_policy": error_msg,
                    "amount_policy": None,
                    "amount_policy_line": None
                }
            }
