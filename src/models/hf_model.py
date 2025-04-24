# src/models/hf_model.py

import json
import logging

from typing import List, Dict, Any
from huggingface_hub import login
from transformers import pipeline
from config import HUGGINGFACE_TOKEN
from models.base import BaseModelClient


login(token=HUGGINGFACE_TOKEN)


logger = logging.getLogger(__name__)

class HuggingFaceModelClient(BaseModelClient):
    def __init__(self, model_name: str):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto"
        )

    def query(self, question: str, context_files: List[str]) -> Dict[str, Any]:
        system_prompt = "You are an insurance expert. Provide clear eligibility judgment, quote from policy, and covered amount."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        outputs = self.pipe(messages, max_new_tokens=512)
        try:
            text = outputs[0]["generated_text"]
            json_start = text.find("{")
            json_obj = text[json_start:]
            return json.loads(json_obj)
        except Exception as e:
            logger.error(f"HF Parsing Error: {e}")
            return {}



