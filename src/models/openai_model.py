# src/models/openai_model.py
from typing import List, Dict, Any

from .base import BaseModelClient
from assistant_manager import client, create_vector_store, create_assistant, update_assistant_vector
import json, time, logging

logger = logging.getLogger(__name__)

class OpenAIModelClient(BaseModelClient):
    def __init__(self, model_name: str, sys_prompt: str, file_paths: List[str]):
        self.vector_store = create_vector_store("RAG_VectorStore", file_paths)
        self.assistant = create_assistant("Insurance Expert Assistant", sys_prompt, model_name)
        update_assistant_vector(self.assistant.id, self.vector_store.id)
        self.thread = client.beta.threads.create()

    def query(self, question: str, context_files: List[str]) -> Dict[str, Any]:
        prompt = f"Question: \"{question}\". Give a Yes/No, quote the supporting text, and mention the amount if relevant."
        client.beta.threads.messages.create(thread_id=self.thread.id, role="user", content=prompt)
        run = client.beta.threads.runs.create(thread_id=self.thread.id, assistant_id=self.assistant.id)

        while client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id).status != "completed":
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=self.thread.id)
        for msg in reversed(messages.data):
            if msg.role == "assistant":
                return json.loads(msg.content[0].text.value.strip())
        return {}

