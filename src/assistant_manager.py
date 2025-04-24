# src/assistant_manager.py

import openai
import logging
from config import OPENAI_API_KEY, VECTOR_STORE_EXPIRATION_DAYS, RESPONSE_FORMAT_PATH
from typing import List, Any

from utils import load_response_schema

logger = logging.getLogger(__name__)

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def create_vector_store(name: str, file_paths: List[str], check: bool =True) -> Any:
    """
    Create a vector store in OpenAI's API, upload PDF files to it, and index them.

    Args:
        name (str): Name of the vector store.
        file_paths (list[str]): List of file paths to upload.
        check (bool): For logging purposes

    Returns:
        OpenAI VectorStore object
    """

    logger.info(f"Creating vector store")
    vector_store = client.vector_stores.create(
        name=name,
        expires_after={"anchor": "last_active_at", "days": VECTOR_STORE_EXPIRATION_DAYS},
    )
    streams = [open(path, "rb") for path in file_paths]
    file_batch = client.vector_stores.file_batches.upload_and_poll(vector_store_id=vector_store.id, files=streams)
    if check:
        logger.info(f"  File Batch Status: {file_batch.status}")
        logger.info(f"  File Batch Counts: {file_batch.file_counts}")

    return vector_store

def create_assistant(name: str, sys_prompt: str, model: str = "gpt-4o") -> Any:
    """
    Create an OpenAI Assistant configured to perform file search using a given system prompt.

    Args:
        name (str): Name of the assistant.
        sys_prompt (str): System-level instructions for the assistant.
        model (str): Model to use for generating responses.

    Returns:
        OpenAI Assistant object
    """
    logger.info(f"Creating assistant")
    response_schema = load_response_schema(RESPONSE_FORMAT_PATH)

    return client.beta.assistants.create(
        name=name,
        instructions=sys_prompt,
        model=model,
        tools=[{"type": "file_search"}],
        response_format={"type": "json_schema", "json_schema": response_schema}
    )


def update_assistant_vector(assistant_id: str, vector_store_id: str) -> None:
    """
    Link an assistant to a vector store so it can use file search during conversation.

    Args:
        assistant_id (str): The ID of the assistant.
        vector_store_id (str): The ID of the vector store.
    """
    logger.info(f"Update assistant vector")
    client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
    )
