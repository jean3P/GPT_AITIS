# src/models/vector_store.py

import logging
import re

import numpy as np
import os
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from config import CACHE_EMBEDDINGS, EMBEDDINGS_DIR

logger = logging.getLogger(__name__)


class LocalVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize a vector store with the specified embedding model.

        Args:
            model_name: The model to use for embeddings or local path.
                        Default is 'sentence-transformers/all-MiniLM-L6-v2'.
        """
        logger.info(f"Initializing vector store with model: {model_name}")

        try:
            # Load the model - works with both local paths and model names
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded sentence-transformers model")

            # Initialize storage for embeddings and text chunks
            self.embeddings = None
            self.text_chunks = []
            self.chunk_metadata = []  # To store source file for each chunk

        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def extract_policy_id(self, path: str) -> str:
        """
        Extract policy ID from the file filename (PDF or TXT).
        Example: "10_nobis_policy.pdf" -> "10" or "10_nobis_policy.txt" -> "10"

        Args:
            path: Path to the file

        Returns:
            The policy ID as a string
        """
        filename = os.path.basename(path)
        match = re.match(r'^(\d+)_', filename)
        if match:
            return match.group(1)
        else:
            logger.warning(f"Could not extract policy ID from filename: {filename}")
            # Return the filename without extension as fallback
            return os.path.splitext(filename)[0]

    def get_file_type(self, path: str) -> str:
        """
        Determine file type based on extension.

        Args:
            path: Path to the file

        Returns:
            File type ('pdf', 'txt', or 'unknown')
        """
        _, ext = os.path.splitext(path.lower())
        if ext == '.pdf':
            return 'pdf'
        elif ext == '.txt':
            return 'txt'
        else:
            return 'unknown'

    def extract_text_from_pdf(self, path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            logger.error(f"Error extracting text from PDF {path}: {e}")
            return ""

    def extract_text_from_txt(self, path: str, encoding: str = 'utf-8') -> str:
        """
        Extract text from a TXT file.

        Args:
            path: Path to the TXT file
            encoding: File encoding (default: utf-8)

        Returns:
            The text content of the file
        """
        try:
            with open(path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encodings if utf-8 fails
            for fallback_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(path, 'r', encoding=fallback_encoding) as file:
                        logger.info(f"Successfully read {path} using {fallback_encoding} encoding")
                        return file.read()
                except UnicodeDecodeError:
                    continue
            logger.error(f"Could not decode {path} with any common encoding")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from TXT {path}: {e}")
            return ""

    def extract_text_from_file(self, path: str) -> str:
        """
        Extract text from a file based on its type.

        Args:
            path: Path to the file

        Returns:
            The extracted text content
        """
        file_type = self.get_file_type(path)

        if file_type == 'pdf':
            return self.extract_text_from_pdf(path)
        elif file_type == 'txt':
            return self.extract_text_from_txt(path)
        else:
            logger.warning(f"Unsupported file type for {path}. Supported types: PDF, TXT")
            return ""

    def chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into smaller chunks."""
        # Simple paragraph-based chunking
        paragraphs = text.split('\n\n')
        chunks = []

        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Fallback to word-based chunking if no paragraphs
        if not chunks:
            words = text.split()
            chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

        return chunks

    def embed(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of text chunks using sentence-transformers."""
        if not texts:
            return np.array([])

        try:
            # sentence-transformers handles batching internally
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return np.array([])

    def index_documents(self, file_paths: Union[List[str], str]):
        """
        Index documents (PDF or TXT) by extracting text and creating embeddings.

        Args:
            file_paths: List of file paths or single file path to index
        """
        # Convert single path to list for uniform processing
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        logger.info(f"Indexing {len(file_paths)} documents")
        all_chunks = []
        self.text_chunks = []
        self.chunk_metadata = []

        # Validate file types
        supported_files = []
        for path in file_paths:
            file_type = self.get_file_type(path)
            if file_type in ['pdf', 'txt']:
                supported_files.append(path)
            else:
                logger.warning(f"Skipping unsupported file: {path} (type: {file_type})")

        if not supported_files:
            logger.error("No supported files found for indexing")
            return

        # Try to load embeddings from cache first
        all_cached = True
        for path in supported_files:
            policy_id = self.extract_policy_id(path)
            cache_path = os.path.join(EMBEDDINGS_DIR, f"policy_{policy_id}_embeddings.npz")

            if not os.path.exists(cache_path) or not CACHE_EMBEDDINGS:
                all_cached = False
                break

        # If all embeddings are cached, load them
        if all_cached and CACHE_EMBEDDINGS:
            try:
                logger.info("Loading embeddings from cache")
                # Load first policy to initialize metadata structure
                policy_id = self.extract_policy_id(supported_files[0])
                cache_path = os.path.join(EMBEDDINGS_DIR, f"policy_{policy_id}_embeddings.npz")
                cached_data = np.load(cache_path, allow_pickle=True)

                self.text_chunks = cached_data['text_chunks'].tolist()
                self.chunk_metadata = cached_data['chunk_metadata'].tolist()
                self.embeddings = cached_data['embeddings']

                # Load and append additional policies if more than one
                for i, path in enumerate(supported_files[1:], 1):
                    policy_id = self.extract_policy_id(path)
                    cache_path = os.path.join(EMBEDDINGS_DIR, f"policy_{policy_id}_embeddings.npz")
                    cached_data = np.load(cache_path, allow_pickle=True)

                    self.text_chunks.extend(cached_data['text_chunks'].tolist())
                    self.chunk_metadata.extend(cached_data['chunk_metadata'].tolist())

                    # Concatenate embeddings
                    if i == 1:  # First additional policy
                        self.embeddings = np.vstack([self.embeddings, cached_data['embeddings']])
                    else:  # Subsequent policies
                        self.embeddings = np.vstack([self.embeddings, cached_data['embeddings']])

                logger.info(f"Successfully loaded cached embeddings for {len(supported_files)} files")
                return
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}")
                logger.info("Falling back to generating embeddings")

        # If cache loading failed or some embeddings weren't cached, process all documents
        for path in supported_files:
            try:
                policy_id = self.extract_policy_id(path)
                file_type = self.get_file_type(path)
                logger.info(f"Processing {file_type.upper()} document: {path}")

                text = self.extract_text_from_file(path)
                if not text:
                    logger.warning(f"No text extracted from {path}")
                    continue

                # Get filename for metadata
                filename = os.path.basename(path)

                # Add prefix to each chunk to help identify source
                chunks = self.chunk_text(text)
                source_prefixed_chunks = [
                    f"[Policy {policy_id}]: {chunk}" for chunk in chunks
                ]

                self.text_chunks.extend(source_prefixed_chunks)
                all_chunks.extend(source_prefixed_chunks)

                # Store metadata for each chunk
                for _ in chunks:
                    self.chunk_metadata.append({
                        "policy_id": policy_id,
                        "source_file": path,
                        "filename": filename,
                        "file_type": file_type
                    })

            except Exception as e:
                logger.error(f"Error processing document {path}: {e}")

        if not all_chunks:
            logger.warning("No text chunks extracted from documents")
            return

        logger.info(f"Creating embeddings for {len(all_chunks)} chunks")
        self.embeddings = self.embed(all_chunks)
        logger.info(
            f"Indexed {len(all_chunks)} chunks with embedding shape {self.embeddings.shape if self.embeddings is not None else 'None'}")

        # Save embeddings to cache for each policy separately
        if CACHE_EMBEDDINGS:
            self._save_embeddings_to_cache(supported_files)

    def _save_embeddings_to_cache(self, file_paths: List[str]):
        """Save embeddings to cache for each policy."""
        try:
            for path in file_paths:
                policy_id = self.extract_policy_id(path)
                cache_path = os.path.join(EMBEDDINGS_DIR, f"policy_{policy_id}_embeddings.npz")

                # Filter chunks and embeddings for this policy
                policy_indices = [i for i, meta in enumerate(self.chunk_metadata) if meta["policy_id"] == policy_id]

                if not policy_indices:
                    logger.warning(f"No chunks found for policy {policy_id}, skipping cache")
                    continue

                policy_chunks = [self.text_chunks[i] for i in policy_indices]
                policy_metadata = [self.chunk_metadata[i] for i in policy_indices]
                policy_embeddings = self.embeddings[policy_indices]

                # Save to file
                np.savez_compressed(
                    cache_path,
                    text_chunks=np.array(policy_chunks, dtype=object),
                    chunk_metadata=np.array(policy_metadata, dtype=object),
                    embeddings=policy_embeddings
                )

                logger.info(f"Cached embeddings for policy {policy_id} at {cache_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}")
            logger.info("Continuing without saving cache")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query vector and all document vectors."""
        if a.size == 0 or b.size == 0:
            return np.array([])

        # Normalize the vectors
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)

        # Avoid division by zero
        a_norm = np.where(a_norm == 0, 1e-10, a_norm)
        b_norm = np.where(b_norm == 0, 1e-10, b_norm)

        a_normalized = a / a_norm
        b_normalized = b / b_norm

        # Compute similarity
        return np.dot(a_normalized, b_normalized.T)

    def retrieve(self, query: str, k: int = 1, policy_id: Optional[str] = None) -> List[str]:
        """
        Retrieve the k most relevant text chunks for a query.

        Args:
            query: The search query
            k: Number of chunks to retrieve
            policy_id: Optional policy ID to filter results by

        Returns:
            List of relevant text chunks
        """
        if self.embeddings is None or self.embeddings.size == 0:
            logger.warning("No embeddings available for retrieval")
            return []

        if not self.text_chunks:
            logger.warning("No text chunks available for retrieval")
            return []

        logger.info(f"Retrieving top {k} chunks for query: {query}")
        query_embedding = self.embed([query])

        if query_embedding.size == 0:
            logger.warning("Failed to create query embedding")
            return []

        # Compute similarity
        similarities = self.cosine_similarity(query_embedding, self.embeddings)

        if similarities.size == 0:
            logger.warning("Failed to compute similarities")
            return []

        similarities = similarities[0]  # Get the first row

        # Filter by policy if specified
        if policy_id:
            # Create mask for chunks from the specified policy
            policy_mask = np.array([
                meta["policy_id"] == policy_id for meta in self.chunk_metadata
            ])

            # Apply mask to similarities (set others to -1 to exclude them)
            filtered_similarities = np.where(policy_mask, similarities, -1)

            # If we have at least one chunk from the specified policy
            if np.max(filtered_similarities) > -1:
                similarities = filtered_similarities

        # Get top k indices
        if len(similarities) < k:
            k = len(similarities)

        top_indices = np.argsort(similarities)[-k:][::-1]

        # Return the text chunks
        return [self.text_chunks[i] for i in top_indices]

    def retrieve_by_policy(self, query: str, k_per_policy: int = 3) -> Dict[str, List[str]]:
        """
        Retrieve chunks for each policy, organized by policy_id.

        Args:
            query: The search query
            k_per_policy: Number of chunks to retrieve per policy

        Returns:
            Dictionary mapping policy_ids to lists of relevant chunks
        """
        if self.embeddings is None or self.embeddings.size == 0:
            logger.warning("No embeddings available for retrieval")
            return {}

        if not self.text_chunks:
            logger.warning("No text chunks available for retrieval")
            return {}

        logger.info(f"Retrieving top {k_per_policy} chunks per policy for query: {query}")
        query_embedding = self.embed([query])

        if query_embedding.size == 0:
            logger.warning("Failed to create query embedding")
            return {}

        # Compute similarity for all chunks
        similarities = self.cosine_similarity(query_embedding, self.embeddings)[0]

        # Group by policy_id
        policy_chunks = {}

        # Get unique policy IDs
        unique_policies = set(meta["policy_id"] for meta in self.chunk_metadata)

        # For each policy, get the top k chunks
        for pid in unique_policies:
            # Find indices for this policy
            policy_indices = [
                i for i, meta in enumerate(self.chunk_metadata)
                if meta["policy_id"] == pid
            ]

            # Get similarities for this policy's chunks
            policy_similarities = [(i, similarities[i]) for i in policy_indices]

            # Sort by similarity (descending)
            policy_similarities.sort(key=lambda x: x[1], reverse=True)

            # Take top k
            top_k_indices = [idx for idx, _ in policy_similarities[:k_per_policy]]

            # Store the chunks
            policy_chunks[pid] = [self.text_chunks[i] for i in top_k_indices]

        return policy_chunks

    def get_indexed_files_info(self) -> Dict[str, Dict]:
        """
        Get information about indexed files.

        Returns:
            Dictionary with file statistics and metadata
        """
        if not self.chunk_metadata:
            return {}

        files_info = {}
        for meta in self.chunk_metadata:
            policy_id = meta["policy_id"]
            if policy_id not in files_info:
                files_info[policy_id] = {
                    "filename": meta["filename"],
                    "file_type": meta["file_type"],
                    "source_file": meta["source_file"],
                    "chunk_count": 0
                }
            files_info[policy_id]["chunk_count"] += 1

        return files_info
