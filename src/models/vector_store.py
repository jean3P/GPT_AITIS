# src/models/vector_store.py

import logging
import numpy as np
import os
from typing import List, Dict, Optional
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
        Extract policy ID from the PDF filename.
        Example: "10_nobis_policy.pdf" -> "10"

        Args:
            path: Path to the PDF file

        Returns:
            The policy ID as a string
        """
        filename = os.path.basename(path)
        import re
        match = re.match(r'^(\d+)_', filename)
        if match:
            return match.group(1)
        else:
            logger.warning(f"Could not extract policy ID from filename: {filename}")
            # Return the filename without extension as fallback
            return os.path.splitext(filename)[0]

    def extract_text_from_pdf(self, path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            logger.error(f"Error extracting text from PDF {path}: {e}")
            return ""

    def chunk_text(self, text: str, max_length: int = 100, overlap: int = 20) -> List[str]:
        """
        Split text into smaller chunks with improved handling of special characters and more
        robust chunking strategies.

        Args:
            text: The text to chunk
            max_length: Maximum length of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        # Normalize whitespace characters (including NBSP)
        import re
        text = re.sub(r'\s+', ' ', text)  # Replace all whitespace sequences with a single space
        text = text.replace('\xa0', ' ')  # Replace NBSP with regular space

        # First attempt paragraph-based chunking (split on double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If we have extremely long paragraphs, split them into sentences
        for i, para in enumerate(paragraphs):
            if len(para) > max_length:
                # Replace paragraph with sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                paragraphs[i:i + 1] = sentences

        # Now create chunks from our units (paragraphs/sentences)
        chunks = []
        current_chunk = ""

        for unit in paragraphs:
            # If this unit alone exceeds max_length, we need to split it further
            if len(unit) > max_length:
                # If current_chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split long unit by character count with overlap
                for i in range(0, len(unit), max_length - overlap):
                    if i > 0:
                        start = i - overlap
                    else:
                        start = 0

                    chunk = unit[start:start + max_length]

                    # Don't add tiny final chunks
                    if len(chunk) < max_length / 4 and chunks and i > 0:
                        # Append to the previous chunk if it would fit
                        if len(chunks[-1]) + len(chunk) <= max_length:
                            chunks[-1] = chunks[-1] + " " + chunk
                        else:
                            chunks.append(chunk)
                    else:
                        chunks.append(chunk)

            # Normal case: try to add the unit to current_chunk
            elif len(current_chunk) + len(unit) + 1 <= max_length:
                if current_chunk:
                    current_chunk += " " + unit
                else:
                    current_chunk = unit

            # If adding would exceed max_length, save current chunk and start a new one
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = unit

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add policy ID prefix to each chunk (moved this out to the index_documents method)

        # Ensure no chunk exceeds max_length
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Split further if needed (should rarely happen at this point)
                words = chunk.split()
                temp = ""
                for word in words:
                    if len(temp) + len(word) + 1 <= max_length:
                        if temp:
                            temp += " " + word
                        else:
                            temp = word
                    else:
                        final_chunks.append(temp)
                        temp = word
                if temp:
                    final_chunks.append(temp)

        # Final verification - any chunk still too long gets truncated (safety measure)
        final_chunks = [chunk[:max_length] for chunk in final_chunks]

        return final_chunks

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

    def index_documents(self, pdf_paths: List[str]):
        """Index PDF documents by extracting text and creating embeddings."""
        logger.info(f"Indexing {len(pdf_paths)} documents")
        all_chunks = []
        self.text_chunks = []
        self.chunk_metadata = []

        # Try to load embeddings from cache first
        all_cached = True
        for path in pdf_paths:
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
                policy_id = self.extract_policy_id(pdf_paths[0])
                cache_path = os.path.join(EMBEDDINGS_DIR, f"policy_{policy_id}_embeddings.npz")
                cached_data = np.load(cache_path, allow_pickle=True)

                self.text_chunks = cached_data['text_chunks'].tolist()
                self.chunk_metadata = cached_data['chunk_metadata'].tolist()
                self.embeddings = cached_data['embeddings']

                # Load and append additional policies if more than one
                for i, path in enumerate(pdf_paths[1:], 1):
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

                logger.info(f"Successfully loaded cached embeddings for {len(pdf_paths)} policies")
                return
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}")
                logger.info("Falling back to generating embeddings")

        # If cache loading failed or some embeddings weren't cached, process all documents
        for path in pdf_paths:
            try:
                policy_id = self.extract_policy_id(path)
                logger.info(f"Processing document: {path}")
                text = self.extract_text_from_pdf(path)
                if not text:
                    logger.warning(f"No text extracted from {path}")
                    continue

                # Get policy ID for metadata
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
                        "filename": filename
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
            self._save_embeddings_to_cache(pdf_paths)

    def _save_embeddings_to_cache(self, pdf_paths: List[str]):
        """Save embeddings to cache for each policy."""
        try:
            for path in pdf_paths:
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

        result_chunks = [self.text_chunks[i] for i in top_indices]

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
