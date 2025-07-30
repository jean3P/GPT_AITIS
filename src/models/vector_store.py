# models/vector_store.py (Enhanced Version)

import logging
import re
import numpy as np
import os
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Import chunking components
from models.chunking.base import ChunkingStrategy
from models.chunking.factory import ChunkingFactory, create_preset_strategy, auto_register_strategies

logger = logging.getLogger(__name__)


class EnhancedLocalVectorStore:
    """
    Enhanced vector store with pluggable chunking strategies.

    Supports multiple chunking approaches through the Strategy pattern,
    making it easy to test and compare different chunking methods.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunking_strategy: str = "simple",
                 chunking_config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced vector store.

        Args:
            model_name: Sentence transformer model for embeddings
            chunking_strategy: Strategy name ('simple', 'structural', 'semantic', etc.)
            chunking_config: Optional configuration for the chunking strategy
        """
        logger.info(f"Initializing EnhancedLocalVectorStore with {chunking_strategy} chunking")

        # Auto-register strategies on first use
        auto_register_strategies()

        # Initialize sentence transformer
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded sentence-transformers model")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            raise RuntimeError(f"Failed to load embedding model: {e}")

        # Initialize chunking strategy
        self.chunking_strategy = self._create_chunking_strategy(
            chunking_strategy, chunking_config
        )

        # Storage for embeddings and chunks
        self.embeddings = None
        self.text_chunks = []
        self.chunk_metadata = []
        self.indexed_files = {}

        logger.info(f"EnhancedLocalVectorStore initialized successfully")

    def _create_chunking_strategy(self, strategy_name: str,
                                  config: Optional[Dict[str, Any]]) -> ChunkingStrategy:
        """Create the chunking strategy."""
        try:
            # Check if it's a preset
            if strategy_name in ['fast', 'balanced', 'comprehensive', 'research']:
                return create_preset_strategy(strategy_name)
            else:
                return ChunkingFactory.create_strategy(strategy_name, config)
        except Exception as e:
            logger.warning(f"Failed to create {strategy_name} strategy: {e}")
            logger.info("Falling back to simple chunking strategy")
            return ChunkingFactory.create_strategy('simple', {'max_length': 512})

    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about the current chunking strategy."""
        strategy_info = self.chunking_strategy.get_strategy_info()
        strategy_info.update({
            'total_chunks': len(self.text_chunks),
            'total_files_indexed': len(self.indexed_files),
            'has_embeddings': self.embeddings is not None,
            'embedding_model': self.model.get_sentence_embedding_dimension()
        })
        return strategy_info

    def switch_chunking_strategy(self, new_strategy: str,
                                 config: Optional[Dict[str, Any]] = None,
                                 reindex: bool = False):
        """
        Switch to a different chunking strategy.

        Args:
            new_strategy: Name of the new strategy
            config: Optional configuration for the new strategy
            reindex: Whether to re-index existing files with new strategy
        """
        logger.info(f"Switching from {self.chunking_strategy.name} to {new_strategy}")

        old_strategy = self.chunking_strategy
        self.chunking_strategy = self._create_chunking_strategy(new_strategy, config)

        if reindex and self.indexed_files:
            logger.info("Re-indexing files with new chunking strategy")
            file_paths = list(self.indexed_files.keys())
            self.index_documents(file_paths)
        else:
            logger.info("Strategy switched. Use reindex=True to re-process existing files.")

    def extract_policy_id(self, path: str) -> str:
        """Extract policy ID from filename."""
        filename = os.path.basename(path)
        match = re.match(r'^(\d+)_', filename)
        if match:
            return match.group(1)
        else:
            logger.warning(f"Could not extract policy ID from filename: {filename}")
            return os.path.splitext(filename)[0]

    def get_file_type(self, path: str) -> str:
        """Determine file type based on extension."""
        _, ext = os.path.splitext(path.lower())
        if ext == '.pdf':
            return 'pdf'
        elif ext == '.txt':
            return 'txt'
        else:
            return 'unknown'

    def extract_text_from_file(self, path: str) -> str:
        """Extract text from a file based on its type."""
        file_type = self.get_file_type(path)

        if file_type == 'pdf':
            return self._extract_text_from_pdf(path)
        elif file_type == 'txt':
            return self._extract_text_from_txt(path)
        else:
            logger.warning(f"Unsupported file type for {path}. Supported types: PDF, TXT")
            return ""

    def _extract_text_from_pdf(self, path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            logger.error(f"Error extracting text from PDF {path}: {e}")
            return ""

    def _extract_text_from_txt(self, path: str, encoding: str = 'utf-8') -> str:
        """Extract text from a TXT file."""
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

    def index_documents(self, file_paths: Union[List[str], str]):
        """Index documents using the current chunking strategy."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        logger.info(f"Indexing {len(file_paths)} documents using {self.chunking_strategy.name} strategy")

        # Reset storage
        self.text_chunks = []
        self.chunk_metadata = []
        self.indexed_files = {}

        all_chunk_results = []

        for path in file_paths:
            try:
                policy_id = self.extract_policy_id(path)
                file_type = self.get_file_type(path)
                logger.info(f"Processing {file_type.upper()} document: {path}")

                text = self.extract_text_from_file(path)
                if not text:
                    logger.warning(f"No text extracted from {path}")
                    continue

                # Use chunking strategy to process text
                chunk_results = self.chunking_strategy.chunk_text(
                    text=text,
                    policy_id=policy_id,
                    max_length=512  # Default, strategy may ignore
                )

                # Store file information
                self.indexed_files[path] = {
                    'policy_id': policy_id,
                    'file_type': file_type,
                    'chunk_count': len(chunk_results),
                    'strategy_used': self.chunking_strategy.name
                }

                all_chunk_results.extend(chunk_results)

                logger.info(f"Created {len(chunk_results)} chunks for policy {policy_id}")

            except Exception as e:
                logger.error(f"Error processing document {path}: {e}")

        if not all_chunk_results:
            logger.warning("No chunks created from documents")
            return

        # Extract texts and metadata
        self.text_chunks = self.chunking_strategy.get_chunk_text_list(all_chunk_results)
        self.chunk_metadata = self.chunking_strategy.get_metadata_list(all_chunk_results)

        logger.info(f"Creating embeddings for {len(self.text_chunks)} chunks")
        self.embeddings = self.embed(self.text_chunks)

        logger.info(f"Successfully indexed {len(self.text_chunks)} chunks using {self.chunking_strategy.name} strategy")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of text chunks."""
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return np.array([])

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and document vectors."""
        if a.size == 0 or b.size == 0:
            return np.array([])

        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)

        a_norm = np.where(a_norm == 0, 1e-10, a_norm)
        b_norm = np.where(b_norm == 0, 1e-10, b_norm)

        a_normalized = a / a_norm
        b_normalized = b / b_norm

        return np.dot(a_normalized, b_normalized.T)

    def retrieve(self, query: str, k: int = 1, policy_id: Optional[str] = None) -> List[str]:
        """Retrieve the k most relevant text chunks for a query."""
        if self.embeddings is None or self.embeddings.size == 0:
            logger.warning("No embeddings available for retrieval")
            return []

        if not self.text_chunks:
            logger.warning("No text chunks available for retrieval")
            return []

        logger.info(f"Retrieving top {k} chunks for query using {self.chunking_strategy.name} strategy")
        query_embedding = self.embed([query])

        if query_embedding.size == 0:
            logger.warning("Failed to create query embedding")
            return []

        similarities = self.cosine_similarity(query_embedding, self.embeddings)
        if similarities.size == 0:
            logger.warning("Failed to compute similarities")
            return []

        similarities = similarities[0]

        # Filter by policy if specified
        if policy_id:
            policy_mask = np.array([
                meta.policy_id == policy_id for meta in self.chunk_metadata
            ])
            if np.any(policy_mask):
                filtered_similarities = np.where(policy_mask, similarities, -1)
                similarities = filtered_similarities

        # Get top k indices
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [self.text_chunks[i] for i in top_indices]

    def get_chunk_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the chunks."""
        if not self.chunk_metadata:
            return {"error": "No chunks available"}

        stats = {
            "strategy": self.chunking_strategy.name,
            "total_chunks": len(self.chunk_metadata),
            "chunk_types": {},
            "coverage_types": {},
            "avg_word_count": 0,
            "chunks_with_amounts": 0,
            "chunks_with_conditions": 0,
            "chunks_with_exclusions": 0,
            "policies_indexed": len(set(meta.policy_id for meta in self.chunk_metadata if meta.policy_id))
        }

        total_words = 0
        for meta in self.chunk_metadata:
            # Count chunk types
            chunk_type = meta.chunk_type
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1

            # Count coverage types
            coverage_type = meta.coverage_type
            stats["coverage_types"][coverage_type] = stats["coverage_types"].get(coverage_type, 0) + 1

            # Accumulate word count
            total_words += meta.word_count

            # Count special characteristics
            if meta.has_amounts:
                stats["chunks_with_amounts"] += 1
            if meta.has_conditions:
                stats["chunks_with_conditions"] += 1
            if meta.has_exclusions:
                stats["chunks_with_exclusions"] += 1

        stats["avg_word_count"] = total_words / len(self.chunk_metadata) if self.chunk_metadata else 0

        return stats

    def get_available_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        return ChunkingFactory.get_available_strategies()

    def compare_strategies(self, strategies: List[str], sample_text: str,
                           policy_id: str = "test") -> Dict[str, Dict[str, Any]]:
        """
        Compare different chunking strategies on sample text.

        Args:
            strategies: List of strategy names to compare
            sample_text: Text to chunk for comparison
            policy_id: Policy ID for testing

        Returns:
            Dictionary mapping strategy names to their results
        """
        comparison_results = {}

        for strategy_name in strategies:
            try:
                # Create strategy instance
                strategy = ChunkingFactory.create_strategy(strategy_name)

                # Chunk the sample text
                chunks = strategy.chunk_text(sample_text, policy_id)

                # Analyze results
                comparison_results[strategy_name] = {
                    "chunk_count": len(chunks),
                    "avg_chunk_length": np.mean([len(chunk.text) for chunk in chunks]),
                    "chunk_types": list(set(chunk.metadata.chunk_type for chunk in chunks)),
                    "coverage_types": list(set(chunk.metadata.coverage_type for chunk in chunks)),
                    "chunks_with_amounts": sum(1 for chunk in chunks if chunk.metadata.has_amounts),
                    "strategy_info": strategy.get_strategy_info()
                }

            except Exception as e:
                comparison_results[strategy_name] = {
                    "error": str(e),
                    "chunk_count": 0
                }

        return comparison_results


# Backward compatibility: alias to existing LocalVectorStore interface
LocalVectorStore = EnhancedLocalVectorStore
