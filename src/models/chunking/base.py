# models/chunking/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    chunk_id: str
    policy_id: Optional[str] = None
    chunk_type: str = "unknown"
    word_count: int = 0
    has_amounts: bool = False
    has_conditions: bool = False
    has_exclusions: bool = False
    section: Optional[str] = None
    coverage_type: str = "general"
    confidence_score: float = 1.0
    extra_data: Optional[Dict[str, Any]] = None


@dataclass
class ChunkResult:
    """Result of chunking operation."""
    text: str
    metadata: ChunkMetadata


class ChunkingStrategy(ABC):
    """
    Abstract base class for all chunking strategies.

    This interface ensures all chunking approaches have consistent behavior
    and can be easily swapped in the vector store.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chunking strategy.

        Args:
            config: Optional configuration dictionary for the strategy
        """
        self.config = config or {}
        self.name = self.__class__.__name__.lower().replace('chunker', '')
        logger.info(f"Initialized {self.name} chunking strategy")

    @abstractmethod
    def chunk_text(self, text: str, policy_id: Optional[str] = None,
                   max_length: int = 512) -> List[ChunkResult]:
        """
        Chunk the given text using this strategy.

        Args:
            text: The text to chunk
            policy_id: Optional policy identifier for metadata
            max_length: Maximum chunk length (strategy may ignore this)

        Returns:
            List of ChunkResult objects containing text and metadata
        """
        pass

    @abstractmethod
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this chunking strategy.

        Returns:
            Dictionary with strategy metadata (name, description, config, etc.)
        """
        pass

    def validate_config(self) -> bool:
        """
        Validate the strategy configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return True

    def get_chunk_text_list(self, chunk_results: List[ChunkResult]) -> List[str]:
        """
        Extract just the text from chunk results for backward compatibility.

        Args:
            chunk_results: List of ChunkResult objects

        Returns:
            List of chunk texts
        """
        return [result.text for result in chunk_results]

    def get_metadata_list(self, chunk_results: List[ChunkResult]) -> List[ChunkMetadata]:
        """
        Extract just the metadata from chunk results.

        Args:
            chunk_results: List of ChunkResult objects

        Returns:
            List of ChunkMetadata objects
        """
        return [result.metadata for result in chunk_results]

    def _create_chunk_id(self, policy_id: Optional[str], chunk_index: int) -> str:
        """Create a unique chunk ID."""
        prefix = f"{policy_id}_" if policy_id else ""
        return f"{prefix}{self.name}_{chunk_index}"

    def _analyze_text_properties(self, text: str) -> Dict[str, bool]:
        """Analyze text for common properties."""
        import re

        return {
            'has_amounts': bool(re.search(r'€\s*\d+|CHF\s*\d+|\d+\s*EUR|maximum.*\d+', text)),
            'has_conditions': any(word in text.lower() for word in
                                  ['if', 'when', 'unless', 'provided that', 'subject to']),
            'has_exclusions': any(word in text.lower() for word in
                                  ['not covered', 'excluded', 'exception', 'does not apply'])
        }


class ChunkingError(Exception):
    """Custom exception for chunking-related errors."""
    pass
