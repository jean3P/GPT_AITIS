# models/chunking/simple_chunker.py

from typing import List, Dict, Any, Optional
import logging

from .base import ChunkingStrategy, ChunkResult, ChunkMetadata

logger = logging.getLogger(__name__)


class SimpleChunker(ChunkingStrategy):
    """
    Simple paragraph-based chunking strategy.

    This implements your current LocalVectorStore chunking approach adapted
    to follow the ChunkingStrategy interface.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simple chunker.

        Config options:
        - max_length: Maximum chunk length in characters (default: 512)
        - overlap: Number of characters to overlap between chunks (default: 0)
        - preserve_paragraphs: Whether to prefer paragraph boundaries (default: True)
        """
        super().__init__(config)

        # Configuration with defaults matching your LocalVectorStore
        self.max_length = self.config.get('max_length', 512)
        self.overlap = self.config.get('overlap', 0)
        self.preserve_paragraphs = self.config.get('preserve_paragraphs', True)

        logger.info(f"SimpleChunker configured: max_length={self.max_length}, "
                    f"overlap={self.overlap}, preserve_paragraphs={self.preserve_paragraphs}")

    def chunk_text(self, text: str, policy_id: Optional[str] = None,
                   max_length: int = 512) -> List[ChunkResult]:
        """
        Chunk text using your current LocalVectorStore approach.

        This replicates your existing LocalVectorStore.chunk_text() method
        but returns ChunkResult objects instead of just strings.
        """
        # Use configured max_length unless overridden
        chunk_max_length = self.config.get('max_length', max_length)

        # Apply your exact chunking logic
        chunks = self._chunk_text_simple(text, chunk_max_length)

        # Convert to ChunkResult objects
        chunk_results = []
        for i, chunk_text in enumerate(chunks):
            # Create metadata for each chunk
            metadata = ChunkMetadata(
                chunk_id=self._create_chunk_id(policy_id, i),
                policy_id=policy_id,
                chunk_type="simple_paragraph",
                word_count=len(chunk_text.split()),
                has_amounts=self._detect_amounts(chunk_text),
                has_conditions=self._detect_conditions(chunk_text),
                has_exclusions=self._detect_exclusions(chunk_text),
                section=None,  # Simple chunker doesn't identify sections
                coverage_type="general",  # Simple chunker doesn't classify coverage
                confidence_score=1.0,
                extra_data={
                    'chunk_method': 'paragraph_based',
                    'chunk_index': i,
                    'original_length': len(chunk_text)
                }
            )

            chunk_results.append(ChunkResult(
                text=chunk_text,
                metadata=metadata
            ))

        logger.info(f"SimpleChunker created {len(chunk_results)} chunks for policy {policy_id}")
        return chunk_results

    def _chunk_text_simple(self, text: str, max_length: int) -> List[str]:
        """
        Your exact chunking logic from LocalVectorStore.chunk_text()
        """
        if self.preserve_paragraphs:
            return self._paragraph_based_chunking(text, max_length)
        else:
            return self._word_based_chunking(text, max_length)

    def _paragraph_based_chunking(self, text: str, max_length: int) -> List[str]:
        """
        Exact implementation from your LocalVectorStore.
        """
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
            chunks = self._word_based_chunking(text, max_length)

        return chunks

    def _word_based_chunking(self, text: str, max_length: int) -> List[str]:
        """
        Fallback word-based chunking from your LocalVectorStore.
        """
        words = text.split()
        chunks = []

        current_chunk_words = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length <= max_length:
                current_chunk_words.append(word)
                current_length += word_length
            else:
                if current_chunk_words:
                    chunks.append(" ".join(current_chunk_words))
                current_chunk_words = [word]
                current_length = word_length

        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        return chunks if chunks else [text[:max_length]]

    def _detect_amounts(self, text: str) -> bool:
        """Detect if text contains monetary amounts or numbers."""
        import re
        # Look for currency symbols, numbers with currency codes, or percentage
        amount_patterns = [
            r'€\s*\d+',  # Euro amounts
            r'CHF\s*\d+',  # Swiss Franc
            r'\d+\s*EUR',  # EUR suffix
            r'\d+\s*CHF',  # CHF suffix
            r'maximum.*\d+',  # Maximum amounts
            r'\d+%',  # Percentages
            r'\d+\.\d+',  # Decimal numbers
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in amount_patterns)

    def _detect_conditions(self, text: str) -> bool:
        """Detect if text contains conditional language."""
        condition_words = [
            'if', 'when', 'unless', 'provided that', 'subject to',
            'conditional', 'depends on', 'only if', 'in case of',
            'se', 'quando', 'solo se', 'purché'  # Italian equivalents
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in condition_words)

    def _detect_exclusions(self, text: str) -> bool:
        """Detect if text contains exclusion language."""
        exclusion_words = [
            'not covered', 'excluded', 'exception', 'does not apply',
            'limitation', 'restriction', 'not eligible', 'shall not',
            'non coperto', 'escluso', 'eccezione', 'limitazione'  # Italian
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in exclusion_words)

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this chunking strategy."""
        return {
            "name": "simple",
            "description": "Simple paragraph-based chunking adapted from LocalVectorStore",
            "type": "rule_based",
            "complexity": "low",
            "performance": "fast",
            "config": self.config,
            "features": [
                "paragraph_splitting",
                "configurable_max_length",
                "word_based_fallback",
                "basic_content_detection"
            ],
            "best_for": [
                "quick_testing",
                "baseline_comparison",
                "simple_documents",
                "fast_processing",
                "compatibility_with_existing_system"
            ],
            "matches_original": "LocalVectorStore.chunk_text()"
        }

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.max_length <= 0:
            logger.error("max_length must be positive")
            return False

        if self.overlap < 0:
            logger.error("overlap cannot be negative")
            return False

        if self.overlap >= self.max_length:
            logger.error("overlap must be less than max_length")
            return False

        return True
