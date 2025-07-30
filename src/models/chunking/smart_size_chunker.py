# models/chunking/smart_size_chunker.py

import re
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from .base import ChunkingStrategy, ChunkResult, ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class ContentSignal:
    """Signals that indicate content importance for chunk sizing."""
    has_amount: bool = False
    has_condition: bool = False
    has_exclusion: bool = False
    has_coverage_statement: bool = False
    has_definition: bool = False
    sentence_count: int = 0
    complexity_score: float = 0.0
    confidence_score: float = 1.0


class SmartSizeChunker(ChunkingStrategy):
    """
    Smart size chunking strategy that adapts chunk size based on content density.

    Key principles:
    - Important content (amounts, conditions, coverage statements) gets larger chunks
    - Simple content gets smaller chunks for efficiency
    - Semantic coherence is prioritized over fixed size limits
    - Preserves complete legal/coverage contexts
    - Uses word-based sizing for consistency with other strategies
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize smart size chunker.

        Config options:
        - base_chunk_words: Base target chunk size in words (default: 80)
        - min_chunk_words: Minimum allowed chunk size in words (default: 20)
        - max_chunk_words: Maximum allowed chunk size in words (default: 200)
        - importance_multiplier: How much to expand important content (default: 1.5)
        - coherence_threshold: Minimum coherence score to maintain (default: 0.7)
        - preserve_complete_clauses: Keep legal clauses intact (default: True)
        - overlap_words: Number of words to overlap between chunks (default: 0)
        """
        super().__init__(config)

        # Use word-based sizing for consistency with other strategies
        self.base_chunk_words = self.config.get('base_chunk_words', 80)
        self.min_chunk_words = self.config.get('min_chunk_words', 20)
        self.max_chunk_words = self.config.get('max_chunk_words', 200)
        self.importance_multiplier = self.config.get('importance_multiplier', 1.5)
        self.coherence_threshold = self.config.get('coherence_threshold', 0.7)
        self.preserve_complete_clauses = self.config.get('preserve_complete_clauses', True)
        self.overlap_words = self.config.get('overlap_words', 0)

        # Compile patterns for content analysis
        self._compile_content_patterns()

        logger.info(f"SmartSizeChunker configured: base_words={self.base_chunk_words}, "
                    f"range=[{self.min_chunk_words}, {self.max_chunk_words}], "
                    f"overlap={self.overlap_words}")

    def _compile_content_patterns(self):
        """Compile regex patterns for content analysis."""

        # Amount patterns (high importance)
        self.amount_patterns = [
            re.compile(r'€\s*\d+(?:[.,]\d+)?', re.IGNORECASE),
            re.compile(r'CHF\s*\d+(?:[.,]\d+)?', re.IGNORECASE),
            re.compile(r'\d+(?:[.,]\d+)?\s*(?:€|EUR|CHF)', re.IGNORECASE),
            re.compile(r'(?:up to|maximum|limit of|fino a)\s*€?\s*\d+', re.IGNORECASE),
            re.compile(r'option\s+\d+\s*€\s*\d+', re.IGNORECASE),
        ]

        # Condition patterns (high importance)
        self.condition_patterns = [
            re.compile(r'\b(?:if|when|unless|provided that|subject to|conditional)\b', re.IGNORECASE),
            re.compile(r'\b(?:se|quando|purché|solo se)\b', re.IGNORECASE),
            re.compile(r'\b(?:only if|in case of|depends on)\b', re.IGNORECASE),
            re.compile(r'\b(?:must|shall|required|mandatory)\b', re.IGNORECASE),
        ]

        # Exclusion patterns (high importance)
        self.exclusion_patterns = [
            re.compile(r'\b(?:not covered|excluded|exception|does not apply)\b', re.IGNORECASE),
            re.compile(r'\b(?:limitation|restriction|not eligible|shall not)\b', re.IGNORECASE),
            re.compile(r'\b(?:non coperto|escluso|eccezione|limitazione)\b', re.IGNORECASE),
            re.compile(r'\b(?:exclude|except|but not|other than)\b', re.IGNORECASE),
        ]

        # Coverage statement patterns (high importance)
        self.coverage_patterns = [
            re.compile(r'\b(?:covered|insured|guarantee|benefit|indemnity)\b', re.IGNORECASE),
            re.compile(r'\b(?:assicurato|coperto|garanzia|beneficio)\b', re.IGNORECASE),
            re.compile(r'\b(?:pays|reimburse|compensate|indemnify)\b', re.IGNORECASE),
            re.compile(r'\b(?:what is (?:covered|insured)|che cosa è assicurato)\b', re.IGNORECASE),
        ]

        # Definition patterns (medium importance)
        self.definition_patterns = [
            re.compile(r'\b(?:means|defined as|refers to|definition)\b', re.IGNORECASE),
            re.compile(r'\b(?:significa|definito come|si riferisce a)\b', re.IGNORECASE),
            re.compile(r'^[A-Z][A-Z\s]+:', re.MULTILINE),  # ALL CAPS definitions
        ]

        # Clause boundary patterns (fixed word boundaries)
        self.clause_boundary_patterns = [
            re.compile(r'^\s*[a-z]\)\s+', re.MULTILINE),  # a) b) c)
            re.compile(r'^\s*\d+\.\s+', re.MULTILINE),  # 1. 2. 3.
            re.compile(r'^\s*[A-Z]\.\s+', re.MULTILINE),  # A. B. C.
            re.compile(r'^\s*\bArticle\s+\d+\b', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*\bSection\s+[A-Z]\b', re.MULTILINE | re.IGNORECASE),
        ]

    def chunk_text(self, text: str, policy_id: Optional[str] = None,
                   max_length: int = 512) -> List[ChunkResult]:
        """
        Chunk text using smart size adaptation.
        Note: max_length parameter is ignored as we use word-based sizing.
        """
        logger.info(f"Starting smart size chunking for policy {policy_id}")

        # Split into sentences for analysis
        sentences = self._split_into_sentences(text)

        # Analyze each sentence for content signals
        sentence_signals = [self._analyze_sentence(sentence) for sentence in sentences]

        # Create chunks based on content importance and coherence
        chunks = self._create_smart_chunks(sentences, sentence_signals)

        # Add overlap if configured
        if self.overlap_words > 0:
            chunks = self._add_overlap(chunks)

        # Convert to ChunkResult objects
        chunk_results = []
        for i, chunk_text in enumerate(chunks):
            chunk_results.append(self._create_chunk_result(
                chunk_text, i, policy_id, len(chunks)
            ))

        logger.info(f"Created {len(chunk_results)} smart-sized chunks")
        return chunk_results

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with improved handling of abbreviations.
        TODO: Consider using nltk.tokenize.sent_tokenize or spaCy for better accuracy.
        """
        # Simple sentence splitting with some abbreviation handling
        # This is a basic implementation - could be enhanced with nltk/spaCy
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Handle common abbreviations that might cause false splits
        merged_sentences = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]

            # Check if this sentence ends with a common abbreviation
            if i < len(sentences) - 1 and self._ends_with_abbreviation(sentence):
                # Merge with next sentence
                sentence = sentence + ' ' + sentences[i + 1]
                i += 2
            else:
                i += 1

            merged_sentences.append(sentence)

        return merged_sentences

    def _ends_with_abbreviation(self, sentence: str) -> bool:
        """Check if sentence ends with common abbreviations."""
        common_abbrevs = ['e.g.', 'i.e.', 'art.', 'sec.', 'etc.', 'vs.', 'cf.']
        sentence_lower = sentence.lower().strip()
        return any(sentence_lower.endswith(abbrev) for abbrev in common_abbrevs)

    def _analyze_sentence(self, sentence: str) -> ContentSignal:
        """Analyze a sentence for content importance signals."""
        signal = ContentSignal()

        # Check for amounts
        signal.has_amount = any(pattern.search(sentence) for pattern in self.amount_patterns)

        # Check for conditions
        signal.has_condition = any(pattern.search(sentence) for pattern in self.condition_patterns)

        # Check for exclusions
        signal.has_exclusion = any(pattern.search(sentence) for pattern in self.exclusion_patterns)

        # Check for coverage statements
        signal.has_coverage_statement = any(pattern.search(sentence) for pattern in self.coverage_patterns)

        # Check for definitions
        signal.has_definition = any(pattern.search(sentence) for pattern in self.definition_patterns)

        # Count sentences (for now, just 1)
        signal.sentence_count = 1

        # Calculate complexity and confidence scores
        signal.complexity_score = self._calculate_complexity_score(sentence, signal)
        signal.confidence_score = self._calculate_confidence_score(sentence, signal)

        return signal

    def _calculate_complexity_score(self, text: str, signal: ContentSignal) -> float:
        """Calculate complexity score for text (can be sentence or chunk)."""
        score = 0.0

        # Base score from content signals
        if signal.has_amount:
            score += 0.3
        if signal.has_condition:
            score += 0.2
        if signal.has_exclusion:
            score += 0.25
        if signal.has_coverage_statement:
            score += 0.2
        if signal.has_definition:
            score += 0.15

        # Length-based complexity (normalize by sentence count for multi-sentence chunks)
        word_count = len(text.split())
        avg_words_per_sentence = word_count / max(1, signal.sentence_count)

        if avg_words_per_sentence > 30:
            score += 0.1
        elif avg_words_per_sentence < 10:
            score -= 0.1

        # Punctuation complexity (normalize by sentence count)
        complex_punct = text.count(',') + text.count('(') + text.count(';')
        punct_per_sentence = complex_punct / max(1, signal.sentence_count)

        if punct_per_sentence > 3:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _calculate_confidence_score(self, text: str, signal: ContentSignal) -> float:
        """Calculate confidence score (different from complexity)."""
        confidence = 1.0

        # Higher confidence for structured content
        if signal.has_amount or signal.has_coverage_statement:
            confidence += 0.1

        # Lower confidence for very short or very long sentences
        word_count = len(text.split())
        if word_count < 5:
            confidence -= 0.2
        elif word_count > 50:
            confidence -= 0.1

        # Higher confidence for clear structure
        if any(pattern.search(text) for pattern in self.clause_boundary_patterns):
            confidence += 0.05

        return max(0.0, min(1.0, confidence))

    def _create_smart_chunks(self, sentences: List[str], signals: List[ContentSignal]) -> List[str]:
        """Create chunks with smart size adaptation."""
        chunks = []
        current_chunk_sentences = []
        current_chunk_words = 0
        current_importance = 0.0

        for i, (sentence, signal) in enumerate(zip(sentences, signals)):
            sentence_words = len(sentence.split())

            # Calculate target chunk size based on importance
            target_words = self._calculate_target_chunk_words(current_importance, signal)

            # Check if we should start a new chunk
            should_split = self._should_split_chunk(
                current_chunk_words, sentence_words, target_words,
                current_chunk_sentences, sentence, signal
            )

            if should_split and current_chunk_sentences:
                # Save current chunk
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = []
                current_chunk_words = 0
                current_importance = 0.0

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_words += sentence_words
            current_importance = max(current_importance, signal.complexity_score)

        # Add final chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))

        return chunks

    def _calculate_target_chunk_words(self, current_importance: float,
                                      new_signal: ContentSignal) -> int:
        """Calculate target chunk size in words based on content importance."""
        # Use the higher importance between current and new content
        max_importance = max(current_importance, new_signal.complexity_score)

        # Scale base size by importance
        if max_importance > 0.6:  # High importance
            target_words = int(self.base_chunk_words * self.importance_multiplier)
        elif max_importance > 0.3:  # Medium importance
            target_words = int(self.base_chunk_words * 1.2)
        else:  # Low importance
            target_words = int(self.base_chunk_words * 0.8)

        # Ensure within bounds and above minimum for embedding quality
        target_words = max(self.min_chunk_words, min(self.max_chunk_words, target_words))

        return target_words

    def _should_split_chunk(self, current_words: int, sentence_words: int,
                            target_words: int, current_sentences: List[str],
                            new_sentence: str, new_signal: ContentSignal) -> bool:
        """Determine if we should split the current chunk."""

        # If we haven't reached minimum size, don't split
        if current_words < self.min_chunk_words:
            return False

        # If adding this sentence would exceed max size, split
        if current_words + sentence_words > self.max_chunk_words:
            return True

        # If we're over target size, consider splitting
        if current_words > target_words:
            # Don't split if this would break clause coherence
            if self.preserve_complete_clauses:
                if self._would_break_clause_coherence(current_sentences, new_sentence):
                    return False
            return True

        # Check for natural boundaries (section breaks, etc.)
        if self._is_natural_boundary(new_sentence):
            return current_words > self.min_chunk_words

        return False

    def _would_break_clause_coherence(self, current_sentences: List[str],
                                      new_sentence: str) -> bool:
        """Check if splitting would break clause coherence."""
        if not current_sentences:
            return False

        # Check if current chunk ends with incomplete clause indicators
        last_sentence = current_sentences[-1]

        # Indicators that the clause continues
        continuation_indicators = [
            'and', 'or', 'but', 'however', 'provided that',
            'subject to', 'unless', 'if', 'when', 'where'
        ]

        # Strip punctuation for better matching
        last_sentence_clean = re.sub(r'[.,;:()]+\s*$', '', last_sentence.lower().strip())

        # If last sentence ends with continuation indicators, don't split
        for indicator in continuation_indicators:
            if last_sentence_clean.endswith(indicator):
                return True

        # Check if new sentence starts with continuation words
        new_sentence_lower = new_sentence.lower().strip()
        continuation_starts = ['and', 'or', 'but', 'however', 'therefore', 'thus']

        for start in continuation_starts:
            if new_sentence_lower.startswith(start):
                return True

        return False

    def _is_natural_boundary(self, sentence: str) -> bool:
        """Check if sentence represents a natural boundary."""
        # Check for clause boundary patterns
        for pattern in self.clause_boundary_patterns:
            if pattern.match(sentence):
                return True

        # Check for section/article headers with word boundaries
        if re.search(r'\b(?:SECTION|ARTICLE|CHAPTER)\b', sentence, re.IGNORECASE):
            return True

        return False

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add word-based overlap between chunks."""
        if len(chunks) <= 1 or self.overlap_words <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no prefix overlap
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk_words = chunks[i - 1].split()
                overlap_words = prev_chunk_words[-self.overlap_words:]

                # Add overlap to current chunk
                current_chunk_words = chunk.split()
                overlapped_chunk = ' '.join(overlap_words + current_chunk_words)
                overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _create_chunk_result(self, text: str, chunk_index: int, policy_id: Optional[str],
                             total_chunks: int) -> ChunkResult:
        """Create a ChunkResult with smart size metadata."""

        # Analyze the chunk for properties
        chunk_signal = self._analyze_chunk(text)

        # Create metadata
        metadata = ChunkMetadata(
            chunk_id=self._create_chunk_id(policy_id, chunk_index),
            policy_id=policy_id,
            chunk_type="smart_size",
            word_count=len(text.split()),
            has_amounts=chunk_signal.has_amount,
            has_conditions=chunk_signal.has_condition,
            has_exclusions=chunk_signal.has_exclusion,
            section=None,  # Could be enhanced to detect sections
            coverage_type=self._infer_coverage_type(text),
            confidence_score=chunk_signal.confidence_score,
            extra_data={
                'chunk_method': 'smart_size',
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'target_words': self._calculate_target_chunk_words(0, chunk_signal),
                'actual_words': len(text.split()),
                'complexity_score': chunk_signal.complexity_score,
                'has_coverage_statement': chunk_signal.has_coverage_statement,
                'has_definition': chunk_signal.has_definition,
                'sentence_count': chunk_signal.sentence_count,
                'overlap_words': self.overlap_words
            }
        )

        return ChunkResult(text=text, metadata=metadata)

    def _analyze_chunk(self, text: str) -> ContentSignal:
        """Analyze an entire chunk for content signals."""
        signal = ContentSignal()

        # Check for amounts
        signal.has_amount = any(pattern.search(text) for pattern in self.amount_patterns)

        # Check for conditions
        signal.has_condition = any(pattern.search(text) for pattern in self.condition_patterns)

        # Check for exclusions
        signal.has_exclusion = any(pattern.search(text) for pattern in self.exclusion_patterns)

        # Check for coverage statements
        signal.has_coverage_statement = any(pattern.search(text) for pattern in self.coverage_patterns)

        # Check for definitions
        signal.has_definition = any(pattern.search(text) for pattern in self.definition_patterns)

        # Count sentences
        signal.sentence_count = len(self._split_into_sentences(text))

        # Calculate complexity and confidence scores
        signal.complexity_score = self._calculate_complexity_score(text, signal)
        signal.confidence_score = self._calculate_confidence_score(text, signal)

        return signal

    def _infer_coverage_type(self, text: str) -> str:
        """Infer coverage type from chunk content."""
        text_lower = text.lower()

        # Coverage type mapping
        coverage_keywords = {
            'medical': ['medical', 'mediche', 'hospital', 'doctor', 'treatment'],
            'baggage': ['baggage', 'bagaglio', 'luggage', 'suitcase'],
            'cancellation': ['cancellation', 'annullamento', 'cancel', 'trip'],
            'delay': ['delay', 'ritardo', 'late', 'postpone'],
            'assistance': ['assistance', 'assistenza', 'help', 'support'],
            'exclusions': ['exclusion', 'esclusione', 'not covered', 'except'],
            'definitions': ['definition', 'definizione', 'means', 'refers to']
        }

        for coverage_type, keywords in coverage_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return coverage_type

        return 'general'

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this chunking strategy."""
        return {
            "name": "smart_size",
            "description": "Adaptive word-based chunking that adjusts size based on content importance and coherence",
            "type": "adaptive",
            "complexity": "high",
            "performance": "medium",
            "config": self.config,
            "features": [
                "content_importance_analysis",
                "adaptive_word_based_sizing",
                "clause_coherence_preservation",
                "natural_boundary_detection",
                "multi_signal_analysis",
                "complexity_and_confidence_scoring",
                "optional_overlap_support"
            ],
            "content_signals": [
                "monetary_amounts",
                "conditions_and_prerequisites",
                "exclusions_and_limitations",
                "coverage_statements",
                "definitions_and_terms",
                "clause_boundaries"
            ],
            "size_adaptation": {
                "base_words": self.base_chunk_words,
                "word_range": [self.min_chunk_words, self.max_chunk_words],
                "importance_multiplier": self.importance_multiplier,
                "overlap_words": self.overlap_words
            },
            "best_for": [
                "insurance_policy_analysis",
                "legal_document_processing",
                "coverage_determination",
                "amount_extraction",
                "condition_analysis"
            ],
            "improvements_over_v1": [
                "word_based_sizing_consistency",
                "fixed_clause_coherence_detection",
                "improved_sentence_splitting",
                "separate_complexity_confidence_scores",
                "optional_overlap_support",
                "better_natural_boundary_detection"
            ],
            "expected_improvement": "Enhanced retrieval precision for complex queries with better context preservation"
        }

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.min_chunk_words <= 0:
            logger.error("min_chunk_words must be positive")
            return False

        if self.max_chunk_words <= self.min_chunk_words:
            logger.error("max_chunk_words must be greater than min_chunk_words")
            return False

        if self.base_chunk_words < self.min_chunk_words or self.base_chunk_words > self.max_chunk_words:
            logger.error("base_chunk_words must be between min and max chunk words")
            return False

        if self.importance_multiplier <= 0:
            logger.error("importance_multiplier must be positive")
            return False

        if self.overlap_words < 0:
            logger.error("overlap_words cannot be negative")
            return False

        if self.overlap_words >= self.min_chunk_words:
            logger.error("overlap_words must be less than min_chunk_words")
            return False

        return True
