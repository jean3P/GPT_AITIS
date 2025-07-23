# models/chunking/section_chunker.py

import re
from typing import List, Dict, Any, Optional, Tuple
import logging

from .base import ChunkingStrategy, ChunkResult, ChunkMetadata

logger = logging.getLogger(__name__)


class SectionChunker(ChunkingStrategy):
    """
    Section/Chapter-based chunking strategy for insurance policies.

    Recognizes structural boundaries like "SECTION A", "CHAPTER 1", etc.
    and creates chunks that preserve full legal context while filtering
    out irrelevant front matter.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize section chunker.

        Config options:
        - max_section_length: Maximum length for a section chunk (default: 2000)
        - min_section_length: Minimum length to consider valid section (default: 50)
        - preserve_subsections: Whether to keep subsections together (default: True)
        - include_front_matter: Whether to include document front matter (default: False)
        - sentence_window_size: Size for sentence-based fallback chunking (default: 3)
        """
        super().__init__(config)

        self.max_section_length = self.config.get('max_section_length', 2000)
        self.min_section_length = self.config.get('min_section_length', 50)
        self.preserve_subsections = self.config.get('preserve_subsections', True)
        self.include_front_matter = self.config.get('include_front_matter', False)
        self.sentence_window_size = self.config.get('sentence_window_size', 3)

        # Compile regex patterns for section detection
        self._compile_section_patterns()

        logger.info(f"SectionChunker configured: max_length={self.max_section_length}, "
                    f"preserve_subsections={self.preserve_subsections}")

    def _compile_section_patterns(self):
        """Compile regex patterns for different section types."""

        # Main section patterns (English)
        self.section_patterns = [
            # SECTION A, SECTION B, etc.
            r'^SECTION\s+[A-Z](?:\s*[-–]\s*.*)?$',
            # CHAPTER 1, CHAPTER 2, etc.
            r'^CHAPTER\s+\d+(?:\s*[-–]\s*.*)?$',
            # Article patterns
            r'^ART\.?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*.*)?$',
            r'^ARTICLE\s+\d+(?:\.\d+)?(?:\s*[-–]\s*.*)?$',
        ]

        # Italian patterns
        self.italian_patterns = [
            # SEZIONE A, SEZIONE B
            r'^SEZIONE\s+[A-Z](?:\s*[-–]\s*.*)?$',
            # CAPITOLO 1, CAPITOLO 2
            r'^CAPITOLO\s+\d+(?:\s*[-–]\s*.*)?$',
            # Italian article patterns
            r'^ART\.?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*.*)?$',
            r'^ARTICOLO\s+\d+(?:\.\d+)?(?:\s*[-–]\s*.*)?$',
        ]

        # Coverage-specific patterns (both languages)
        self.coverage_patterns = [
            # What is insured sections
            r'^(?:Che cosa è assicurato\?|What is insured\?).*$',
            r'^(?:Che cosa NON è assicurato\?|What is NOT insured\?).*$',

            # Common coverage types
            r'^(?:Assistenza in Viaggio|Travel Assistance).*$',
            r'^(?:Spese mediche|Medical Expenses).*$',
            r'^(?:Bagaglio|Baggage).*$',
            r'^(?:Annullamento|Cancellation).*$',
            r'^(?:Interruzione|Interruption).*$',
            r'^(?:Ritardo|Delay).*$',

            # Exclusions and limitations
            r'^(?:Esclusioni|Exclusions).*$',
            r'^(?:Limitazioni|Limitations).*$',
            r'^(?:Ci sono limiti|Are there limits).*$',
        ]

        # Compile all patterns
        self.compiled_patterns = []
        for pattern_list in [self.section_patterns, self.italian_patterns, self.coverage_patterns]:
            self.compiled_patterns.extend([re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                                           for pattern in pattern_list])

        # Pattern for identifying front matter to skip
        self.front_matter_patterns = [
            re.compile(r'^(?:INDEX|INDICE)$', re.IGNORECASE),
            re.compile(r'^(?:The Information Set|Il presente documento)', re.IGNORECASE),
            re.compile(r'^(?:Before signing|Prima della sottoscrizione)', re.IGNORECASE),
            re.compile(r'^(?:Last update|Ultimo aggiornamento)', re.IGNORECASE),
        ]

    def chunk_text(self, text: str, policy_id: Optional[str] = None,
                   max_length: int = 512) -> List[ChunkResult]:
        """
        Chunk text using section/chapter boundaries.
        """
        logger.info(f"Starting section-based chunking for policy {policy_id}")

        # Split text into lines for processing
        lines = text.split('\n')

        # Find section boundaries
        sections = self._identify_sections(lines)

        # Create chunks from sections
        chunk_results = []
        for i, section in enumerate(sections):
            section_chunks = self._process_section(section, policy_id, i)
            chunk_results.extend(section_chunks)

        logger.info(f"Created {len(chunk_results)} chunks from {len(sections)} sections")
        return chunk_results

    def _identify_sections(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Identify section boundaries in the text."""
        sections = []
        current_section = None
        section_content = []
        front_matter_ended = not self.include_front_matter

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()

            # Skip empty lines
            if not line_stripped:
                if current_section:
                    section_content.append(line)
                continue

            # Check if this is a section header
            is_section_header, section_type, section_title = self._is_section_header(line_stripped)

            # Check if we're still in front matter
            if not front_matter_ended:
                if is_section_header or self._is_main_content_start(line_stripped):
                    front_matter_ended = True
                else:
                    continue  # Skip front matter

            if is_section_header:
                # Save previous section if it exists
                if current_section and section_content:
                    current_section['content'] = '\n'.join(section_content)
                    if len(current_section['content'].strip()) >= self.min_section_length:
                        sections.append(current_section)

                # Start new section
                current_section = {
                    'title': section_title,
                    'type': section_type,
                    'start_line': line_num,
                    'header': line_stripped,
                    'content': ''
                }
                section_content = []
            else:
                # Add to current section content
                if current_section:
                    section_content.append(line)
                elif front_matter_ended:
                    # Create a default section for content without clear headers
                    current_section = {
                        'title': 'General Content',
                        'type': 'general',
                        'start_line': line_num,
                        'header': 'General Content',
                        'content': ''
                    }
                    section_content = [line]

        # Don't forget the last section
        if current_section and section_content:
            current_section['content'] = '\n'.join(section_content)
            if len(current_section['content'].strip()) >= self.min_section_length:
                sections.append(current_section)

        return sections

    def _is_section_header(self, line: str) -> Tuple[bool, str, str]:
        """
        Check if a line is a section header.
        Returns: (is_header, section_type, clean_title)
        """
        for pattern in self.compiled_patterns:
            if pattern.match(line):
                # Determine section type based on pattern content
                line_lower = line.lower()

                if any(word in line_lower for word in ['section', 'sezione']):
                    section_type = 'section'
                elif any(word in line_lower for word in ['chapter', 'capitolo']):
                    section_type = 'chapter'
                elif any(word in line_lower for word in ['article', 'articolo', 'art']):
                    section_type = 'article'
                elif any(word in line_lower for word in ['esclusioni', 'exclusions']):
                    section_type = 'exclusions'
                elif any(word in line_lower for word in ['assicurato', 'insured', 'covered']):
                    section_type = 'coverage'
                elif any(word in line_lower for word in ['annullamento', 'cancellation']):
                    section_type = 'cancellation'
                elif any(word in line_lower for word in ['bagaglio', 'baggage']):
                    section_type = 'baggage'
                elif any(word in line_lower for word in ['mediche', 'medical']):
                    section_type = 'medical'
                elif any(word in line_lower for word in ['assistenza', 'assistance']):
                    section_type = 'assistance'
                else:
                    section_type = 'general'

                # Clean title (remove extra whitespace, normalize)
                clean_title = ' '.join(line.split())

                return True, section_type, clean_title

        return False, '', ''

    def _is_main_content_start(self, line: str) -> bool:
        """Check if line indicates start of main content (end of front matter)."""
        indicators = [
            'che cosa è assicurato',
            'what is insured',
            'section a',
            'sezione a',
            'definitions',
            'definizioni'
        ]

        line_lower = line.lower()
        return any(indicator in line_lower for indicator in indicators)

    def _process_section(self, section: Dict[str, Any], policy_id: Optional[str],
                         section_index: int) -> List[ChunkResult]:
        """Process a single section into chunks."""

        section_text = section['content'].strip()
        section_header = section['header']
        section_type = section['type']

        # If section is small enough, return as single chunk
        if len(section_text) <= self.max_section_length:
            return [self._create_section_chunk(
                text=f"{section_header}\n\n{section_text}",
                section=section,
                policy_id=policy_id,
                chunk_index=0,
                total_chunks=1
            )]

        # Section is too large, need to split
        if self.preserve_subsections:
            # Try to split on subsection boundaries first
            subsection_chunks = self._split_by_subsections(section_text, section_header)
            if len(subsection_chunks) > 1 and all(len(chunk) <= self.max_section_length
                                                  for chunk in subsection_chunks):
                return [
                    self._create_section_chunk(
                        text=chunk_text,
                        section=section,
                        policy_id=policy_id,
                        chunk_index=i,
                        total_chunks=len(subsection_chunks)
                    )
                    for i, chunk_text in enumerate(subsection_chunks)
                ]

        # Fall back to sentence-based chunking
        sentence_chunks = self._split_by_sentences(section_text, section_header)
        return [
            self._create_section_chunk(
                text=chunk_text,
                section=section,
                policy_id=policy_id,
                chunk_index=i,
                total_chunks=len(sentence_chunks)
            )
            for i, chunk_text in enumerate(sentence_chunks)
        ]

    def _split_by_subsections(self, text: str, header: str) -> List[str]:
        """Split text by subsection markers without duplicating headers."""

        # Look for subsection patterns like "a)", "1.", bullet points, etc.
        subsection_patterns = [
            r'^[a-z]\)\s+',  # a) b) c)
            r'^\d+\.\s+',  # 1. 2. 3.
            r'^•\s+',  # bullet points
            r'^-\s+',  # dashes
            r'^[A-Z]+\.\s+',  # A. B. C.
            r'^ARTICLE\s+\d+',  # ARTICLE 1.1, 1.2, etc.
            r'^ART\.?\s+\d+',  # ART. 1.1, 1.2, etc.
        ]

        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []

        # First chunk gets the header
        first_chunk = True

        for line in lines:
            line_stripped = line.strip()

            # Check if this line starts a new subsection
            is_subsection = False
            for pattern in subsection_patterns:
                if re.match(pattern, line_stripped):
                    is_subsection = True
                    break

            if is_subsection and len('\n'.join(current_chunk_lines)) > 100:
                # Save current chunk
                if first_chunk:
                    # First chunk gets the header
                    chunk_text = f"{header}\n\n" + '\n'.join(current_chunk_lines)
                    first_chunk = False
                else:
                    # Subsequent chunks don't repeat the full header
                    chunk_text = '\n'.join(current_chunk_lines)

                chunks.append(chunk_text)
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)

        # Add final chunk
        if current_chunk_lines:
            if first_chunk:
                # If we never split, include the header
                chunk_text = f"{header}\n\n" + '\n'.join(current_chunk_lines)
            else:
                # This is a continuation chunk
                chunk_text = '\n'.join(current_chunk_lines)
            chunks.append(chunk_text)

        # If we didn't split successfully, return the original with header
        return chunks if len(chunks) > 1 else [f"{header}\n\n{text}"]

    def _split_by_sentences(self, text: str, header: str) -> List[str]:
        """Split text using sentence-based windowing without duplicating headers."""

        # Simple sentence splitting (could be enhanced with spaCy/NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk_sentences = []
        current_length = 0
        first_chunk = True

        for sentence in sentences:
            sentence_length = len(sentence) + 1

            # Account for header length only in first chunk
            header_length = len(header) + 2 if first_chunk else 0
            total_length = current_length + sentence_length + header_length

            if total_length > self.max_section_length and len(current_chunk_sentences) > 0:
                # Save current chunk
                if first_chunk:
                    chunk_text = f"{header}\n\n" + '\n'.join(current_chunk_sentences)
                    first_chunk = False
                else:
                    chunk_text = '\n'.join(current_chunk_sentences)

                chunks.append(chunk_text)
                current_chunk_sentences = [sentence]
                current_length = sentence_length
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length

        # Add final chunk
        if current_chunk_sentences:
            if first_chunk:
                chunk_text = f"{header}\n\n" + '\n'.join(current_chunk_sentences)
            else:
                chunk_text = '\n'.join(current_chunk_sentences)
            chunks.append(chunk_text)

        return chunks if chunks else [f"{header}\n\n{text}"]

    def _create_section_chunk(self, text: str, section: Dict[str, Any],
                              policy_id: Optional[str], chunk_index: int,
                              total_chunks: int) -> ChunkResult:
        """Create a ChunkResult for a section."""

        # Analyze text properties
        text_properties = self._analyze_text_properties(text)

        # Create metadata
        metadata = ChunkMetadata(
            chunk_id=self._create_chunk_id(policy_id, chunk_index),
            policy_id=policy_id,
            chunk_type=f"section_{section['type']}",
            word_count=len(text.split()),
            has_amounts=text_properties['has_amounts'],
            has_conditions=text_properties['has_conditions'],
            has_exclusions=text_properties['has_exclusions'],
            section=section['title'],
            coverage_type=self._infer_coverage_type(section, text),
            confidence_score=self._calculate_confidence_score(section, text),
            extra_data={
                'section_type': section['type'],
                'section_title': section['title'],
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'section_header': section['header'],
                'original_section_length': len(section['content'])
            }
        )

        return ChunkResult(text=text, metadata=metadata)

    def _infer_coverage_type(self, section: Dict[str, Any], text: str) -> str:
        """Infer the type of coverage this section relates to."""

        section_title_lower = section['title'].lower()
        text_lower = text.lower()

        # Map keywords to coverage types
        coverage_mapping = {
            'medical': ['mediche', 'medical', 'spese mediche', 'health'],
            'baggage': ['bagaglio', 'baggage', 'luggage'],
            'cancellation': ['annullamento', 'cancellation', 'cancel'],
            'delay': ['ritardo', 'delay', 'flight delay'],
            'assistance': ['assistenza', 'assistance', 'help'],
            'exclusions': ['esclusioni', 'exclusions', 'not covered'],
            'definitions': ['definitions', 'definizioni', 'glossary'],
            'general_conditions': ['conditions', 'condizioni', 'terms']
        }

        # Check section title first
        for coverage_type, keywords in coverage_mapping.items():
            for keyword in keywords:
                if keyword in section_title_lower:
                    return coverage_type

        # Check text content
        for coverage_type, keywords in coverage_mapping.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:  # Multiple keyword matches
                return coverage_type

        return section['type'] if section['type'] != 'general' else 'general'

    def _calculate_confidence_score(self, section: Dict[str, Any], text: str) -> float:
        """Calculate confidence score for this chunk."""

        score = 1.0

        # Higher confidence for well-structured sections
        if section['type'] in ['section', 'chapter', 'article']:
            score += 0.1

        # Higher confidence for coverage-specific sections
        if any(keyword in section['title'].lower()
               for keyword in ['coverage', 'guarantee', 'benefit', 'assicur']):
            score += 0.1

        # Lower confidence for very short sections
        if len(text) < 100:
            score -= 0.2

        # Higher confidence for sections with clear structure
        if re.search(r'[a-z]\)\s|^\d+\.\s|^•\s', text, re.MULTILINE):
            score += 0.1

        return max(0.0, min(1.0, score))

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this chunking strategy."""
        return {
            "name": "section",
            "description": "Section/Chapter-based chunking for insurance policies with structural awareness",
            "type": "structural",
            "complexity": "medium",
            "performance": "medium",
            "config": self.config,
            "features": [
                "section_boundary_detection",
                "multi_language_support",
                "coverage_type_inference",
                "subsection_preservation",
                "front_matter_filtering",
                "sentence_fallback_chunking",
                "confidence_scoring"
            ],
            "supported_patterns": [
                "SECTION A/B/C",
                "CHAPTER 1/2/3",
                "ARTICLE patterns",
                "Coverage-specific headers",
                "Italian equivalents"
            ],
            "best_for": [
                "insurance_policies",
                "legal_documents",
                "structured_contracts",
                "multi_language_documents",
                "coverage_analysis",
                "eligibility_determination"
            ],
            "expected_improvement": "+10-15pp retrieval precision & justification IoU"
        }

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.max_section_length <= 0:
            logger.error("max_section_length must be positive")
            return False

        if self.min_section_length < 0:
            logger.error("min_section_length cannot be negative")
            return False

        if self.sentence_window_size <= 0:
            logger.error("sentence_window_size must be positive")
            return False

        if self.min_section_length >= self.max_section_length:
            logger.error("min_section_length must be less than max_section_length")
            return False

        return True
