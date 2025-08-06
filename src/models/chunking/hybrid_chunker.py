# models/chunking/hybrid_chunker.py

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .base import ChunkingStrategy, ChunkResult, ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class HybridSection:
    """Represents a section with both structural and semantic information."""
    header: str
    content: str
    section_type: str
    start_position: int
    end_position: int
    embedding: Optional[np.ndarray] = None
    subsections: List['HybridSection'] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []


class HybridChunker(ChunkingStrategy):
    """
    Advanced hybrid chunking strategy that combines:
    1. Structural awareness (sections, chapters, articles)
    2. Semantic coherence (embedding-based similarity)
    3. Cross-reference detection
    4. Coverage-specific optimization

    This strategy is specifically optimized for insurance policies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Configuration
        self.max_chunk_words = self.config.get('max_chunk_words', 250)
        self.min_chunk_words = self.config.get('min_chunk_words', 50)
        self.overlap_words = self.config.get('overlap_words', 20)
        self.semantic_threshold = self.config.get('semantic_threshold', 0.75)
        self.embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.include_cross_references = self.config.get('include_cross_references', True)
        self.preserve_tables = self.config.get('preserve_tables', True)

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

        # Compile patterns
        self._compile_patterns()

        # Coverage type keywords for insurance domain
        self.coverage_keywords = {
            'baggage': ['baggage', 'luggage', 'bagaglio', 'belongings', 'suitcase', 'personal effects'],
            'medical': ['medical', 'hospital', 'doctor', 'mediche', 'treatment', 'illness', 'injury', 'health'],
            'cancellation': ['cancellation', 'annullamento', 'cancel', 'refund', 'rinuncia', 'trip cancellation'],
            'delay': ['delay', 'ritardo', 'late', 'postpone', 'delayed', 'postponed'],
            'assistance': ['assistance', 'assistenza', 'help', 'support', 'emergency', '24/7', 'helpline'],
            'exclusion': ['excluded', 'not covered', 'escluso', 'exception', 'esclusioni', 'limitation'],
            'general': ['general', 'definitions', 'definizioni', 'terms', 'conditions']
        }

        logger.info(f"HybridChunker initialized with max_words={self.max_chunk_words}")

    def _compile_patterns(self):
        """Compile regex patterns for structural and content detection."""

        # Main section patterns (multilingual)
        self.section_patterns = [
            # English patterns
            (r'^SECTION\s+[A-Z](?:\s*[-–]\s*(.*))?$', 'section'),
            (r'^CHAPTER\s+\d+(?:\s*[-–]\s*(.*))?$', 'chapter'),
            (r'^ARTICLE?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*(.*))?$', 'article'),

            # Italian patterns
            (r'^SEZIONE\s+[A-Z](?:\s*[-–]\s*(.*))?$', 'section'),
            (r'^CAPITOLO\s+\d+(?:\s*[-–]\s*(.*))?$', 'chapter'),
            (r'^ARTICOLO?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*(.*))?$', 'article'),

            # Coverage-specific patterns
            (r'^(?:What is |Che cosa è )?(?:insured|assicurato|covered|coperto).*\??$', 'coverage'),
            (r'^(?:What is NOT |Che cosa NON è )?(?:insured|assicurato|covered|coperto).*\??$', 'exclusion'),
            (r'^(?:Exclusions?|Esclusioni?)(?:\s*[-–]\s*(.*))?$', 'exclusion'),
            (r'^(?:Definitions?|Definizioni?)(?:\s*[-–]\s*(.*))?$', 'definition'),

            # Subsection patterns
            (r'^\s*[a-z]\)\s+(.*)$', 'subsection'),
            (r'^\s*\d+\.\s+(.*)$', 'subsection'),
            (r'^\s*[A-Z]\.\s+(.*)$', 'subsection'),
            (r'^\s*[-•]\s+(.*)$', 'subsection'),
        ]

        # Compile patterns
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), type_)
            for pattern, type_ in self.section_patterns
        ]

        # Cross-reference patterns
        self.xref_patterns = [
            re.compile(r'(?:see|vedere|cfr\.?)\s+(?:article|articolo|section|sezione)\s+(\d+(?:\.\d+)?)',
                       re.IGNORECASE),
            re.compile(r'(?:as per|secondo|come da)\s+(?:article|articolo|section|sezione)\s+(\d+(?:\.\d+)?)',
                       re.IGNORECASE),
            re.compile(r'(?:refer to|riferimento a)\s+(?:article|articolo|section|sezione)\s+(\d+(?:\.\d+)?)',
                       re.IGNORECASE),
        ]

        # Table detection patterns
        self.table_patterns = [
            re.compile(r'^\s*(?:Option|Opzione)\s+\d+.*€\s*\d+', re.IGNORECASE | re.MULTILINE),
            re.compile(r'^\s*\|.*\|.*\|', re.MULTILINE),  # Markdown tables
            re.compile(r'(?:Amount|Importo|Limit|Limite).*\n.*€\s*\d+', re.IGNORECASE),
        ]

    def chunk_text(self, text: str, policy_id: Optional[str] = None,
                   max_length: int = 512) -> List[ChunkResult]:
        """
        Create hybrid chunks that combine structural and semantic approaches.
        """
        logger.info(f"Starting hybrid chunking for policy {policy_id}")

        # Step 1: Extract hierarchical structure
        sections = self._extract_hierarchical_sections(text)
        logger.info(f"Extracted {len(sections)} top-level sections")

        # Step 2: Generate embeddings for sections if model available
        if self.embedding_model:
            self._generate_section_embeddings(sections)

        # Step 3: Create initial chunks from sections
        chunks = []
        for section_idx, section in enumerate(sections):
            section_chunks = self._process_section(section, policy_id, section_idx)
            chunks.extend(section_chunks)

        # Step 4: Add cross-reference chunks if enabled
        if self.include_cross_references:
            xref_chunks = self._create_cross_reference_chunks(sections, chunks, policy_id)
            chunks.extend(xref_chunks)

        # Step 5: Add overlap between chunks
        if self.overlap_words > 0:
            chunks = self._add_chunk_overlap(chunks)

        logger.info(f"Created {len(chunks)} hybrid chunks")
        return chunks

    def _extract_hierarchical_sections(self, text: str) -> List[HybridSection]:
        """Extract sections with hierarchical structure."""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        current_position = 0

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()

            # Skip empty lines
            if not line_stripped:
                if current_section:
                    current_content.append(line)
                current_position += len(line) + 1
                continue

            # Check if this is a section header
            section_info = self._identify_section_header(line_stripped)

            if section_info:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content)
                    current_section.end_position = current_position
                    if self._is_valid_section(current_section):
                        sections.append(current_section)

                # Start new section
                header, section_type = section_info
                current_section = HybridSection(
                    header=header,
                    content='',
                    section_type=section_type,
                    start_position=current_position,
                    end_position=current_position
                )
                current_content = []
            else:
                current_content.append(line)

            current_position += len(line) + 1

        # Don't forget the last section
        if current_section:
            current_section.content = '\n'.join(current_content)
            current_section.end_position = current_position
            if self._is_valid_section(current_section):
                sections.append(current_section)

        # Extract subsections for each section
        for section in sections:
            section.subsections = self._extract_subsections(section)

        return sections

    def _identify_section_header(self, line: str) -> Optional[Tuple[str, str]]:
        """Check if a line is a section header and return (header, type)."""
        for pattern, section_type in self.compiled_patterns:
            match = pattern.match(line)
            if match:
                return line, section_type
        return None

    def _extract_subsections(self, section: HybridSection) -> List[HybridSection]:
        """Extract subsections from a section's content."""
        subsections = []
        lines = section.content.split('\n')
        current_subsection = None
        current_content = []

        for line in lines:
            # Check for subsection patterns
            for pattern, type_ in self.compiled_patterns:
                if type_ == 'subsection':
                    match = pattern.match(line)
                    if match:
                        # Save previous subsection
                        if current_subsection:
                            current_subsection.content = '\n'.join(current_content)
                            if len(current_subsection.content.strip()) > 20:
                                subsections.append(current_subsection)

                        # Start new subsection
                        current_subsection = HybridSection(
                            header=line.strip(),
                            content='',
                            section_type='subsection',
                            start_position=0,
                            end_position=0
                        )
                        current_content = []
                        break
            else:
                if current_subsection:
                    current_content.append(line)

        # Add last subsection
        if current_subsection and current_content:
            current_subsection.content = '\n'.join(current_content)
            if len(current_subsection.content.strip()) > 20:
                subsections.append(current_subsection)

        return subsections

    def _is_valid_section(self, section: HybridSection) -> bool:
        """Check if a section has enough content to be valid."""
        word_count = len(section.content.split())
        return word_count >= 10  # Minimum 10 words

    def _generate_section_embeddings(self, sections: List[HybridSection]):
        """Generate embeddings for all sections."""
        if not self.embedding_model:
            return

        for section in sections:
            # Generate embedding for section
            section_text = f"{section.header}\n{section.content[:500]}"  # Use header + first 500 chars
            section.embedding = self.embedding_model.encode(section_text)

            # Generate embeddings for subsections
            for subsection in section.subsections:
                subsection_text = f"{subsection.header}\n{subsection.content[:300]}"
                subsection.embedding = self.embedding_model.encode(subsection_text)

    def _process_section(self, section: HybridSection, policy_id: str,
                         section_idx: int) -> List[ChunkResult]:
        """Process a section into chunks."""
        chunks = []

        # Determine if section is small enough to be a single chunk
        word_count = len(section.content.split())

        if word_count <= self.max_chunk_words:
            # Create single chunk for the section
            chunk = self._create_section_chunk(
                section, policy_id, f"{section_idx}_0",
                include_header=True
            )
            chunks.append(chunk)
        else:
            # Split section into multiple chunks
            if section.subsections:
                # Use subsections as natural boundaries
                chunks.extend(self._chunk_by_subsections(section, policy_id, section_idx))
            else:
                # Use semantic chunking within the section
                chunks.extend(self._chunk_by_semantics(section, policy_id, section_idx))

        return chunks

    def _chunk_by_subsections(self, section: HybridSection, policy_id: str,
                              section_idx: int) -> List[ChunkResult]:
        """Create chunks based on subsections."""
        chunks = []

        # Create a chunk for the section introduction (before first subsection)
        intro_chunk = self._create_intro_chunk(section, policy_id, f"{section_idx}_intro")
        if intro_chunk:
            chunks.append(intro_chunk)

        # Group subsections into chunks
        current_chunk_subsections = []
        current_word_count = 0

        for i, subsection in enumerate(section.subsections):
            subsection_words = len(subsection.content.split())

            # Check if adding this subsection would exceed max size
            if current_word_count + subsection_words > self.max_chunk_words and current_chunk_subsections:
                # Create chunk from current subsections
                chunk = self._create_subsection_chunk(
                    section, current_chunk_subsections, policy_id,
                    f"{section_idx}_{len(chunks)}"
                )
                chunks.append(chunk)

                # Start new chunk
                current_chunk_subsections = [subsection]
                current_word_count = subsection_words
            else:
                # Add to current chunk
                current_chunk_subsections.append(subsection)
                current_word_count += subsection_words

        # Create final chunk
        if current_chunk_subsections:
            chunk = self._create_subsection_chunk(
                section, current_chunk_subsections, policy_id,
                f"{section_idx}_{len(chunks)}"
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_semantics(self, section: HybridSection, policy_id: str,
                            section_idx: int) -> List[ChunkResult]:
        """Create chunks using semantic similarity."""
        chunks = []

        # Split into sentences
        sentences = self._split_into_sentences(section.content)

        if not sentences:
            return chunks

        # Generate embeddings if available
        if self.embedding_model:
            embeddings = self.embedding_model.encode(sentences)
            chunks_data = self._create_semantic_chunks(
                sentences, embeddings, section.header
            )
        else:
            # Fallback to simple sentence grouping
            chunks_data = self._create_simple_chunks(sentences, section.header)

        # Convert to ChunkResult objects
        for i, chunk_data in enumerate(chunks_data):
            chunk_text = chunk_data['text']
            chunk = self._create_chunk_result(
                text=chunk_text,
                section=section,
                policy_id=policy_id,
                chunk_id=f"{section_idx}_{i}",
                metadata_extra={
                    'semantic_score': chunk_data.get('score', 0.0),
                    'sentence_count': chunk_data.get('sentence_count', 0)
                }
            )
            chunks.append(chunk)

        return chunks

    def _create_semantic_chunks(self, sentences: List[str], embeddings: np.ndarray,
                                section_header: str) -> List[Dict[str, Any]]:
        """Create chunks based on semantic similarity."""
        chunks_data = []
        current_chunk = []
        current_embedding = None

        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if not current_chunk:
                # Start new chunk
                current_chunk = [sentence]
                current_embedding = embedding
            else:
                # Calculate similarity with current chunk
                similarity = cosine_similarity(
                    [current_embedding],
                    [embedding]
                )[0][0]

                # Check if we should add to current chunk
                current_words = sum(len(s.split()) for s in current_chunk)
                sentence_words = len(sentence.split())

                if (similarity >= self.semantic_threshold and
                        current_words + sentence_words <= self.max_chunk_words):
                    # Add to current chunk
                    current_chunk.append(sentence)
                    # Update embedding (average)
                    current_embedding = (current_embedding * len(current_chunk) + embedding) / (len(current_chunk) + 1)
                else:
                    # Save current chunk and start new one
                    chunk_text = f"{section_header}\n\n{' '.join(current_chunk)}"
                    chunks_data.append({
                        'text': chunk_text,
                        'score': similarity,
                        'sentence_count': len(current_chunk)
                    })

                    current_chunk = [sentence]
                    current_embedding = embedding

        # Add final chunk
        if current_chunk:
            chunk_text = f"{section_header}\n\n{' '.join(current_chunk)}"
            chunks_data.append({
                'text': chunk_text,
                'score': 1.0,
                'sentence_count': len(current_chunk)
            })

        return chunks_data

    def _create_simple_chunks(self, sentences: List[str], section_header: str) -> List[Dict[str, Any]]:
        """Create chunks without embeddings."""
        chunks_data = []
        current_chunk = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_words + sentence_words > self.max_chunk_words and current_chunk:
                # Save current chunk
                chunk_text = f"{section_header}\n\n{' '.join(current_chunk)}"
                chunks_data.append({
                    'text': chunk_text,
                    'score': 0.0,
                    'sentence_count': len(current_chunk)
                })

                current_chunk = [sentence]
                current_words = sentence_words
            else:
                current_chunk.append(sentence)
                current_words += sentence_words

        # Add final chunk
        if current_chunk:
            chunk_text = f"{section_header}\n\n{' '.join(current_chunk)}"
            chunks_data.append({
                'text': chunk_text,
                'score': 0.0,
                'sentence_count': len(current_chunk)
            })

        return chunks_data

    def _create_cross_reference_chunks(self, sections: List[HybridSection],
                                       existing_chunks: List[ChunkResult],
                                       policy_id: str) -> List[ChunkResult]:
        """Create chunks that link related sections."""
        xref_chunks = []

        # Find sections that reference each other
        references = self._find_cross_references(sections)

        # Create chunks for important cross-references
        for ref in references:
            if self._is_important_reference(ref):
                xref_chunk = self._create_xref_chunk(ref, policy_id, len(xref_chunks))
                if xref_chunk:
                    xref_chunks.append(xref_chunk)

        # Find coverage-exclusion pairs
        coverage_exclusion_pairs = self._find_coverage_exclusion_pairs(sections)

        for pair in coverage_exclusion_pairs:
            pair_chunk = self._create_coverage_exclusion_chunk(
                pair, policy_id, len(xref_chunks)
            )
            if pair_chunk:
                xref_chunks.append(pair_chunk)

        logger.info(f"Created {len(xref_chunks)} cross-reference chunks")
        return xref_chunks

    def _find_cross_references(self, sections: List[HybridSection]) -> List[Dict[str, Any]]:
        """Find cross-references between sections."""
        references = []

        for i, section in enumerate(sections):
            # Search for references in section content
            for pattern in self.xref_patterns:
                for match in pattern.finditer(section.content):
                    ref_target = match.group(1)

                    # Find target section
                    target_section = self._find_section_by_number(sections, ref_target)
                    if target_section:
                        references.append({
                            'source': section,
                            'target': target_section,
                            'reference': match.group(0),
                            'context': section.content[max(0, match.start() - 50):match.end() + 50]
                        })

        return references

    def _find_coverage_exclusion_pairs(self, sections: List[HybridSection]) -> List[
        Tuple[HybridSection, HybridSection]]:
        """Find related coverage and exclusion sections."""
        pairs = []

        coverage_sections = [s for s in sections if s.section_type in ['coverage', 'section']
                             and self._is_coverage_section(s)]
        exclusion_sections = [s for s in sections if s.section_type == 'exclusion'
                              or 'exclusion' in s.header.lower()]

        for coverage in coverage_sections:
            # Find related exclusions
            coverage_type = self._detect_coverage_type(coverage)

            for exclusion in exclusion_sections:
                if self._sections_are_related(coverage, exclusion, coverage_type):
                    pairs.append((coverage, exclusion))

        return pairs

    def _create_xref_chunk(self, reference: Dict[str, Any], policy_id: str,
                           chunk_idx: int) -> Optional[ChunkResult]:
        """Create a chunk for a cross-reference."""
        source = reference['source']
        target = reference['target']

        # Create combined text
        combined_text = (
            f"CROSS-REFERENCE: {source.header} → {target.header}\n\n"
            f"FROM {source.header}:\n"
            f"{reference['context']}\n\n"
            f"REFERENCED {target.header}:\n"
            f"{target.content[:300]}..."
        )

        # Create metadata
        metadata = ChunkMetadata(
            chunk_id=self._create_chunk_id(policy_id, f"xref_{chunk_idx}"),
            policy_id=policy_id,
            chunk_type="cross_reference",
            word_count=len(combined_text.split()),
            has_amounts=self._has_amounts(combined_text),
            has_conditions=self._has_conditions(combined_text),
            has_exclusions=self._has_exclusions(combined_text),
            section=f"CrossRef: {source.header}",
            coverage_type=self._detect_coverage_type_from_text(combined_text),
            confidence_score=0.9,
            extra_data={
                'reference_type': 'cross_reference',
                'source_section': source.header,
                'target_section': target.header,
                'is_hybrid': True
            }
        )

        return ChunkResult(text=combined_text, metadata=metadata)

    def _create_coverage_exclusion_chunk(self, pair: Tuple[HybridSection, HybridSection],
                                         policy_id: str, chunk_idx: int) -> Optional[ChunkResult]:
        """Create a chunk linking coverage with its exclusions."""
        coverage, exclusion = pair
        coverage_type = self._detect_coverage_type(coverage)

        # Extract key parts
        coverage_summary = self._extract_key_sentences(coverage.content, max_sentences=3)
        exclusion_summary = self._extract_key_sentences(exclusion.content, max_sentences=3)

        combined_text = (
            f"COVERAGE & EXCLUSIONS: {coverage_type.upper()}\n\n"
            f"WHAT IS COVERED ({coverage.header}):\n"
            f"{coverage_summary}\n\n"
            f"WHAT IS NOT COVERED ({exclusion.header}):\n"
            f"{exclusion_summary}"
        )

        # Create metadata
        metadata = ChunkMetadata(
            chunk_id=self._create_chunk_id(policy_id, f"pair_{chunk_idx}"),
            policy_id=policy_id,
            chunk_type="coverage_exclusion_pair",
            word_count=len(combined_text.split()),
            has_amounts=self._has_amounts(combined_text),
            has_conditions=self._has_conditions(combined_text),
            has_exclusions=True,  # Always true for these chunks
            section=f"Coverage-Exclusion: {coverage_type}",
            coverage_type=coverage_type,
            confidence_score=0.95,
            extra_data={
                'reference_type': 'coverage_exclusion_pair',
                'coverage_section': coverage.header,
                'exclusion_section': exclusion.header,
                'is_complete_context': True,
                'is_hybrid': True
            }
        )

        return ChunkResult(text=combined_text, metadata=metadata)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be enhanced with NLTK/spaCy
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _detect_coverage_type(self, section: HybridSection) -> str:
        """Detect the type of coverage from a section."""
        text = f"{section.header} {section.content}".lower()

        scores = {}
        for coverage_type, keywords in self.coverage_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[coverage_type] = score

        if scores:
            return max(scores, key=scores.get)
        return 'general'

    def _detect_coverage_type_from_text(self, text: str) -> str:
        """Detect coverage type from text."""
        text_lower = text.lower()

        scores = {}
        for coverage_type, keywords in self.coverage_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[coverage_type] = score

        if scores:
            return max(scores, key=scores.get)
        return 'general'

    def _is_coverage_section(self, section: HybridSection) -> bool:
        """Check if a section describes coverage."""
        indicators = ['covered', 'insured', 'coperto', 'assicurato', 'guarantee', 'benefit']
        text_lower = f"{section.header} {section.content[:200]}".lower()
        return any(indicator in text_lower for indicator in indicators)

    def _sections_are_related(self, section1: HybridSection, section2: HybridSection,
                              coverage_type: str = None) -> bool:
        """Check if two sections are related."""
        # Check if they mention the same coverage type
        if coverage_type:
            keywords = self.coverage_keywords.get(coverage_type, [])
            section2_text = f"{section2.header} {section2.content}".lower()
            if any(keyword in section2_text for keyword in keywords):
                return True

        # Check for explicit references
        if any(section1.header in section2.content or section2.header in section1.content
               for _ in [1]):
            return True

        # Check embedding similarity if available
        if self.embedding_model and section1.embedding is not None and section2.embedding is not None:
            similarity = cosine_similarity([section1.embedding], [section2.embedding])[0][0]
            return similarity > 0.7

        return False

    def _extract_key_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Extract the most important sentences from text."""
        sentences = self._split_into_sentences(text)

        if len(sentences) <= max_sentences:
            return ' '.join(sentences)

        # Simple importance scoring based on keywords
        important_keywords = ['must', 'shall', 'covered', 'not covered', 'excluded',
                              'required', 'maximum', 'limit', '€', 'EUR', 'CHF']

        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for keyword in important_keywords if keyword in sentence.lower())
            scored_sentences.append((score, sentence))

        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        selected = [sent for _, sent in scored_sentences[:max_sentences]]

        # Return in original order
        return ' '.join(sent for sent in sentences if sent in selected)

    def _find_section_by_number(self, sections: List[HybridSection], number: str) -> Optional[HybridSection]:
        """Find a section by its number."""
        for section in sections:
            if number in section.header:
                return section
        return None

    def _is_important_reference(self, reference: Dict[str, Any]) -> bool:
        """Check if a cross-reference is important enough to create a chunk."""
        # References to exclusions or conditions are always important
        important_keywords = ['exclusion', 'condition', 'requirement', 'limit', 'deductible']
        context_lower = reference['context'].lower()
        return any(keyword in context_lower for keyword in important_keywords)

    def _add_chunk_overlap(self, chunks: List[ChunkResult]) -> List[ChunkResult]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.overlap_words <= 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_words = chunks[i - 1].text.split()
                if len(prev_words) > self.overlap_words:
                    overlap_text = ' '.join(prev_words[-self.overlap_words:])

                    # Add overlap to current chunk
                    new_text = f"{overlap_text}\n[...]\n{chunk.text}"

                    # Update metadata
                    new_chunk = ChunkResult(
                        text=new_text,
                        metadata=chunk.metadata
                    )
                    new_chunk.metadata.word_count = len(new_text.split())
                    new_chunk.metadata.extra_data['has_overlap'] = True
                    new_chunk.metadata.extra_data['overlap_words'] = self.overlap_words

                    overlapped_chunks.append(new_chunk)
                else:
                    overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _create_section_chunk(self, section: HybridSection, policy_id: str,
                              chunk_id: str, include_header: bool = True) -> ChunkResult:
        """Create a chunk from a section."""
        if include_header:
            text = f"{section.header}\n\n{section.content}"
        else:
            text = section.content

        return self._create_chunk_result(
            text=text,
            section=section,
            policy_id=policy_id,
            chunk_id=chunk_id
        )

    def _create_intro_chunk(self, section: HybridSection, policy_id: str,
                            chunk_id: str) -> Optional[ChunkResult]:
        """Create a chunk for section introduction (before subsections)."""
        # Find content before first subsection
        if not section.subsections:
            return None

        first_subsection_pos = section.content.find(section.subsections[0].header)
        if first_subsection_pos <= 0:
            return None

        intro_text = section.content[:first_subsection_pos].strip()
        if len(intro_text.split()) < 20:  # Too short
            return None

        text = f"{section.header}\n\n{intro_text}"

        return self._create_chunk_result(
            text=text,
            section=section,
            policy_id=policy_id,
            chunk_id=chunk_id,
            metadata_extra={'chunk_part': 'introduction'}
        )

    def _create_subsection_chunk(self, section: HybridSection,
                                 subsections: List[HybridSection],
                                 policy_id: str, chunk_id: str) -> ChunkResult:
        """Create a chunk from subsections."""
        # Combine subsection texts
        subsection_texts = []
        for subsection in subsections:
            subsection_texts.append(f"{subsection.header}\n{subsection.content}")

        text = f"{section.header}\n\n" + "\n\n".join(subsection_texts)

        return self._create_chunk_result(
            text=text,
            section=section,
            policy_id=policy_id,
            chunk_id=chunk_id,
            metadata_extra={
                'subsection_count': len(subsections),
                'subsection_headers': [s.header for s in subsections]
            }
        )

    def _create_chunk_result(self, text: str, section: HybridSection,
                             policy_id: str, chunk_id: str,
                             metadata_extra: Dict[str, Any] = None) -> ChunkResult:
        """Create a ChunkResult with metadata."""
        # Analyze text properties
        has_amounts = self._has_amounts(text)
        has_conditions = self._has_conditions(text)
        has_exclusions = self._has_exclusions(text)

        # Create base metadata
        metadata = ChunkMetadata(
            chunk_id=self._create_chunk_id(policy_id, chunk_id),
            policy_id=policy_id,
            chunk_type=f"hybrid_{section.section_type}",
            word_count=len(text.split()),
            has_amounts=has_amounts,
            has_conditions=has_conditions,
            has_exclusions=has_exclusions,
            section=section.header,
            coverage_type=self._detect_coverage_type(section),
            confidence_score=self._calculate_confidence_score(section, text),
            extra_data={
                'section_type': section.section_type,
                'chunking_method': 'hybrid',
                'has_embedding': section.embedding is not None,
                'is_hybrid': True
            }
        )

        # Add extra metadata if provided
        if metadata_extra:
            metadata.extra_data.update(metadata_extra)

        return ChunkResult(text=text, metadata=metadata)

    def _has_amounts(self, text: str) -> bool:
        """Check if text contains monetary amounts."""
        amount_patterns = [
            r'€\s*\d+',
            r'EUR\s*\d+',
            r'CHF\s*\d+',
            r'\d+\s*(?:€|EUR|CHF)',
            r'(?:maximum|limit|up to).*\d+',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in amount_patterns)

    def _has_conditions(self, text: str) -> bool:
        """Check if text contains conditions."""
        condition_keywords = [
            'if', 'when', 'provided that', 'subject to', 'must', 'shall',
            'required', 'only if', 'unless', 'condition', 'prerequisite'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in condition_keywords)

    def _has_exclusions(self, text: str) -> bool:
        """Check if text contains exclusions."""
        exclusion_keywords = [
            'not covered', 'excluded', 'exception', 'does not apply',
            'limitation', 'restriction', 'not eligible', 'exclusion'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in exclusion_keywords)

    def _calculate_confidence_score(self, section: HybridSection, text: str) -> float:
        """Calculate confidence score for the chunk."""
        score = 1.0

        # Higher confidence for well-structured sections
        if section.section_type in ['section', 'chapter', 'article']:
            score += 0.1

        # Higher confidence for coverage-specific sections
        if section.section_type in ['coverage', 'exclusion']:
            score += 0.15

        # Lower confidence for very short sections
        if len(text.split()) < 50:
            score -= 0.2

        # Higher confidence if section has subsections (well-structured)
        if section.subsections:
            score += 0.1

        # Higher confidence if embedding is available
        if section.embedding is not None:
            score += 0.05

        return max(0.0, min(1.0, score))

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this chunking strategy."""
        return {
            "name": "hybrid",
            "description": "Advanced hybrid chunking combining structural and semantic approaches",
            "type": "hybrid",
            "complexity": "very high",
            "performance": "medium",
            "config": self.config,
            "features": [
                "hierarchical_section_extraction",
                "semantic_coherence_analysis",
                "cross_reference_detection",
                "coverage_exclusion_pairing",
                "multilingual_support",
                "table_preservation",
                "intelligent_overlap",
                "embedding_based_similarity"
            ],
            "supported_structures": [
                "sections_chapters_articles",
                "coverage_exclusion_pairs",
                "subsections_and_lists",
                "cross_references",
                "tables_and_amounts"
            ],
            "optimization_for": [
                "insurance_policies",
                "legal_documents",
                "structured_contracts",
                "multilingual_content"
            ],
            "expected_benefits": [
                "better_context_preservation",
                "improved_retrieval_accuracy",
                "reduced_false_negatives",
                "comprehensive_coverage_understanding"
            ]
        }

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.max_chunk_words <= 0:
            logger.error("max_chunk_words must be positive")
            return False

        if self.min_chunk_words <= 0:
            logger.error("min_chunk_words must be positive")
            return False

        if self.min_chunk_words >= self.max_chunk_words:
            logger.error("min_chunk_words must be less than max_chunk_words")
            return False

        if self.overlap_words < 0:
            logger.error("overlap_words cannot be negative")
            return False

        if self.semantic_threshold < 0 or self.semantic_threshold > 1:
            logger.error("semantic_threshold must be between 0 and 1")
            return False

        return True
