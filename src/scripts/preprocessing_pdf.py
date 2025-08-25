import os.path
import fitz  # PyMuPDF
import camelot
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from enum import Enum

from config import DOCUMENT_DIR

try:
    from pymupdf4llm import to_markdown

    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    logging.warning("PyMuPDF4LLM not available. Using basic text extraction.")


class DocumentType(Enum):
    """Supported document types."""
    INSURANCE_POLICY = "insurance_policy"
    CONTRACT = "contract"
    LEGAL_DOCUMENT = "legal_document"
    TECHNICAL_MANUAL = "technical_manual"
    GENERIC = "generic"


@dataclass
class DocumentChunk:
    """A structured chunk of document content for RAG."""
    chunk_id: str
    chunk_type: str
    title: str
    content: str
    page_number: int
    section_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'chunk_id': self.chunk_id,
            'chunk_type': self.chunk_type,
            'title': self.title,
            'content': self.content,
            'page_number': self.page_number,
            'section_path': self.section_path,
            'metadata': self.metadata,
            'tables': self.tables
        }

    def to_rag_dict(self) -> Dict[str, Any]:
        """Convert to RAG-optimized dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'title': self.title,
            'content': self.content,
            'page_number': self.page_number,
            'section_path': self.section_path,
            'chunk_type': self.chunk_type,
            'language': self.metadata.get('language', 'unknown'),
            'content_type': self.metadata.get('content_type', 'general'),
            'semantic_type': self.metadata.get('semantic_type', 'general'),
            'word_count': self.metadata.get('word_count', 0),
            'has_tables': len(self.tables) > 0,
            'tables': self.tables,
            'quality_score': self.metadata.get('quality_score', 0.0),
            'context_preserved': self.metadata.get('context_preserved', False)
        }


@dataclass
class CompanyProfile:
    """Enhanced company-specific document structure profile."""
    name: str
    document_types: List[DocumentType]
    section_patterns: Dict[int, List[str]]
    table_indicators: List[str] = field(default_factory=list)
    header_patterns: List[str] = field(default_factory=list)
    footer_patterns: List[str] = field(default_factory=list)
    language_hints: List[str] = field(default_factory=list)
    special_formatting: Dict[str, Any] = field(default_factory=dict)
    ignore_patterns: List[str] = field(default_factory=list)
    question_patterns: List[str] = field(default_factory=list)
    semantic_indicators: Dict[str, List[str]] = field(default_factory=dict)
    content_type_patterns: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """Enhanced configuration for PDF processing pipeline."""
    table_accuracy_threshold: float = 75.0
    preserve_structure: bool = True
    extract_metadata: bool = True
    debug_mode: bool = False

    # Enhanced RAG-optimized settings
    max_chunk_size: int = 1500  # Increased for better context
    min_chunk_size: int = 150  # Increased minimum for better quality
    optimal_chunk_size: int = 1000  # Sweet spot for RAG

    # Context preservation settings
    overlap_size: int = 200  # Character overlap between chunks
    preserve_semantic_units: bool = True
    maintain_context: bool = True

    # Quality settings
    min_quality_score: float = 0.6
    require_complete_sentences: bool = True
    preserve_lists: bool = True
    preserve_qa_pairs: bool = True

    # Document processing
    company_profile: Optional[CompanyProfile] = None
    document_type: DocumentType = DocumentType.GENERIC

    # Enhanced language settings
    primary_language: Optional[str] = None
    auto_detect_language: bool = True
    supported_languages: List[str] = field(default_factory=lambda: [
        'english', 'italian', 'french', 'german', 'spanish', 'portuguese'
    ])

    # Advanced processing
    custom_section_patterns: List[Tuple[int, str]] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=list)
    table_detection_strategy: str = "adaptive"

    # Output formatting
    include_page_numbers: bool = True
    normalize_text: bool = True
    enhance_readability: bool = True


class EnhancedCompanyProfileManager:
    """Enhanced profile manager with improved multi-language support and semantic understanding."""

    def __init__(self):
        self.profiles = self._initialize_enhanced_profiles()

    def _initialize_enhanced_profiles(self) -> Dict[str, CompanyProfile]:
        profiles = {}

        # Enhanced English Insurance Profile
        profiles["english_insurance"] = CompanyProfile(
            name="English Insurance Enhanced",
            document_types=[DocumentType.INSURANCE_POLICY],
            section_patterns={
                # Question-style headers (HIGHEST PRIORITY)
                0: [
                    r'^(What\s+is\s+(?:insured|covered))\?\s*(.*)$',
                    r'^(What\s+is\s+NOT\s+(?:insured|covered))\?\s*(.*)$',
                    r'^(Are\s+there\s+(?:coverage\s+)?limits)\?\s*(.*)$',
                    r'^(What\s+are\s+my\s+obligations)\?\s*(.*)$',
                    r'^(What\s+are\s+the\s+(?:insurer\'?s\s+)?obligations)\?\s*(.*)$',
                    r'^(What\s+to\s+do\s+in\s+case\s+of\s+(?:a\s+)?claim)\?\s*(.*)$',
                    r'^(When\s+and\s+how\s+do\s+I\s+pay)\?\s*(.*)$',
                    r'^(When\s+does\s+(?:the\s+)?cover(?:age)?\s+start\s+and\s+end)\?\s*(.*)$',
                    r'^(How\s+can\s+I\s+cancel\s+(?:the\s+)?policy)\?\s*(.*)$',
                    r'^(Who\s+is\s+this\s+product\s+for)\?\s*(.*)$',
                    r'^(What\s+costs\s+do\s+I\s+have\s+to\s+bear)\?\s*(.*)$',
                    r'^(HOW\s+CAN\s+I\s+COMPLAIN.*RESOLVE.*DISPUTES)\?\s*(.*)$',
                ],
                1: [
                    # Main sections
                    r'^SECTION\s+([A-Z\d]+(?:\s+[IV]+)?)\s*[-–—:]*\s*(.*)$',
                    r'^PART\s+([A-Z\d]+(?:\s+[IV]+)?)\s*[-–—:]*\s*(.*)$',
                    r'^CHAPTER\s+([A-Z\d]+(?:\s+[IV]+)?)\s*[-–—:]*\s*(.*)$',
                    r'^([A-Z][A-Z\s]*[A-Z])\s*$',  # All caps titles
                ],
                2: [
                    # Articles and subsections
                    r'^ARTICLE\s+(\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                    r'^ART\.\s*(\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                    r'^Art\.\s*(\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                    r'^(\d+\.\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                ],
                3: [
                    # Sub-items
                    r'^([a-z])\)\s*(.+)$',
                    r'^([A-Z])\.\s*(.+)$',
                    r'^([A-Z]\d+)\s*[-–—:]*\s*(.+)$',
                    r'^([A-Z]\.\d+)\s*[-–—:]*\s*(.+)$',
                ],
            },
            question_patterns=[
                r'What\s+is\s+(?:insured|covered)\?',
                r'What\s+is\s+NOT\s+(?:insured|covered)\?',
                r'Are\s+there\s+(?:coverage\s+)?limits\?',
                r'What\s+are\s+my\s+obligations\?',
                r'What\s+are\s+the\s+(?:insurer\'?s\s+)?obligations\?',
                r'What\s+to\s+do\s+in\s+case\s+of\s+(?:a\s+)?claim\?',
                r'When\s+and\s+how\s+do\s+I\s+pay\?',
                r'When\s+does\s+(?:the\s+)?cover(?:age)?\s+start\s+and\s+end\?',
                r'How\s+can\s+I\s+cancel\s+(?:the\s+)?policy\?',
                r'Who\s+is\s+this\s+product\s+for\?',
                r'What\s+costs\s+do\s+I\s+have\s+to\s+bear\?',
                r'HOW\s+CAN\s+I\s+COMPLAIN.*RESOLVE.*DISPUTES\?',
            ],
            table_indicators=['COVERAGE', 'LIMITS', 'PREMIUMS', 'CONDITIONS', 'OPTIONS', 'BENEFITS'],
            semantic_indicators={
                'coverage': ['insured', 'covered', 'protection', 'benefit', 'guarantee'],
                'exclusions': ['excluded', 'not covered', 'limitations', 'restrictions'],
                'obligations': ['must', 'shall', 'required', 'obligation', 'duty'],
                'procedures': ['procedure', 'process', 'steps', 'how to', 'method'],
                'definitions': ['means', 'definition', 'refers to', 'defined as', 'shall mean']
            },
            content_type_patterns={
                'coverage': [r'cover(?:age|ed|s)', r'insur(?:ed|ance)', r'protect(?:ion|ed)', r'benefit'],
                'exclusions': [r'exclud(?:ed|es|ing)', r'not\s+cover(?:ed|age)', r'limitation', r'restriction'],
                'claims': [r'claim', r'loss', r'damage', r'incident', r'accident'],
                'premium': [r'premium', r'payment', r'cost', r'fee', r'charge'],
                'assistance': [r'assistance', r'help', r'support', r'service', r'aid']
            },
            header_patterns=[
                r'--- Page \d+ ---',
                r'INSURANCE CONTRACT',
                r'NOBIS GROUP',
                r'POLICY CONDITIONS'
            ],
            ignore_patterns=[
                r'^Last update:.*',
                r'^Nobis Compagnia di Assicurazioni S\.p\.A\.',
                r'^The Legal Representative.*',
                r'^dr\. Giorgio Introvigne.*',
                r'^Conditions Assicurazioni Filo diretto Travel.*',
                r'^Model \d+.*edition.*',
                r'^\s*$',
                r'^Page \d+ of \d+$',
                r'^www\.',
                r'^.*@.*\..*$',  # Email addresses
                r'^\d{2}\.\d{2}\.\d{4}$',  # Dates
            ],
            language_hints=['english', 'insurance', 'policy', 'coverage', 'section', 'article', 'what is']
        )

        # Enhanced Italian Insurance Profile
        profiles["italian_insurance"] = CompanyProfile(
            name="Italian Insurance Enhanced",
            document_types=[DocumentType.INSURANCE_POLICY],
            section_patterns={
                # Question-style headers (HIGHEST PRIORITY) - Enhanced patterns
                0: [
                    r'^(Che\s+cosa\s+(?:è|sono)\s*assicurat[oi])\?\s*(.*)$',
                    r'^(Che\s+cosa\s+NON\s+(?:è|sono)\s*assicurat[oi])\?\s*(.*)$',
                    r'^(Ci\s+sono\s+limiti\s+di\s+copertura)\?\s*(.*)$',
                    r'^(Che\s+obblighi\s+ho)\?\s*(.*)$',
                    r"^(Quali\s+obblighi\s+ha\s+l['’]impresa)\?\s*(.*)$",
                    r'^(Cosa\s+fare\s+in\s+caso\s+di\s+sinistro)\?\s*(.*)$',
                    r'^(Quando\s+e\s+come\s+devo\s+pagare)\?\s*(.*)$',
                    r'^(Quando\s+comincia\s+la\s+copertura\s+e\s+quando\s+finisce)\?\s*(.*)$',
                    r'^(Come\s+posso\s+disdire\s+la\s+polizza)\?\s*(.*)$',
                    r'^(A\s+chi\s+è\s+rivolto\s+questo\s+prodotto)\?\s*(.*)$',
                    r'^(Quali\s+costi\s+devo\s+sostenere)\?\s*(.*)$',
                    r'^(COME\s+POSSO\s+PRESENTARE.*RECLAMI.*CONTROVERSIE)\?\s*(.*)$',
                        # Additional Italian question patterns
                    r'^(Quando\s+comincia.*quando\s+finisce)\?\s*(.*)$',
                    r'^(Come\s+posso\s+disdire)\?\s*(.*)$',
                ],
            1: [
                # Main sections - Enhanced patterns
                r'^SEZIONE\s+([A-Z\d]+(?:\s+[IV]+)?)\s*[-–—:]*\s*(.*)$',
                r'^PARTE\s+([A-Z\d]+(?:\s+[IV]+)?)\s*[-–—:]*\s*(.*)$',
                r'^CAPITOLO\s+([A-Z\d]+(?:\s+[IV]+)?)\s*[-–—:]*\s*(.*)$',
                r'^GARANZIE?\s+([A-Z\d\s]+)\s*$',
                r'^([A-Z][A-ZÀÈÉÌÒÙ\s]*[A-ZÀÈÉÌÒÙ])\s*$',  # All caps with Italian accents
                # Specific Italian insurance sections
                r'^(ASSISTENZA\s+IN\s+VIAGGIO)\s*(.*)$',
                r'^(SPESE\s+MEDICHE)\s*(.*)$',
                r'^(BAGAGLIO)\s*(.*)$',
                r'^(ANNULLAMENTO\s+VIAGGIO)\s*(.*)$',
            ],
            2: [
                # Articles and subsections - Enhanced
                r'^ARTICOLO\s+(\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                r'^ART\.\s*(\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                r'^Art\.\s*(\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                r'^(\d+\.\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                # Italian-specific patterns
                r'^([A-Z]\d*)\s*[-–—:]*\s*(.+)$',  # Like "B1", "C.1"
            ],
            3: [
                # Sub-items - Enhanced
                r'^([a-z])\)\s*(.+)$',
                r'^([A-Z])\.\s*(.+)$',
                r'^(\d+\.\d+(?:\.\d+)*)\s*[-–—:]*\s*(.+)$',
                r'^([A-Z]\d+)\s*[-–—:]*\s*(.+)$',
                r'^([A-Z]\.\d+)\s*[-–—:]*\s*(.+)$',
                # Lettered sub-items common in Italian docs
                r'^([a-z])\s*[-–—:]\s*(.+)$',
            ],
        },
        question_patterns = [
            r'Che\s+cosa\s+(?:è|sono)\s*assicurat[oi]\?',
            r'Che\s+cosa\s+NON\s+(?:è|sono)\s*assicurat[oi]\?',
            r'Ci\s+sono\s+limiti\s+di\s+copertura\?',
            r'Che\s+obblighi\s+ho\?',
            r"Quali\s+obblighi\s+ha\s+l['’]impresa\?",
            r'Cosa\s+fare\s+in\s+caso\s+di\s+sinistro\?',
            r'Quando\s+e\s+come\s+devo\s+pagare\?',
            r'Quando\s+comincia.*copertura.*quando\s+finisce\?',
            r'Come\s+posso\s+disdire.*polizza\?',
            r'A\s+chi\s+è\s+rivolto\s+questo\s+prodotto\?',
            r'Quali\s+costi\s+devo\s+sostenere\?',
            r'COME\s+POSSO\s+PRESENTARE.*RECLAMI\?',
        ],
        table_indicators = ['COPERTURA', 'LIMITI', 'PREMI', 'CONDIZIONI', 'OPZIONI', 'GARANZIE', 'MASSIMALI'],
        semantic_indicators = {
            'copertura': ['assicurato', 'coperto', 'protezione', 'garanzia', 'beneficio'],
            'esclusioni': ['escluso', 'non coperto', 'limitazioni', 'restrizioni', 'non assicurato'],
            'obblighi': ['deve', 'dovrà', 'obbligo', 'dovere', 'tenuto'],
            'procedure': ['procedura', 'processo', 'modalità', 'come fare', 'metodo'],
            'definizioni': ['significa', 'definizione', 'si intende', 'definito come', 'comprende']
        },
        content_type_patterns = {
            'copertura': [r'copertur[ao]', r'assicurat[oi]', r'garanzi[ao]', r'protezion[ei]', r'benefici[oi]'],
            'esclusioni': [r'esclus[oi]', r'non\s+(?:coperto|assicurato)', r'limitazion[ei]', r'restrizion[ei]'],
            'sinistri': [r'sinistro', r'danno', r'perdita', r'incidente', r'evento'],
            'premio': [r'premio', r'pagamento', r'costo', r'tariffa', r'importo'],
            'assistenza': [r'assistenza', r'aiuto', r'supporto', r'servizio', r'soccorso']
        },
        header_patterns = [
            r'Pag\.\s*\d+\s*di\s*\d+',
            r'--- Page \d+ ---',
            r'CONTRATTO ASSICURATIVO',
            r'POLIZZA',
            r'CONDIZIONI DI ASSICURAZIONE'
        ],
        ignore_patterns = [
            r'^Ultimo aggiornamento:.*',
            r'^Inter Partner Assistance S\.A\..*',
            r'^AXA.*Assistance.*',
            r'^Nobis.*',
            r'^Il Rappresentante Legale.*',
            r'^Condizioni.*',
            r'^Modello \d+.*edizione.*',
            r'^www\.',
            r'^\s*$',
            r'^Pag\.\s*\d+\s*di\s*\d+\s*$',
            r'^.*@.*\..*$',  # Email addresses
            r'^\d{2}\.\d{2}\.\d{4}$',  # Dates
            r'^Via\s+.*\d+.*$',  # Addresses
            r'^Codice\s+Fiscale.*$',
            r'^Partita\s+IVA.*$',
        ],
        language_hints = ['italian', 'assicurazione', 'polizza', 'garanzia', 'sezione', 'articolo', 'che cosa',
                          'società']
        )

        return profiles

    def get_profile(self, profile_name: str) -> Optional[CompanyProfile]:
        """Get a specific company profile."""
        return self.profiles.get(profile_name.lower())

    def detect_best_profile(self, text_sample: str) -> str:
        """Enhanced auto-detection with better language recognition."""
        text_lower = text_sample.lower()

        italian_score = 0
        english_score = 0

        # Enhanced Italian indicators with weights
        italian_indicators = {
            'che cosa': 15, 'assicurato': 10, 'polizza': 10, 'garanzia': 8, 'sezione': 5,
            'articolo': 5, 'copertura': 8, 'sinistro': 8, 'premio': 5, 'contratto': 5,
            'viaggio': 3, 'bagaglio': 3, 'spese mediche': 10, 'assistenza': 5,
            'rimborso': 5, 'indennizzo': 5, 'esclusioni': 8, 'società': 3,
            'dell\'assicurato': 12, 'centrale operativa': 10, 'quando comincia': 8,
            'come posso': 8, 'quali obblighi': 8, 'cosa fare': 5
        }

        # Enhanced English indicators with weights
        english_indicators = {
            'what is': 15, 'insured': 10, 'policy': 10, 'coverage': 8, 'section': 5,
            'article': 5, 'insurance': 10, 'contract': 5, 'travel': 3, 'baggage': 3,
            'medical expenses': 10, 'assistance': 5, 'reimbursement': 5, 'claims': 8,
            'exclusions': 8, 'company': 3, 'insured person': 12, 'when does': 8,
            'how can': 8, 'what are my': 8, 'what to do': 5, 'limits': 5
        }

        # Count weighted matches
        for indicator, weight in italian_indicators.items():
            if indicator in text_lower:
                italian_score += text_lower.count(indicator) * weight

        for indicator, weight in english_indicators.items():
            if indicator in text_lower:
                english_score += text_lower.count(indicator) * weight

        # Specific pattern bonuses
        if re.search(r'che cosa (?:è|sono)\s*assicurat[oi]\?', text_lower):
            italian_score += 25
        if re.search(r'what is (?:insured|covered)\?', text_lower):
            english_score += 25

        # Company-specific patterns with higher weights
        if any(pattern in text_lower for pattern in ['axa', 'inter partner assistance', 'centrale operativa']):
            italian_score += 15
        if any(pattern in text_lower for pattern in ['nobis', 'filo diretto']):
            english_score += 10

        # Language-specific character patterns
        italian_chars = (text_lower.count('à') + text_lower.count('è') + text_lower.count('é') +
                         text_lower.count('ì') + text_lower.count('ò') + text_lower.count('ù'))
        italian_score += italian_chars * 5

        # Document structure patterns
        if re.search(r'pag\.\s*\d+\s*di\s*\d+', text_lower):
            italian_score += 10
        if re.search(r'page \d+ of \d+', text_lower):
            english_score += 10

        # Final decision with logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced language detection scores - Italian: {italian_score}, English: {english_score}")

        if italian_score > english_score:
            return 'italian_insurance'
        elif english_score > italian_score:
            return 'english_insurance'
        else:
            # Enhanced tie-breaking
            if any(word in text_lower for word in ['assicurato', 'polizza', 'società', 'garanzia']):
                return 'italian_insurance'
            return 'italian_insurance'  # Default to Italian for insurance docs

    def list_profiles(self) -> List[str]:
        """List available profiles."""
        return list(self.profiles.keys())

class EnhancedAdvancedPDFPreprocessor:
    """Production-ready PDF preprocessing pipeline with enhanced chunk quality."""

    # Enhanced character normalization map
    NORMALIZATION_MAP = {
        "\u200b": "", "\ufeff": "", "\xad": "", "\u00a0": " ",
        "\u2009": " ", "\u202f": " ", "\u2013": "-", "\u2014": "--",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2022": "•", "\u2026": "...", "\u20ac": "€",
        # Preserve Italian accented characters properly
    }

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.profile_manager = EnhancedCompanyProfileManager()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better Italian support and readability."""
        if not text:
            return ""

        if not self.config.normalize_text:
            return text.strip()

        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Normalize special characters (but preserve Italian accents)
        for char, replacement in self.NORMALIZATION_MAP.items():
            text = text.replace(char, replacement)

        # Fix broken hyphenation while preserving intentional hyphens
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

        # Enhanced whitespace normalization
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Clean punctuation spacing
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s*\n', r'\1\n', text)

        # Enhanced Italian text fixes
        text = re.sub(r'è\s+assicurat', 'è assicurat', text)
        text = re.sub(r'à\s+([aeiou])', r'à\1', text)
        text = re.sub(r'ò\s+([aeiou])', r'ò\1', text)

        # Fix common formatting issues
        text = re.sub(r'(\w)\s*\n\s*(\w)', r'\1 \2', text)  # Fix broken words across lines
        text = re.sub(r'\n\s*([.,:;])', r'\1', text)  # Fix punctuation on new lines

        # Enhanced readability improvements
        if self.config.enhance_readability:
            # Ensure proper spacing after punctuation
            text = re.sub(r'([.!?])\s*([A-ZÀÈÉÌÒÙ])', r'\1 \2', text)
            # Fix number formatting
            text = re.sub(r'(\d)\s+([.,])\s*(\d)', r'\1\2\3', text)

        return text.strip()

    def extract_with_pymupdf4llm(self, doc: fitz.Document) -> str:
        """Extract text with enhanced fallback strategy."""
        try:
            if PYMUPDF4LLM_AVAILABLE:
                markdown_content = to_markdown(doc)
                return self.clean_text(markdown_content)
            else:
                return self._enhanced_fallback_extraction(doc)
        except Exception as e:
            self.logger.warning(f"Primary extraction failed: {e}")
            return self._enhanced_fallback_extraction(doc)

    def _enhanced_fallback_extraction(self, doc: fitz.Document) -> str:
        """Enhanced fallback text extraction with better page separation."""
        text_blocks = []
        for page_num in range(doc.page_count):
            page = doc[page_num]

            # Try different text extraction methods in order of preference
            text_methods = [
                # Method 1: Preserve ligatures and whitespace
                lambda: page.get_text(flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE),
                # Method 2: Standard text extraction
                lambda: page.get_text("text"),
                # Method 3: Block-based extraction
                lambda: page.get_text("blocks"),
                # Method 4: Dict-based extraction (most detailed)
                lambda: page.get_text("dict")
            ]

            extracted_text = ""
            for method_idx, method in enumerate(text_methods):
                try:
                    result = method()

                    if isinstance(result, list):  # blocks method
                        result = "\n".join(
                            [block[4] if len(block) > 4 else str(block) for block in result if block])
                    elif isinstance(result, dict):  # dict method
                        result = self._extract_text_from_dict(result)

                    if result and result.strip():
                        extracted_text = result
                        self.logger.debug(f"Page {page_num + 1}: Used extraction method {method_idx + 1}")
                        break

                except Exception as e:
                    self.logger.debug(f"Extraction method {method_idx + 1} failed on page {page_num + 1}: {e}")
                    continue

            if extracted_text.strip():
                if self.config.include_page_numbers:
                    text_blocks.append(f"--- Page {page_num + 1} ---\n{extracted_text}")
                else:
                    text_blocks.append(extracted_text)

        return self.clean_text("\n\n".join(text_blocks))

    def _extract_text_from_dict(self, text_dict: dict) -> str:
        """Extract text from PyMuPDF dict format while preserving structure."""
        text_parts = []

        if 'blocks' in text_dict:
            for block in text_dict['blocks']:
                if 'lines' in block:
                    for line in block['lines']:
                        if 'spans' in line:
                            line_text = ''
                            for span in line['spans']:
                                if 'text' in span:
                                    line_text += span['text']
                            if line_text.strip():
                                text_parts.append(line_text)

        return '\n'.join(text_parts)

    def _enhanced_fix_concatenated_markers(self, text: str) -> str:
        """Enhanced fix for concatenated page markers with better pattern recognition."""

        # Enhanced patterns for both languages
        patterns_to_fix = [
            # English patterns
            (r'(--- Page \d+ ---)\s*([A-Z])', r'\1\n\2'),
            (r'(--- Page \d+ ---)\s*(SECTION\s+[A-Z\d]+)', r'\1\n\2'),
            (r'(--- Page \d+ ---)\s*(ARTICLE\s+\d+)', r'\1\n\2'),
            (r'(--- Page \d+ ---)\s*(What\s+)', r'\1\n\2'),

            # Italian patterns - Enhanced
            (r'(Pag\.\s*\d+\s*di\s*\d+)\s*([A-ZÀÈÉÌÒÙ][a-zàèéìòù])', r'\1\n\2'),
            (r'(Pag\.\s*\d+\s*di\s*\d+)\s*(Che\s+cosa)', r'\1\n\2'),
            (r'(--- Page \d+ ---)\s*(Che\s+cosa)', r'\1\n\2'),
            (r'(\d+\s*di\s*\d+)\s*([A-ZÀÈÉÌÒÙ][a-zàèéìòù])', r'\1\n\2'),
            (r'(Pag\.\s*\d+\s*di\s*\d+)\s*(SEZIONE)', r'\1\n\2'),
            (r'(Pag\.\s*\d+\s*di\s*\d+)\s*(ARTICOLO)', r'\1\n\2'),

            # Fix spacing in question patterns (Italian)
            (r'Che\s+cosa\s+è\s*assicurat', 'Che cosa è assicurat'),
            (r'Che\s+cosa\s+NON\s+è\s*assicurat', 'Che cosa NON è assicurat'),
            (r'Ci\s+sono\s+limiti\s+di\s*copertura', 'Ci sono limiti di copertura'),

            # Fix spacing in question patterns (English)
            (r'What\s+is\s+insured', 'What is insured'),
            (r'What\s+is\s+NOT\s+insured', 'What is NOT insured'),
            (r'Are\s+there\s+coverage\s*limits', 'Are there coverage limits'),

            # General improvements
            (r'([.!?])\s*([A-ZÀÈÉÌÒÙ])', r'\1\n\2'),  # New sentence, new line
            (r'(\w)\s*\n\s*([a-zàèéìòù])', r'\1 \2'),  # Fix broken words
        ]

        for pattern, replacement in patterns_to_fix:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.MULTILINE)

        # Ensure proper page markers
        text = re.sub(r'--- Page (\d+) --([^-])', r'--- Page \1 ---\n\2', text)
        text = re.sub(r'(Pag\.\s*\d+\s*di\s*\d+)([A-ZÀÈÉÌÒÙ])', r'\1\n\2', text)

        self.logger.debug("Enhanced concatenated marker fixes applied")
        return text

    def _enhanced_detect_structure_element(self, line: str) -> Tuple[int, str, str]:
        """Enhanced structure detection with better semantic understanding."""
        line = line.strip()

        if not line or not self.config.company_profile:
            return 0, "", line

        # Check all pattern levels in priority order
        for level in sorted(self.config.company_profile.section_patterns.keys()):
            patterns = self.config.company_profile.section_patterns[level]

            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()

                    if len(groups) >= 2:
                        section_id = groups[0].strip()
                        title_part = groups[1].strip() if groups[1] else ""
                    elif len(groups) == 1:
                        # For patterns that capture the entire title
                        if level == 0:  # Question patterns
                            section_id = ""
                            title_part = groups[0].strip()
                        else:
                            section_id = groups[0].strip()
                            title_part = ""
                    else:
                        section_id = ""
                        title_part = line.strip()

                    # Enhanced title cleaning
                    if title_part:
                        title_part = re.sub(r'^[-–—:\s]+', '', title_part)
                        title_part = re.sub(r'[-–—:\s]+$', '', title_part)
                        title_part = self.clean_text(title_part)

                    # Build enhanced full title
                    if level == 0:  # Question patterns - use as-is
                        full_title = title_part if title_part else line.strip()
                        # Ensure question mark is included
                        if not full_title.endswith('?') and '?' in line:
                            full_title += '?'
                    elif section_id and title_part:
                        full_title = f"{section_id} - {title_part}"
                    elif section_id:
                        full_title = section_id
                    elif title_part:
                        full_title = title_part
                    else:
                        full_title = line.strip()

                    # Adjust level for question patterns
                    actual_level = 1 if level == 0 else level

                    self.logger.info(
                        f"🎯 Enhanced structure detected: Level {actual_level}, Pattern: '{pattern[:50]}...', Title: '{full_title}'")
                    return actual_level, section_id, full_title

        return 0, "", line

    def _enhanced_should_ignore_line(self, line: str) -> bool:
        """Enhanced line filtering with better semantic understanding."""
        if not self.config.company_profile or not line.strip():
            return True

        line_clean = line.strip()

        # Check ignore patterns
        for pattern in self.config.company_profile.ignore_patterns:
            if re.match(pattern, line_clean, re.IGNORECASE):
                self.logger.debug(f"Ignoring line (pattern match): {line_clean[:50]}...")
                return True

        # Enhanced filters
        if len(line_clean) < 3:
            return True

        # Ignore pure numbers
        if re.match(r'^\d+\s*$', line_clean):
            return True

        # Ignore pure punctuation
        if re.match(r'^[^\w\s]*$', line_clean):
            return True

        # Ignore URLs and email addresses
        if re.match(r'^(?:https?://|www\.|.*@.*\..*)$', line_clean, re.IGNORECASE):
            return True

        # Ignore pure dates
        if re.match(r'^\d{1,2}[./]\d{1,2}[./]\d{2,4}$', line_clean):
            return True

        return False

    def create_enhanced_smart_chunks(self, text: str, tables_by_page: Dict[int, List[Dict]]) -> List[DocumentChunk]:
        """Enhanced smart chunking with guaranteed high-quality results."""
        self.logger.info("Starting enhanced smart chunking with quality optimization")

        # Apply enhanced preprocessing
        text = self._enhanced_fix_concatenated_markers(text)

        # Try structured chunking with enhanced quality
        structured_chunks = self._try_enhanced_structured_chunking(text, tables_by_page)

        if structured_chunks and len(structured_chunks) > 1:
            # Validate and enhance chunk quality
            quality_chunks = self._enhance_chunk_quality(structured_chunks)
            if quality_chunks:
                self.logger.info(f"✅ Successfully created {len(quality_chunks)} enhanced quality chunks")
                return quality_chunks

        # Enhanced fallback chunking
        self.logger.warning("Structure detection suboptimal, using enhanced intelligent fallback")
        fallback_chunks = self._create_enhanced_intelligent_fallback_chunks(text, tables_by_page)
        quality_chunks = self._enhance_chunk_quality(fallback_chunks)

        return quality_chunks if quality_chunks else fallback_chunks

    def _try_enhanced_structured_chunking(self, text: str, tables_by_page: Dict[int, List[Dict]]) -> List[
        DocumentChunk]:
        """Enhanced structured chunking with better semantic preservation."""
        try:
            hierarchical_structure = self._enhanced_parse_document_structure(text)

            if not hierarchical_structure:
                self.logger.warning("No hierarchical structure detected")
                return []

            chunks = self._create_enhanced_chunks_from_structure(hierarchical_structure, tables_by_page)
            valid_chunks = [chunk for chunk in chunks if self._validate_enhanced_chunk(chunk)]

            if len(valid_chunks) < len(chunks):
                self.logger.warning(f"Enhanced validation filtered out {len(chunks) - len(valid_chunks)} chunks")

            return valid_chunks

        except Exception as e:
            self.logger.error(f"Enhanced structured chunking failed: {e}")
            return []

    def _enhanced_parse_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced document structure parsing with better semantic understanding."""
        try:
            self.logger.info("Starting enhanced document structure parsing")
            structural_elements = self._enhanced_parse_text_with_page_tracking(text)

            self.logger.info(f"Identified {len(structural_elements)} structural elements")

            structure_count = sum(1 for e in structural_elements if e['type'] == 'structure')
            content_count = sum(1 for e in structural_elements if e['type'] == 'content')
            self.logger.info(f"Structure elements: {structure_count}, Content elements: {content_count}")

            if structure_count == 0:
                self.logger.warning("No structural elements found")
                return []

            hierarchical_structure = self._build_enhanced_hierarchy(structural_elements)
            self.logger.info(
                f"Built enhanced hierarchical structure with {len(hierarchical_structure)} top-level sections")

            return hierarchical_structure

        except Exception as e:
            self.logger.error(f"Enhanced document structure parsing failed: {e}")
            return []

    def _enhanced_parse_text_with_page_tracking(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced text parsing with better semantic understanding and page tracking."""
        elements = []
        current_page = 1
        current_content = []

        if not text or not isinstance(text, str):
            self.logger.warning("Invalid text input")
            return []

        lines = text.split('\n')
        self.logger.debug(f"Processing {len(lines)} lines")

        i = 0
        while i < len(lines):
            original_line = lines[i]
            line = original_line.strip()

            if not line:
                if current_content:
                    current_content.append("")
                i += 1
                continue

            # Check for page markers
            page_patterns = [
                r'^---\s*Page\s+(\d+)\s*---\s*$',  # English
                r'^Pag\.\s*(\d+)\s*di\s*\d+\s*$',  # Italian
            ]

            page_found = False
            for pattern in page_patterns:
                page_match = re.match(pattern, line)
                if page_match:
                    new_page = int(page_match.group(1))
                    if new_page != current_page:
                        current_page = new_page
                        self.logger.debug(f"📄 Found page marker: Page {current_page}")
                    page_found = True
                    break

            if page_found:
                i += 1
                continue

            # Skip ignored patterns
            if self._enhanced_should_ignore_line(line):
                i += 1
                continue

            # Detect structural elements
            level, section_id, title = self._enhanced_detect_structure_element(line)

            if level > 0:
                self.logger.info(f"🎯 Structure detected on page {current_page}: Level {level}, '{line}'")

                # Save previous content if substantial
                if current_content:
                    content_text = '\n'.join(current_content).strip()
                    if content_text and len(content_text) >= self.config.min_chunk_size:
                        elements.append({
                            'type': 'content',
                            'level': 999,
                            'title': 'Content Block',
                            'content': content_text,
                            'page': current_page,
                            'section_id': '',
                            'full_line': '',
                            'semantic_type': self._detect_content_semantic_type(content_text)
                        })
                    current_content = []

                # Look ahead for continuation content
                continuation_content = []
                j = i + 1
                while j < len(lines) and j < i + 10:  # Look ahead max 10 lines
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue

                    # If it's another structural element or page marker, stop
                    if (self._enhanced_detect_structure_element(next_line)[0] > 0 or
                            any(re.match(pattern, next_line) for pattern in page_patterns)):
                        break

                    # If it's not ignored, add to continuation
                    if not self._enhanced_should_ignore_line(next_line):
                        continuation_content.append(lines[j].rstrip())
                    j += 1

                # Add structural element with any immediate continuation
                structure_content = '\n'.join(continuation_content).strip() if continuation_content else ''

                elements.append({
                    'type': 'structure',
                    'level': level,
                    'title': title,
                    'content': structure_content,
                    'page': current_page,
                    'section_id': section_id,
                    'full_line': line,
                    'semantic_type': self._detect_content_semantic_type(title + ' ' + structure_content)
                })

                # Skip the processed continuation lines
                i = j

            else:
                # Accumulate content
                current_content.append(original_line.rstrip())
                i += 1

        # Save remaining content
        if current_content:
            content_text = '\n'.join(current_content).strip()
            if content_text and len(content_text) >= self.config.min_chunk_size:
                elements.append({
                    'type': 'content',
                    'level': 999,
                    'title': 'Content Block',
                    'content': content_text,
                    'page': current_page,
                    'section_id': '',
                    'full_line': '',
                    'semantic_type': self._detect_content_semantic_type(content_text)
                })

        return elements

    def _detect_content_semantic_type(self, content: str) -> str:
        """Detect semantic type of content for better categorization."""
        if not content:
            return 'general'

        content_lower = content.lower()

        # Use profile-specific patterns if available
        if self.config.company_profile and hasattr(self.config.company_profile, 'content_type_patterns'):
            for content_type, patterns in self.config.company_profile.content_type_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower, re.IGNORECASE):
                        return content_type

        # Fallback to general patterns
        if re.search(r'esclus|exclud', content_lower):
            return "exclusions"
        elif re.search(r'copertur|coverage|garanzia', content_lower):
            return "coverage"
        elif re.search(r'limite|limit', content_lower):
            return "limits"
        elif re.search(r'premio|premium|pagamento|payment', content_lower):
            return "premium"
        elif re.search(r'sinistro|claim|denuncia', content_lower):
            return "claims"
        elif re.search(r'assistenza|assistance', content_lower):
            return "assistance"
        elif re.search(r'rimborso|reimbursement', content_lower):
            return "reimbursement"
        elif re.search(r'definizione|definition|significa|means', content_lower):
            return "definitions"
        elif re.search(r'procedura|procedure|modalità|method', content_lower):
            return "procedures"
        else:
            return "general"

    def _build_enhanced_hierarchy(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build enhanced hierarchical structure with better semantic organization."""
        hierarchy = []
        stack = []

        for element in elements:
            if element['type'] == 'structure':
                level = element['level']

                # Pop stack until we find the right parent level
                while stack and stack[-1]['level'] >= level:
                    popped = stack.pop()
                    self.logger.debug(f"Popped from stack: {popped['title']} (level {popped['level']})")

                # Create enhanced section
                section = {
                    'level': level,
                    'title': element['title'],
                    'section_id': element['section_id'],
                    'content': element['content'],
                    'page': element['page'],
                    'subsections': [],
                    'full_path': [],
                    'semantic_type': element.get('semantic_type', 'general'),
                    'quality_indicators': self._assess_section_quality(element)
                }

                # Build full path
                section['full_path'] = [s['title'] for s in stack] + [element['title']]

                # Add to parent or root
                if stack:
                    parent = stack[-1]
                    parent['subsections'].append(section)
                    self.logger.debug(f"Added '{element['title']}' as child of '{parent['title']}'")
                else:
                    hierarchy.append(section)
                    self.logger.debug(f"Added '{element['title']}' as root section")

                stack.append(section)

            elif element['type'] == 'content':
                if stack:
                    current_section = stack[-1]
                    if current_section['content']:
                        current_section['content'] += '\n\n' + element['content']
                    else:
                        current_section['content'] = element['content']

                    # Update semantic type if more specific
                    if element.get('semantic_type', 'general') != 'general':
                        current_section['semantic_type'] = element['semantic_type']

                    self.logger.debug(f"Added content to section '{current_section['title']}'")
                else:
                    # Create enhanced default section for orphaned content
                    section = {
                        'level': 1,
                        'title': 'Document Content',
                        'section_id': '',
                        'content': element['content'],
                        'page': element['page'],
                        'subsections': [],
                        'full_path': ['Document Content'],
                        'semantic_type': element.get('semantic_type', 'general'),
                        'quality_indicators': {'is_orphaned': True}
                    }
                    hierarchy.append(section)
                    stack.append(section)
                    self.logger.debug("Created enhanced default section for orphaned content")

        return hierarchy

    def _assess_section_quality(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of a section for better chunk creation."""
        indicators = {
            'has_title': bool(element.get('title', '').strip()),
            'title_length': len(element.get('title', '')),
            'has_content': bool(element.get('content', '').strip()),
            'content_length': len(element.get('content', '')),
            'has_section_id': bool(element.get('section_id', '').strip()),
            'is_question': element.get('title', '').endswith('?'),
            'semantic_type': element.get('semantic_type', 'general'),
            'page_number': element.get('page', 0)
        }

        # Calculate quality score
        score = 0.0
        if indicators['has_title']:
            score += 0.3
        if indicators['has_content']:
            score += 0.3
        if indicators['has_section_id']:
            score += 0.1
        if indicators['is_question']:
            score += 0.1
        if indicators['semantic_type'] != 'general':
            score += 0.1
        if 50 <= indicators['title_length'] <= 200:
            score += 0.1

        indicators['quality_score'] = score
        return indicators

    def _create_enhanced_chunks_from_structure(self, hierarchy: List[Dict[str, Any]],
                                               tables_by_page: Dict[int, List[Dict]]) -> List[DocumentChunk]:
        """Create enhanced chunks from hierarchical structure with better quality control."""
        chunks = []
        chunk_counter = 0

        def process_section_enhanced(section: Dict[str, Any], parent_path: List[str] = None) -> None:
            nonlocal chunk_counter

            parent_path = parent_path or []
            section_title = section['title']
            current_path = parent_path + [section_title]
            page_num = section.get('page', 1)

            # Enhanced content processing
            if section.get('content', '').strip():
                content = section['content'].strip()

                # Intelligent content splitting with semantic preservation
                content_chunks = self._enhanced_split_content_intelligently(content, section_title,
                                                                            section.get('semantic_type', 'general'))

                for i, chunk_content in enumerate(content_chunks):
                    if len(chunk_content.strip()) >= self.config.min_chunk_size:
                        title = section_title if len(content_chunks) == 1 else f"{section_title} (Part {i + 1})"

                        chunk = self._create_enhanced_quality_chunk(
                            chunk_counter, title, chunk_content, page_num,
                            current_path, tables_by_page, "enhanced_structured",
                            section.get('semantic_type', 'general'),
                            section.get('quality_indicators', {})
                        )
                        chunks.append(chunk)
                        chunk_counter += 1

            # Process subsections
            for subsection in section.get('subsections', []):
                process_section_enhanced(subsection, current_path)

        for section in hierarchy:
            process_section_enhanced(section)

        return chunks

    def _enhanced_split_content_intelligently(self, content: str, title: str, semantic_type: str) -> List[str]:
        """Enhanced content splitting that preserves semantic units and context."""

        # If content is already optimal size, return as-is
        if len(content) <= self.config.optimal_chunk_size:
            return [content]

        # Detect if this is a special semantic unit that shouldn't be split
        if self._is_protected_semantic_unit(content, semantic_type):
            # If it's too large but protected, try minimal splitting
            if len(content) > self.config.max_chunk_size:
                return self._minimal_split_protected_content(content)
            else:
                return [content]

        chunks = []

        # Step 1: Split by enhanced semantic boundaries
        semantic_blocks = self._split_by_enhanced_semantic_boundaries(content, semantic_type)

        current_chunk = ""

        for block in semantic_blocks:
            # Try to add entire semantic block
            test_chunk = (current_chunk + "\n\n" + block).strip() if current_chunk else block

            if len(test_chunk) <= self.config.max_chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if substantial
                if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Handle oversized block
                if len(block) > self.config.max_chunk_size:
                    block_chunks = self._safe_split_large_semantic_block(block, semantic_type)
                    chunks.extend(block_chunks[:-1])
                    current_chunk = block_chunks[-1] if block_chunks else ""
                else:
                    current_chunk = block

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())

        # Enhanced context preservation with semantic overlap
        return self._add_enhanced_semantic_overlap(chunks, semantic_type)

    def _is_protected_semantic_unit(self, content: str, semantic_type: str) -> bool:
        """Check if content represents a protected semantic unit that shouldn't be split."""

        # Lists with multiple items (preserve complete lists)
        if re.search(r'^\s*[•\-\*\d+\.\)]\s+.*(\n\s*[•\-\*\d+\.\)]\s+.*){2,}', content, re.MULTILINE):
            return True

        # Q&A pairs (preserve question-answer structure)
        if re.search(r'[^?]*\?[^\n]*\n[^\n?]*[^\?]\n', content):
            return True

        # Tables or structured data
        if content.count('|') >= 4 or content.count('\t') >= 4:
            return True

        # Definitions (based on semantic type and content patterns)
        if semantic_type == 'definitions' or re.search(
                r'(significa|significa che|definisce|definition|means|refers to)', content, re.IGNORECASE):
            return True

        # Procedures or step-by-step instructions
        if semantic_type == 'procedures' or re.search(r'(procedura|procedure|modalità|step|fase|point)', content,
                                                      re.IGNORECASE):
            return True

        # Legal clauses or conditions
        if re.search(r'(condizione|condition|clausola|clause|termine|term)', content, re.IGNORECASE):
            return True

        return False

    def _minimal_split_protected_content(self, content: str) -> List[str]:
        """Minimally split protected content while preserving semantic integrity."""

        # Try to find natural break points that don't break semantic units

        # Split at double line breaks (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', content)
        if len(paragraphs) > 1:
            chunks = []
            current_chunk = ""

            for para in paragraphs:
                test_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para

                if len(test_chunk) <= self.config.max_chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para

            if current_chunk:
                chunks.append(current_chunk.strip())

            return [chunk for chunk in chunks if len(chunk.strip()) >= self.config.min_chunk_size]

        # If no paragraph breaks, return as single chunk (better to have oversized than broken)
        return [content]

    def _split_by_enhanced_semantic_boundaries(self, content: str, semantic_type: str) -> List[str]:
        """Split content by enhanced semantic boundaries specific to content type."""

        # Enhanced patterns based on semantic type
        if semantic_type == 'exclusions':
            # For exclusions, split by numbered items or bullet points
            if re.search(r'^\s*\d+\.\s+', content, re.MULTILINE):
                blocks = re.split(r'(?=^\s*\d+\.\s+)', content, flags=re.MULTILINE)
            elif re.search(r'^\s*[•\-]\s+', content, re.MULTILINE):
                blocks = re.split(r'(?=^\s*[•\-]\s+)', content, flags=re.MULTILINE)
            else:
                blocks = re.split(r'\n\s*\n', content)

        elif semantic_type == 'coverage':
            # For coverage, split by benefit types or coverage areas
            blocks = re.split(r'(?=\n[A-ZÀÈÉÌÒÙ][A-ZÀÈÉÌÒÙa-zàèéìòù\s]*:?\n)', content)
            if len(blocks) == 1:
                blocks = re.split(r'\n\s*\n', content)

        elif semantic_type == 'procedures':
            # For procedures, preserve step sequences
            if re.search(r'^\s*\d+\.\s+', content, re.MULTILINE):
                # Keep numbered steps together in logical groups
                blocks = re.split(r'(?=^\s*1\.\s+)', content, flags=re.MULTILINE)
            else:
                blocks = re.split(r'\n\s*\n', content)

        else:
            # Default semantic splitting
            blocks = re.split(r'\n\s*\n+', content)

        # Clean and filter blocks
        refined_blocks = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Further semantic refinement if needed
            if len(block) > self.config.max_chunk_size * 1.5:
                # Only split if really necessary and safe
                sub_blocks = self._safe_semantic_subsplit(block, semantic_type)
                refined_blocks.extend(sub_blocks)
            else:
                refined_blocks.append(block)

        return [block for block in refined_blocks if block.strip()]

    def _safe_semantic_subsplit(self, block: str, semantic_type: str) -> List[str]:
        """Safely split a large semantic block while preserving meaning."""

        # Try sentence-based splitting first
        sentences = self._enhanced_split_into_sentences(block)

        if len(sentences) <= 1:
            return [block]  # Can't split safely

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

            if len(test_chunk) <= self.config.max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _enhanced_split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting for multilingual content with better accuracy."""

        # Enhanced sentence boundary patterns for Italian and English
        patterns = [
            # Standard sentence endings followed by capital letters
            r'([.!?]+)\s+(?=[A-ZÀÈÉÌÒÙ])',
            # Italian specific patterns
            r'([.!?]+)\s+(?=(?:Il|La|Lo|Gli|Le|Un|Una|Che|Come|Quando|Dove|Perché|Se)\s)',
            # English specific patterns
            r'([.!?]+)\s+(?=(?:The|A|An|What|How|When|Where|Why|If)\s)',
            # List item boundaries (preserve list structure)
            r';\s*(?=\n|\s*[A-ZÀÈÉÌÒÙ])',
        ]

        # Try each pattern and use the one that gives the best split
        best_sentences = [text]

        for pattern in patterns:
            candidate_sentences = re.split(pattern, text)

            # Reconstruct sentences with punctuation
            reconstructed = []
            i = 0
            while i < len(candidate_sentences):
                sentence = candidate_sentences[i].strip()

                # Add back punctuation if it was captured
                if i + 1 < len(candidate_sentences) and re.match(r'^[.!?]+$', candidate_sentences[i + 1]):
                    sentence += candidate_sentences[i + 1]
                    i += 2
                else:
                    i += 1

                if sentence and len(sentence) > 10:  # Avoid tiny fragments
                    reconstructed.append(sentence)

            # Use this split if it's better than current best
            if len(reconstructed) > len(best_sentences) and len(reconstructed) > 1:
                # Check quality of split
                avg_length = sum(len(s) for s in reconstructed) / len(reconstructed)
                if 20 < avg_length < 500:  # Reasonable sentence lengths
                    best_sentences = reconstructed

        return best_sentences

    def _safe_split_large_semantic_block(self, block: str, semantic_type: str) -> List[str]:
        """Safely split a large semantic block while preserving context."""

        # First try enhanced semantic subsplitting
        sub_blocks = self._safe_semantic_subsplit(block, semantic_type)

        if all(len(sb) <= self.config.max_chunk_size for sb in sub_blocks):
            return sub_blocks

        # If still too large, use enhanced sentence-based splitting
        final_chunks = []

        for sub_block in sub_blocks:
            if len(sub_block) <= self.config.max_chunk_size:
                final_chunks.append(sub_block)
            else:
                # Enhanced sentence splitting with clause preservation
                sentence_chunks = self._split_by_enhanced_clauses(sub_block)
                final_chunks.extend(sentence_chunks)

        return final_chunks

    def _split_by_enhanced_clauses(self, text: str) -> List[str]:
        """Split text by enhanced clause detection while preserving meaning."""

        if len(text) <= self.config.max_chunk_size:
            return [text]

        # Enhanced clause boundary markers for Italian and English
        clause_patterns = [
            # Italian clause markers
            r',\s+(?=(?:che|il quale|la quale|dove|quando|come|perché|mentre|anche se|tuttavia|inoltre|quindi|pertanto)\s)',
            r';\s+',
            r':\s+(?=[A-ZÀÈÉÌÒÙ])',
            r'\s+(?:e|o|ma|però|tuttavia|inoltre|quindi|pertanto|mentre)\s+',
            r'\s+–\s+',
            r'\s+—\s+',

            # English clause markers
            r',\s+(?=(?:which|who|where|when|how|because|while|although|however|furthermore|therefore|thus)\s)',
            r'\s+(?:and|or|but|however|furthermore|therefore|thus|while)\s+',
        ]

        best_split = None
        best_balance = float('inf')

        for pattern in clause_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))

            for match in matches:
                split_pos = match.end()
                left_part = text[:split_pos].strip()
                right_part = text[split_pos:].strip()

                # Check if split creates reasonable chunks
                if (self.config.min_chunk_size <= len(left_part) <= self.config.max_chunk_size and
                        len(right_part) >= self.config.min_chunk_size):

                    # Prefer splits that create balanced chunks
                    balance = abs(len(left_part) - len(right_part))
                    if balance < best_balance:
                        best_balance = balance
                        best_split = (left_part, right_part)

        if best_split:
            left_part, right_part = best_split
            chunks = [left_part]

            # Recursively split right part if still too long
            if len(right_part) > self.config.max_chunk_size:
                chunks.extend(self._split_by_enhanced_clauses(right_part))
            else:
                chunks.append(right_part)

            return chunks

        # Last resort: preserve as much semantic meaning as possible
        return self._preserve_semantic_meaning_split(text)

    def _preserve_semantic_meaning_split(self, text: str) -> List[str]:
        """Last resort splitting that tries to preserve semantic meaning."""

        # If we must split, try to do it at the most meaningful boundary

        # Try to split at paragraph boundaries first
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            return self._group_paragraphs_by_size(paragraphs)

        # Try to split at sentence boundaries
        sentences = self._enhanced_split_into_sentences(text)
        if len(sentences) > 1:
            return self._group_sentences_by_size(sentences)

        # Absolute last resort: split but add clear continuation markers
        mid_point = len(text) // 2

        # Try to find a word boundary near the midpoint
        best_split = mid_point
        for offset in range(0, 100):  # Look within 100 characters
            for pos in [mid_point - offset, mid_point + offset]:
                if 0 < pos < len(text) and text[pos].isspace():
                    best_split = pos
                    break
            if best_split != mid_point:
                break

        left_part = text[:best_split].strip()
        right_part = text[best_split:].strip()

        # Add continuation markers to indicate the split
        if not left_part.endswith(('.', '!', '?', ';', ':')):
            left_part += "..."

        return [left_part, right_part]

    def _group_paragraphs_by_size(self, paragraphs: List[str]) -> List[str]:
        """Group paragraphs into chunks of appropriate size."""
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            test_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para

            if len(test_chunk) <= self.config.max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Handle oversized paragraph
                if len(para) > self.config.max_chunk_size:
                    chunks.extend(self._preserve_semantic_meaning_split(para))
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _group_sentences_by_size(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks of appropriate size."""
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            test_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence

            if len(test_chunk) <= self.config.max_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Handle oversized sentence
                if len(sentence) > self.config.max_chunk_size:
                    chunks.extend(self._preserve_semantic_meaning_split(sentence))
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _add_enhanced_semantic_overlap(self, chunks: List[str], semantic_type: str) -> List[str]:
        """Add enhanced semantic overlap between chunks to preserve context."""

        if len(chunks) <= 1:
            return chunks

        # Adjust overlap size based on semantic type
        if semantic_type in ['definitions', 'procedures', 'coverage']:
            overlap_size = min(self.config.overlap_size * 2, 300)  # More overlap for important content
        elif semantic_type == 'exclusions':
            overlap_size = min(self.config.overlap_size, 150)  # Less overlap for exclusions
        else:
            overlap_size = self.config.overlap_size

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Extract meaningful overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = self._extract_semantic_overlap(prev_chunk, overlap_size, semantic_type)

                if overlap_text:
                    # Add context separator
                    separator = "\n\n[Continued from previous section]\n"
                    overlapped_chunk = f"{overlap_text.strip()}{separator}{chunk}"
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _extract_semantic_overlap(self, text: str, max_overlap_size: int, semantic_type: str) -> str:
        """Extract semantically meaningful overlap from text."""

        if len(text) <= max_overlap_size:
            return ""  # Don't overlap if the entire chunk is small

        # Get the last portion of text for overlap
        tail = text[-max_overlap_size * 2:]  # Look at more text to find good boundary

        # Find semantic boundaries for different content types
        if semantic_type == 'definitions':
            # For definitions, try to include the complete definition
            sentences = self._enhanced_split_into_sentences(tail)
            if len(sentences) > 1:
                # Include last complete definition/explanation
                return sentences[-1] if len(sentences[-1]) <= max_overlap_size else ""

        elif semantic_type == 'procedures':
            # For procedures, include the last complete step or instruction
            if re.search(r'\d+\.\s+', tail):
                # Find last numbered step
                steps = re.split(r'(?=\d+\.\s+)', tail)
                if steps and len(steps[-1]) <= max_overlap_size:
                    return steps[-1].strip()

        elif semantic_type in ['coverage', 'exclusions']:
            # For coverage/exclusions, include last complete item or clause
            if re.search(r'[•\-\*]\s+', tail):
                items = re.split(r'(?=[•\-\*]\s+)', tail)
                if items and len(items[-1]) <= max_overlap_size:
                    return items[-1].strip()

        # Default: extract last complete sentence(s)
        sentences = self._enhanced_split_into_sentences(tail)
        if len(sentences) > 1:
            last_sentence = sentences[-1]
            if len(last_sentence) <= max_overlap_size:
                return last_sentence
            elif len(sentences) > 2:
                # Try last two sentences if they fit
                last_two = sentences[-2] + " " + sentences[-1]
                if len(last_two) <= max_overlap_size:
                    return last_two

        return ""

    def _create_enhanced_intelligent_fallback_chunks(self, text: str, tables_by_page: Dict[int, List[Dict]]) -> \
    List[DocumentChunk]:
        """Enhanced intelligent fallback chunking with better quality control."""
        chunks = []
        chunk_counter = 0

        elements = self._enhanced_parse_text_with_page_tracking(text)

        if not elements:
            return self._create_enhanced_basic_page_chunks(text, tables_by_page)

        current_section_title = "Document Content"
        current_section_content = []
        current_page = 1
        current_semantic_type = "general"

        for element in elements:
            if element['type'] == 'structure':
                # Save previous content
                if current_section_content:
                    content_text = '\n'.join(current_section_content).strip()
                    if len(content_text) >= self.config.min_chunk_size:
                        chunk = self._create_enhanced_quality_chunk(
                            chunk_counter, current_section_title, content_text,
                            current_page, [current_section_title], tables_by_page,
                            "enhanced_intelligent_fallback", current_semantic_type
                        )
                        chunks.append(chunk)
                        chunk_counter += 1

                # Start new section
                current_section_title = element['title']
                current_section_content = []
                current_page = element['page']
                current_semantic_type = element.get('semantic_type', 'general')

            elif element['type'] == 'content':
                current_section_content.append(element['content'])
                current_page = element['page']
                if element.get('semantic_type', 'general') != 'general':
                    current_semantic_type = element['semantic_type']

        # Save final content
        if current_section_content:
            content_text = '\n'.join(current_section_content).strip()
            if len(content_text) >= self.config.min_chunk_size:
                chunk = self._create_enhanced_quality_chunk(
                    chunk_counter, current_section_title, content_text,
                    current_page, [current_section_title], tables_by_page,
                    "enhanced_intelligent_fallback", current_semantic_type
                )
                chunks.append(chunk)
                chunk_counter += 1

        self.logger.info(f"Created {len(chunks)} enhanced intelligent fallback chunks")
        return chunks

    def _create_enhanced_basic_page_chunks(self, text: str, tables_by_page: Dict[int, List[Dict]]) -> List[
        DocumentChunk]:
        """Enhanced basic page-based chunking with better semantic preservation."""
        chunks = []
        chunk_counter = 0
        current_page = 1
        current_content = []

        lines = text.split('\n')

        for line in lines:
            # Check for page markers
            page_patterns = [
                r'^---\s*Page\s+(\d+)\s*---\s*$',
                r'^Pag\.\s*(\d+)\s*di\s*\d+\s*$',
            ]

            page_found = False
            for pattern in page_patterns:
                page_match = re.match(pattern, line.strip())
                if page_match:
                    # Save previous page content
                    if current_content:
                        content_text = '\n'.join(current_content).strip()
                        if len(content_text) >= self.config.min_chunk_size:
                            semantic_type = self._detect_content_semantic_type(content_text)
                            chunk = self._create_enhanced_quality_chunk(
                                chunk_counter, f"Page {current_page}", content_text,
                                current_page, [f"Page {current_page}"], tables_by_page,
                                "enhanced_basic_page", semantic_type
                            )
                            chunks.append(chunk)
                            chunk_counter += 1

                    # Start new page
                    current_page = int(page_match.group(1))
                    current_content = []
                    page_found = True
                    break

            if not page_found:
                current_content.append(line)

        # Save final page
        if current_content:
            content_text = '\n'.join(current_content).strip()
            if len(content_text) >= self.config.min_chunk_size:
                semantic_type = self._detect_content_semantic_type(content_text)
                chunk = self._create_enhanced_quality_chunk(
                    chunk_counter, f"Page {current_page}", content_text,
                    current_page, [f"Page {current_page}"], tables_by_page,
                    "enhanced_basic_page", semantic_type
                )
                chunks.append(chunk)
                chunk_counter += 1

        self.logger.info(f"Created {len(chunks)} enhanced basic page chunks")
        return chunks

    def _determine_level_from_title(self, title: str) -> int:
        """Determine hierarchical level from title with enhanced pattern recognition."""
        if not title:
            return 999

        title_clean = title.strip()

        # Level 0: Questions (highest priority semantic sections)
        if title_clean.endswith('?'):
            return 0

        # Level 1: Main sections (SECTION, SEZIONE, etc.)
        if re.search(r'^(?:SECTION|SEZIONE|PARTE|PART|CAPITOLO|CHAPTER)\s+[A-Z\d]+', title_clean, re.IGNORECASE):
            return 1

        # Level 1: All caps titles (likely main sections)
        if re.match(r'^[A-ZÀÈÉÌÒÙ][A-ZÀÈÉÌÒÙ\s]*[A-ZÀÈÉÌÒÙ]$', title_clean):
            return 1

        # Level 1: Italian insurance specific sections
        italian_main_sections = [
            r'^(?:ASSISTENZA\s+IN\s+VIAGGIO|SPESE\s+MEDICHE|BAGAGLIO|ANNULLAMENTO\s+VIAGGIO)',
            r'^(?:GARANZIE?|COPERTURE?|ESCLUSIONI|CONDIZIONI)'
        ]
        for pattern in italian_main_sections:
            if re.search(pattern, title_clean, re.IGNORECASE):
                return 1

        # Level 2: Articles (ARTICLE, ARTICOLO, ART.)
        if re.search(r'^(?:ARTICLE|ARTICOLO|ART\.)\s+\d+', title_clean, re.IGNORECASE):
            return 2

        # Level 2: Numbered sections (1.0, 2.0, etc.)
        if re.match(r'^\d+\.\d+\s*[-–—:]*\s*.+$', title_clean):
            return 2

        # Level 2: Letter-number combinations (A1, B.1, etc.)
        if re.match(r'^[A-Z]\d*\s*[-–—:]*\s*.+$', title_clean):
            return 2

        # Level 3: Detailed subsections (1.2.3, A.1.2, etc.)
        if re.match(r'^\d+\.\d+\.\d+', title_clean):
            return 3

        # Level 3: Letter-dot-number (A.1, B.2, etc.)
        if re.match(r'^[A-Z]\.\d+\s*[-–—:]*\s*.+$', title_clean):
            return 3

        # Level 4: Lettered sub-items (a), b), etc.)
        if re.match(r'^[a-z]\)\s*.+$', title_clean):
            return 4

        # Level 4: Single letter subsections (A., B., etc.)
        if re.match(r'^[A-Z]\.\s*.+$', title_clean):
            return 4

        # Default: content level
        return 999

    def _extract_section_id_from_title(self, title: str) -> str:
        """Extract section ID from title with comprehensive pattern matching."""
        if not title:
            return ""

        title_clean = title.strip()

        # Pattern 1: Section/Article with Roman or Arabic numerals
        # SECTION A, SEZIONE I, ARTICLE 1, ARTICOLO 2
        match = re.match(
            r'^(?:SECTION|SEZIONE|PARTE|PART|CAPITOLO|CHAPTER|ARTICLE|ARTICOLO|ART\.)\s+([A-Z\d]+(?:\s+[IV]+)?)',
            title_clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 2: Numbered sections (1.2.3, 2.1, etc.)
        match = re.match(r'^(\d+(?:\.\d+)*)', title_clean)
        if match:
            return match.group(1)

        # Pattern 3: Letter-number combinations (A1, B2, etc.)
        match = re.match(r'^([A-Z]\d*)', title_clean)
        if match:
            return match.group(1)

        # Pattern 4: Letter-dot-number (A.1, B.2, etc.)
        match = re.match(r'^([A-Z]\.\d+)', title_clean)
        if match:
            return match.group(1)

        # Pattern 5: Single letters with parentheses (a), b), etc.)
        match = re.match(r'^([a-z])\)', title_clean)
        if match:
            return match.group(1)

        # Pattern 6: Single letters with dots (A., B., etc.)
        match = re.match(r'^([A-Z])\.', title_clean)
        if match:
            return match.group(1)

        # Pattern 7: Italian specific patterns (C.1 - Oggetto dell'assicurazione)
        match = re.match(r'^([A-Z]\.?\d*)\s*[-–—:]*\s*', title_clean)
        if match:
            return match.group(1)

        # Pattern 8: Roman numerals
        match = re.match(r'^([IVX]+)', title_clean)
        if match:
            return match.group(1)

        # No identifiable section ID
        return ""

    def _create_enhanced_quality_chunk(self, chunk_id: int, title: str, content: str, page: int,
                                       section_path: List[str], tables_by_page: Dict[int, List[Dict]],
                                       method: str, semantic_type: str = "general",
                                       quality_indicators: Dict[str, Any] = None) -> DocumentChunk:
        """Create an enhanced quality document chunk with comprehensive metadata."""

        # Enhanced content cleaning and validation
        content = self.clean_text(content)

        # Calculate quality score
        quality_score = self._calculate_chunk_quality_score(title, content, semantic_type, quality_indicators or {})

        # Enhanced metadata
        enhanced_metadata = {
            'level': self._determine_level_from_title(title),
            'section_id': self._extract_section_id_from_title(title),
            'word_count': len(content.split()),
            'char_count': len(content),
            'sentence_count': len(self._enhanced_split_into_sentences(content)),
            'chunk_method': method,
            'parent_section': section_path[-2] if len(section_path) > 1 else None,
            'document_hierarchy': section_path.copy(),
            'language': self._detect_chunk_language(content),
            'content_type': self._classify_content_type(title, content),
            'semantic_type': semantic_type,
            'quality_score': quality_score,
            'has_questions': bool(re.search(r'\?', content)),
            'has_lists': bool(re.search(r'^\s*[•\-\*\d+\.\)]\s', content, re.MULTILINE)),
            'has_definitions': bool(
                re.search(r'(significa|significa che|definisce|definition|means|refers to)', content,
                          re.IGNORECASE)),
            'has_procedures': bool(
                re.search(r'(procedura|procedure|modalità|step|fase|point)', content, re.IGNORECASE)),
            'context_preserved': method in ['enhanced_structured', 'enhanced_intelligent_fallback'],
            'is_complete_concept': self._is_complete_concept(content, semantic_type),
            'readability_score': self._calculate_readability_score(content),
            'topic_keywords': self._extract_topic_keywords(content, semantic_type),
        }

        chunk = DocumentChunk(
            chunk_id=f"chunk_{chunk_id:04d}",
            chunk_type=self._determine_enhanced_chunk_type(title, content, semantic_type),
            title=title,
            content=content,
            page_number=page,
            section_path=section_path.copy(),
            metadata=enhanced_metadata,
            tables=tables_by_page.get(page, [])
        )

        return chunk

    def _calculate_chunk_quality_score(self, title: str, content: str, semantic_type: str,
                                       quality_indicators: Dict[str, Any]) -> float:
        """Calculate a comprehensive quality score for the chunk."""

        score = 0.0

        # Title quality (20% of score)
        if title and title.strip():
            score += 0.1
            if 10 <= len(title) <= 100:
                score += 0.05
            if title.endswith('?'):  # Questions are valuable
                score += 0.05

        # Content quality (40% of score)
        if content and content.strip():
            score += 0.1
            word_count = len(content.split())
            if self.config.min_chunk_size <= len(content) <= self.config.optimal_chunk_size:
                score += 0.1
            if word_count >= 20:  # Substantial content
                score += 0.05
            if re.search(r'[.!?]', content):  # Has proper punctuation
                score += 0.05
            if not re.search(r'\.\.\.|…', content):  # Not truncated
                score += 0.1

        # Semantic value (20% of score)
        if semantic_type != 'general':
            score += 0.1
            if semantic_type in ['coverage', 'exclusions', 'definitions', 'procedures']:
                score += 0.1  # High-value semantic types

        # Structure and completeness (20% of score)
        if self._is_complete_concept(content, semantic_type):
            score += 0.1
        if quality_indicators.get('has_section_id', False):
            score += 0.05
        if not quality_indicators.get('is_orphaned', False):
            score += 0.05

        return min(1.0, score)  # Cap at 1.0

    def _is_complete_concept(self, content: str, semantic_type: str) -> bool:
        """Check if the content represents a complete concept."""

        # Check for truncation indicators
        if content.endswith('...') or content.endswith('…'):
            return False

        # Check for incomplete sentences
        if not re.search(r'[.!?]\s*$', content.strip()):
            # Allow exceptions for lists and structured content
            if not re.search(r'^\s*[•\-\*\d+\.\)]\s', content, re.MULTILINE):
                return False

        # Semantic type specific checks
        if semantic_type == 'definitions':
            # Should have definition indicators
            if not re.search(r'(significa|significa che|definisce|definition|means|refers to)', content,
                             re.IGNORECASE):
                return False

        elif semantic_type == 'procedures':
            # Should have procedural indicators
            if not re.search(r'(procedura|procedure|modalità|step|fase|point|come|how)', content, re.IGNORECASE):
                return False

        return True

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate a basic readability score for the content."""

        if not content:
            return 0.0

        words = content.split()
        sentences = self._enhanced_split_into_sentences(content)

        if not words or not sentences:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Simple readability metric (inverse of average sentence length)
        # Shorter sentences are generally more readable
        if avg_words_per_sentence <= 15:
            readability = 1.0
        elif avg_words_per_sentence <= 25:
            readability = 0.8
        elif avg_words_per_sentence <= 35:
            readability = 0.6
        else:
            readability = 0.4

        # Bonus for proper punctuation and structure
        if re.search(r'[.!?]', content):
            readability += 0.1

        # Penalty for excessive jargon (very long words)
        long_words = [w for w in words if len(w) > 12]
        if len(long_words) / len(words) > 0.1:  # More than 10% long words
            readability -= 0.2

        return max(0.0, min(1.0, readability))

    def _extract_topic_keywords(self, content: str, semantic_type: str) -> List[str]:
        """Extract key topic keywords from content."""

        keywords = []
        content_lower = content.lower()

        # Use profile-specific semantic indicators if available
        if self.config.company_profile and hasattr(self.config.company_profile, 'semantic_indicators'):
            for topic, indicators in self.config.company_profile.semantic_indicators.items():
                for indicator in indicators:
                    if indicator in content_lower:
                        keywords.append(topic)
                        break

        # Add semantic type as keyword
        if semantic_type != 'general':
            keywords.append(semantic_type)

        # Extract important domain-specific terms
        domain_terms = {
            'italian': [
                'assicurato', 'polizza', 'garanzia', 'copertura', 'sinistro', 'premio',
                'rimborso', 'indennizzo', 'esclusioni', 'assistenza', 'viaggio', 'bagaglio'
            ],
            'english': [
                'insured', 'policy', 'coverage', 'claim', 'premium', 'reimbursement',
                'exclusions', 'assistance', 'travel', 'baggage', 'benefits'
            ]
        }

        language = self._detect_chunk_language(content)
        if language in domain_terms:
            for term in domain_terms[language]:
                if term in content_lower:
                    keywords.append(term)

        return list(set(keywords))  # Remove duplicates

    def _enhance_chunk_quality(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Enhance chunk quality through post-processing validation and improvement."""

        if not chunks:
            return chunks

        enhanced_chunks = []

        for chunk in chunks:
            # Skip low-quality chunks if quality filtering is enabled
            quality_score = chunk.metadata.get('quality_score', 0.0)
            if quality_score < self.config.min_quality_score:
                self.logger.warning(f"Filtered out low-quality chunk: {chunk.title} (score: {quality_score:.2f})")
                continue

            # Enhance content readability if enabled
            if self.config.enhance_readability:
                chunk.content = self._enhance_content_readability(chunk.content)

            # Validate semantic completeness if required
            if self.config.preserve_semantic_units:
                if not self._is_complete_concept(chunk.content, chunk.metadata.get('semantic_type', 'general')):
                    self.logger.warning(f"Chunk may have incomplete semantic content: {chunk.title}")
                    # Try to merge with next chunk if possible
                    # (This would require more complex logic to implement properly)

            # Ensure complete sentences if required
            if self.config.require_complete_sentences:
                chunk.content = self._ensure_complete_sentences(chunk.content)

            enhanced_chunks.append(chunk)

        self.logger.info(f"Enhanced chunk quality: {len(chunks)} -> {len(enhanced_chunks)} chunks")
        return enhanced_chunks

    def _enhance_content_readability(self, content: str) -> str:
        """Enhance content readability through formatting improvements."""

        # Ensure proper spacing after punctuation
        content = re.sub(r'([.!?])(\w)', r'\1 \2', content)

        # Fix common formatting issues
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Normalize line breaks

        # Ensure proper punctuation spacing
        content = re.sub(r'\s+([.,;:!?])', r'\1', content)

        # Fix number formatting
        content = re.sub(r'(\d)\s+([.,])\s*(\d)', r'\1\2\3', content)

        return content.strip()

    def _ensure_complete_sentences(self, content: str) -> str:
        """Ensure content ends with complete sentences."""

        content = content.strip()

        # If content doesn't end with proper punctuation
        if not re.search(r'[.!?]\s*$', content):
            # Check if it's a list or structured content (these are OK without final punctuation)
            if re.search(r'^\s*[•\-\*\d+\.\)]\s', content, re.MULTILINE):
                return content  # Lists are OK as-is

            # For regular text, try to find the last complete sentence
            sentences = self._enhanced_split_into_sentences(content)
            if len(sentences) > 1:
                # Keep only complete sentences
                complete_sentences = []
                for sentence in sentences:
                    if re.search(r'[.!?]\s*$', sentence.strip()):
                        complete_sentences.append(sentence)
                    else:
                        break  # Stop at first incomplete sentence

                if complete_sentences:
                    content = ' '.join(complete_sentences).strip()

        return content

    def _validate_enhanced_chunk(self, chunk: DocumentChunk) -> bool:
        """Enhanced chunk validation with comprehensive quality checks."""

        if not chunk.content or not chunk.content.strip():
            return False

        content_length = len(chunk.content.strip())

        # Basic size validation
        if content_length < self.config.min_chunk_size:
            return False

        if content_length > self.config.max_chunk_size * 1.2:  # Allow some flexibility
            return False

        # Content quality validation
        if not re.search(r'\w+', chunk.content):
            return False

        # Language validation
        if chunk.metadata.get('language', 'unknown') == 'unknown':
            # Try to re-detect language
            chunk.metadata['language'] = self._detect_chunk_language(chunk.content)

        # Quality score validation
        quality_score = chunk.metadata.get('quality_score', 0.0)
        if quality_score < self.config.min_quality_score:
            return False

        # Semantic completeness validation (if enabled)
        if self.config.preserve_semantic_units:
            semantic_type = chunk.metadata.get('semantic_type', 'general')
            if not self._is_complete_concept(chunk.content, semantic_type):
                return False

        return True

    def _determine_enhanced_chunk_type(self, title: str, content: str, semantic_type: str) -> str:
        """Enhanced chunk type determination with semantic understanding."""

        title_lower = title.lower()
        content_lower = content.lower()

        # Question-based sections (highest priority)
        if title.endswith('?'):
            if re.search(r'che cosa.*assicurat', title_lower) or re.search(r'what.*insured', title_lower):
                return "coverage_section"
            elif re.search(r'che cosa non.*assicurat', title_lower) or re.search(r'what.*not.*insured',
                                                                                 title_lower):
                return "exclusions_section"
            elif re.search(r'limiti.*copertura', title_lower) or re.search(r'coverage.*limits', title_lower):
                return "limits_section"
            elif re.search(r'obblighi', title_lower) or re.search(r'obligations', title_lower):
                return "obligations_section"
            elif re.search(r'sinistro|claims?', title_lower):
                return "claims_section"
            elif re.search(r'pagare.*premio|payment', title_lower):
                return "payment_section"
            else:
                return "question_section"

        # Semantic type-based classification
        if semantic_type == 'coverage':
            return "coverage_section"
        elif semantic_type == 'exclusions':
            return "exclusions_section"
        elif semantic_type == 'claims':
            return "claims_section"
        elif semantic_type == 'premium':
            return "payment_section"
        elif semantic_type == 'assistance':
            return "assistance_section"
        elif semantic_type == 'definitions':
            return "definitions_section"
        elif semantic_type == 'procedures':
            return "procedures_section"

        # Traditional structural classification
        if re.search(r'section|sezione', title_lower):
            return "section"
        elif re.search(r'article|articolo', title_lower):
            return "article"
        elif re.search(r'^\d+\.\d+', title):
            return "subsection"
        elif re.search(r'^[A-Z]\.\d+', title):
            return "clause"
        else:
            return "content_block"

    def _classify_content_type(self, title: str, content: str) -> str:
        """Enhanced content type classification for RAG optimization."""

        content_lower = content.lower()
        title_lower = title.lower()

        # Use enhanced semantic detection
        semantic_type = self._detect_content_semantic_type(content)
        if semantic_type != 'general':
            return semantic_type

        # Enhanced pattern matching
        patterns = {
            'exclusions': [
                r'esclus[oi]', r'non\s+(?:coperto|assicurato)', r'limitazion[ei]', r'restrizion[ei]',
                r'exclud[ed]', r'not\s+(?:covered|insured)', r'limitation[s]?', r'restriction[s]?'
            ],
            'coverage': [
                r'copertur[ao]', r'assicurat[oi]', r'garanzi[ao]', r'protezion[ei]', r'benefici[oi]',
                r'cover(?:age|ed|s)', r'insur(?:ed|ance)', r'protect(?:ion|ed)', r'benefit[s]?'
            ],
            'limits': [
                r'limite[i]?', r'massimale[i]?', r'limit[s]?', r'maximum', r'cap'
            ],
            'premium': [
                r'premio', r'pagamento', r'costo', r'tariffa', r'importo',
                r'premium', r'payment', r'cost', r'fee', r'charge'
            ],
            'claims': [
                r'sinistro', r'danno', r'perdita', r'incidente', r'evento', r'denuncia',
                r'claim[s]?', r'damage', r'loss', r'incident', r'accident', r'report'
            ],
            'assistance': [
                r'assistenza', r'aiuto', r'supporto', r'servizio', r'soccorso',
                r'assistance', r'help', r'support', r'service', r'aid'
            ],
            'reimbursement': [
                r'rimborso', r'indennizzo', r'risarcimento',
                r'reimbursement', r'refund', r'compensation'
            ]
        }

        # Score each category
        scores = {}
        for category, category_patterns in patterns.items():
            score = 0
            for pattern in category_patterns:
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                score += matches * 2  # Weight content matches more

                # Also check title
                title_matches = len(re.findall(pattern, title_lower, re.IGNORECASE))
                score += title_matches * 3  # Weight title matches even more

            if score > 0:
                scores[category] = score

        # Return highest scoring category
        if scores:
            return max(scores, key=scores.get)

        return "general"

    def extract_tables_from_page(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables with improved accuracy and better structure."""
        tables = []
        try:
            strategy = self.config.table_detection_strategy
            if strategy == "aggressive":
                methods = [("lattice", 60), ("stream", 60)]
            elif strategy == "conservative":
                methods = [("lattice", 90)]
            else:
                methods = [("lattice", self.config.table_accuracy_threshold),
                           ("stream", self.config.table_accuracy_threshold)]

            for method, threshold in methods:
                try:
                    if method == "lattice":
                        camelot_tables = camelot.read_pdf(
                            pdf_path, pages=str(page_num), flavor="lattice",
                            line_scale=40, table_areas=None
                        )
                    else:
                        camelot_tables = camelot.read_pdf(
                            pdf_path, pages=str(page_num), flavor="stream",
                            edge_tol=500
                        )

                    for i, table in enumerate(camelot_tables):
                        if table.accuracy >= threshold:
                            structured_table = self._structure_enhanced_table(table.df, i, method, table.accuracy,
                                                                              page_num)
                            tables.append(structured_table)
                            self.logger.debug(f"Extracted {method} table {i} with accuracy {table.accuracy:.1f}%")

                    if tables and strategy == "adaptive":
                        break

                except Exception as e:
                    self.logger.debug(f"Table extraction with {method} failed: {e}")
                    continue

        except Exception as e:
            self.logger.warning(f"Table extraction failed for page {page_num}: {e}")

        return tables

    def _structure_enhanced_table(self, df: pd.DataFrame, table_id: int, method: str, accuracy: float,
                                  page_num: int) -> Dict[str, Any]:
        """Structure table data with enhanced processing and validation."""

        # Clean empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        if df.empty:
            return {
                "table_id": table_id, "page": page_num, "method": method,
                "accuracy": accuracy, "headers": [], "data": [],
                "raw_shape": (0, 0), "format": "structured_table",
                "metadata": {"header_row": 0, "data_rows": 0, "columns": 0}
            }

        # Enhanced header detection
        header_row = self._find_enhanced_header_row(df)

        if header_row < len(df):
            headers = df.iloc[header_row].fillna('').astype(str).tolist()
            data_rows = df.iloc[header_row + 1:] if header_row < len(df) - 1 else pd.DataFrame()
        else:
            headers = [f"Column_{i + 1}" for i in range(len(df.columns))]
            data_rows = df

        # Enhanced header cleaning
        headers = [self._clean_table_header(str(h), i) for i, h in enumerate(headers)]

        # Enhanced data structuring
        structured_data = []
        for _, row in data_rows.iterrows():
            row_data = {}
            for header, value in zip(headers, row):
                if pd.notna(value) and str(value).strip():
                    clean_value = self.clean_text(str(value))
                    if clean_value and clean_value != 'nan':
                        row_data[header] = clean_value
            if row_data:
                structured_data.append(row_data)

        return {
            "table_id": table_id, "page": page_num, "method": method,
            "accuracy": accuracy, "headers": headers, "data": structured_data,
            "raw_shape": df.shape, "format": "structured_table",
            "metadata": {
                "header_row": header_row, "data_rows": len(structured_data),
                "columns": len(headers), "quality_score": self._assess_table_quality(headers, structured_data)
            }
        }

    def _clean_table_header(self, header: str, index: int) -> str:
        """Clean and enhance table headers."""

        header = self.clean_text(header)

        if not header or header.lower() in ['nan', 'none', '']:
            return f"Column_{index + 1}"

        # Remove common artifacts
        header = re.sub(r'^\s*[-–—]+\s*', '', header)
        header = re.sub(r'\s*[-–—]+\s*$', '', header)

        # Limit length
        if len(header) > 50:
            header = header[:47] + "..."

        return header.strip() or f"Column_{index + 1}"

    def _find_enhanced_header_row(self, df: pd.DataFrame) -> int:
        """Find optimal header row with enhanced detection."""

        if len(df) <= 1:
            return 0

        scores = []
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            score = 0

            # Basic filled ratio
            filled_ratio = row.notna().sum() / len(row)
            score += filled_ratio * 10

            # Check for header-like content
            for cell in row:
                if pd.notna(cell):
                    cell_str = str(cell).strip()

                    # Common table header indicators
                    header_indicators = [
                        'OPTION', 'AMOUNT', 'TYPE', 'NAME', 'COVERAGE', 'LIMIT', 'PREMIUM',
                        'COPERTURA', 'GARANZIA', 'LIMITE', 'PREMIO', 'TIPO', 'NOME', 'IMPORTO'
                    ]

                    if any(indicator in cell_str.upper() for indicator in header_indicators):
                        score += 5

                    # Reasonable header length
                    if 3 <= len(cell_str) <= 50:
                        score += 2

                    # Proper capitalization
                    if cell_str.istitle() or cell_str.isupper():
                        score += 1

                    # Not purely numeric (headers usually aren't)
                    if not cell_str.replace('.', '').replace(',', '').isdigit():
                        score += 1

            scores.append(score)

        return scores.index(max(scores)) if scores else 0

    def _assess_table_quality(self, headers: List[str], data: List[Dict[str, Any]]) -> float:
        """Assess table quality score."""

        score = 0.0

        # Header quality
        if headers:
            score += 0.3
            unique_headers = len(set(headers))
            if unique_headers == len(headers):  # All unique
                score += 0.2

        # Data quality
        if data:
            score += 0.3

            # Check data consistency
            if len(data) > 1:
                avg_fields = sum(len(row) for row in data) / len(data)
                if avg_fields >= len(headers) * 0.7:  # Most fields filled
                    score += 0.2

        return min(1.0, score)

    def _detect_chunk_language(self, content: str) -> str:
        """Enhanced language detection for individual chunks."""

        content_lower = content.lower()

        # Enhanced Italian indicators with weights
        italian_indicators = {
            'che': 3, 'della': 3, 'sono': 3, 'assicurato': 10, 'garanzia': 8, 'polizza': 10,
            'articolo': 5, 'società': 5, 'copertura': 8, 'sinistro': 8, 'rimborso': 5,
            'assistenza': 5, 'viaggio': 3, 'bagaglio': 5, 'spese': 3, 'mediche': 5
        }

        # Enhanced English indicators with weights
        english_indicators = {
            'the': 2, 'and': 2, 'of': 2, 'insured': 10, 'coverage': 8, 'policy': 10,
            'article': 5, 'company': 5, 'claim': 8, 'assistance': 5, 'travel': 3,
            'baggage': 5, 'medical': 5, 'expenses': 5, 'benefits': 5
        }

        italian_score = 0
        english_score = 0

        # Count weighted occurrences
        words = re.findall(r'\b\w+\b', content_lower)
        word_set = set(words)

        for word, weight in italian_indicators.items():
            if word in word_set:
                italian_score += weight

        for word, weight in english_indicators.items():
            if word in word_set:
                english_score += weight

        # Character-level indicators
        italian_chars = (content_lower.count('à') + content_lower.count('è') + content_lower.count('é') +
                         content_lower.count('ì') + content_lower.count('ò') + content_lower.count('ù'))
        italian_score += italian_chars * 5

        # Determine language
        if italian_score > english_score * 1.2:  # Require clear majority
            return 'italian'
        elif english_score > italian_score * 1.2:
            return 'english'
        else:
            return 'unknown'

    def extract_document_metadata(self, doc: fitz.Document, text_sample: str = "") -> Dict[str, Any]:
        """Extract comprehensive document metadata with enhanced analysis."""

        return {
            "page_count": doc.page_count,
            "title": doc.metadata.get('title', '') or self._extract_title_from_content(text_sample),
            "author": doc.metadata.get('author', ''),
            "subject": doc.metadata.get('subject', ''),
            "creator": doc.metadata.get('creator', ''),
            "producer": doc.metadata.get('producer', ''),
            "creation_date": doc.metadata.get('creationDate', ''),
            "mod_date": doc.metadata.get('modDate', ''),
            "is_pdf": doc.is_pdf,
            "is_encrypted": doc.is_encrypted,
            "detected_language": self._detect_document_language(text_sample),
            "company_profile": self.config.company_profile.name if self.config.company_profile else "unknown",
            "document_type": self.config.document_type.value,
            "processing_config": {
                "table_accuracy_threshold": self.config.table_accuracy_threshold,
                "max_chunk_size": self.config.max_chunk_size,
                "min_chunk_size": self.config.min_chunk_size,
                "optimal_chunk_size": self.config.optimal_chunk_size,
                "overlap_size": self.config.overlap_size,
                "preserve_semantic_units": self.config.preserve_semantic_units,
                "min_quality_score": self.config.min_quality_score
            },
            "quality_indicators": {
                "estimated_readability": self._estimate_document_readability(text_sample),
                "structure_complexity": self._estimate_structure_complexity(text_sample),
                "content_density": len(text_sample.split()) / max(1, text_sample.count('\n')),
            }
        }

    def _extract_title_from_content(self, text: str) -> str:
        """Enhanced title extraction from content."""

        lines = text.split('\n')[:15]  # Look at more lines

        for line in lines:
            line = line.strip()
            if 10 < len(line) < 200:
                # Enhanced title indicators
                title_indicators = [
                    'INSURANCE', 'ASSICURAZIONE', 'CONTRACT', 'CONTRATTO',
                    'POLICY', 'POLIZZA', 'CONDITIONS', 'CONDIZIONI',
                    'TRAVEL', 'VIAGGIO', 'PROTECTION', 'PROTEZIONE'
                ]

                if any(keyword in line.upper() for keyword in title_indicators):
                    return self.clean_text(line)

        return ""

    def _detect_document_language(self, text: str) -> str:
        """Enhanced document language detection with better accuracy."""

        text_lower = text.lower()

        # Enhanced Italian indicators with higher precision
        italian_indicators = {
            'che cosa': 20, 'assicurato': 15, 'polizza': 15, 'garanzia': 12, 'sezione': 8,
            'articolo': 8, 'copertura': 12, 'sinistro': 12, 'premio': 8, 'contratto': 8,
            'della': 5, 'sono': 5, 'società': 8, 'centrale operativa': 15, 'spese mediche': 15,
            'assistenza': 8, 'quando comincia': 12, 'come posso': 12, 'quali obblighi': 12,
            'cosa fare': 8, 'rimborso': 8, 'indennizzo': 8, 'esclusioni': 12
        }

        # Enhanced English indicators with higher precision
        english_indicators = {
            'what is': 20, 'insured': 15, 'policy': 15, 'coverage': 12, 'section': 8,
            'article': 8, 'insurance': 15, 'contract': 8, 'travel': 5, 'baggage': 8,
            'medical expenses': 15, 'assistance': 8, 'when does': 12, 'how can': 12,
            'what are my': 12, 'what to do': 8, 'reimbursement': 8, 'claims': 12,
            'exclusions': 12, 'company': 5, 'insured person': 15
        }

        italian_score = 0
        english_score = 0

        # Count weighted matches with context
        for indicator, weight in italian_indicators.items():
            count = text_lower.count(indicator)
            italian_score += count * weight

        for indicator, weight in english_indicators.items():
            count = text_lower.count(indicator)
            english_score += count * weight

        # Enhanced pattern bonuses
        if re.search(r'che cosa (?:è|sono)\s*assicurat[oi]\?', text_lower):
            italian_score += 30
        if re.search(r'what is (?:insured|covered)\?', text_lower):
            english_score += 30

        # Company-specific patterns with higher weights
        if any(pattern in text_lower for pattern in ['axa', 'inter partner assistance', 'centrale operativa']):
            italian_score += 25
        if any(pattern in text_lower for pattern in ['nobis', 'filo diretto']):
            english_score += 15

        # Enhanced character-level indicators
        italian_chars = (text_lower.count('à') + text_lower.count('è') + text_lower.count('é') +
                         text_lower.count('ì') + text_lower.count('ò') + text_lower.count('ù'))
        italian_score += italian_chars * 8

        # Document structure patterns
        if re.search(r'pag\.\s*\d+\s*di\s*\d+', text_lower):
            italian_score += 15
        if re.search(r'page \d+ of \d+', text_lower):
            english_score += 15

        self.logger.info(f"Enhanced language detection scores - Italian: {italian_score}, English: {english_score}")

        # More decisive thresholds
        if italian_score > english_score * 1.2:
            return 'italian'
        elif english_score > italian_score * 1.2:
            return 'english'
        else:
            # Enhanced tie-breaking with more specific indicators
            specific_italian = any(
                word in text_lower for word in ['assicurato', 'polizza', 'società', 'garanzia', 'che cosa'])
            specific_english = any(word in text_lower for word in ['insured', 'policy', 'company', 'what is'])

            if specific_italian and not specific_english:
                return 'italian'
            elif specific_english and not specific_italian:
                return 'english'
            else:
                return 'italian'  # Default to Italian for insurance docs

    def _estimate_document_readability(self, text: str) -> float:
        """Estimate document readability score."""

        if not text:
            return 0.0

        words = text.split()
        sentences = self._enhanced_split_into_sentences(text)

        if not words or not sentences:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Readability based on sentence length and structure
        if avg_words_per_sentence <= 20:
            readability = 0.9
        elif avg_words_per_sentence <= 30:
            readability = 0.7
        elif avg_words_per_sentence <= 40:
            readability = 0.5
        else:
            readability = 0.3

        # Adjust for document structure
        if re.search(r'^\s*[•\-\*\d+\.\)]\s', text, re.MULTILINE):
            readability += 0.1  # Lists improve readability

        return min(1.0, readability)

    def _estimate_structure_complexity(self, text: str) -> str:
        """Estimate document structure complexity."""

        if not text:
            return "unknown"

        # Count structural elements
        sections = len(re.findall(r'^\s*(?:SECTION|SEZIONE|PARTE|CAPITOLO)', text, re.MULTILINE | re.IGNORECASE))
        articles = len(re.findall(r'^\s*(?:ARTICLE|ARTICOLO|ART\.)', text, re.MULTILINE | re.IGNORECASE))
        questions = len(re.findall(r'[^?]*\?', text))
        lists = len(re.findall(r'^\s*[•\-\*\d+\.\)]\s', text, re.MULTILINE))

        total_elements = sections + articles + questions + lists

        if total_elements >= 20:
            return "high"
        elif total_elements >= 10:
            return "medium"
        elif total_elements >= 5:
            return "low"
        else:
            return "minimal"

    def process_pdf(self, pdf_path: str, auto_detect_profile: bool = True) -> Dict[str, Any]:
        """Enhanced PDF processing with comprehensive quality control and separate output files."""

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.logger.info(f"Starting enhanced PDF processing with quality optimization: {pdf_path}")

        doc = fitz.open(str(pdf_path))

        try:
            # Enhanced text extraction
            full_text = self.extract_with_pymupdf4llm(doc)
            self.logger.info(f"Extracted {len(full_text)} characters of text")

            if not full_text.strip():
                raise ValueError("No text content extracted from PDF")

            # Enhanced profile detection
            if auto_detect_profile and full_text:
                detected_profile = self.profile_manager.detect_best_profile(
                    full_text[:8000])  # Use more text for detection
                profile = self.profile_manager.get_profile(detected_profile)
                if profile:
                    self.config.company_profile = profile
                    self.logger.info(f"Applied enhanced profile: {detected_profile}")

            # Extract enhanced metadata
            text_sample = full_text[:5000] if full_text else ""
            metadata = self.extract_document_metadata(doc, text_sample)

            # Extract tables with enhanced processing
            tables_by_page = {}
            total_tables = 0
            for page_num in range(1, doc.page_count + 1):
                tables = self.extract_tables_from_page(str(pdf_path), page_num)
                if tables:
                    tables_by_page[page_num] = tables
                    total_tables += len(tables)

            self.logger.info(f"Extracted {total_tables} tables across {len(tables_by_page)} pages")

            # Create enhanced smart chunks with quality optimization
            document_chunks = self.create_enhanced_smart_chunks(full_text, tables_by_page)

            if not document_chunks:
                raise ValueError("Failed to create any chunks from document")

            # Final validation and quality enhancement
            document_chunks = [chunk for chunk in document_chunks if self._validate_enhanced_chunk(chunk)]

            if not document_chunks:
                raise ValueError("All chunks were filtered out during quality validation")

            # Build comprehensive processing stats
            processing_stats = {
                "total_pages": doc.page_count,
                "total_chunks": len(document_chunks),
                "total_tables": total_tables,
                "detected_language": metadata.get("detected_language", "unknown"),
                "company_profile": metadata.get("company_profile", "unknown"),
                "avg_chunk_size": sum(len(chunk.content) for chunk in document_chunks) // len(
                    document_chunks) if document_chunks else 0,
                "avg_quality_score": sum(chunk.metadata.get('quality_score', 0) for chunk in document_chunks) / len(
                    document_chunks) if document_chunks else 0,
                "high_quality_chunks": sum(
                    1 for chunk in document_chunks if chunk.metadata.get('quality_score', 0) >= 0.8),
                "section_distribution": self._analyze_enhanced_section_distribution(document_chunks),
                "content_type_distribution": self._analyze_content_type_distribution(document_chunks),
                "semantic_type_distribution": self._analyze_semantic_type_distribution(document_chunks),
                "language_distribution": self._analyze_language_distribution(document_chunks),
                "quality_distribution": self._analyze_quality_distribution(document_chunks),
                "tables_per_page": {page: len(tables) for page, tables in tables_by_page.items()},
                "processing_method": "enhanced_quality_optimized_chunking",
                "readability_metrics": {
                    "avg_readability": sum(
                        chunk.metadata.get('readability_score', 0) for chunk in document_chunks) / len(
                        document_chunks) if document_chunks else 0,
                    "chunks_with_complete_concepts": sum(
                        1 for chunk in document_chunks if chunk.metadata.get('is_complete_concept', False)),
                    "chunks_with_context": sum(
                        1 for chunk in document_chunks if chunk.metadata.get('context_preserved', False))
                }
            }

            result = {
                "source_file": str(pdf_path),
                "metadata": metadata,
                "chunks": [chunk.to_dict() for chunk in document_chunks],
                "rag_chunks": [chunk.to_rag_dict() for chunk in document_chunks],
                "tables_by_page": tables_by_page,
                "processing_stats": processing_stats
            }

            self.logger.info(f"✅ Successfully created {len(document_chunks)} enhanced quality-optimized chunks")
            self.logger.info(f"📊 Average quality score: {processing_stats['avg_quality_score']:.2f}")
            self.logger.info(f"🏆 High-quality chunks: {processing_stats['high_quality_chunks']}")

            return result

        finally:
            doc.close()

    def _analyze_enhanced_section_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Analyze enhanced chunk type distribution."""
        distribution = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            distribution[chunk_type] = distribution.get(chunk_type, 0) + 1
        return distribution

    def _analyze_content_type_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Analyze content type distribution."""
        distribution = {}
        for chunk in chunks:
            content_type = chunk.metadata.get('content_type', 'unknown')
            distribution[content_type] = distribution.get(content_type, 0) + 1
        return distribution

    def _analyze_semantic_type_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Analyze semantic type distribution."""
        distribution = {}
        for chunk in chunks:
            semantic_type = chunk.metadata.get('semantic_type', 'unknown')
            distribution[semantic_type] = distribution.get(semantic_type, 0) + 1
        return distribution

    def _analyze_language_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Analyze language distribution."""
        distribution = {}
        for chunk in chunks:
            language = chunk.metadata.get('language', 'unknown')
            distribution[language] = distribution.get(language, 0) + 1
        return distribution

    def _analyze_quality_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Analyze quality score distribution."""
        distribution = {
            "excellent (0.9-1.0)": 0,
            "good (0.8-0.9)": 0,
            "fair (0.6-0.8)": 0,
            "poor (0.4-0.6)": 0,
            "very_poor (0.0-0.4)": 0
        }

        for chunk in chunks:
            score = chunk.metadata.get('quality_score', 0.0)
            if score >= 0.9:
                distribution["excellent (0.9-1.0)"] += 1
            elif score >= 0.8:
                distribution["good (0.8-0.9)"] += 1
            elif score >= 0.6:
                distribution["fair (0.6-0.8)"] += 1
            elif score >= 0.4:
                distribution["poor (0.4-0.6)"] += 1
            else:
                distribution["very_poor (0.0-0.4)"] += 1

        return distribution

    def export_enhanced_separate_files(self, result: Dict[str, Any], base_filename: str) -> Dict[str, str]:
        """Export enhanced separate JSON files with comprehensive metadata."""

        # Enhanced RAG-optimized chunks file
        rag_chunks_file = f"{base_filename}_enhanced_rag_chunks.json"
        enhanced_rag_data = {
            "chunks": result["rag_chunks"],
            "processing_info": {
                "total_chunks": len(result["rag_chunks"]),
                "avg_quality_score": result["processing_stats"]["avg_quality_score"],
                "high_quality_chunks": result["processing_stats"]["high_quality_chunks"],
                "processing_method": result["processing_stats"]["processing_method"]
            }
        }
        with open(rag_chunks_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_rag_data, f, indent=2, ensure_ascii=False)

        # Enhanced metadata file
        metadata_file = f"{base_filename}_enhanced_metadata.json"
        metadata_content = {
            "source_file": result["source_file"],
            "document_metadata": result["metadata"],
            "tables_by_page": result["tables_by_page"],
            "quality_metrics": {
                "readability_metrics": result["processing_stats"]["readability_metrics"],
                "quality_distribution": result["processing_stats"]["quality_distribution"]
            }
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_content, f, indent=2, ensure_ascii=False)

        # Comprehensive stats file
        stats_file = f"{base_filename}_enhanced_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(result["processing_stats"], f, indent=2, ensure_ascii=False)

        # Quality analysis file
        quality_file = f"{base_filename}_quality_analysis.json"
        quality_analysis = {
            "overall_quality": {
                "avg_quality_score": result["processing_stats"]["avg_quality_score"],
                "high_quality_chunks": result["processing_stats"]["high_quality_chunks"],
                "total_chunks": result["processing_stats"]["total_chunks"]
            },
            "quality_distribution": result["processing_stats"]["quality_distribution"],
            "semantic_distribution": result["processing_stats"]["semantic_type_distribution"],
            "readability_metrics": result["processing_stats"]["readability_metrics"],
            "recommendations": self._generate_quality_recommendations(result["processing_stats"])
        }
        with open(quality_file, "w", encoding="utf-8") as f:
            json.dump(quality_analysis, f, indent=2, ensure_ascii=False)

        # Complete result file (legacy compatibility)
        complete_file = f"{base_filename}_complete_enhanced.json"
        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump({
                "source_file": result["source_file"],
                "metadata": result["metadata"],
                "chunks": result["chunks"],
                "tables_by_page": result["tables_by_page"],
                "processing_stats": result["processing_stats"]
            }, f, indent=2, ensure_ascii=False)

        return {
            "rag_chunks": rag_chunks_file,
            "metadata": metadata_file,
            "stats": stats_file,
            "quality_analysis": quality_file,
            "complete": complete_file
        }

    def _generate_quality_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations based on processing stats."""

        recommendations = []

        avg_quality = stats.get("avg_quality_score", 0.0)
        total_chunks = stats.get("total_chunks", 0)
        high_quality_chunks = stats.get("high_quality_chunks", 0)

        if total_chunks == 0:
            recommendations.append("No chunks were created - check document content and processing parameters")
            return recommendations

        high_quality_ratio = high_quality_chunks / total_chunks

        # Overall quality assessment
        if avg_quality < 0.6:
            recommendations.append(
                "Document has low overall quality - consider adjusting chunk size or reviewing source document")
        elif avg_quality < 0.8:
            recommendations.append("Document has moderate quality - some optimization possible")
        else:
            recommendations.append("Document has good overall quality")

        # High-quality chunk ratio
        if high_quality_ratio < 0.3:
            recommendations.append(
                "Low percentage of high-quality chunks - consider increasing min_chunk_size or improving semantic detection")
        elif high_quality_ratio < 0.6:
            recommendations.append("Moderate percentage of high-quality chunks - fine-tuning recommended")
        else:
            recommendations.append("Good percentage of high-quality chunks")

        # Readability analysis
        readability_metrics = stats.get("readability_metrics", {})
        avg_readability = readability_metrics.get("avg_readability", 0.0)

        if avg_readability < 0.5:
            recommendations.append(
                "Low readability detected - consider content simplification or better sentence boundary detection")
        elif avg_readability < 0.7:
            recommendations.append("Moderate readability - acceptable for technical documents")
        else:
            recommendations.append("Good readability scores")

        # Context preservation
        chunks_with_context = readability_metrics.get("chunks_with_context", 0)
        context_ratio = chunks_with_context / total_chunks if total_chunks > 0 else 0

        if context_ratio < 0.5:
            recommendations.append(
                "Low context preservation - consider enabling semantic overlap or adjusting chunking strategy")
        else:
            recommendations.append("Good context preservation")

        # Semantic distribution
        semantic_dist = stats.get("semantic_type_distribution", {})
        general_chunks = semantic_dist.get("general", 0)
        general_ratio = general_chunks / total_chunks if total_chunks > 0 else 0

        if general_ratio > 0.7:
            recommendations.append(
                "High percentage of 'general' semantic types - consider improving semantic classification patterns")
        elif general_ratio > 0.5:
            recommendations.append("Moderate semantic classification - some improvement possible")
        else:
            recommendations.append("Good semantic classification")

        return recommendations

def main():
    """Enhanced main function with comprehensive quality reporting and separate file outputs."""
    print("=== Enhanced Production-Ready Multi-Language RAG PDF Processor ===")
    print("🚀 Now with Advanced Quality Optimization and Semantic Understanding!")

    # Enhanced production configuration
    config = ProcessingConfig(
        table_accuracy_threshold=75.0,
        preserve_structure=True,
        extract_metadata=True,
        max_chunk_size=1500,  # Increased for better context
        min_chunk_size=150,  # Increased for better quality
        optimal_chunk_size=1000,  # Sweet spot for RAG
        overlap_size=200,  # Enhanced context preservation
        preserve_semantic_units=True,
        maintain_context=True,
        min_quality_score=0.6,  # Quality threshold
        require_complete_sentences=True,
        preserve_lists=True,
        preserve_qa_pairs=True,
        debug_mode=True,
        auto_detect_language=True,
        enhance_readability=True
    )

    preprocessor = EnhancedAdvancedPDFPreprocessor(config)

    # Process your documents
    name = "18_Nobis - Baggage loss EN"  # Update with your filename
    pdf_path = os.path.join(DOCUMENT_DIR, f"{name}.pdf")

    try:
        print(f"\n🔍 Processing: {pdf_path}")
        print("⚙️ Enhanced Quality Optimization Enabled")

        result = preprocessor.process_pdf(pdf_path, auto_detect_profile=True)

        # Export enhanced separate files
        output_files = preprocessor.export_enhanced_separate_files(result, name)

        # Comprehensive processing summary
        stats = result["processing_stats"]

        print(f"\n📊 ENHANCED PROCESSING RESULTS:")
        print(f"  📄 Document: {result['source_file']}")
        print(f"  🏢 Profile: {stats['company_profile']}")
        print(f"  🌍 Language: {stats['detected_language']}")
        print(f"  📑 Pages: {stats['total_pages']}")
        print(f"  🧩 Total Chunks: {stats['total_chunks']}")
        print(f"  📊 Tables: {stats['total_tables']}")
        print(f"  📏 Avg Chunk Size: {stats['avg_chunk_size']} chars")
        print(f"  ⭐ Avg Quality Score: {stats['avg_quality_score']:.2f}")
        print(f"  🏆 High-Quality Chunks: {stats['high_quality_chunks']}/{stats['total_chunks']}")

        print(f"\n📋 CHUNK TYPE DISTRIBUTION:")
        for chunk_type, count in stats['section_distribution'].items():
            print(f"  - {chunk_type}: {count} chunks")

        print(f"\n🏷️ CONTENT TYPE DISTRIBUTION:")
        for content_type, count in stats.get('content_type_distribution', {}).items():
            print(f"  - {content_type}: {count} chunks")

        print(f"\n🧠 SEMANTIC TYPE DISTRIBUTION:")
        for semantic_type, count in stats.get('semantic_type_distribution', {}).items():
            print(f"  - {semantic_type}: {count} chunks")

        print(f"\n🌍 LANGUAGE DISTRIBUTION:")
        for language, count in stats.get('language_distribution', {}).items():
            print(f"  - {language}: {count} chunks")

        print(f"\n📈 QUALITY DISTRIBUTION:")
        for quality_range, count in stats.get('quality_distribution', {}).items():
            print(f"  - {quality_range}: {count} chunks")

        print(f"\n📚 READABILITY METRICS:")
        readability = stats.get('readability_metrics', {})
        print(f"  - Average Readability: {readability.get('avg_readability', 0):.2f}")
        print(f"  - Complete Concepts: {readability.get('chunks_with_complete_concepts', 0)} chunks")
        print(f"  - Context Preserved: {readability.get('chunks_with_context', 0)} chunks")

        print(f"\n📖 SAMPLE ENHANCED QUALITY CHUNKS:")
        # Sort chunks by quality score and show the best ones
        sorted_chunks = sorted(result['rag_chunks'],
                               key=lambda x: x.get('quality_score', 0),
                               reverse=True)

        for i, chunk in enumerate(sorted_chunks[:5]):
            print(f"  {i + 1}. [{chunk['chunk_type'].upper()}] {chunk['title'][:60]}...")
            print(f"     📍 Page {chunk['page_number']} | {chunk['word_count']} words")
            print(f"     ⭐ Quality: {chunk.get('quality_score', 0):.2f}")
            print(f"     🌍 Language: {chunk.get('language', 'unknown')}")
            print(f"     🏷️ Content: {chunk.get('content_type', 'unknown')}")
            print(f"     🧠 Semantic: {chunk.get('semantic_type', 'unknown')}")
            print(f"     🗂️ Path: {' > '.join(chunk['section_path'])}")
            print(f"     📝 Preview: {chunk['content'][:80]}...")
            print()

        # Quality recommendations
        quality_analysis_file = output_files.get('quality_analysis', '')
        if quality_analysis_file and os.path.exists(quality_analysis_file):
            with open(quality_analysis_file, 'r', encoding='utf-8') as f:
                quality_data = json.load(f)
                recommendations = quality_data.get('recommendations', [])

                if recommendations:
                    print(f"💡 QUALITY RECOMMENDATIONS:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec}")

        print(f"\n✅ ENHANCED PROCESSING COMPLETED SUCCESSFULLY!")
        print(
            f"🎯 Quality Optimization: {stats['high_quality_chunks']}/{stats['total_chunks']} high-quality chunks")
        print(f"\n📁 OUTPUT FILES:")
        for file_type, filename in output_files.items():
            print(f"  - {file_type.upper().replace('_', ' ')}: {filename}")

        print(f"\n🚀 Ready for Advanced RAG Applications!")
        print("   - Enhanced semantic understanding")
        print("   - Improved context preservation")
        print("   - Quality-optimized chunking")
        print("   - Multi-language support")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()