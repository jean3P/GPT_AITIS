# src/models/vector_store.py

import logging
import re

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

    def extract_text_from_file(self, path: str) -> str:
        """Extract text from PDF or TXT file."""
        file_extension = os.path.splitext(path)[1].lower()

        if file_extension == '.pdf':
            return self.extract_text_from_pdf(path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return ""

    def extract_text_from_txt(self, path: str) -> str:
        """Extract text from a TXT file."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Successfully extracted {len(text)} characters from TXT: {path}")
            return text
        except UnicodeDecodeError:
            # Try different encodings if UTF-8 fails
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(path, 'r', encoding=encoding) as file:
                        text = file.read()
                    logger.info(f"Successfully extracted text from TXT using {encoding}: {path}")
                    return text
                except UnicodeDecodeError:
                    continue
            logger.error(f"Could not decode TXT file with any encoding: {path}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text from TXT {path}: {e}")
            return ""

    def extract_text_from_pdf(self, path: str) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            logger.error(f"Error extracting text from PDF {path}: {e}")
            return ""

    def chunk_text(self, text: str, max_length: int = 300) -> List[str]:
        """
        Split text into smaller chunks with thorough cleaning.
        """
        import re

        # Step 1: Aggressive cleaning
        # Remove page markers and === patterns ===
        text = re.sub(r'(=== PAGE \d+ ===|--- Page \d+ ---|Pag\.\s*\d+\s*di\s*\d+)', '', text)

        # Remove all patterns like === something ===
        text = re.sub(r'=== .* ===', '', text)

        # Remove document headers/footers patterns - GENERALIZED
        header_patterns = [
            r'.*?Polizza\s+Collettiva.*?(?:Pagina|ed\.).*',
            r'DIPA_.*?(?:Pagina|ed\.).*',
            r'.*?Alidays.*?ed\.\d+.*',
            r'.*?Condizioni per l\'Assicurato.*?ed\.\d+.*',
            r'.*?"ALI HEALTH \d+".*',
            r'.*?Pagina \d+ di \d+.*',
            r'.*?ed\.\s*\d+.*Pagina.*',
        ]

        for pattern in header_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove document metadata lines - NEW
        text = re.sub(r'.*Total pages:\s*\d+.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'.*Extracted from:.*\.pdf.*', '', text, flags=re.IGNORECASE)

        # Replace multiple newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove lines that are just whitespace, numbers, or separators
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Keep lines that have meaningful content (more than 3 chars, not just numbers/symbols)
            if len(line) > 3 and re.search(r'[a-zA-Z]{3,}', line):
                clean_lines.append(line)
            elif len(line) == 0 and clean_lines and clean_lines[-1] != '':
                clean_lines.append('')  # Keep paragraph breaks

        # Rejoin and normalize
        text = '\n'.join(clean_lines)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max double newlines

        # Step 2: Simple paragraph-based chunking
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < 10:  # Skip very short paragraphs
                continue

            # Replace internal newlines with spaces in paragraphs
            paragraph = re.sub(r'\n+', ' ', paragraph)
            paragraph = re.sub(r'\s+', ' ', paragraph)  # Normalize whitespace

            if len(current_chunk) + len(paragraph) + 2 < max_length:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Step 3: Final filtering
        final_chunks = []
        for chunk in chunks:
            # Only keep substantial chunks
            if len(chunk) >= 20 and len(re.findall(r'\b\w+\b', chunk)) >= 5:
                final_chunks.append(chunk)

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

    def classify_chunk_content(self, chunk: str) -> Dict[str, any]:
        """Enhanced metadata for precise coverage determination"""

        metadata = {
            # Core content classification (bilingual)
            'content_types': self._get_content_types(chunk),

            # Critical for coverage determination
            'contains_coverage_grant': self._has_coverage_granting_language(chunk),
            'contains_amount_specification': self._has_specific_amounts(chunk),
            'contains_conditions': self._has_conditional_language(chunk),
            'contains_exclusions': self._has_exclusion_language(chunk),
            'contains_procedures': self._has_procedural_requirements(chunk),

            # For exact quote extraction
            'monetary_values': self._extract_monetary_values(chunk),
            'coverage_triggers': self._extract_coverage_triggers(chunk),
            'condition_keywords': self._extract_condition_keywords(chunk),

            # Chunk quality indicators
            'is_complete_clause': self._looks_like_complete_clause(chunk),
            'cross_references': self._extract_cross_references(chunk)
        }

        return metadata

    def _get_content_types(self, chunk: str) -> List[str]:
        """Simple content classification (Italian & English)"""
        tags = []

        chunk_lower = chunk.lower()

        # Medical/Health coverage - Italian & English
        medical_terms = [
            # Italian
            'medic', 'spese mediche', 'ricovero', 'ospedale', 'assistenza sanitaria',
            'rimpatrio', 'emergenza', 'cure', 'sanitari', 'malattia', 'infortunio',
            # English
            'medical', 'hospital', 'healthcare', 'treatment', 'emergency', 'illness',
            'injury', 'repatriation', 'medical expenses'
        ]
        if any(term in chunk_lower for term in medical_terms):
            tags.append('medical')

        # Baggage coverage - Italian & English
        baggage_terms = [
            # Italian
            'bagaglio', 'effetti personali', 'smarrimento', 'furto', 'valigie',
            'danneggiamento', 'beni personali', 'oggetti',
            # English
            'baggage', 'luggage', 'personal effects', 'belongings', 'theft',
            'loss', 'damage', 'stolen', 'missing'
        ]
        if any(term in chunk_lower for term in baggage_terms):
            tags.append('baggage')

        # Exclusions - Italian & English
        exclusion_terms = [
            # Italian
            'esclus', 'non coperto', 'non è coperto', 'limitazioni', 'non opera',
            'non si applica', 'sono esclusi', 'non previsto',
            # English
            'exclusion', 'excluded', 'not covered', 'limitation', 'does not apply',
            'not applicable', 'restrictions'
        ]
        if any(term in chunk_lower for term in exclusion_terms):
            tags.append('exclusions')

        # Contact/Claims info - Italian & English
        contact_terms = [
            # Italian
            'telefono', 'email', 'contatto', 'sinistro', 'denuncia', 'centrale operativa',
            'assistenza', 'numero verde', 'recapito',
            # English
            'phone', 'telephone', 'contact', 'claim', 'report', 'assistance center',
            'helpline', 'emergency number'
        ]
        if any(term in chunk_lower for term in contact_terms):
            tags.append('contact')

        # Coverage limits/amounts - Italian & English
        financial_terms = [
            # Italian
            '€', 'euro', 'eur', 'massimale', 'franchigia', 'limite', 'importo',
            'rimborso', 'indennizzo', 'premio', 'costo',
            # English
            'amount', 'limit', 'maximum', 'deductible', 'coverage limit',
            'reimbursement', 'premium', 'cost', 'sum insured'
        ]
        if any(term in chunk_lower for term in financial_terms):
            tags.append('financial')

        # Travel/Trip related - Italian & English
        travel_terms = [
            # Italian
            'viaggio', 'vacanza', 'turismo', 'destinazione', 'partenza', 'rientro',
            'soggiorno', 'trasporto', 'volo', 'aereo',
            # English
            'travel', 'trip', 'vacation', 'journey', 'destination', 'departure',
            'return', 'flight', 'transport', 'accommodation'
        ]
        if any(term in chunk_lower for term in travel_terms):
            tags.append('travel')

        # Legal/Contractual terms - Italian & English
        legal_terms = [
            # Italian
            'contratto', 'polizza', 'assicurazione', 'contraente', 'assicurato',
            'condizioni', 'clausola', 'articolo', 'normativa',
            # English
            'contract', 'policy', 'insurance', 'policyholder', 'insured',
            'terms', 'conditions', 'clause', 'article', 'regulation'
        ]
        if any(term in chunk_lower for term in legal_terms):
            tags.append('legal')

        return tags or ['general']

    def _has_coverage_granting_language(self, chunk: str) -> bool:
        """Detect language that grants coverage"""
        granting_patterns = [
            # Italian
            r'la società (copre|rimborsa|indennizza|garantisce|prende a carico)',
            r'l\'assicurazione (prevede|copre)',
            r'è previsto (il rimborso|l\'indennizzo)',
            r'in caso di.*la società',
            r'la garanzia (prevede|copre)',
            r'rimborso.*fino a',

            # English
            r'(covered|reimbursed|indemnified|compensated)',
            r'the company (will|shall) (pay|reimburse|cover)',
            r'in the event of.*coverage',
            r'benefits include',
            r'coverage includes'
        ]

        return any(re.search(pattern, chunk, re.IGNORECASE) for pattern in granting_patterns)

    def _has_specific_amounts(self, chunk: str) -> bool:
        """Detect specific monetary amounts (not just currency symbols)"""
        amount_patterns = [
            r'€\s*[\d.,]+',  # €1,000 or €1.000,50
            r'EUR\s*[\d.,]+',
            r'euro\s*[\d.,]+',
            r'massimo.*€\s*[\d.,]+',
            r'limite.*€\s*[\d.,]+',
            r'opzione.*€\s*[\d.,]+',
            r'fino a.*€\s*[\d.,]+',
            r'importo.*€\s*[\d.,]+'
        ]

        return any(re.search(pattern, chunk, re.IGNORECASE) for pattern in amount_patterns)

    def _has_conditional_language(self, chunk: str) -> bool:
        """Detect conditional requirements"""
        conditional_patterns = [
            # Italian
            r'a condizione che',
            r'purché',
            r'se e solo se',
            r'è necessario che',
            r'entro \d+ (ore|giorni)',
            r'previo',
            r'salvo',
            r'ad eccezione',

            # English
            r'provided that',
            r'subject to',
            r'on condition that',
            r'within \d+ (hours|days)',
            r'must be reported',
            r'except',
            r'unless'
        ]

        return any(re.search(pattern, chunk, re.IGNORECASE) for pattern in conditional_patterns)

    def _has_exclusion_language(self, chunk: str) -> bool:
        """Detect exclusion language"""
        exclusion_patterns = [
            # Italian
            r'sono esclus',
            r'non è coperto',
            r'non sono coperti',
            r'esclude',
            r'limitazioni',
            r'non si applica',
            r'non opera',

            # English
            r'are excluded',
            r'not covered',
            r'excludes',
            r'does not cover',
            r'limitations',
            r'not applicable'
        ]

        return any(re.search(pattern, chunk, re.IGNORECASE) for pattern in exclusion_patterns)

    def _has_procedural_requirements(self, chunk: str) -> bool:
        """Detect procedural requirements"""
        procedural_patterns = [
            # Italian
            r'denuncia.*entro',
            r'segnalazione.*entro',
            r'deve essere denunciato',
            r'property irregularity report',
            r'pir',
            r'centrale operativa',

            # English
            r'must be reported',
            r'report.*within',
            r'notify.*within',
            r'assistance center',
            r'claim.*procedures'
        ]

        return any(re.search(pattern, chunk, re.IGNORECASE) for pattern in procedural_patterns)

    def _extract_monetary_values(self, chunk: str) -> List[Dict]:
        """Extract specific monetary amounts with context"""
        amounts = []

        # More precise pattern matching
        patterns = [
            (r'€\s*([\d.,]+)', 'euro'),
            (r'EUR\s*([\d.,]+)', 'euro'),
            (r'([\d.,]+)\s*euro', 'euro')
        ]

        for pattern, currency in patterns:
            matches = re.finditer(pattern, chunk, re.IGNORECASE)
            for match in matches:
                # Get context around the amount
                start = max(0, match.start() - 50)
                end = min(len(chunk), match.end() + 50)
                context = chunk[start:end].strip()

                amounts.append({
                    'amount': match.group(1),
                    'currency': currency,
                    'context': context,
                    'position': match.start()
                })

        return amounts

    def _extract_coverage_triggers(self, chunk: str) -> List[str]:
        """Extract phrases that trigger coverage"""
        triggers = []

        trigger_patterns = [
            # Italian
            r'in caso di ([^.]{1,80})',
            r'qualora ([^.]{1,80})',
            r'nel caso in cui ([^.]{1,80})',

            # English
            r'in the event of ([^.]{1,80})',
            r'if ([^.]{1,80})',
            r'when ([^.]{1,80})'
        ]

        for pattern in trigger_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            triggers.extend([match.strip() for match in matches])

        return triggers

    def _extract_condition_keywords(self, chunk: str) -> List[str]:
        """Extract condition-related keywords"""
        keywords = []

        condition_words = [
            # Italian
            'condizione', 'requisito', 'necessario', 'obbligatorio', 'entro',
            'previo', 'salvo', 'purché',
            # English
            'condition', 'requirement', 'necessary', 'mandatory', 'within',
            'provided', 'unless', 'subject'
        ]

        chunk_lower = chunk.lower()
        for word in condition_words:
            if word in chunk_lower:
                keywords.append(word)

        return keywords

    def _looks_like_complete_clause(self, chunk: str) -> bool:
        """Check if chunk contains a complete policy clause"""
        # Look for complete sentence structure
        has_subject_verb = bool(
            re.search(r'(la società|l\'assicurazione|the company|coverage|garanzia)', chunk, re.IGNORECASE))
        has_ending = chunk.strip().endswith('.') or chunk.strip().endswith(';')
        reasonable_length = 50 <= len(chunk) <= 1000

        return has_subject_verb and has_ending and reasonable_length

    def _extract_cross_references(self, chunk: str) -> List[str]:
        """Extract references to other articles/sections"""
        references = []

        ref_patterns = [
            # Italian
            r'art\w*\.?\s*(\d+)',
            r'articolo\s*(\d+)',
            r'sezione\s*([A-Z]\.?\d*)',
            r'paragrafo\s*([A-Z]\.?\d*)',

            # English
            r'article\s*(\d+)',
            r'section\s*([A-Z]\.?\d*)',
            r'clause\s*(\d+)'
        ]

        for pattern in ref_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            references.extend(matches)

        return references

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
                text = self.extract_text_from_file(path)
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
                for chunk in source_prefixed_chunks:
                    enhanced_metadata = self.classify_chunk_content(chunk)
                    enhanced_metadata.update({
                        "policy_id": policy_id,
                        "source_file": path,
                        "filename": filename,
                        "chunk_length": len(chunk)
                    })
                    self.chunk_metadata.append(enhanced_metadata)

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

            # Get similarities for this policy's chunks with coverage scoring
            policy_similarities = []
            for i in policy_indices:
                base_similarity = similarities[i]

                # Apply coverage relevance boost
                coverage_boost = self._calculate_coverage_boost(self.chunk_metadata[i])
                final_score = base_similarity + coverage_boost

                policy_similarities.append((i, final_score))

            # Sort by enhanced similarity (descending)
            policy_similarities.sort(key=lambda x: x[1], reverse=True)

            # Take top k
            top_k_indices = [idx for idx, _ in policy_similarities[:k_per_policy]]

            # Store the chunks
            policy_chunks[pid] = [self.text_chunks[i] for i in top_k_indices]

        return policy_chunks

    def _calculate_coverage_boost(self, metadata: Dict) -> float:
        """Calculate boost score based on coverage-relevant metadata"""
        boost = 0.0

        # Boost chunks that grant coverage
        if metadata.get('contains_coverage_grant', False):
            boost += 0.1

        # Boost chunks with specific amounts
        if metadata.get('contains_amount_specification', False):
            boost += 0.08

        # Boost complete clauses
        if metadata.get('is_complete_clause', False):
            boost += 0.05

        # Boost chunks with conditions (helps eligibility decisions)
        if metadata.get('contains_conditions', False):
            boost += 0.03

        # Boost financial content
        if 'financial' in metadata.get('content_types', []):
            boost += 0.02

        return boost

