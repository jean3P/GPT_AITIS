# Insurance Policy Analysis Pipeline with Phi-4 and RAG

## System Overview

This document provides a comprehensive explanation of the insurance policy analysis pipeline implemented using Microsoft's Phi-4 model with Retrieval-Augmented Generation (RAG). The system is designed to analyze insurance policies and provide accurate responses to coverage-related queries by combining vector-based retrieval with contextual persona extraction.

## Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  PDF Insurance  │────►│ Vector Indexing │────►│ Question Input  │
│    Policies     │     │                 │     │                 │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│ JSON Extraction │◄────│   Phi-4 Model   │◄────│  Prompt + RAG   │
│                 │     │                 │     │   + Persona     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│                 │
│  JSON Response  │
│                 │
│                 │
└─────────────────┘
```

## 1. RAG Implementation in Detail

### Document Indexing

The system begins by processing PDF insurance policy documents:

1. **Text Extraction**: Converts PDFs to plain text using PyMuPDF (fitz)
2. **Text Chunking**: Breaks text into smaller, semantically meaningful chunks (typically paragraph-based)
3. **Embedding Generation**: Creates vector embeddings for each chunk using SentenceTransformer's MiniLM-L6-v2 model
4. **Metadata Association**: Associates each chunk with its source policy ID and filename

```python
# Core document indexing functionality
def index_documents(self, pdf_paths: List[str]):
    all_chunks = []
    self.text_chunks = []
    self.chunk_metadata = []

    for path in pdf_paths:
        # Extract text from PDF
        text = self.extract_text_from_pdf(path)
        
        # Get policy ID for metadata
        policy_id = self.extract_policy_id(path)
        
        # Chunk the text into manageable pieces
        chunks = self.chunk_text(text)
        
        # Add source prefix to each chunk
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
                "filename": os.path.basename(path)
            })

    # Create embeddings for all chunks
    self.embeddings = self.embed(all_chunks)
```

### Retrieval Process

When a question is received, the system retrieves the most relevant policy sections:

1. **Query Embedding**: Converts the user's question into a vector using the same embedding model
2. **Similarity Calculation**: Computes cosine similarity between the question vector and all document chunk vectors
3. **Policy Filtering**: Optionally filters results to focus on a specific policy ID
4. **Top-K Selection (K = 3)**: Returns the most relevant chunks based on similarity scores

```python
def retrieve(self, query: str, k: int = 3, policy_id: Optional[str] = None) -> List[str]:
    # Create embedding for the query
    query_embedding = self.embed([query])
    
    # Compute similarity between query and all document chunks
    similarities = self.cosine_similarity(query_embedding, self.embeddings)[0]
    
    # Optional filtering by policy ID
    if policy_id:
        policy_mask = np.array([
            meta["policy_id"] == policy_id for meta in self.chunk_metadata
        ])
        filtered_similarities = np.where(policy_mask, similarities, -1)
        
        if np.max(filtered_similarities) > -1:
            similarities = filtered_similarities
    
    # Get top k most similar chunks
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    # Return the corresponding text chunks
    return [self.text_chunks[i] for i in top_indices]
```

## 2. Persona Extraction

A distinctive feature of this system is its ability to extract personas from questions, enhancing contextual understanding:

1. **Rule-Based Extraction**: Uses patterns to identify relationships, locations, and affected people
2. **LLM-Based Backup**: Falls back to Phi-4 model for complex cases
3. **Location Detection**: Identifies where events occurred (airport, hotel, home, abroad, etc.)
4. **Relationship Mapping**: Determines relationships between policyholders and affected people

Key persona information extracted:
- Policy user (who owns the policy)
- Affected person (who experienced the event)
- Location (where the incident occurred)
- Whether the incident happened abroad
- Relationship to policyholder
- Whether the affected person is likely covered
- Other mentioned people and their relationships

```python
def extract_personas(self, question: str) -> Dict[str, Any]:
    # First try rule-based extraction
    rule_based_result = self.rule_based_extractor.extract(question)

    # Try LLM-based approach as a backup
    llm_result = self.llm_based_extractor.extract(question)

    # Use LLM result if available, otherwise use rule-based
    if llm_result:
        return llm_result
    else:
        return rule_based_result
```

### Persona Extraction Prompt

For complex persona extraction cases, the system uses a specialized prompt with Phi-4:

```
Analyze this insurance query and extract ONLY information about the people involved and their locations:

Query: "{question}"

1. Who is making the insurance claim (the primary policy user/policyholder)?
2. Who actually experienced the event, accident, health issue, loss, or damage that is the subject of the claim?
3. WHERE was the affected person when the event occurred? (e.g., at home, abroad, at the airport, in a hotel)
4. What is the relationship between the policyholder and the affected person?
5. Is the affected person covered by the policy? (Usually yes if they are the policyholder, spouse, or dependent)
6. Who else is mentioned but NOT making the claim or experiencing the event?
7. Total number of people in the scenario?
8. Number of people actually covered by the insurance or making claims?

IMPORTANT: Carefully distinguish between these roles:
- The POLICYHOLDER (who owns the policy and usually makes the claim)
- The AFFECTED PERSON (who actually experienced the event, accident, illness, loss, or damage)
- OTHER PEOPLE who are merely mentioned but not directly involved

Pay special attention to LOCATION information, which is often critical for insurance claims:
- Was the event domestic or international?
- Was the person in transit (airport, train station, etc.)?
- Was the person at a specific venue (hotel, resort, hospital)?
- Was the person in their home country or abroad?

Examples to consider:
- "At the airport my baggage was lost" → Location: airport
- "During our vacation in Spain my daughter got sick" → Location: abroad (Spain)
- "My house was damaged by a storm" → Location: home/domestic
- "While staying at the hotel, my wallet was stolen" → Location: hotel

IMPORTANT: The answer must ONLY be valid JSON in this EXACT format:
{
  "personas": {
    "policy_user": "Who is making the claim/policyholder",
    "affected_person": "Who experienced the event/accident/illness/loss/damage",
    "location": "Where the affected person was when the event occurred",
    "is_abroad": true or false (whether the event occurred outside home country),
    "relationship_to_policyholder": "Relationship between affected person and policyholder (self, spouse, child, etc.)",
    "is_affected_covered": true or false (whether the affected person is likely covered),
    "mentioned_people": "Who else is mentioned but not a claimant or affected",
    "total_count": number of ALL people mentioned,
    "claimant_count": number of people actually claiming/using the insurance,
    "relationship": "Relationships between all mentioned people"
  }
}
```

### Persona Formatting

Once extracted, persona information is formatted for inclusion in the main prompt:

```python
def format_persona_text(personas_info: Dict[str, Any]) -> str:
    persona_text = "IMPORTANT PERSONA INFORMATION FROM THE QUESTION:\n"
    
    # Extract persona details
    policy_user = personas_info["personas"]["policy_user"]
    affected_person = personas_info["personas"].get("affected_person", f"{policy_user} (inferred)")
    location = personas_info["personas"].get("location", "unspecified location")
    is_abroad = personas_info["personas"].get("is_abroad", False)
    relationship_to_policyholder = personas_info["personas"].get("relationship_to_policyholder", "self (inferred)")
    is_affected_covered = personas_info["personas"].get("is_affected_covered", True)
    mentioned_people = personas_info["personas"]["mentioned_people"]
    total_count = personas_info["personas"]["total_count"]
    claimant_count = personas_info["personas"]["claimant_count"]
    relationship = personas_info["personas"]["relationship"]
    
    # Build detailed persona text with all extracted information
    persona_text += f"- Primary policy user/claimant: {policy_user}\n"
    persona_text += f"- Person who experienced the event/accident: {affected_person}\n"
    persona_text += f"- Location where the event occurred: {location}\n"
    # ... additional formatting ...
    
    # Add location-specific guidance
    if is_abroad:
        persona_text += "When determining coverage, check if the policy covers events occurring abroad and any special conditions or exclusions for international coverage.\n"
    
    # Add person-specific guidance
    if relationship_to_policyholder == "self":
        persona_text += "Also check provisions that apply to the policyholder directly.\n"
    elif relationship_to_policyholder in ["spouse/partner", "child/dependent"]:
        persona_text += "Also verify if family members/dependents are covered and under what conditions.\n"
    
    return persona_text
```

## 3. Prompt Engineering (Standard Prompt)

The system uses a carefully crafted prompt to guide Phi-4's responses. The standard prompt template is:

```
You are an expert assistant helping users understand their insurance coverage.
Given a question and access to a policy document, follow these instructions:

1. Determine if the case is covered:
   - Answer with one of the following:
     - "Yes - it's covered"
     - "No - not relevant"
     - "No - not covered"
     - "No - condition(s) not met"
     - "Maybe"
2. If the answer is "Yes - it's covered" or "No - not covered":
   - Quote the exact sentence(s) from the policy that support your decision.
3. If the answer is "Yes - it's covered":
   - State the **maximum amount** the policy will cover in this case.
   - Quote the exact sentence from the policy that specifies this amount.
4. Is the answer is "No - not relevant":
   - Use when the question asks about something the policy doesn't address at all
5. If the answer is "No - condition(s) not met":
   - Use when coverage exists but specific required conditions aren't satisfied.
6. If the answer is "Maybe":
   - Use only when the policy is genuinely ambiguous about this specific situation.

Return the answer in JSON format with the following fields:
{
  "answer": {
    "eligibility": "Yes - it's covered | No - not relevant | No - not covered | No - condition(s) not met | Maybe",
    "eligibility_policy": "Quoted text from policy",
    "amount_policy": "Amount like '1000 CHF' or null",
    "amount_policy_line": "Quoted policy text or null"
  }
}
```

The final prompt sent to Phi-4 combines:
1. The base system prompt (shown above)
2. Relevant policy sections retrieved via RAG
3. The user's question
4. Formatted persona information
5. JSON format instructions

## 4. Phi-4 Model Integration

Microsoft's Phi-4 model is used for processing the enhanced prompts:

1. **Model Loading**: The model is loaded using Hugging Face's AutoModelForCausalLM
2. **Pipeline Creation**: A text-generation pipeline is created with appropriate settings
3. **Inference**: The model generates responses based on the combined prompt
4. **Parameter Settings**: Uses parameters like max_new_tokens=768, do_sample=False for deterministic outputs

```python
def _initialize_pipeline(self, model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Create pipeline with loaded model and tokenizer
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        return pipe
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        raise
```

## 5. JSON Response Extraction

The system extracts structured JSON from Phi-4's responses using multiple fallback methods:

1. **Solution Block Extraction**: First tries to extract JSON from Phi's Solution blocks
2. **Code Block Extraction**: Falls back to extracting from markdown code blocks
3. **Generic Text Extraction**: Uses regex patterns to find JSON-like structures in text
4. **Validation & Fixing**: Checks structure and fixes common JSON formatting issues

```python
def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
    # Try multiple extraction methods
    return self._try_all_json_extraction_methods(text)

def _try_all_json_extraction_methods(self, generated: str) -> Optional[Dict[str, Any]]:
    # Try solution block extraction (specific to Phi pattern)
    solution_json = self._extract_json_from_solution_block(generated)
    if solution_json:
        return solution_json

    # Try code block extraction
    code_block_json = self._extract_json_from_code_block(generated)
    if code_block_json:
        return code_block_json

    # Try generic JSON extraction
    generic_json = self._extract_json_from_text(generated)
    if generic_json:
        return generic_json

    return None
```

## 6. Sample Query Processing Flow

Let's walk through how a sample insurance query is processed:

### Example Question:
"My daughter broke her arm during our vacation in Spain. Is this covered by our travel insurance?"

### Processing Steps:

1. **Persona Extraction**:
   - Policy User: "policyholder (parent)"
   - Affected Person: "daughter of policyholder"
   - Location: "abroad (Spain)"
   - Is Abroad: true
   - Relationship: "parent-child"
   - Is Affected Covered: true (as family member)

2. **Relevant Context Retrieval**:
   The system retrieves policy sections related to:
   - Medical treatment abroad
   - Coverage for family members
   - Emergency medical expenses
   - Limitations or exclusions for injuries

3. **Prompt Construction**:
   ```
   [System prompt]
   
   Relevant policy information:
   [Policy 10]: In case of accident or illness abroad requiring hospitalization or emergency medical treatment, the policy covers medical expenses up to 1,000,000 CHF per person per trip...
   [Policy 10]: The policy covers the policyholder and family members listed on the policy, including spouse/partner and dependent children under 21 years of age...
   [Policy 10]: Sports injuries are covered unless they result from professional competitions or high-risk activities as defined in Section 4.3...
   
   Question: My daughter broke her arm during our vacation in Spain. Is this covered by our travel insurance?
   
   IMPORTANT PERSONA INFORMATION FROM THE QUESTION:
   - Primary policy user/claimant: policyholder (parent)
   - Person who experienced the event/accident: daughter of policyholder
   - Location where the event occurred: abroad (Spain)
   - Event occurred abroad: Yes
   - Relationship to policyholder: child/dependent
   - The affected person is likely covered under this policy
   - Other people mentioned (not policy users): None specifically mentioned
   - Total number of people mentioned: 2
   - Number of people actually claiming/covered: 2
   - Relationships between all people: parent-child
   
   When determining coverage, check if the policy covers events occurring abroad and any special conditions or exclusions for international coverage.
   Also verify if family members/dependents are covered and under what conditions.
   Take into account all of these factors when assessing eligibility for coverage.
   
   [JSON format instructions]
   
   Then the json Solution is:
   ```

4. **Model Output Processing**:
   - Phi-4 generates a response including a JSON object
   - The JSON extractor identifies and extracts the valid JSON
   - The system validates the JSON structure and content

5. **Final Response**:
   ```json
   {
     "answer": {
       "eligibility": "Yes - it's covered",
       "eligibility_policy": "In case of accident or illness abroad requiring hospitalization or emergency medical treatment, the policy covers medical expenses up to 1,000,000 CHF per person per trip.",
       "amount_policy": "1,000,000 CHF",
       "amount_policy_line": "In case of accident or illness abroad requiring hospitalization or emergency medical treatment, the policy covers medical expenses up to 1,000,000 CHF per person per trip."
     }
   }
   ```

## 7. Running the Pipeline

The pipeline can be executed using command-line arguments:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 --prompt standard --num-questions 10
```

Key parameters:
- `--model`: Model provider (hf for Hugging Face/Phi-4)
- `--model-name`: Specific model identifier
- `--prompt`: Template to use (standard, detailed, etc.)
- `--batch`: Process all policies together
- `--num-questions`: Number of questions to process
- `--output-dir`: Directory for JSON output
- `--log-level`: Logging detail level

## Conclusion

The insurance policy analysis pipeline combines the strengths of:

1. **Retrieval-Augmented Generation** to provide relevant policy context
2. **Persona extraction** to understand the human context of queries
3. **Phi-4's reasoning capabilities** to interpret policies and determine coverage
4. **Structured JSON output** for consistent, machine-readable responses

This system demonstrates an effective implementation of context-aware LLM applications for specialized document analysis, significantly improving the accuracy and specificity of insurance policy interpretation.
