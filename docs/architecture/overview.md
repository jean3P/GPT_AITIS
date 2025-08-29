# GPT_AITIS Architecture Overview

## System Architecture

```mermaid
graph TB
    %% Input Sources
    subgraph Input["📥 Input Layer"]
        P[Policy PDFs]
        Q[Questions Dataset]
        C[Configuration]
    end

    %% Processing Core
    subgraph Processing["⚙️ Processing Core"]
        TE[Text Extractor]
        CS[Chunking Strategies]
        VS[Vector Store]
        RAG[RAG Pipeline]
    end

    %% Model Layer
    subgraph Models["🤖 Model Layer"]
        OAI[OpenAI<br/>GPT-4/3.5]
        HF[HuggingFace<br/>Phi-4/Qwen]
        OR[OpenRouter<br/>Cloud Models]
    end

    %% Output Layer
    subgraph Output["📤 Output Layer"]
        VER[Verifier]
        JSON[JSON Results]
        EVAL[Evaluation]
    end

    %% Connections
    P --> TE
    Q --> RAG
    C --> RAG
    
    TE --> CS
    CS --> VS
    VS --> RAG
    
    RAG --> OAI
    RAG --> HF
    RAG --> OR
    
    OAI --> VER
    HF --> VER
    OR --> VER
    
    VER --> JSON
    JSON --> EVAL

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2
    classDef process fill:#fff3e0,stroke:#f57c00
    classDef model fill:#f3e5f5,stroke:#7b1fa2
    classDef output fill:#e8f5e9,stroke:#388e3c
    
    class P,Q,C input
    class TE,CS,VS,RAG process
    class OAI,HF,OR model
    class VER,JSON,EVAL output
```

## Core Components

### 📥 **Input Layer**
- **Policy PDFs**: Insurance policy documents to analyze
- **Questions Dataset**: Excel/JSON files with coverage queries
- **Configuration**: Model selection, prompts, and processing parameters

### ⚙️ **Processing Core**
- **Text Extractor**: Converts PDFs to processable text
- **Chunking Strategies**: 7 strategies (Simple, Section, Smart Size, Semantic, Graph, Semantic Graph, Hybrid)
- **Vector Store**: ChromaDB-based embeddings for similarity search
- **RAG Pipeline**: Orchestrates retrieval and analysis flow

### 🤖 **Model Layer**
- **OpenAI**: GPT-4, GPT-3.5-turbo via API
- **HuggingFace**: Local models (Phi-4, Qwen) with GPU support
- **OpenRouter**: Cloud access to various LLMs

### 📤 **Output Layer**
- **Verifier**: Multi-pass validation and error correction
- **JSON Results**: Structured output with coverage decisions
- **Evaluation**: Metrics calculation against ground truth

## Key Features

### 🎯 **Smart Chunking**
The system offers 7 intelligent chunking strategies to handle different document structures:
- **Simple**: Basic fixed-size chunks
- **Section**: Structure-aware segmentation
- **Smart Size**: Adaptive sizing based on content importance
- **Semantic**: Embedding-based coherent segments
- **Graph**: Entity-relationship aware chunking
- **Semantic Graph**: Hybrid semantic-graph approach
- **Hybrid**: Combined strategy for maximum accuracy

### 🔍 **RAG Pipeline**
1. **Retrieve** relevant policy sections (k=1,3,5 chunks)
2. **Filter** irrelevant information
3. **Extract** personas (claimant, location, relationships)
4. **Generate** coverage decision with justification

### ✅ **Verification System**
- Two-pass verification to ensure accuracy
- Validates responses against policy text
- Corrects inconsistencies automatically

## Processing Flow

```mermaid
graph TD
    A[Question] --> B{RAG Mode?}
    B -->|Yes| C[Retrieve Chunks]
    B -->|No| D[Full Policy]
    C --> E[Filter & Extract]
    D --> E
    E --> F[Model Inference]
    F --> G[Verification]
    G --> H[JSON Output]
    H --> I[Evaluation]
    
    style A fill:#e3f2fd
    style H fill:#e8f5e9
    style I fill:#e8f5e9
```

## Output Structure

```json
{
  "policy_id": "travel_insurance_01",
  "question": "Is theft covered while traveling?",
  "outcome": "yes",
  "justification": "Section 4.2: Theft of personal belongings is covered up to $2,500 per incident",
  "payment": "$2,500 maximum per claim, $100 deductible",
}
```

## Usage Modes

### 🚀 **Single Query**
```bash
python src/main.py --model openai --model-name gpt-4 --rag-strategy semantic
```

### 📦 **Batch Processing**
```bash
python src/main.py --model hf --model-name microsoft/phi-4 --batch --k 5
```

### 🔬 **Research Mode**
```bash
python src/main.py --rag-strategy semantic_graph --verify --use-persona --filter-irrelevant
```
