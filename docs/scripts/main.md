# Main Pipeline Script

The `main.py` script is the primary entry point for running insurance policy analysis in the GPT_AITIS system. It orchestrates the entire analysis pipeline, from loading models to generating structured JSON outputs.

## Overview

The main pipeline script provides:

- **Multi-model support**: OpenAI, HuggingFace, and OpenRouter models
- **Flexible analysis modes**: RAG-based retrieval or complete policy analysis
- **Advanced features**: Verification, relevance filtering, and persona extraction
- **Batch processing**: Analyze multiple policies and questions efficiently
- **Structured outputs**: JSON results organized by timestamp and configuration

## Command-Line Arguments

### Basic Arguments

```bash
# Model Selection
--model {openai,hf,openrouter}     # Model provider
--model-name MODEL_NAME             # Specific model to use

# Analysis Mode
--k K                               # Number of chunks for RAG (default: 3)
--complete-policy                   # Use complete policy instead of RAG

# Prompt Selection
--prompt PROMPT_NAME                # Prompt template (default: standard)

# Processing Options
--batch                             # Process all policies together
--log-level {DEBUG,INFO,WARNING}   # Logging verbosity
```

### Filtering Arguments

```bash
# Question Filtering
--questions "1,2,3,4,5"            # Specific question IDs to process
--num-questions N                   # Limit to first N questions

# Policy Filtering
--policy-id "18"                   # Process specific policy only
```

### Advanced Features

```bash
# Verification
--verifier                         # Enable result verification
--verifier-iterations N            # Number of verification passes (default: 1)

# Relevance Filtering
--filter-irrelevant               # Filter out irrelevant questions
--prompt-relevant PROMPT_NAME     # Relevance filter prompt (default: relevance_filter_v1)

# RAG Strategy
--rag-strategy {simple,section,semantic,graph,hybrid}  # Chunking strategy

# Persona Extraction
--use-persona                     # Enable persona extraction
```

## Model Providers

### 1. OpenAI Models

```bash
python src/main.py --model openai --model-name gpt-4o \
    --prompt standard --k 3 --batch
```

**Available Models**:
- `gpt-4o` - Latest GPT-4 model
- `gpt-4` - Standard GPT-4
- `gpt-3.5-turbo` - Faster, cost-effective option

**Requirements**:
- API key in `.env` file: `OPENAI_API_KEY=sk-...`

### 2. HuggingFace Models (Local)

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch
```

**Available Models**:
- `microsoft/phi-4` - Efficient 14B parameter model
- `Qwen/Qwen2.5-7B` - Smaller Qwen model
- `Qwen/Qwen2.5-32B` - Large Qwen model

**Requirements**:
- Models downloaded to `/cluster/scratch/$USER/models/`
- Sufficient GPU memory

### 3. OpenRouter Models (Cloud)

```bash
python src/main.py --model openrouter --model-name qwen/qwen-2.5-72b-instruct \
    --prompt precise_v4_qwen --k 3 --batch
```

**Available Models**:
- `qwen/qwen-2.5-72b-instruct` - Large Qwen model
- `anthropic/claude-3-opus` - Claude 3 Opus
- Various other cloud models

**Requirements**:
- API key in `.env`: `OPENROUTER_API_KEY=...`

## Analysis Modes

### RAG Mode (Default)

Retrieves relevant chunks from policy documents:

```bash
# Basic RAG with k=3 chunks
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch

# Different k values
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 1 --batch  # Single chunk

python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 5 --batch  # Five chunks
```

### Complete Policy Mode

Analyzes entire policy documents:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --complete-policy --batch
```

**Considerations**:
- Requires models with large context windows
- Higher computational cost
- May exceed token limits for some models

## RAG Strategies

### Available Strategies

1. **Simple** (default)
   ```bash
   --rag-strategy simple
   ```
   - Basic paragraph-based chunking
   - Fixed size chunks

2. **Section-based**
   ```bash
   --rag-strategy section
   ```
   - Preserves document structure
   - Respects section boundaries

3. **Semantic**
   ```bash
   --rag-strategy semantic
   ```
   - Groups semantically similar content
   - Uses embeddings for coherence

4. **Graph-based**
   ```bash
   --rag-strategy graph
   ```
   - Entity and relationship aware
   - Advanced structural understanding

5. **Hybrid**
   ```bash
   --rag-strategy hybrid
   ```
   - Combines multiple strategies
   - Balanced approach

### Strategy Comparison

```bash
# Test different strategies
for strategy in simple semantic hybrid; do
    python src/main.py --model hf --model-name microsoft/phi-4 \
        --prompt precise_v4 --k 3 --rag-strategy $strategy --batch
done
```

## Advanced Features

### Result Verification

Adds a second pass to catch and correct errors:

```bash
# Single verification iteration
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch \
    --verifier --verifier-iterations 1

# Multiple iterations (for critical accuracy)
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch \
    --verifier --verifier-iterations 2
```

**Verification Process**:
1. Initial analysis generates result
2. Verifier reviews result against policy text
3. Corrections applied if errors found
4. Process logged for transparency

### Relevance Filtering

Pre-filters questions unrelated to policy coverage:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch \
    --filter-irrelevant --prompt-relevant relevance_filter_v2
```

**Benefits**:
- Reduces false positives
- Improves processing efficiency
- Better handling of out-of-scope questions

### Persona Extraction

Analyzes who is claiming and who is affected:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch \
    --use-persona
```

**Extracts**:
- Policy holder identity
- Affected person
- Location of incident
- Relationships between parties

## Batch Processing

### Default Batch Mode

Processes all policies efficiently:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch
```

**Advantages**:
- Single model load for all policies
- Consistent processing
- Organized output structure

### Individual Processing

For debugging or specific cases:

```bash
# Single policy, single question
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 \
    --questions "1" --policy-id "18"

# Don't use --batch for individual processing
```

## Output Structure

### Directory Organization

```
resources/results/json_output/
└── microsoft_phi-4/                    # Model name
    ├── k=3/                           # RAG configuration
    │   └── 25-01-25--14-30-00/       # Timestamp
    │       └── precise_v4/            # Prompt name
    │           ├── policy_10_results.json
    │           ├── policy_18_results.json
    │           └── ...
    └── complete-policy/               # Alternative mode
        └── 25-01-25--16-00-00/
            └── precise_v4/
                └── ...
```

### JSON Output Format

Each policy generates a structured JSON file:

```json
{
  "policy_id": "18",
  "questions": [
    {
      "request_id": "1",
      "question": "At the airport my baggage was lost...",
      "outcome": "Yes",
      "outcome_justification": "In the event that the air carrier fails to deliver...",
      "payment_justification": "Option 1 € 150,00 Option 2 € 350,00..."
    },
    // ... more questions
  ]
}
```

## Common Use Cases

### 1. Basic Analysis

Standard RAG-based analysis:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch
```

### 2. High-Accuracy Configuration

Maximum accuracy with verification:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v5 --k 5 --batch \
    --verifier --verifier-iterations 2 \
    --filter-irrelevant --use-persona
```

### 3. Quick Testing

Test on subset of questions:

```bash
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 \
    --questions "1,2,3" --policy-id "18"
```

### 4. Model Comparison

Run same configuration on different models:

```bash
# Phi-4
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch

# Qwen
python src/main.py --model hf --model-name Qwen/Qwen2.5-7B \
    --prompt precise_v4_qwen --k 3 --batch

# OpenAI
python src/main.py --model openai --model-name gpt-4o \
    --prompt standard --k 3 --batch
```

### 5. Prompt Testing

Compare different prompts:

```bash
for prompt in precise_v4 precise_v5 standard; do
    python src/main.py --model hf --model-name microsoft/phi-4 \
        --prompt $prompt --k 3 --batch
done
```

## SLURM Integration

For HPC environments:

```bash
#!/bin/bash
#SBATCH --job-name=insurance_analysis
#SBATCH --gres=gpu:a40_48gb:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

# Activate environment
cd ~/repos/GPT_AITIS
source .venv/bin/activate

# Run analysis
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch \
    --log-level INFO
```

## Integration with Evaluation

Output structure designed for seamless evaluation:

```bash
# Step 1: Generate outputs
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch

# Step 2: Evaluate immediately
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 --latest

# Step 3: Generate tables
python src/scripts/latex_tables.py \
    --models microsoft_phi-4
```

The main pipeline script provides a flexible, powerful interface for insurance policy analysis, supporting various models, strategies, and configurations to meet different research and production needs.