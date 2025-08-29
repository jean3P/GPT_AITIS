# Configuration

This guide covers all configuration options for the GPT_AITIS system, from environment variables to model-specific settings.

## Environment Variables

### Setting Up `.env`

Create a `.env` file in the project root with your API keys and configuration:

```bash
# Required API Keys
OPENAI_API_KEY=sk-...                  # For OpenAI models
HUGGINGFACE_TOKEN=hf_...               # For HuggingFace models
OPENROUTER_API_KEY=sk-or-v1-...        # For OpenRouter API

# Optional OpenRouter Configuration
OPENROUTER_SITE_URL=https://your-site.com   # Your application URL
OPENROUTER_SITE_NAME=YourAppName           # Your application name
```

### HuggingFace Cache Configuration

For HPC environments, set cache directories in your job scripts:

```bash
export HF_HUB_CACHE="/cluster/scratch/cache/huggingface/hub"
export HF_ASSETS_CACHE="/cluster/scratch/cache/huggingface/assets"
```

## Model Configuration

### Available Models

Models are configured in `src/config.py`. The system supports multiple model providers:

#### Local Models
```python
MODEL_PATHS = {
    # Phi models
    "microsoft/phi-4": "/cluster/scratch/$USER/models/phi-4",
    "phi-4": "/cluster/scratch/$USER/models/phi-4",
    
    # Qwen models
    "Qwen/Qwen2.5-32B": "/cluster/scratch/$USER/models/qwen2.5-32b",
    "Qwen/Qwen2.5-7B": "/cluster/scratch/$USER/models/qwen2.5-7b",
    "qwen2.5-32b": "/cluster/scratch/$USER/models/qwen2.5-32b",
    "qwen2.5-7b": "/cluster/scratch/$USER/models/qwen2.5-7b",
}
```

#### OpenRouter Models
```python
OPENROUTER_MODELS = {
    # Qwen models
    "qwen/qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-32b-instruct": "qwen/qwen-2.5-32b-instruct",
    
    # Other models
    "anthropic/claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o": "openai/gpt-4o",
    "meta-llama/llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
}
```

### Model-Specific Parameters

Each model has specific generation parameters:

```python
MODEL_CONFIGS = {
    "microsoft/phi-4": {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "max_new_tokens": 1020,
        "temperature": 0.1,
        "do_sample": False,
        "repetition_penalty": 1.05,
        "pad_token_id": None
    },
    "Qwen/Qwen2.5-32B": {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "max_new_tokens": 1024,
        "do_sample": True,
        "repetition_penalty": 1.0,
        "pad_token_id": None
    }
}
```

### Adding New Models

To add a new model:

1. Add the model path to `MODEL_PATHS`:
```python
MODEL_PATHS["new-model"] = "/path/to/model"
```

2. Add model configuration to `MODEL_CONFIGS`:
```python
MODEL_CONFIGS["new-model"] = {
    "torch_dtype": "auto",
    "device_map": "auto",
    "max_new_tokens": 1024,
    # ... other parameters
}
```

## Chunking Strategy Configuration

### Available Strategies

Configure chunking strategies when initializing the vector store:

#### Simple Chunking
```python
{
    "max_length": 512,
    "overlap": 0,
    "preserve_paragraphs": True
}
```

#### Section-Based Chunking
```python
{
    "max_section_length": 2000,
    "min_section_length": 50,
    "preserve_subsections": True,
    "include_front_matter": False,
    "sentence_window_size": 3
}
```

#### Smart Size Chunking
```python
{
    "base_chunk_words": 80,
    "min_chunk_words": 20,
    "max_chunk_words": 200,
    "importance_multiplier": 1.5,
    "coherence_threshold": 0.7,
    "preserve_complete_clauses": True,
    "overlap_words": 0
}
```

#### Semantic Chunking
```python
{
    "embedding_model": "all-MiniLM-L6-v2",
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_value": 75,
    "min_chunk_sentences": 2,
    "max_chunk_sentences": 15,
    "preserve_paragraph_boundaries": True,
    "device": "cpu"
}
```

### Chunking Presets

Use predefined configurations for common scenarios:

```python
CHUNKING_PRESETS = {
    'fast': {
        'strategy': 'simple',
        'config': {'max_length': 256, 'overlap': 25}
    },
    'balanced': {
        'strategy': 'section',
        'config': {
            'max_section_length': 1500,
            'preserve_subsections': True
        }
    },
    'comprehensive': {
        'strategy': 'section',
        'config': {
            'max_section_length': 2500,
            'preserve_subsections': True,
            'include_front_matter': True,
            'sentence_window_size': 5
        }
    },
    'semantic_focused': {
        'strategy': 'semantic',
        'config': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'breakpoint_threshold_value': 85,
            'min_chunk_sentences': 3,
            'max_chunk_sentences': 12
        }
    }
}
```

## Prompt Configuration

### Available Prompts

Prompts are defined in `src/prompts/insurance_prompts.py`:

| Prompt Name | Description | Use Case |
|------------|-------------|----------|
| `standard` | Basic coverage determination | General analysis |
| `detailed` | Includes persona handling | Complex scenarios |
| `precise` | Strict quote requirements | High accuracy needs |
| `precise_v2` | Enhanced precision | Better handling of conditions |
| `precise_v3` | Minimal format, strict copying | Reduced hallucination |
| `precise_v4` | Deterministic with timing checks | Complete validation |
| `precise_v5` | Reduces false negatives | Better coverage detection |
| `precise_v5_qwen` | Qwen-optimized v5 | Qwen models |
| `relevance_filter_v1` | Query relevance checking | Pre-filtering |
| `relevance_filter_v2` | Enhanced relevance detection | Better filtering |

### Selecting Prompts

Choose prompts based on your model and requirements:

```bash
# For Phi-4 models
python src/main.py --model hf --model-name microsoft/phi-4 --prompt precise_v5

# For Qwen models
python src/main.py --model qwen --model-name Qwen/Qwen2.5-32B --prompt precise_v5_qwen

# For OpenRouter models
python src/main.py --model openrouter --model-name qwen/qwen-2.5-72b-instruct --prompt precise_v4_qwen
```

## Application Settings

### Core Constants

Configure application behavior in `src/config.py`:

```python
# File paths
DATASET_PATH = "resources/questions/questions.xlsx"
DOCUMENT_DIR = "resources/documents/policies/"
GT_PATH = "resources/ground_truth/"
EVALUATION_RESULTS_PATH = "resources/results/"

# Vector store settings
VECTOR_STORE_EXPIRATION_DAYS = 30
VECTOR_NAME_PREFIX = "AITIS_"

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_PATH = "/cluster/scratch/$USER/models/embeddings/sentence-transformers_all-MiniLM-L6-v2"
CACHE_EMBEDDINGS = True

# Output settings
JSON_PATH = "resources/results/json_output"
LOG_DIR = "resources/results/logs"
```

### Logging Configuration

Configure logging levels and output:

```python
# In your scripts or via CLI
--log-level DEBUG    # Most verbose
--log-level INFO     # Standard logging
--log-level WARNING  # Warnings only
--log-level ERROR    # Errors only
```

Logging configuration in code:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
```

## RAG Pipeline Configuration

### Context Retrieval Settings

Configure the number of context chunks to retrieve:

```bash
# Default: 3 chunks
python src/main.py --k 3

# More context for complex queries
python src/main.py --k 5

# Complete policy mode (no chunking)
python src/main.py --complete-policy
```

### Persona Extraction

**Goal:** Identifies WHO is making the claim and WHO was affected in the question to better determine eligibility.

*Example:* "My daughter broke her leg during our trip" → The system identifies the claimant is "the parent" and the affected person is "the daughter", which helps determine if dependents are covered.

```bash
python src/main.py --persona
```

### Relevance Filtering

**Goal:** Checks if the user's question is about a type of loss/event (medical, baggage, cancellation, etc.) that this specific policy covers.

*Example:* If the policy only covers "baggage loss" and someone asks "Will you pay for my hospital bills?", it's filtered as irrelevant because medical expenses aren't covered by this baggage-only policy.

```bash
python src/main.py --filter-irrelevant --prompt-relevant relevance_filter_v2
```

### Verification Settings

**Goal:** Double-checks the model's answer by having it review its own response to catch and correct mistakes.

*Example:* After answering "Yes, covered", the system asks itself "Is this really correct based on the policy?" and can change to "No, not covered" if it finds an error in its reasoning.


```bash
python src/main.py --verifier --verifier-iterations 1
```

## Output Configuration

### Directory Structure

Output directories follow this pattern:
```
{output_dir}/{model_name}/{mode}/{timestamp}/{prompt_name}/
```

Where:
- `model_name`: Sanitized model name (e.g., `microsoft_phi-4`)
- `mode`: Either `k=N` or `complete-policy`
- `timestamp`: Format `DD-MM-YY--HH-MM-SS`
- `prompt_name`: Selected prompt template

### Custom Output Directory

Specify custom output location:

```bash
python src/main.py --output-dir /custom/path/to/results
```

## 🛠️ Troubleshooting Configuration

### Common Configuration Issues

??? failure "Model Loading Fails"
    ```bash
    # Check model path exists
    ls -la /cluster/scratch/$USER/models/
    
    # Verify model directory permissions
    chmod -R 755 /cluster/scratch/$USER/models/
    
    # Check GPU memory availability
    nvidia-smi
    
    # Test model loading in isolation
    python -c "from transformers import AutoModelForCausalLM; print('Model loads OK')"
    ```

??? failure "API Keys Not Working"
    ```bash
    # Verify .env file is loaded
    python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OpenAI:', 'OPENAI_API_KEY' in os.environ)"
    
    # Check API key format
    # OpenAI: Should start with 'sk-'
    # HuggingFace: Should start with 'hf_'
    # OpenRouter: Should start with 'sk-or-v1-'
    
    # Test API connection
    python scripts/test_api_connection.py
    ```

??? failure "Out of Memory Errors"
    ```bash
    # Enable CPU offloading for large models
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    
    # Use smaller batch size
    python src/main.py --batch-size 1
    
    # Try quantized models
    python src/main.py --model-name microsoft/phi-4 --load-in-8bit
    ```

??? failure "Chunking Strategy Errors"
    ```bash
    # Test chunking in isolation
    python -c "from models.chunking import ChunkingFactory; c = ChunkingFactory.create_strategy('simple'); print('OK')"
    
    # Use simpler strategy if semantic fails
    python src/main.py --rag-strategy simple
    
    # Verify embedding model is downloaded
    ls -la /cluster/scratch/$USER/models/embeddings/
    ```

??? failure "HuggingFace Cache Issues"
    ```bash
    # Clear corrupted cache
    rm -rf $HF_HUB_CACHE/*
    
    # Set custom cache location
    export HF_HUB_CACHE="/cluster/scratch/$USER/cache/huggingface/hub"
    export HF_ASSETS_CACHE="/cluster/scratch/$USER/cache/huggingface/assets"
    
    # Download models with resume capability
    python scripts/download_models.py --resume
    ```

??? failure "OpenRouter Rate Limiting"
    ```bash
    # Check current usage
    curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
         https://openrouter.ai/api/v1/auth/key
    
    # Reduce request rate
    python src/main.py --openrouter-rpm 6  # 6 requests per minute
    
    # Add delay between requests
    python src/main.py --request-delay 10  # 10 seconds between requests
    ```

### Debug Mode

??? info "Enable Verbose Logging"
    ```bash
    # Maximum verbosity
    python src/main.py --log-level DEBUG
    
    # Log to file for analysis
    python src/main.py --log-level DEBUG --log-file debug.log
    
    # Debug specific component
    TRANSFORMERS_VERBOSITY=debug python src/main.py
    ```

