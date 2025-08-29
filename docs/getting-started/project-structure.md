# Project Structure

The GPT_AITIS project follows a modular structure designed for scalability and maintainability. This page provides a comprehensive overview of the directory layout and file organization.

## Directory Overview

```
GPT_AITIS/
├── src/                      # Main source code and scripts
├── resources/                # Data and resources
├── docs/                     # Documentation
├── .venv/                    # Virtual environment
├── .env                      # Environment variables
├── pyproject.toml           # Project configuration
├── mkdocs.yml               # Documentation config
└── SLURM job scripts        # HPC execution scripts
```

## Core Directories

### `/src` - Source Code

The main application code organized by functionality:

```
src/
├── main.py                   # Main entry point
├── rag_runner.py            # RAG pipeline orchestration
├── config.py                # Configuration management
├── utils.py                 # Utility functions
├── output_formatter.py      # Result formatting
├── logging_utils.py         # Logging configuration
├── assistant_manager.py     # OpenAI assistant management
│
├── models/                  # Model implementations
│   ├── __init__.py
│   ├── base.py             # Base model interface
│   ├── factory.py          # Model factory pattern
│   ├── hf_model.py         # HuggingFace models
│   ├── openai_model.py     # OpenAI integration
│   ├── qwen_model.py       # Qwen-specific models
│   ├── openrouter_model.py # OpenRouter API
│   ├── shared_client.py    # Shared model utilities
│   ├── vector_store.py     # Vector storage
│   ├── verifier.py         # Result verification
│   │
│   ├── chunking/           # Text chunking strategies
│   │   ├── __init__.py
│   │   ├── base.py         # Base chunking interface
│   │   ├── factory.py      # Chunking factory
│   │   ├── simple_chunker.py
│   │   ├── section_chunker.py
│   │   ├── smart_size_chunker.py
│   │   ├── semantic_chunker.py
│   │   ├── graph_chunker.py
│   │   ├── semantic_graph_chunker.py
│   │   └── hybrid_chunker.py
│   │
│   ├── persona/            # Persona extraction
│   │   ├── __init__.py
│   │   ├── extractor.py    # Main extractor
│   │   ├── rule_based.py   # Rule-based extraction
│   │   ├── llm_based.py    # LLM-based extraction
│   │   ├── location.py     # Location detection
│   │   └── formatters.py   # Output formatting
│   │
│   └── json_utils/         # JSON processing
│       ├── __init__.py
│       └── extractors.py   # JSON extraction utilities
│
├── prompts/                # Prompt templates
│   ├── __init__.py
│   ├── insurance_prompts.py    # Insurance-specific prompts
│   └── verification_prompts.py # Verification prompts
│
└── scripts/                # Utility scripts
    ├── latex_tables.py     # Get tables in Latex code
    └── evaluate_results.py # Evaluation script
```

### `/resources` - Data and Resources

All data files, configurations, and results:

```
resources/
├── questions/              # Test questions
│   └── questions.xlsx      # Insurance queries dataset
│
├── documents/             # Insurance policies
│   └── policies/          # PDF policy documents
│       ├── 1_policy.pdf
│       ├── 2_policy.pdf
│       └── ...
│
├── ground_truth/          # Expected answers
│   ├── 1_gt.xlsx
│   ├── 2_gt.xlsx
│   └── ...
│
├── raw_ground_truth/      # Original GT files
│
├── results/               # Analysis results
│   ├── json_output/       # JSON results by model
│   │   └── model_name/
│   │       └── k=3/
│   │           └── timestamp/
│   │               └── prompt_name/
│   │                   ├── policy_1_results.json
│   │                   ├── policy_2_results.json
│   │                   └── ...
│   │
│   ├── evaluation_results/    # Evaluation outputs
│   │
│   └── logs/                 # Application logs
│
└── embeddings/               # Cached embeddings
```

### `/docs` - Documentation

MkDocs-based documentation:

```
docs/
├── index.md                  # Documentation home
├── getting-started/          # Getting started guides
│   ├── installation.md
│   ├── quickstart.md
│   ├── project-structure.md
│   └── configuration.md
├── user-guide/              # User documentation
├── evaluation/              # Evaluation guides
├── architecture/            # System architecture
├── models/                  # Model documentation
├── api/                     # API reference
├── examples/                # Usage examples
└── troubleshooting/         # Problem solving
```

## Configuration Files

### Root Configuration Files

- **`.env`** - Environment variables:
  ```bash
  OPENAI_API_KEY=your_key_here
  HUGGINGFACE_TOKEN=your_token_here
  OPENROUTER_API_KEY=your_key_here
  OPENROUTER_SITE_URL=optional_site_url
  OPENROUTER_SITE_NAME=optional_site_name
  ```

- **`pyproject.toml`** - Project dependencies and configuration
- **`mkdocs.yml`** - Documentation configuration

### SLURM Job Scripts

HPC job submission scripts in the root directory:

- **`download_embedding_model.sh`** - Download embedding models
- **`download_models.sh`** - Download LLM models

## Key File Descriptions

### Core Application Files

| File | Description |
|------|-------------|
| `src/main.py` | CLI entry point, argument parsing, pipeline execution |
| `src/rag_runner.py` | RAG pipeline orchestration (run_rag, run_batch_rag) |
| `src/config.py` | Central configuration, model paths, constants |
| `src/utils.py` | Utility functions (read_questions, list_policy_paths) |
| `src/output_formatter.py` | JSON result formatting and directory management |
| `src/logging_utils.py` | Logging configuration and setup |
| `src/assistant_manager.py` | OpenAI assistant and vector store management |

### Model Implementation Files

| File | Description |
|------|-------------|
| `models/base.py` | BaseModelClient abstract interface |
| `models/factory.py` | Model creation factory pattern |
| `models/hf_model.py` | HuggingFaceModelClient implementation |
| `models/openai_model.py` | OpenAIModelClient for API integration |
| `models/qwen_model.py` | QwenModelClient with Qwen optimizations |
| `models/openrouter_model.py` | OpenRouterModelClient for cloud models |
| `models/shared_client.py` | SharedModelClient for memory efficiency |
| `models/vector_store.py` | LocalVectorStore and EnhancedLocalVectorStore |
| `models/verifier.py` | ResultVerifier for output verification |

### Chunking Strategy Files

| File | Description |
|------|-------------|
| `chunking/base.py` | ChunkingStrategy abstract base class |
| `chunking/factory.py` | ChunkingFactory and preset configurations |
| `chunking/simple_chunker.py` | Basic paragraph-based chunking |
| `chunking/section_chunker.py` | Structure-aware section chunking |
| `chunking/smart_size_chunker.py` | Content-aware adaptive chunking |
| `chunking/semantic_chunker.py` | Embedding-based semantic chunking |

### Persona Extraction Files

| File | Description |
|------|-------------|
| `persona/extractor.py` | Main PersonaExtractor coordinator |
| `persona/rule_based.py` | RuleBasedExtractor using patterns |
| `persona/llm_based.py` | LLMBasedExtractor using language models |
| `persona/location.py` | LocationDetector for event locations |
| `persona/formatters.py` | Format persona information for prompts |

### Prompt Files

| File | Description |
|------|-------------|
| `prompts/insurance_prompts.py` | InsurancePrompts class with all prompt templates |
| `prompts/verification_prompts.py` | VerificationPrompts for result checking |

### Scripts

| Script                        | Purpose                                     |
|-------------------------------|---------------------------------------------|
| `scripts/evaluate_results.py` | Evaluate model outputs against ground truth |
| `scripts/latex_tables.py`     | Obtain the tables in Latex code             |

## Data Flow

1. **Input Phase**
    - Questions loaded from `resources/questions/questions.xlsx`
    - Policy PDFs from `resources/documents/policies/`

2. **Processing Phase**
    - Text extraction and chunking based on selected strategy
    - Vector store creation and embedding generation
    - Context retrieval using similarity search
    - LLM query with retrieved context

3. **Output Phase**
    - JSON results saved to `resources/results/json_output/`
    - Organized by model, parameters, timestamp, and prompt

4. **Evaluation Phase**
    - Compare outputs with ground truth files
    - Generate evaluation metrics and reports

## Output Directory Structure

Results are organized hierarchically:

```
resources/results/json_output/
└── {model_name}/              # e.g., "microsoft_phi-4"
    └── {mode}/                # "k=3" or "complete-policy"
        └── {timestamp}/       # "22-01-25--14-30-45"
            └── {prompt_name}/ # "precise_v4"
                ├── policy_1_results.json
                ├── policy_2_results.json
                └── ...
```

## Model Configuration

Models are configured in `src/config.py`:

```python
# Model paths
MODEL_PATHS = {
    "microsoft/phi-4": "/cluster/scratch/user/models/phi-4",
    "Qwen/Qwen2.5-32B": "/cluster/scratch/user/models/qwen2.5-32b",
    ...
}

# Model-specific configurations
MODEL_CONFIGS = {
    "microsoft/phi-4": {
        "torch_dtype": "auto",
        "device_map": "auto",
        "max_new_tokens": 1020,
        ...
    }
}
```

## Environment Variables

Required environment variables in `.env`:

```bash
# API Keys
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...
OPENROUTER_API_KEY=sk-or-v1-...

# Optional OpenRouter settings
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME=YourAppName

# HuggingFace cache (set in job scripts)
HF_HUB_CACHE=/cluster/scratch/cache/huggingface/hub
HF_ASSETS_CACHE=/cluster/scratch/cache/huggingface/assets
```

## File Naming Conventions

### Data Files
- Policy PDFs: `{id}_{name}.pdf` (e.g., `18_Nobis - Baggage loss EN.pdf`)
- Ground truth: `{id}_gt.xlsx`
- Results: `policy_{id}_results.json`

## Best Practices

1. **Code Organization**
    - Keep all source code under `src/`
    - Group related functionality in subdirectories
    - Use `__init__.py` files for proper package structure

2. **Data Management**
    - Input data goes in `resources/documents/`
    - Results are timestamped and organized by parameters
    - Logs are kept separate from results

3. **Configuration**
    - Sensitive data in `.env` (never commit)
    - Model configs centralized in `config.py`
    - Prompts organized by purpose

4. **Version Control**
    - `.gitignore` should exclude: `.env` files, model files (`*.bin`, `*.safetensors`), result directories, log files and cache directories.