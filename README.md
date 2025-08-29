# GPT_AITIS - Insurance Policy Analysis with LLMs and RAG

An advanced system for automated insurance coverage determination using Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). The system analyzes insurance policy documents, answers coverage-related questions with precise policy citations, and provides comprehensive evaluation tools for assessing model performance.

## 🚀 Features

- **Multi-Model Support**: OpenAI GPT-4, Microsoft Phi-4, Qwen models (local & OpenRouter)
- **Advanced RAG Strategies**: Simple, Section-based, Smart-size, Semantic, Graph-based, Hybrid chunking
- **Persona Extraction**: Identifies claimants, affected persons, locations, and relationships
- **Verification System**: Optional multi-iteration verification to correct model outputs
- **Batch Processing**: Efficiently process multiple policies and questions
- **Relevance Filtering**: Pre-screens queries to avoid processing unrelated questions
- **Comprehensive Evaluation**: Automated evaluation against ground truth with detailed metrics
- **Visual Analytics**: Dashboard generation for performance visualization

## 📋 Prerequisites

- Python 3.12+
- CUDA-capable GPU (for local models)
- [UV](https://github.com/astral-sh/uv) package manager
- PDF insurance policy documents
- Ground truth data for evaluation

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/GPT_AITIS.git
cd GPT_AITIS
```

### 2. Install UV (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Virtual Environment and Install Dependencies
```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies from pyproject.toml
uv pip install -e .
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
# OpenAI API (if using GPT models)
OPENAI_API_KEY=your_openai_api_key

# HuggingFace (for model downloads)
HUGGINGFACE_TOKEN=your_huggingface_token

# OpenRouter API (if using OpenRouter models)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_SITE_URL=http://localhost:3000  # Optional
OPENROUTER_SITE_NAME=YourSiteName  # Optional
```

## 📚 Documentation

The project includes comprehensive MkDocs documentation covering installation, usage, API reference, and development guides.

### Running Documentation Locally

1. **Install MkDocs** (if not already installed):
```bash
uv pip install mkdocs mkdocs-material mkdocs-mermaid2-plugin
```

2. **Start the MkDocs development server**:
```bash
# From the project root directory
mkdocs serve
```

3. **For remote server access** (e.g., HPC cluster):

If you're running MkDocs on a remote server, set up an SSH tunnel to view the documentation on your local machine:

```bash
# From your local machine
ssh -L 8000:localhost:8000 your_username@slurm-login01.isc.heia-fr.ch

# Then in the SSH session, navigate to the project and run:
cd /path/to/GPT_AITIS
source .venv/bin/activate
mkdocs serve
```

4. **Access the documentation**:

Open your web browser and navigate to:
```
http://localhost:8000
```

### Documentation Structure

The documentation includes:
- **Getting Started**: Installation, quick start guide
- **User Guide**: Detailed usage instructions, configuration options
- **API Reference**: Complete API documentation for all modules
- **Development**: Contributing guidelines, architecture overview
- **Examples**: Sample configurations and use cases

### Building Static Documentation

To build the documentation as static HTML files:
```bash
mkdocs build
```

The built documentation will be available in the `site/` directory.

## 📁 Project Structure

```
GPT_AITIS/
├── src/
│   ├── main.py                 # Main entry point
│   ├── rag_runner.py          # RAG pipeline orchestration
│   ├── models/                # Model implementations
│   │   ├── openai_model.py    # OpenAI GPT integration
│   │   ├── hf_model.py        # HuggingFace models
│   │   ├── qwen_model.py      # Qwen-specific implementation
│   │   ├── openrouter_model.py # OpenRouter API
│   │   ├── vector_store.py    # Document embedding & retrieval
│   │   ├── verifier.py        # Result verification system
│   │   └── chunking/          # Advanced chunking strategies
│   ├── prompts/               # Prompt templates
│   └── scripts/               # Utility scripts
│       ├── evaluate_results.py # Model evaluation
│       ├── create_dashboard.py # Generate visual analytics
│       └── compare_models.py   # Model comparison tools
├── docs/                      # MkDocs documentation source
│   ├── index.md              # Documentation homepage
│   ├── getting-started/      # Installation and setup guides
│   ├── user-guide/           # Usage documentation
│   ├── api/                  # API reference
│   └── development/          # Developer documentation
├── mkdocs.yml                # MkDocs configuration
├── resources/
│   ├── documents/policies/    # Place PDF policies here
│   ├── questions/            # Questions Excel file
│   ├── ground_truth/         # Ground truth JSON files
│   ├── raw_ground_truth/     # Raw GT Excel files
│   └── results/              # All outputs
│       ├── json_output/      # Model predictions
│       ├── evaluation_results/ # Evaluation metrics
│       ├── eval_new/         # Custom evaluation outputs
│       ├── logs/             # Execution logs
│       └── dashboard.html    # Visual analytics dashboard
├── pyproject.toml            # Project dependencies
└── .env                      # Environment variables
```

## 📄 Data Preparation

### 1. Insurance Policy PDFs
Place your insurance policy PDF files in `resources/documents/policies/`:
- Files should be named with pattern: `{ID}_{PolicyName}.pdf`
- Example: `10_Nobis_Travel_Insurance.pdf`, `18_Baggage_Loss_Policy.pdf`

### 2. Questions File
Create an Excel file at `resources/questions/questions.xlsx` with columns:
- `Id`: Question ID (numeric)
- `Questions`: The insurance-related question

Example:
| Id | Questions |
|----|-----------|
| 1  | My baggage was lost at the airport. Can I claim? |
| 2  | I got sick during my trip to Spain. Is medical treatment covered? |

### 3. Ground Truth Data
For evaluation, prepare ground truth files in `resources/ground_truth/`:
- Format: `policy_{ID}_gt.json`
- Structure should match the output format (see Output Structure section)

## 🏃‍♂️ Running the System

### Basic Command Structure
```bash
python src/main.py [OPTIONS]
```

### Common Usage Examples

#### 1. Process All Questions with OpenAI GPT-4
```bash
python src/main.py \
    --model openai \
    --model-name gpt-4o \
    --prompt standard \
    --k 3 \
    --batch
```

#### 2. Use Local Phi-4 Model with Semantic RAG
```bash
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --k 3 \
    --rag-strategy semantic \
    --batch
```

#### 3. Process Specific Questions with Verification
```bash
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --questions "1,2,3,4,5" \
    --k 3 \
    --verifier \
    --verifier-iterations 2 \
    --batch
```

#### 4. Use OpenRouter for Qwen Model
```bash
python src/main.py \
    --model openrouter \
    --model-name "qwen/qwen-2.5-72b-instruct" \
    --prompt precise_v4_qwen \
    --k 3 \
    --batch
```

### 📌 Key Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model provider: `openai`, `hf`, `qwen`, `openrouter` | `hf` |
| `--model-name` | Specific model name/path | `microsoft/phi-4` |
| `--prompt` | Prompt template name | `standard` |
| `--batch` | Process all policies in batch mode | False |
| `--k` | Number of context chunks to retrieve | 3 |
| `--rag-strategy` | RAG strategy: `simple`, `section`, `smart_size`, `semantic`, `graph`, `hybrid` | `simple` |
| `--questions` | Comma-separated question IDs | All questions |
| `--policy-id` | Process specific policy ID only | All policies |
| `--complete-policy` | Use entire policy document (no RAG) | False |
| `--verifier` | Enable result verification | False |
| `--verifier-iterations` | Number of verification passes | 1 |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

## 📊 Evaluation System

### Running Evaluations

After generating predictions, evaluate model performance against ground truth:

```bash
python ./src/scripts/evaluate_results.py \
    --models qwen3-235b-a22b \
    --json-path ./resources/results/json_output/ \
    --gt-path ./resources/ground_truth/ \
    --output-dir ./resources/results/eval_new/
```

#### Evaluation Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--models` | Model names to evaluate (comma-separated) | `qwen3-235b-a22b,microsoft_phi-4` |
| `--json-path` | Path to model outputs | `./resources/results/json_output/` |
| `--gt-path` | Path to ground truth files | `./resources/ground_truth/` |
| `--output-dir` | Where to save evaluation results | `./resources/results/eval_new/` |
| `--detailed` | Generate detailed per-question analysis | Flag option |
| `--policies` | Specific policy IDs to evaluate | `10,18,20` |

### Evaluation Output Structure

```
resources/results/eval_new/
├── {model_name}/
│   ├── summary_metrics.json      # Overall performance metrics
│   ├── per_policy_results.json   # Policy-level breakdown
│   ├── per_question_results.json # Question-level analysis
│   └── confusion_matrix.csv      # Classification confusion matrix
└── comparison_report.html         # Multi-model comparison
```

### Metrics Provided

- **Accuracy**: Overall correctness of eligibility decisions
- **Precision/Recall/F1**: Per-class metrics (Yes, No - Unrelated, No - conditions not met)
- **Justification Quality**: 
  - IoU (Intersection over Union) for quoted text
  - Exact match percentage
  - Partial match scores
- **Payment Accuracy**: Correctness of payment amount extraction

## 📊 Output Structure

### Model Predictions
Results are saved in a timestamped directory structure:
```
resources/results/json_output/
└── {model_name}/
    └── k={k}/                    # or complete-policy/
        └── {DD-MM-YY--HH-MM-SS}/
            └── {prompt_name}/
                └── policy_{ID}_results.json
```

### JSON Output Format
```json
{
  "policy_id": "10",
  "questions": [
    {
      "request_id": "1",
      "question": "My baggage was lost at the airport. Can I claim?",
      "outcome": "Yes",
      "outcome_justification": "In the event that the air carrier fails to deliver...",
      "payment_justification": "Option 1 € 150,00 Option 2 € 350,00"
    }
  ]
}
```

### Ground Truth Format
Ground truth files should follow the same structure:
```json
{
  "policy_id": "10",
  "questions": [
    {
      "request_id": "1",
      "question": "My baggage was lost at the airport. Can I claim?",
      "outcome": "Yes",
      "outcome_justification": "Expected justification text from policy",
      "payment_justification": "Expected payment amount"
    }
  ]
}
```

## 🚀 Running on HPC/SLURM

For HPC environments, use the provided SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=insurance_rag_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40_48gb:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Activate environment
cd /path/to/GPT_AITIS
source .venv/bin/activate

# Run analysis
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --k 3 \
    --batch \
    --rag-strategy semantic

# Run evaluation
python ./src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --json-path ./resources/results/json_output/ \
    --gt-path ./resources/ground_truth/ \
    --output-dir ./resources/results/eval_new/

# Generate dashboard
python ./src/scripts/create_dashboard.py \
    --eval-dir ./resources/results/eval_new/ \
    --output ./resources/results/dashboard.html
```

## 📈 Performance Tips

1. **GPU Memory Management**
   - For large models: 32GB+ GPU recommended
   - Use `--complete-policy` cautiously with large documents

2. **Evaluation Efficiency**
   - Use `--policies` to evaluate specific policies during development
   - Enable `--detailed` only when needed (increases output size)

3. **Result Organization**
   - Use consistent naming for easy evaluation
   - Archive timestamped results regularly

## 🐛 Troubleshooting

### Common Issues

1. **Evaluation Script Not Found**
   ```bash
   # Ensure you're in the project root
   cd /path/to/GPT_AITIS
   python ./src/scripts/evaluate_results.py --help
   ```

2. **Ground Truth Mismatch**
   - Ensure ground truth JSON structure matches output format
   - Check policy IDs match between predictions and ground truth
   - Verify question IDs are consistent

3. **Memory Issues During Evaluation**
   ```bash
   # Process policies in batches
   python ./src/scripts/evaluate_results.py \
       --models your_model \
       --policies "1,2,3,4,5" \
       --json-path ./resources/results/json_output/ \
       --gt-path ./resources/ground_truth/
   ```

## 📚 Complete Workflow Example

### End-to-End Pipeline
```bash
# 1. Prepare data
cp your_policies/*.pdf resources/documents/policies/
cp questions.xlsx resources/questions/
cp ground_truth/*.json resources/ground_truth/

# 2. Run model predictions
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --k 3 \
    --rag-strategy semantic \
    --batch \
    --verifier

# 3. Evaluate results
python ./src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --json-path ./resources/results/json_output/ \
    --gt-path ./resources/ground_truth/ \
    --output-dir ./resources/results/eval_new/ \
    --detailed

# 4. Generate visualizations
python ./src/scripts/create_dashboard.py \
    --eval-dir ./resources/results/eval_new/ \
    --output ./resources/results/dashboard.html

# 5. View results
open ./resources/results/dashboard.html  # macOS
# or
xdg-open ./resources/results/dashboard.html  # Linux
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure evaluation scripts work with your changes
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.