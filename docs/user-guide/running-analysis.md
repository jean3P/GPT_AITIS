# Running Analysis

This guide covers how to run insurance policy analysis using GPT_AITIS, from basic usage to advanced configurations.

## Basic Usage

### Simplest Command
```bash
python src/main.py
```
This runs with default settings:

- Model: microsoft/phi-4
- RAG strategy: simple
- All available questions
- All policies in the documents directory

### Typical Analysis
```bash
python src/main.py \
  --model hf \
  --model-name microsoft/phi-4 \
  --k 3 \
  --questions "1,2,3,4,5"
```

## Command Line Options

### Model Selection

#### `--model` (required)
Choose the model provider:

- `openai`: OpenAI API models
- `hf`: HuggingFace models (local or downloaded)
- `qwen`: Qwen models (alias for hf with Qwen)
- `openrouter`: OpenRouter API for cloud models

```bash
# OpenAI
python src/main.py --model openai --model-name gpt-4o

# HuggingFace
python src/main.py --model hf --model-name microsoft/phi-4

# OpenRouter
python src/main.py --model openrouter --model-name qwen/qwen-2.5-72b-instruct
```

#### `--model-name`
Specific model to use. Examples:

- OpenAI: `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`
- HuggingFace: `microsoft/phi-4`, `Qwen/Qwen2.5-32B`
- OpenRouter: `qwen/qwen-2.5-72b-instruct`, `anthropic/claude-3.5-sonnet`

### Processing Options

#### `--batch`
Process all policies in a single run (more efficient):
```bash
python src/main.py --batch
```

Without `--batch`, each policy is processed individually.

#### `--num-questions`
Limit the number of questions to process:
```bash
python src/main.py --num-questions 10  # Process first 10 questions
```

#### `--questions`
Process specific questions by ID:
```bash
python src/main.py --questions "1,5,10,15"
```

#### `--policy-id`
Process only a specific policy:
```bash
python src/main.py --policy-id 18  # Only process policy 18
```

### RAG Configuration

#### `--rag-strategy`
Choose the document chunking strategy:
```bash
python src/main.py --rag-strategy semantic
```

Available strategies:

- `simple`: Basic paragraph-based chunking
- `section`: Structural section-based chunking
- `smart_size`: Adaptive chunking based on content
- `semantic`: Embedding-based semantic chunking

#### `--k`
Number of chunks to retrieve (default: 3):
```bash
python src/main.py --k 5  # Retrieve top 5 chunks
```

#### `--complete-policy`
Use entire policy document instead of RAG:
```bash
python src/main.py --complete-policy
```

### Prompt Selection

#### `--prompt`
Choose the prompt template:
```bash
python src/main.py --prompt precise_v4
```

Available prompts:

- `standard`: Basic prompt
- `detailed`: More comprehensive
- `precise`: Improved accuracy
- `precise_v2` through `precise_v5`: Iterative improvements
- Model-specific: `precise_v3_qwen`, `precise_v4_qwen`, `precise_v3_phi-4_v2`

### Quality Features

#### `--persona`
Enable persona extraction from questions:
```bash
python src/main.py --persona
```

This identifies:

- Who is making the claim
- Who experienced the event
- Where it occurred
- Coverage relationships

#### `--filter-irrelevant`
Pre-filter obviously irrelevant queries:
```bash
python src/main.py --filter-irrelevant --prompt-relevant relevance_filter_v2
```

#### `--verifier`
Enable result verification:
```bash
python src/main.py --verifier --verifier-iterations 2
```

### Output Configuration

#### `--output-dir`
Specify output directory:
```bash
python src/main.py --output-dir /path/to/results
```

#### `--log-level`
Set logging verbosity:
```bash
python src/main.py --log-level DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

## Complete Examples

### Example 1: Quick Test
Test a single question on one policy:
```bash
python src/main.py \
  --model hf \
  --model-name microsoft/phi-4 \
  --policy-id 18 \
  --questions "1" \
  --k 3
```

### Example 2: Production Analysis
Full analysis with quality checks:
```bash
python src/main.py \
  --model openai \
  --model-name gpt-4o \
  --batch \
  --rag-strategy semantic \
  --k 5 \
  --prompt precise_v4 \
  --persona \
  --filter-irrelevant \
  --verifier \
  --verifier-iterations 1 \
  --log-level INFO
```

### Example 3: Model Comparison
Run same analysis with different models:
```bash
# Phi-4
python src/main.py --model hf --model-name microsoft/phi-4 \
  --prompt precise_v3_phi-4_v2 --batch --output-dir results/phi4

# Qwen 32B
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B \
  --prompt precise_v4_qwen --batch --output-dir results/qwen32b

# GPT-4o
python src/main.py --model openai --model-name gpt-4o \
  --prompt precise_v4 --batch --output-dir results/gpt4o
```

### Example 4: Complete Policy Mode
For models with large context windows:
```bash
python src/main.py \
  --model openrouter \
  --model-name qwen/qwen-2.5-72b-instruct \
  --complete-policy \
  --prompt precise_v4_qwen \
  --batch
```

## Understanding Output

### Directory Structure
Output follows this pattern:
```
output_dir/
├── model_name/
│   ├── k=3/                    # or complete-policy/
│   │   ├── DD-MM-YY--HH-MM-SS/
│   │   │   ├── prompt_name/
│   │   │   │   ├── policy_10_results.json
│   │   │   │   ├── policy_18_results.json
│   │   │   │   └── ...
```

### JSON Output Format
Each policy gets a JSON file:
```json
{
  "policy_id": "18",
  "questions": [
    {
      "request_id": "1",
      "question": "During my vacation, my luggage containing...",
      "outcome": "Yes",
      "outcome_justification": "In the event that the air carrier fails to deliver the Insured's Baggage...",
      "payment_justification": "Option 1 € 150,00 Option 2 € 350,00 Option 3 € 500,00"
    }
  ]
}
```

### Outcome Types
- **"Yes"**: Coverage confirmed with supporting policy text
- **"No - Unrelated event"**: Query unrelated to policy coverage
- **"No - condition(s) not met"**: Related but conditions not satisfied

## Advanced Workflows

### Workflow 1: Iterative Refinement
Start simple and refine:
```bash
# 1. Quick test with simple RAG
python src/main.py --model hf --questions "1" --k 3

# 2. Try semantic chunking
python src/main.py --model hf --questions "1" --k 3 --rag-strategy semantic

# 3. Add more context
python src/main.py --model hf --questions "1" --k 5 --rag-strategy semantic

# 4. Enable verification
python src/main.py --model hf --questions "1" --k 5 --rag-strategy semantic --verifier
```

### Workflow 2: Debugging Failures
When results are incorrect:
```bash
# 1. Enable debug logging
python src/main.py --log-level DEBUG --questions "problematic_id"

# 2. Try complete policy to rule out RAG issues
python src/main.py --complete-policy --questions "problematic_id"

# 3. Test different prompts
python src/main.py --prompt precise_v5 --questions "problematic_id"

# 4. Enable persona extraction
python src/main.py --persona --questions "problematic_id"
```

### Workflow 3: Batch Processing Pipeline
For processing many policies:
```bash
#!/bin/bash
# process_all.sh

MODELS=("microsoft/phi-4" "Qwen/Qwen2.5-32B")
STRATEGIES=("semantic" "hybrid")

for model in "${MODELS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    echo "Processing with $model using $strategy"
    python src/main.py \
      --model hf \
      --model-name "$model" \
      --batch \
      --rag-strategy "$strategy" \
      --k 3 \
      --output-dir "results/${model//\//_}_${strategy}"
  done
done
```

## Performance Optimization

### Memory Management
For large models:
```bash
# Use environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run with optimized settings
python src/main.py --model hf --model-name Qwen/Qwen2.5-32B
```

### Batch Size Considerations
- Use `--batch` for multiple policies
- Process questions in groups for memory efficiency
- Consider splitting large question sets

### GPU Utilization
Monitor GPU usage:
```bash
# In another terminal
watch -n 1 nvidia-smi

# Run analysis
python src/main.py --batch
```

## 🛠️ Troubleshooting Common Issues

### Performance Issues

??? failure "Out of Memory Errors"
    ```bash
    # Solution 1: Reduce context chunks
    python src/main.py --k 1
    
    # Solution 2: Use smaller model
    python src/main.py --model-name Qwen/Qwen2.5-7B
    
    # Solution 3: Process fewer questions at once
    python src/main.py --num-questions 5
    
    # Solution 4: Enable CPU offloading
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    ```

??? failure "Slow Processing Speed"
    ```bash
    # Solution 1: Use simple RAG strategy
    python src/main.py --rag-strategy simple
    
    # Solution 2: Disable verification
    # Don't use --verifier flag
    
    # Solution 3: Enable batch mode
    python src/main.py --batch
    
    # Solution 4: Use GPU acceleration
    python src/main.py --device cuda
    ```

??? failure "Poor Results Quality"
    ```bash
    # Solution 1: Increase context chunks
    python src/main.py --k 5
    
    # Solution 2: Try different RAG strategy
    python src/main.py --rag-strategy section  # or semantic, hybrid
    
    # Solution 3: Use complete policy (no chunking)
    python src/main.py --complete-policy
    
    # Solution 4: Enable all quality features
    python src/main.py --persona --verifier --filter-irrelevant
    
    # Solution 5: Use more powerful model
    python src/main.py --model openai --model-name gpt-4o
    ```

## 📚 Best Practices

### Getting Started

??? success "Start Simple, Add Complexity Gradually"
    ```bash
    # Step 1: Basic configuration
    python src/main.py --model hf --k 3
    
    # Step 2: Add chunking strategy
    python src/main.py --model hf --k 3 --rag-strategy section
    
    # Step 3: Add quality features
    python src/main.py --model hf --k 3 --rag-strategy semantic --persona
    
    # Step 4: Add verification
    python src/main.py --model hf --k 3 --rag-strategy semantic --persona --verifier
    ```

### Model Selection

??? tip "Choose Models Based on Your Needs"
    ```yaml
    Quick Testing:
      Model: microsoft/phi-4
      Command: python src/main.py --model hf --model-name microsoft/phi-4
      
    Better Accuracy:
      Model: Qwen/Qwen2.5-32B or larger
      Command: python src/main.py --model qwen --model-name Qwen/Qwen2.5-32B
      
    Production Use:
      Model: OpenAI GPT-4o with verification
      Command: python src/main.py --model openai --model-name gpt-4o --verifier
    ```

### Quality Assurance

??? success "Always Validate Your Results"
    ```bash
    # Step 1: Run analysis with clear output directory
    python src/main.py --batch --output-dir results/experiment_1
    
    # Step 2: Run evaluation against ground truth
    python scripts/evaluate_results.py \
        --json-path results/experiment_1/ \
        --gt-path resources/ground_truth/ \
        --output-dir evaluation/experiment_1/
    
    # Step 3: Generate visual dashboard
    python scripts/create_dashboard.py \
        --eval-dir evaluation/experiment_1/ \
        --output dashboard_exp1.html
    ```

### Documentation

??? tip "Document Your Experiments"
    ```bash
    # Create experiment directory
    mkdir -p experiments/2025_01_15_semantic_k5
    
    # Save command used
    echo "python src/main.py --model hf --k 5 --rag-strategy semantic --batch" \
        > experiments/2025_01_15_semantic_k5/command.txt
    
    # Save configuration
    cp .env experiments/2025_01_15_semantic_k5/config.env
    
    # Add notes
    cat > experiments/2025_01_15_semantic_k5/notes.md << EOF
    Experiment: Semantic chunking with k=5
    Date: 2025-01-15
    Purpose: Test if more context improves accuracy
    Result: [Add results after evaluation]
    EOF
    ```

### Performance Optimization

??? info "Optimize for Your Hardware"
    ```bash
    # For limited GPU memory (< 24GB)
    python src/main.py \
        --model hf \
        --model-name microsoft/phi-4 \
        --k 3 \
        --batch-size 1
    
    # For powerful GPUs (40GB+)
    python src/main.py \
        --model qwen \
        --model-name Qwen/Qwen2.5-32B \
        --k 5 \
        --batch \
        --verifier
    
    # For CPU-only systems
    python src/main.py \
        --model openrouter \
        --model-name qwen/qwen-2.5-72b-instruct \
        --k 3
    ```

### Production Checklist

??? success "Before Going to Production"
    ```bash
    # 1. Test with sample data
    python src/main.py --policy-id 18 --questions "1,2,3" --verifier
    
    # 2. Validate configuration
    python scripts/validate_config.py
    
    # 3. Run comprehensive evaluation
    python scripts/evaluate_results.py --detailed
    
    # 4. Check resource usage
    python scripts/profile_memory.py
    
    # 5. Set up monitoring
    python src/main.py --log-level INFO --log-file production.log
    ```

## Next Steps

- Learn about [Model Selection](model-selection.md) for choosing the right model
- Explore [RAG Strategies](rag-strategies.md) in detail
- Understand [Prompt Engineering](prompts.md) for better results
- Set up [Result Verification](verification.md) for quality assurance
- Try [Batch Processing](batch-processing.md) for efficiency