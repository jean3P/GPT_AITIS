# Batch Processing

Batch processing allows you to analyze multiple insurance policies with a set of questions in a single run.

## Basic Batch Processing

### Command Structure

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --k 3
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--batch` | Enable batch processing mode | False |
| `--num-questions` | Limit number of questions to process | All |
| `--questions` | Specific question IDs to process | All |
| `--policy-id` | Process only specific policy | All |

## Processing Strategies

### 1. All Policies, All Questions

Process every policy with every question in your dataset:

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5
```

### 2. Specific Questions Across All Policies

Process selected questions for all policies:

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --questions "1,2,3,4,5" \
    --prompt precise_v5
```

### 3. Limited Question Set

Process first N questions for testing:

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --num-questions 10 \
    --prompt precise_v5
```

### 4. Single Policy Batch

Process one policy with all questions:

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --policy-id "18" \
    --prompt precise_v5
```

## Advanced Batch Options

### With RAG Strategy

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --rag-strategy semantic \
    --k 5 \
    --prompt precise_v5
```

### With Verification

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --verifier \
    --verifier-iterations 2 \
    --prompt precise_v5
```

### Complete Policy Mode

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --complete-policy \
    --prompt precise_v5
```

### With Relevance Filtering

```bash
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --filter-irrelevant \
    --prompt-relevant relevance_filter_v2 \
    --prompt precise_v5
```

## Output Organization

### Directory Structure

Batch processing creates organized output directories:

```
resources/results/json_output/
├── microsoft_phi-4/
│   ├── k=3/
│   │   └── DD-MM-YY--HH-MM-SS/
│   │       └── precise_v5/
│   │           ├── policy_10_results.json
│   │           ├── policy_18_results.json
│   │           └── policy_20_results.json
│   └── complete-policy/
│       └── DD-MM-YY--HH-MM-SS/
│           └── precise_v5/
│               └── policy_18_results.json
```

### JSON Output Format

Each policy gets its own JSON file:

```json
{
  "policy_id": "18",
  "questions": [
    {
      "request_id": "1",
      "question": "My baggage was lost at the airport...",
      "outcome": "Yes",
      "outcome_justification": "In the event that the air carrier...",
      "payment_justification": "€ 150,00"
    },
    {
      "request_id": "2",
      "question": "I got sick during my vacation...",
      "outcome": "No - Unrelated event",
      "outcome_justification": "",
      "payment_justification": null
    }
  ]
}
```

## Performance Optimization

### 1. Memory Management

For large batches, monitor memory usage:

```bash
# Use logging to track progress
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --log-level INFO \
    --prompt precise_v5
```

### 2. GPU Utilization

Ensure efficient GPU usage:

```bash
# Check GPU availability first
nvidia-smi

# Run batch processing
python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5
```

### 3. Chunking Strategy Selection

Choose appropriate chunking for batch processing:

- **Simple**: Fast, baseline performance
- **Semantic**: Better accuracy, slower processing
- **Smart Size**: Balance between speed and accuracy

```bash
# Fast processing
python src/main.py --batch --rag-strategy simple --k 3

# Accurate processing
python src/main.py --batch --rag-strategy semantic --k 5
```

## HPC/SLURM Batch Jobs

### Basic SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=batch_analysis
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

python src/main.py --batch \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --k 3
```

## Monitoring Batch Progress

### Real-time Monitoring

```bash
# Watch log file
tail -f resources/results/logs/swereflect_*.log

# Monitor specific policy processing
grep "Processing policy ID: 18" resources/results/logs/*.log
```

### Progress Indicators

The system logs progress at multiple levels:

```
INFO: Processing policy ID: 18 from file: 18_Nobis.pdf
INFO: Processing question 1/30
INFO: Retrieved 3 context chunks
INFO: ✓ Processed question 1 for policy 18
INFO: Saved JSON for policy 18
```

## Error Handling

### Partial Failures

The batch processor continues even if individual queries fail:

```json
{
  "request_id": "5",
  "question": "...",
  "outcome": "Error",
  "outcome_justification": "Model inference timeout",
  "payment_justification": ""
}
```

### Recovery Strategies

1. **Resume Processing**: Re-run with specific questions that failed
   ```bash
   python src/main.py --batch --questions "5,12,18"
   ```

2. **Skip Problematic Policies**: Exclude specific policies
   ```bash
   # Process all except policy 20
   python src/main.py --batch --policy-id "10" 
   python src/main.py --batch --policy-id "18"
   ```

## Batch Processing Workflows

### 1. Model Comparison

```bash
# Run same questions on different models
for model in "microsoft/phi-4" "Qwen/Qwen2.5-7B"; do
    python src/main.py --batch \
        --model hf \
        --model-name "$model" \
        --questions "1,2,3,4,5"
done
```

### 2. RAG Strategy Evaluation

```bash
# Compare different RAG strategies
for strategy in simple section semantic smart_size; do
    python src/main.py --batch \
        --rag-strategy "$strategy" \
        --output-dir "results/rag_comparison/$strategy"
done
```

### 3. Prompt Testing

```bash
# Test different prompts
for prompt in precise_v4 precise_v5 standard; do
    python src/main.py --batch \
        --prompt "$prompt" \
        --output-dir "results/prompt_comparison/$prompt"
done
```

## Next Steps

- Learn about [Evaluation](../evaluation/overview.md) to analyze batch results
- Explore [Model Comparison](../evaluation/comparison.md) techniques
- Set up [Automated Dashboards](../evaluation/dashboards.md) for results
- Configure [HPC workflows](../examples/hpc.md) for large-scale processing