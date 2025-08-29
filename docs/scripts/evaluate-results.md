# Evaluation Script

The evaluation script:

- **Compares** model outputs with human-annotated ground truth
- **Calculates** multiple performance metrics (accuracy, similarity, classification)
- **Supports** various filtering options (model, k-value, prompt, timestamp)
- **Generates** detailed CSV results and JSON summaries
- **Enables** multi-model comparison and analysis

## Command-Line Arguments

### Model Selection

```bash
--models MODEL1 MODEL2 ...    # Specific models to evaluate
                             # If not specified, evaluates all available models

# Examples:
--models microsoft_phi-4
--models microsoft_phi-4 qwen_qwen-2.5-72b-instruct
```

### Configuration Filtering

```bash
--k K                        # Specific k value (e.g., 3 for k=3)
--complete-policy           # Evaluate complete-policy results instead of RAG
--prompt PROMPT_NAME        # Specific prompt to evaluate (e.g., precise_v4)
```

### Timestamp Selection

```bash
--latest                    # Use the latest experiment for each configuration
--date DATE_STRING         # Specific date to evaluate
                          # Can be partial: "25-01-25" or full: "25-01-25--14-30-00"
```

### Path Configuration

```bash
--json-path PATH           # Path to model outputs (default: from config.py)
--gt-path PATH            # Path to ground truth (default: from config.py)
--output-dir PATH         # Where to save evaluation results (default: from config.py)
```

## Input Structure

### Model Output Files

Expected structure in `resources/results/json_output/`:

```
microsoft_phi-4/
├── k=3/
│   ├── 25-01-25--14-30-00/
│   │   └── precise_v4/
│   │       ├── policy_10_results.json
│   │       └── policy_18_results.json
│   └── 25-01-25--16-00-00/
│       └── precise_v5/
│           └── ...
└── complete-policy/
    └── 25-01-25--18-00-00/
        └── precise_v4/
            └── ...
```

### Ground Truth Files

Expected in `resources/ground_truth/`:

```
GT_policy_10.json
GT_policy_18.json
GT_policy_20.json
...
```

Ground truth format:
```json
{
  "policy_id": "18",
  "questions": [
    {
      "request_id": "1",
      "question": "Can I claim if my luggage is lost?",
      "outcome": "Yes",
      "outcome_justification": "The Company will indemnify...",
      "payment_justification": "€ 150,00"
    }
  ]
}
```

## Metrics Calculated

### 1. Accuracy Metrics

- **Exact Outcome Match**: Percentage where model outcome equals ground truth
- **Classification Accuracy**: Overall multi-class classification accuracy

### 2. Similarity Metrics

- **String Edit Distance (SED)**: Character-level similarity (0-1)
- **Intersection over Union (IoU)**: Word-level overlap (0-1)
- **Payment Similarity**: Accuracy of amount extraction

### 3. Classification Metrics

- **Confusion Matrix**: 3x3 matrix for outcome categories
- **Precision/Recall/F1**: Per-category performance metrics
- **Support**: Number of instances per category

## Basic Usage

### Evaluate Latest Results

```bash
# Evaluate latest results for a specific model
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --latest

# Evaluate latest results for specific k value
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --k 3 \
    --latest
```

### Evaluate Specific Configuration

```bash
# Evaluate specific prompt and k value
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --k 3 \
    --prompt precise_v4 \
    --latest

# Evaluate complete-policy mode
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --complete-policy \
    --latest
```

### Compare Multiple Models

```bash
# Compare two models with same configuration
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --k 3 \
    --prompt precise_v4 \
    --latest
```

## Advanced Usage

### Evaluate Specific Timestamp

```bash
# Evaluate specific experiment by date
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --date "25-01-25" \
    --k 3

# Full timestamp specification
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --date "25-01-25--14-30-00" \
    --k 3
```

### Evaluate All Experiments

```bash
# Evaluate all experiments for a model (no --latest or --date)
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4

# This will process all timestamps and configurations
```

### Custom Paths

```bash
# Use custom paths for evaluation
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --json-path /custom/path/to/outputs \
    --gt-path /custom/path/to/ground_truth \
    --output-dir /custom/path/to/results
```

## Output Files

### Directory Structure

Results are saved to `resources/results/evaluation_results/MODEL_NAME/`:

```
microsoft_phi-4/
├── evaluation_results_k=3_exp_25-01-25--14-30-00_precise_v4_25-01-2025_15-45-30.csv
├── evaluation_summary_k=3_exp_25-01-25--14-30-00_precise_v4_25-01-2025_15-45-30.json
└── classification_metrics_k=3_exp_25-01-25--14-30-00_precise_v4_25-01-2025_15-45-30.json
```

### File Naming Convention

Files include configuration details:
- `k=3` or `complete-policy`: Analysis mode
- `exp_25-01-25--14-30-00`: Experiment timestamp
- `precise_v4`: Prompt name
- `25-01-2025_15-45-30`: Evaluation timestamp

### CSV Results File

Detailed row-by-row comparison:

```csv
model_name,k_value,timestamp,prompt_name,policy_id,request_id,question,output_outcome,gt_outcome,outcome_match,justification_similarity,justification_iou,payment_similarity
microsoft_phi-4,3,25-01-25--14-30-00,precise_v4,18,1,"Can I claim...",Yes,Yes,0,0.856,0.812,0.923
```

### JSON Summary File

Aggregated metrics:

```json
{
  "model_name": "microsoft_phi-4",
  "k_configuration": "k=3",
  "experiment_timestamp": "25-01-25--14-30-00",
  "prompt_name": "precise_v4",
  "total_output_questions": 540,
  "total_evaluated_questions": 540,
  "outcome_classification": {
    "accuracy": 0.852,
    "category_metrics": {
      "Yes": {
        "precision": 0.89,
        "recall": 0.91,
        "f1-score": 0.90,
        "support": 280
      },
      ...
    }
  },
  "exact_outcome_match_percentage": 85.2,
  "avg_justification_similarity": 0.823,
  "avg_justification_iou": 0.756,
  "avg_payment_similarity": 0.891
}
```

### Classification Metrics File

Detailed classification analysis:

```json
{
  "confusion_matrix": [
    [255, 15, 10],
    [12, 168, 20],
    [8, 7, 45]
  ],
  "classification_report": {
    "Yes": {
      "precision": 0.89,
      "recall": 0.91,
      "f1-score": 0.90,
      "support": 280
    },
    ...
  },
  "k_configuration": "k=3",
  "experiment_timestamp": "25-01-25--14-30-00",
  "prompt_name": "precise_v4"
}
```

## Console Output

### Progress Information

```
🔍 Available Model Directories Found:
  - microsoft_phi-4 (with k=1 (2 experiments), k=3 (5 experiments))
  - qwen_qwen-2.5-72b-instruct (with k=3 (3 experiments))

🚀 Evaluating model: microsoft_phi-4
📂 Model directory: resources/results/json_output/microsoft_phi-4
🎯 Mode: k=3
📝 Mode: Prompt=precise_v4
📅 Mode: Latest experiment

  📁 Checking k=3 subdirectory...
    📅 Using latest timestamp: 25-01-25--14-30-00
      📁 Found 27 files in timestamp=25-01-25--14-30-00, prompt=precise_v4

  ✅ Selected 27 files for evaluation:
    - Policy 10: policy_10_results.json (k=3, timestamp=25-01-25--14-30-00, prompt=precise_v4)
    - Policy 18: policy_18_results.json (k=3, timestamp=25-01-25--14-30-00, prompt=precise_v4)
    ...
```

### Evaluation Summary

```
=== EVALUATION RESULTS FOR MICROSOFT_PHI-4 (k=3) ===
Experiment Timestamp: 25-01-25--14-30-00
Prompt Name: precise_v4
Total Questions in Output: 540
Total Questions Evaluated: 540
Outcome Classification Accuracy: 0.8520 (85.20%)
Average Justification IoU: 0.7560

💾 Results saved:
  CSV: resources/results/evaluation_results/microsoft_phi-4/evaluation_results_k=3_exp_25-01-25--14-30-00_precise_v4_25-01-2025_15-45-30.csv
  Summary: resources/results/evaluation_results/microsoft_phi-4/evaluation_summary_k=3_exp_25-01-25--14-30-00_precise_v4_25-01-2025_15-45-30.json
  Metrics: resources/results/evaluation_results/microsoft_phi-4/classification_metrics_k=3_exp_25-01-25--14-30-00_precise_v4_25-01-2025_15-45-30.json
```

### Model Comparison

When evaluating multiple models:

```
=== MODEL COMPARISON ===
Metric                              | microsoft_phi-4 (k=3)                    | qwen_qwen-2.5-72b-instruct (k=3)
-----------------------------------|------------------------------------------|------------------------------------------
Outcome Accuracy (%)               | 85.20                                    | 87.50
Justification IoU                  | 0.7560                                   | 0.7890
```

## Practical Examples

### 1. Complete Evaluation Pipeline

```bash
# Step 1: Generate outputs
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch

# Step 2: Evaluate results
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --k 3 \
    --prompt precise_v4 \
    --latest

# Step 3: Generate LaTeX tables
python src/scripts/latex_tables.py \
    --models microsoft_phi-4
```

### 2. Compare K Values

```bash
# Evaluate different k values
for k in 1 3 5; do
    python src/scripts/evaluate_results.py \
        --models microsoft_phi-4 \
        --k $k \
        --latest
done
```

### 3. Compare Prompts

```bash
# Evaluate all prompts for a model
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --k 3
# This evaluates all prompts if multiple exist
```

### 4. Batch Model Comparison

```bash
#!/bin/bash
# compare_models.sh

MODELS=("microsoft_phi-4" "qwen_qwen-2.5-72b-instruct" "openai_gpt-4o")

# Evaluate all models
python src/scripts/evaluate_results.py \
    --models ${MODELS[@]} \
    --k 3 \
    --prompt precise_v4 \
    --latest
```

## Error Handling

### Common Errors

1. **No Output Files Found**
   ```
   Error: No output files found in resources/results/json_output/microsoft_phi-4
          (Looking specifically for k=3)
   ```
   **Solution**: Ensure you've run main.py with matching configuration

2. **No Ground Truth Files**
   ```
   Error: No ground truth files found in resources/ground_truth/
   ```
   **Solution**: Verify GT files exist with correct naming

3. **Mismatched Questions**
   ```
   ⚠️  Request 25 not found in ground truth
   ```
   **Solution**: Ensure ground truth covers all questions

4. **Multiple Files for Same Policy**
   ```
   ⚠️  Policy 18: Found 2 files, using policy_18_results.json
   ```
   **Note**: Script handles this automatically

## Integration with LaTeX Tables

Results from this script are used by `latex_tables.py`:

```bash
# Run evaluation first
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --k 3 --latest

# Then generate tables
python src/scripts/latex_tables.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct
```

The evaluation script ensures all metrics are properly calculated and stored for downstream analysis and reporting.