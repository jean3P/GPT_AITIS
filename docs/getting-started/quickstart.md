# Quick Start Guide

Get up and running with GPT_AITIS in 5 minutes! This guide will walk you through analyzing your first insurance policy.

## 🚀 Prerequisites

Before starting, ensure you have:

- ✅ [Installed GPT_AITIS](installation.md)
- ✅ Activated the virtual environment
- ✅ Configured API keys in `.env` file
- ✅ At least one insurance policy PDF

## 📋 Step 1: Prepare Your Data

### 1.1 Add Insurance Policy PDFs

Place your insurance policy PDFs in the `resources/documents/policies/` directory:

```bash
# Create the directory if it doesn't exist
mkdir -p resources/documents/policies/

# Copy your PDFs (naming convention: {ID}_{Name}.pdf)
cp /path/to/your/policy.pdf resources/documents/policies/10_Sample_Policy.pdf
```

!!! tip "Naming Convention"
    Use the format `{ID}_{PolicyName}.pdf` where ID is a number:
    - ✅ `10_Travel_Insurance.pdf`
    - ✅ `18_Medical_Coverage.pdf`
    - ❌ `TravelInsurance.pdf` (missing ID)

### 1.2 Create Questions File

Create an Excel file with your questions at `resources/questions/questions.xlsx`:

```python
import pandas as pd

# Create sample questions
questions = pd.DataFrame({
    'Id': [1, 2, 3, 4, 5],
    'Questions': [
        "My baggage was lost at the airport. Can I claim?",
        "I got sick during my trip to Spain. Is medical treatment covered?",
        "My flight was delayed by 6 hours. Am I eligible for compensation?",
        "My camera was stolen from my hotel room. Is this covered?",
        "I had to cancel my trip due to illness. Can I get a refund?"
    ]
})

# Save to Excel
questions.to_excel('resources/questions/questions.xlsx', index=False)
```

Or create it manually with Excel/LibreOffice with these columns:
- `Id`: Numeric question ID
- `Questions`: The insurance-related question

## 🏃‍♂️ Step 2: Run Your First Analysis

### Option A: Using OpenAI GPT-4 (Simplest)

```bash
python src/main.py \
    --model openai \
    --model-name gpt-4o \
    --prompt standard \
    --k 3 \
    --batch
```

### Option B: Using Local Phi-4 Model (No API needed)

```bash
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --k 3 \
    --batch
```

### Option C: Process Specific Questions Only

```bash
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --questions "1,2,3" \
    --k 3
```

## 📊 Step 3: Understanding the Output

### Output Location

Results are saved in a timestamped directory:
```
resources/results/json_output/
└── microsoft_phi-4/
    └── k=3/
        └── 09-11-24--14-35-22/
            └── precise_v5/
                └── policy_10_results.json
```

### Output Format

Each policy gets a JSON file with results:

```json
{
  "policy_id": "10",
  "questions": [
    {
      "request_id": "1",
      "question": "My baggage was lost at the airport. Can I claim?",
      "outcome": "Yes",
      "outcome_justification": "In the event that the air carrier fails to deliver the Insured's Baggage within 24 hours from the Insured's arrival at the scheduled destination",
      "payment_justification": "Option 1 € 150,00 Option 2 € 350,00 Option 3 € 500,00"
    }
  ]
}
```

### Key Fields Explained

| Field | Description | Example Values |
|-------|-------------|----------------|
| `outcome` | Coverage decision | `"Yes"`, `"No - Unrelated event"`, `"No - condition(s) not met"` |
| `outcome_justification` | Quote from policy supporting decision | Exact text from policy |
| `payment_justification` | Amount covered (if applicable) | `"€ 500,00"` or `null` |

## 🎯 Step 4: Quick Evaluation (Optional)

If you have ground truth data, evaluate your results:

```bash
# Prepare ground truth file
# Format: resources/ground_truth/policy_10_gt.json

# Run evaluation
python ./src/scripts/evaluate_results.py \
    --models microsoft_phi-4 \
    --json-path ./resources/results/json_output/ \
    --gt-path ./resources/ground_truth/ \
    --output-dir ./resources/results/eval_new/

# View summary metrics
cat ./resources/results/eval_new/microsoft_phi-4/summary_metrics.json
```

## 🎨 Step 5: Visualizing Results

### Generate LaTeX Tables for Academic Papers

For academic publications or reports, generate professional LaTeX tables:

```bash
# Generate LaTeX tables for a single model
python ./src/scripts/latex_tables.py \
    --input-results ./resources/results/evaluation_results/ \
    --models microsoft_phi-4 \
    --output ./results_tables.tex

# Generate comparison tables for multiple models
python ./src/scripts/latex_tables.py \
    --input-results ./resources/results/evaluation_results/ \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --output ./comparison_tables.tex
```

#### LaTeX Table Types Generated

1. **Evaluation Summary Table**: Overall performance metrics
2. **Confusion Matrix**: Classification accuracy breakdown
3. **Classification Metrics**: Precision, Recall, F1 by category
4. **Model Comparison**: Side-by-side performance comparison

#### Complete Example: LaTeX Report Generation

```bash
# 1. Run analysis for two models
python src/main.py --model hf --model-name microsoft/phi-4 --batch
python src/main.py --model openrouter --model-name qwen/qwen-2.5-72b-instruct --batch

# 2. Run evaluation for both models
python ./src/scripts/evaluate_results.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --json-path ./resources/results/json_output/ \
    --gt-path ./resources/ground_truth/

# 3. Generate comparison LaTeX tables
python ./src/scripts/latex_tables.py \
    --input-results ./resources/results/evaluation_results/ \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --output ./model_comparison.tex

# 4. Include in your LaTeX document
echo '\input{model_comparison.tex}' >> your_paper.tex
```

#### Sample LaTeX Output

The script generates tables like this:

```latex
\begin{table}[H]
\centering
\caption{Model Comparison: microsoft\_phi-4 vs qwen\_qwen-2.5-72b-instruct}
\label{tab:model_comparison}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{microsoft\_phi-4} & \textbf{qwen\_qwen-2.5-72b-instruct} \\
\midrule
Predicted Outcome (Accu.) & 85.42\% & 91.67\% \\
Justification Outcome (IoU) & 0.6543 & 0.7234 \\
\bottomrule
\end{tabular}
\end{table}
```

!!! tip "LaTeX Integration"
    The generated `.tex` file can be directly included in your LaTeX document using `\input{filename.tex}` or copied into your paper. Make sure to include the required packages: `\usepackage{booktabs}` and `\usepackage{multirow}`.

## 🔍 Monitoring Progress

Watch the logs for progress:

```bash
# Run with debug logging
python src/main.py \
    --model hf \
    --model-name microsoft/phi-4 \
    --prompt precise_v5 \
    --k 3 \
    --batch \
    --log-level DEBUG

# In another terminal, watch the log file
tail -f resources/results/logs/rag_pipeline_*.log
```

## 🐛 Common Quick Start Issues

??? failure "No PDF files found"
    ```bash
    # Check if PDFs are in the correct directory
    ls -la resources/documents/policies/
    
    # Ensure files have .pdf extension (case sensitive)
    # Rename if needed: mv file.PDF file.pdf
    ```

??? failure "Model not found"
    ```bash
    # For local models, download first
    ./download_models.sh phi-4
    
    # Or use cloud models
    python src/main.py --model openai --model-name gpt-4o
    ```

??? failure "Out of GPU memory"
    ```bash
    # Reduce batch size or use CPU
    export CUDA_VISIBLE_DEVICES=""  # Force CPU
    
    # Or use smaller k value
    python src/main.py --k 1
    ```

## 📈 Sample Results Interpretation

Here's what good results look like:

✅ **Good Result**:
```json
{
  "outcome": "Yes",
  "outcome_justification": "Baggage delay coverage applies when baggage is delayed for more than 4 hours",
  "payment_justification": "Maximum benefit: €200 per person"
}
```

❌ **Poor Result**:
```json
{
  "outcome": "Yes",
  "outcome_justification": "The policy covers various travel incidents",
  "payment_justification": null
}
```
*Issue: Vague justification, no specific policy quote*

## 🎯 Next Steps

Now that you've run your first analysis:

1. **[Configure Advanced Settings](configuration.md)** - Optimize for your use case
2. **[Explore RAG Strategies](../user-guide/rag-strategies.md)** - Improve retrieval quality  
3. **[Try Different Models](../user-guide/model-selection.md)** - Compare performance
4. **[Enable Verification](../user-guide/verification.md)** - Improve accuracy
5. **[Set Up Evaluation](../evaluation/overview.md)** - Measure performance

## 🚀 Quick Command Reference

```bash
# Basic analysis
python src/main.py --model hf --model-name microsoft/phi-4 --batch

# With specific questions
python src/main.py --model hf --model-name microsoft/phi-4 --questions "1,2,3"

# With verification
python src/main.py --model hf --model-name microsoft/phi-4 --batch --verifier

# With semantic RAG
python src/main.py --model hf --model-name microsoft/phi-4 --batch --rag-strategy semantic

# Complete policy mode (no RAG)
python src/main.py --model hf --model-name microsoft/phi-4 --batch --complete-policy

# Evaluation
python ./src/scripts/evaluate_results.py --models microsoft_phi-4

# LaTeX tables
python ./src/scripts/latex_tables.py --models microsoft_phi-4
```

---

!!! success "Congratulations!"
    You've successfully run your first insurance policy analysis with GPT_AITIS! The system has analyzed your policy, answered questions with specific policy citations, and saved the results in JSON format.

!!! tip "Pro Tip"
    Start with a small set of questions (3-5) to verify everything works correctly before processing larger batches. This helps identify any issues early and saves processing time.