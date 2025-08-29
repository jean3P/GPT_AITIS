# LaTeX Tables Script

The `latex_tables.py` script generates publication-ready LaTeX tables from evaluation results. It creates formatted tables suitable for academic papers, including individual model performance tables and multi-model comparison tables.

## Overview

The LaTeX tables script:

- **Reads** evaluation results from `evaluate_results.py`
- **Generates** multiple types of formatted LaTeX tables
- **Supports** individual model analysis and multi-model comparison
- **Creates** publication-ready output with proper formatting
- **Handles** special characters and LaTeX escaping

## Command-Line Arguments

```bash
--input-results PATH      # Directory containing evaluation results 
                         # (default: resources/results/evaluation_results/)

--models MODEL1 MODEL2    # Model names to generate tables for
                         # If not specified, processes all available models

--output PATH            # Output file path for LaTeX tables
                         # (default: auto-generated with timestamp)

--compare-only           # Only generate comparison tables (requires multiple models)
                         # Skips individual model tables
```

## Prerequisites

### Required Evaluation Files

The script expects three files per model from `evaluate_results.py`:

1. **CSV Results**: `evaluation_results_*.csv`
2. **JSON Summary**: `evaluation_summary_*.json`
3. **Classification Metrics**: `classification_metrics_*.json`

Directory structure:
```
resources/results/evaluation_results/
├── microsoft_phi-4/
│   ├── evaluation_results_k=3_exp_25-01-25--14-30-00_precise_v4_*.csv
│   ├── evaluation_summary_k=3_exp_25-01-25--14-30-00_precise_v4_*.json
│   └── classification_metrics_k=3_exp_25-01-25--14-30-00_precise_v4_*.json
└── qwen_qwen-2.5-72b-instruct/
    └── ...
```

## Table Types Generated

### 1. Evaluation Summary Table

Key performance metrics for a single model:

```latex
\begin{table}[H]
\centering
\caption{Evaluation Summary Table - microsoft\_phi-4}
\label{tab:evaluation_summary_microsoft_phi_4}
\begin{tabular}{@{}lp{2cm}@{}}
\toprule
\textbf{Field} & \textbf{Result} \\
\midrule
Predicted Outcome (Accu.) & \textbf{85.20\%} \\
Justification Outcome (SED) & \textbf{0.8234} \\
Justification Payment (SED) & \textbf{0.8910} \\
Justification Outcome (IoU) & \textbf{0.7560} \\
\bottomrule
\end{tabular}
\end{table}
```

### 2. Confusion Matrix Table

Detailed classification breakdown:

```latex
\begin{table}[H]
\centering
\caption{Confusion Matrix of Outcome Classifications - microsoft\_phi-4}
\label{tab:confusion_matrix_microsoft_phi_4}
\begin{tabular}{lccc|c}
\toprule
\multirow{2}{*}{\textbf{Actual Outcome}} & \multicolumn{3}{c}{\textbf{Predicted Outcome}} & \multirow{2}{*}{\textbf{Total}} \\
\cmidrule{2-4}
& \textbf{Yes} & \textbf{\begin{tabular}[c]{@{}c@{}}No - Unrelated\\event\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}No - condition(s)\\not met\end{tabular}} & \\
\midrule
\textbf{Yes} & 255 & 15 & 10 & 280 \\
\textbf{No \- Unrelated event} & 12 & 168 & 20 & 200 \\
\textbf{No \- condition(s) not met} & 8 & 7 & 45 & 60 \\
\midrule
\textbf{Total} & 275 & 190 & 75 & 540 \\
\bottomrule
\end{tabular}
\end{table}
```

### 3. Classification Metrics Table

Per-category performance metrics:

```latex
\begin{table}[H]
\centering
\caption{Performance Metrics by Outcome Category - microsoft\_phi-4}
\label{tab:classification_metrics_microsoft_phi_4}
\begin{tabular}{lccccc}
\toprule
\textbf{Outcome Category} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} & \textbf{Accuracy} \\
\midrule
Yes & 0.9273 & 0.9107 & 0.9189 & 280 & 0.8520 \\
No \- Unrelated event & 0.8842 & 0.8400 & 0.8615 & 200 & \\
No \- condition(s) not met & 0.6000 & 0.7500 & 0.6667 & 60 & \\
\midrule
\textbf{Weighted Average} & 0.8691 & 0.8520 & 0.8596 & 540 & \\
\bottomrule
\multicolumn{6}{p{14cm}}{\textit{Note:} Overall outcome classification accuracy: 0.8520 (85.20\%).} \\
\end{tabular}
\end{table}
```

### 4. Model Comparison Table

Side-by-side comparison of multiple models:

```latex
\begin{table}[H]
\centering
\caption{Model Comparison: microsoft\_phi-4 vs qwen\_qwen-2.5-72b-instruct}
\label{tab:model_comparison}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Metric} & \textbf{microsoft\_phi-4} & \textbf{qwen\_qwen-2.5-72b-instruct} \\
\midrule
Predicted Outcome (Accu.) & 85.20\% & 87.50\% \\
Justification Outcome (SED) & 0.8234 & 0.8512 \\
Justification Payment (SED) & 0.8910 & 0.9023 \\
Justification Outcome (IoU) & 0.7560 & 0.7890 \\
\bottomrule
\end{tabular}
\end{table}
```

### 5. Classification Comparison Table

Detailed classification metrics comparison:

```latex
\begin{table}[H]
\centering
\caption{Classification Metrics Comparison: microsoft\_phi-4 vs qwen\_qwen-2.5-72b-instruct}
\label{tab:classification_comparison}
\begin{tabular}{lccccccc}
\toprule
\multirow{2}{*}{\textbf{Outcome Category}} & \multicolumn{3}{c}{\textbf{microsoft\_phi-4}} & \multicolumn{3}{c}{\textbf{qwen\_qwen-2.5-72b-instruct}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
& \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\
\midrule
Yes & 0.927 & 0.911 & 0.919 & 0.941 & 0.925 & 0.933 \\
No \- Unrelated event & 0.884 & 0.840 & 0.862 & 0.895 & 0.870 & 0.882 \\
No \- condition(s) not met & 0.600 & 0.750 & 0.667 & 0.652 & 0.783 & 0.712 \\
\midrule
\textbf{Overall Accuracy} & \multicolumn{3}{c}{0.8520} & \multicolumn{3}{c}{0.8750} \\
\bottomrule
\end{tabular}
\end{table}
```

## Basic Usage

### Generate Tables for Single Model

```bash
# Generate all tables for one model
python src/scripts/latex_tables.py \
    --models microsoft_phi-4

# Output saved to auto-generated filename:
# latex_tables_microsoft_phi-4_25-01-2025_16-30-45.tex
```

### Generate Tables for Multiple Models

```bash
# Generate individual and comparison tables
python src/scripts/latex_tables.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct

# Output saved to:
# latex_tables_comparison_25-01-2025_16-30-45.tex
```

### Generate Comparison Only

```bash
# Skip individual tables, only generate comparisons
python src/scripts/latex_tables.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --compare-only
```

### Custom Output Path

```bash
# Specify output file location
python src/scripts/latex_tables.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --output results/tables/model_comparison.tex
```

## Advanced Usage

### Process All Available Models

```bash
# Generate tables for all models with evaluation results
python src/scripts/latex_tables.py

# The script will automatically find all models in:
# resources/results/evaluation_results/
```

### Custom Input Directory

```bash
# Use evaluation results from custom location
python src/scripts/latex_tables.py \
    --input-results /path/to/custom/evaluation/results \
    --models microsoft_phi-4
```

### Integration with Full Pipeline

```bash
#!/bin/bash
# Full evaluation pipeline

# 1. Generate model outputs
python src/main.py --model hf --model-name microsoft/phi-4 \
    --prompt precise_v4 --k 3 --batch

python src/main.py --model hf --model-name Qwen/Qwen2.5-72B \
    --prompt precise_v4_qwen --k 3 --batch

# 2. Evaluate results
python src/scripts/evaluate_results.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --k 3 --latest

# 3. Generate LaTeX tables
python src/scripts/latex_tables.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct
```

## Output Format

### File Structure

The generated `.tex` file contains:

```latex
% LaTeX Tables for Insurance Policy Analysis Evaluation
% Generated on 2025-01-25 16:30:45
% Models: microsoft_phi-4, qwen_qwen-2.5-72b-instruct

% === COMPARISON TABLES ===

[Model comparison table]
[Classification comparison table]

% === INDIVIDUAL TABLES FOR MICROSOFT_PHI-4 ===

[Evaluation summary table]
[Confusion matrix]
[Classification metrics table]

% === INDIVIDUAL TABLES FOR QWEN_QWEN-2.5-72B-INSTRUCT ===

[Evaluation summary table]
[Confusion matrix]
[Classification metrics table]
```

### LaTeX Requirements

Tables use these LaTeX packages:

- `booktabs` - For professional table rules
- `multirow` - For spanning cells
- `float` - For `[H]` placement specifier

Include in your LaTeX preamble:
```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{float}
```

### Including in Papers

```latex
% In your LaTeX document
\input{path/to/latex_tables_comparison_25-01-2025_16-30-45.tex}

% Or include specific tables
\input{tables/model_comparison.tex}
```

## Special Features

### LaTeX Character Escaping

The script automatically escapes special characters:

```python
# Characters automatically escaped:
_ → \_
% → \%
$ → \$
# → \#
& → \&
{ → \{
} → \}
^ → \textasciicircum{}
~ → \textasciitilde{}
```

### Metric Formatting

- **Percentages**: Displayed with % symbol (e.g., 85.20%)
- **Decimals**: Shown to 4 decimal places for similarity metrics
- **Integers**: Used for counts in confusion matrix

### Dynamic Column Sizing

Tables adjust based on number of models:

- 2 models: Balanced columns
- 3+ models: Compressed format
- Long model names: Automatic wrapping

## Error Handling

### Common Errors

1. **Missing Evaluation Files**
   ```
   Missing evaluation files in microsoft_phi-4. Missing: CSV files (evaluation_results_*.csv)
   ```
   **Solution**: Run `evaluate_results.py` first

2. **Insufficient Models for Comparison**
   ```
   At least 2 models required for comparison
   ```
   **Solution**: Specify multiple models or skip `--compare-only`

3. **Model Not Found**
   ```
   Warning: Model 'phi-3' not found in available models
   ```
   **Solution**: Check exact model name in evaluation results

### Troubleshooting

1. **Check Available Models**
   ```bash
   ls -la resources/results/evaluation_results/
   ```

2. **Verify Evaluation Files**
   ```bash
   # Check if all required files exist
   ls -la resources/results/evaluation_results/microsoft_phi-4/
   ```

3. **Test with Single Model**
   ```bash
   # Generate tables for one model first
   python src/scripts/latex_tables.py --models microsoft_phi-4
   ```

## Best Practices

### 1. Consistent Evaluation

Always use the same ground truth and evaluation parameters:

```bash
# Evaluate all models with same configuration
for model in microsoft_phi-4 qwen_qwen-2.5-72b-instruct; do
    python src/scripts/evaluate_results.py \
        --models $model --k 3 --prompt precise_v4 --latest
done

# Then generate comparison tables
python src/scripts/latex_tables.py
```

### 2. Table Selection

For papers, typically include:

- Model comparison table (if multiple models)
- Best model's confusion matrix
- Skip individual summaries if space-constrained

### 3. Caption Customization

Edit generated captions for clarity:
```latex
% Original
\caption{Model Comparison: microsoft\_phi-4 vs qwen\_qwen-2.5-72b-instruct}

% Customized
\caption{Performance comparison between Phi-4 and Qwen-2.5-72B models on insurance policy analysis}
```

### 4. Results Presentation

Organize tables logically in papers:

1. Overall comparison table
2. Detailed metrics for best model
3. Error analysis (confusion matrix)

## Integration Tips

### For Conference Papers

```bash
# Generate compact comparison only
python src/scripts/latex_tables.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct openai_gpt-4o \
    --compare-only \
    --output results/tables/conference_comparison.tex
```

### For Journal Articles

```bash
# Generate comprehensive analysis
python src/scripts/latex_tables.py \
    --models microsoft_phi-4 qwen_qwen-2.5-72b-instruct \
    --output results/tables/journal_full_analysis.tex
```

### For Thesis/Reports

```bash
# Process all available models
python src/scripts/latex_tables.py \
    --output results/tables/thesis_complete_evaluation.tex
```

The LaTeX tables script provides publication-ready output that seamlessly integrates with academic writing workflows, ensuring professional presentation of evaluation results.