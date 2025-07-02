# scripts/latex_tables.py

import os
import json
import pandas as pd
import argparse
import sys
from datetime import datetime
from typing import Dict, List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EVALUATION_RESULTS_FILES_PATH


def find_most_recent_evaluation_files(eval_dir: str) -> Dict[str, str]:
    """Find the most recent evaluation files based on timestamp in filename."""
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    all_files = os.listdir(eval_dir)

    # Get all files by type
    csv_files = [f for f in all_files if f.startswith("evaluation_results_") and f.endswith(".csv")]
    summary_files = [f for f in all_files if f.startswith("evaluation_summary_") and f.endswith(".json")]
    metrics_files = [f for f in all_files if f.startswith("classification_metrics_") and f.endswith(".json")]

    missing_types = []
    if not csv_files:
        missing_types.append("CSV files (evaluation_results_*.csv)")
    if not summary_files:
        missing_types.append("Summary files (evaluation_summary_*.json)")
    if not metrics_files:
        missing_types.append("Metrics files (classification_metrics_*.json)")

    if missing_types:
        raise FileNotFoundError(f"Missing evaluation files in {eval_dir}. Missing: {', '.join(missing_types)}")

    # Sort by timestamp and get most recent
    csv_files.sort(reverse=True)
    summary_files.sort(reverse=True)
    metrics_files.sort(reverse=True)

    return {
        'csv': os.path.join(eval_dir, csv_files[0]),
        'summary': os.path.join(eval_dir, summary_files[0]),
        'metrics': os.path.join(eval_dir, metrics_files[0])
    }


def load_model_data(results_dir: str, model_name: str) -> Dict:
    """Load evaluation data for a specific model."""
    model_dir = os.path.join(results_dir, model_name)

    try:
        recent_files = find_most_recent_evaluation_files(model_dir)

        # Read the files
        results_df = pd.read_csv(recent_files['csv'])

        with open(recent_files['summary'], 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        with open(recent_files['metrics'], 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

        return {
            'results_df': results_df,
            'summary_data': summary_data,
            'metrics_data': metrics_data,
            'files': recent_files
        }

    except Exception as e:
        raise Exception(f"Error loading data for model '{model_name}': {str(e)}")


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    if not isinstance(text, str):
        return str(text)

    # Common LaTeX escape sequences
    replacements = {
        '_': '\\_',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '&': '\\&',
        '{': '\\{',
        '}': '\\}',
        '^': '\\textasciicircum{}',
        '~': '\\textasciitilde{}',
    }

    for char, escape in replacements.items():
        text = text.replace(char, escape)

    return text


def generate_evaluation_summary_table(summary_data: Dict, model_name: Optional[str] = None) -> str:
    """Generate LaTeX table for evaluation summary with specific metrics."""
    predicted_outcome = summary_data.get('exact_outcome_match_percentage', 0)
    justification_outcome = summary_data.get('avg_justification_similarity', 0)
    justification_payment = summary_data.get('avg_payment_similarity', 0)
    justification_iou = summary_data.get('avg_justification_iou', 0)

    title_suffix = f" - {escape_latex(model_name)}" if model_name else ""
    label_suffix = f"_{model_name.lower().replace('-', '_').replace('.', '_')}" if model_name else ""

    latex = r'''\begin{table}[H]
\centering
\caption{Evaluation Summary Table''' + title_suffix + r'''}
\label{tab:evaluation_summary''' + label_suffix + r'''}
\begin{tabular}{@{}lp{2cm}@{}}
\toprule
\textbf{Field} & \textbf{Result} \\
\midrule
Predicted Outcome (Accu.) & \textbf{''' + f"{predicted_outcome:.2f}\\%" + r'''} \\
Justification Outcome (SED) &  \textbf{''' + f"{justification_outcome:.4f}" + r'''} \\
Justification Payment (SED) &  \textbf{''' + f"{justification_payment:.4f}" + r'''} \\
Justification Outcome (IoU) &  \textbf{''' + f"{justification_iou:.4f}" + r'''} \\
\bottomrule
\end{tabular}
\end{table}'''

    return latex


def generate_model_comparison_table(model_data_list: List[Dict], model_names: List[str]) -> str:
    """Generate LaTeX table comparing multiple models."""
    if len(model_data_list) < 2:
        raise ValueError("At least 2 models required for comparison")

    # Extract metrics for all models
    def extract_metrics(summary_data):
        return {
            'predicted_outcome': summary_data.get('exact_outcome_match_percentage', 0),
            'justification_outcome': summary_data.get('avg_justification_similarity', 0),
            'justification_payment': summary_data.get('avg_payment_similarity', 0),
            'justification_iou': summary_data.get('avg_justification_iou', 0)
        }

    all_metrics = [extract_metrics(data['summary_data']) for data in model_data_list]

    # Create table header with model names
    escaped_names = [escape_latex(name) for name in model_names]
    header_cols = 'l' + 'c' * len(model_names)

    latex = r'''\begin{table}[H]
\centering
\caption{Model Comparison: ''' + ' vs '.join(escaped_names) + r'''}
\label{tab:model_comparison}
\begin{tabular}{@{}''' + header_cols + r'''@{}}
\toprule
\textbf{Metric} & ''' + ' & '.join([f"\\textbf{{{name}}}" for name in escaped_names]) + r''' \\
\midrule'''

    # Add rows for each metric
    metrics_info = [
        ("Predicted Outcome (Accu.)", "predicted_outcome", ":.2f", "%"),
        ("Justification Outcome (SED)", "justification_outcome", ":.4f", ""),
        ("Justification Payment (SED)", "justification_payment", ":.4f", ""),
        ("Justification Outcome (IoU)", "justification_iou", ":.4f", "")
    ]

    for metric_name, key, fmt, suffix in metrics_info:
        values = [f"{metrics[key]:{fmt[1:]}}{suffix}" for metrics in all_metrics]
        latex += f"\n{metric_name} & " + " & ".join(values) + r" \\"

    latex += r'''
\bottomrule
\end{tabular}
\end{table}'''

    return latex


def generate_comparison_classification_metrics_table(model_data_list: List[Dict], model_names: List[str]) -> str:
    """Generate LaTeX table comparing classification metrics between multiple models."""
    if len(model_data_list) < 2:
        raise ValueError("At least 2 models required for comparison")

    valid_outcomes = [
        "Yes",
        "No - Unrelated event",
        "No - condition(s) not met"
    ]

    escaped_names = [escape_latex(name) for name in model_names]

    # Create dynamic column specification
    col_spec = "l" + "ccc" * len(model_names)

    latex = r'''\begin{table}[H]
\centering
\caption{Classification Metrics Comparison: ''' + ' vs '.join(escaped_names) + r'''}
\label{tab:classification_comparison}
\begin{tabular}{''' + col_spec + r'''}
\toprule
\multirow{2}{*}{\textbf{Outcome Category}}'''

    # Add headers for each model
    for i, name in enumerate(escaped_names):
        latex += f" & \\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}"

    latex += r''' \\
\cmidrule(lr){2-4}'''

    # Add cmidrules for additional models
    for i in range(1, len(model_names)):
        start_col = 2 + i * 3
        end_col = start_col + 2
        latex += f" \\cmidrule(lr){{{start_col}-{end_col}}}"

    latex += r'''
'''

    # Add sub-headers (Prec., Rec., F1 for each model)
    subheaders = []
    for _ in model_names:
        subheaders.extend([r"\textbf{Prec.}", r"\textbf{Rec.}", r"\textbf{F1}"])

    latex += " & " + " & ".join(subheaders) + r" \\"
    latex += r'''
\midrule'''

    # Add rows for each outcome category
    for outcome in valid_outcomes:
        outcome_escaped = outcome.replace('-', '\\-')
        row_data = [outcome_escaped]

        for model_data in model_data_list:
            report = model_data['metrics_data'].get('classification_report', {})
            precision = report.get(outcome, {}).get('precision', 0)
            recall = report.get(outcome, {}).get('recall', 0)
            f1 = report.get(outcome, {}).get('f1-score', 0)

            row_data.extend([f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])

        latex += f"\n{' & '.join(row_data)} \\\\"

    # Add overall accuracy comparison
    latex += r'''
\midrule
\textbf{Overall Accuracy}'''

    for model_data in model_data_list:
        accuracy = model_data['summary_data'].get('outcome_classification', {}).get('accuracy', 0)
        latex += f" & \\multicolumn{{3}}{{c}}{{{accuracy:.4f}}}"

    latex += r''' \\
\bottomrule
\end{tabular}
\end{table}'''

    return latex


def generate_confusion_matrix_table(metrics_data: Dict, model_name: Optional[str] = None) -> str:
    """Generate LaTeX table for confusion matrix."""
    conf_matrix = metrics_data.get('confusion_matrix', [])

    if not conf_matrix:
        return "% Error: Confusion matrix data not found"

    valid_outcomes = [
        "Yes",
        "No - Unrelated event",
        "No - condition(s) not met"
    ]

    # Calculate totals
    row_totals = [sum(row) for row in conf_matrix]
    col_totals = [sum(conf_matrix[row_idx][col_idx] for row_idx in range(len(conf_matrix)))
                  for col_idx in range(len(conf_matrix[0]))]
    total = sum(row_totals)

    title_suffix = f" - {escape_latex(model_name)}" if model_name else ""
    label_suffix = f"_{model_name.lower().replace('-', '_').replace('.', '_')}" if model_name else ""

    latex = r'''\begin{table}[H]
\centering
\caption{Confusion Matrix of Outcome Classifications''' + title_suffix + r'''}
\label{tab:confusion_matrix''' + label_suffix + r'''}
\begin{tabular}{lccc|c}
\toprule
\multirow{2}{*}{\textbf{Actual Outcome}} & \multicolumn{3}{c}{\textbf{Predicted Outcome}} & \multirow{2}{*}{\textbf{Total}} \\
\cmidrule{2-4}
& \textbf{Yes} & \textbf{\begin{tabular}[c]{@{}c@{}}No - Unrelated\\event\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}No - condition(s)\\not met\end{tabular}} & \\
\midrule'''

    for i, outcome in enumerate(valid_outcomes):
        if i < len(conf_matrix) and len(conf_matrix[i]) >= len(valid_outcomes):
            row = [str(conf_matrix[i][j]) for j in range(len(valid_outcomes))]
            row_str = " & ".join(row)
            latex += f"\n\\textbf{{{outcome.replace('-', '\\-')}}} & {row_str} & {row_totals[i]} \\\\"

    latex += r'''
\midrule
\textbf{Total} & ''' + " & ".join([str(val) for val in col_totals[:len(valid_outcomes)]]) + f" & {total}" + r''' \\
\bottomrule
\end{tabular}
\end{table}'''

    return latex


def generate_classification_metrics_table(metrics_data: Dict, summary_data: Dict,
                                          model_name: Optional[str] = None) -> str:
    """Generate LaTeX table for classification metrics."""
    report = metrics_data.get('classification_report', {})

    valid_outcomes = [
        "Yes",
        "No - Unrelated event",
        "No - condition(s) not met"
    ]

    accuracy = summary_data.get('outcome_classification', {}).get('accuracy', 0)

    title_suffix = f" - {escape_latex(model_name)}" if model_name else ""
    label_suffix = f"_{model_name.lower().replace('-', '_').replace('.', '_')}" if model_name else ""

    latex = r'''\begin{table}[H]
\centering
\caption{Performance Metrics by Outcome Category''' + title_suffix + r'''}
\label{tab:classification_metrics''' + label_suffix + r'''}
\begin{tabular}{lccccc}
\toprule
\textbf{Outcome Category} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} & \textbf{Accuracy} \\
\midrule'''

    # Add rows for each outcome category
    for i, outcome in enumerate(valid_outcomes):
        if outcome in report:
            precision = report[outcome].get('precision', 0)
            recall = report[outcome].get('recall', 0)
            f1 = report[outcome].get('f1-score', 0)
            support = report[outcome].get('support', 0)

            outcome_escaped = outcome.replace('-', '\\-')
            accuracy_cell = f"{accuracy:.4f}" if i == 0 else ""
            latex += f"\n{outcome_escaped} & {precision:.4f} & {recall:.4f} & {f1:.4f} & {int(support)} & {accuracy_cell} \\\\"

    # Add weighted average row
    if 'weighted avg' in report:
        w_precision = report['weighted avg'].get('precision', 0)
        w_recall = report['weighted avg'].get('recall', 0)
        w_f1 = report['weighted avg'].get('f1-score', 0)
        w_support = report['weighted avg'].get('support', 0)

        latex += r'''
\midrule
\textbf{Weighted Average} & ''' + f"{w_precision:.4f} & {w_recall:.4f} & {w_f1:.4f} & {int(w_support)} &" + r''' \\
\bottomrule
\multicolumn{6}{p{14cm}}{\textit{Note:} Overall outcome classification accuracy: ''' + f"{accuracy:.4f}" + r''' (''' + f"{summary_data.get('exact_outcome_match_percentage', 0):.2f}\\%" + r''').} \\
\end{tabular}
\end{table}'''

    return latex


def get_available_models(results_dir: str) -> List[str]:
    """Get list of available model directories."""
    if not os.path.exists(results_dir):
        return []

    models = []
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Check if directory contains evaluation files
            try:
                find_most_recent_evaluation_files(item_path)
                models.append(item)
            except FileNotFoundError:
                continue

    return sorted(models)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables for evaluation results')
    parser.add_argument('--input-results', default=EVALUATION_RESULTS_FILES_PATH,
                        help='Directory containing the evaluation results')
    parser.add_argument('--models', nargs='+',
                        help='Model names to generate tables for (e.g., phi-4 qwen-2-5-72b-instruct). '
                             'If not specified, all available models will be processed.')
    parser.add_argument('--output',
                        help='Output file path for LaTeX tables (default: auto-generated)')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only generate comparison tables (requires multiple models)')

    args = parser.parse_args()

    try:
        # Get available models
        available_models = get_available_models(args.input_results)

        if not available_models:
            print(f"No models with evaluation results found in {args.input_results}")
            return

        # Determine which models to process
        if args.models:
            models_to_process = []
            for model in args.models:
                if model in available_models:
                    models_to_process.append(model)
                else:
                    print(f"Warning: Model '{model}' not found in available models: {available_models}")

            if not models_to_process:
                print("No valid models specified")
                return
        else:
            models_to_process = available_models
            print(f"No specific models specified. Processing all available models: {models_to_process}")

        # Load data for all models
        model_data_list = []
        for model_name in models_to_process:
            print(f"Loading data for model: {model_name}")
            try:
                model_data = load_model_data(args.input_results, model_name)
                model_data_list.append(model_data)
            except Exception as e:
                print(f"Error loading data for model '{model_name}': {str(e)}")
                continue

        if not model_data_list:
            print("No valid model data loaded")
            return

        print(f"Successfully loaded data for {len(model_data_list)} models")

        # Generate LaTeX tables
        all_latex = f"""% LaTeX Tables for Insurance Policy Analysis Evaluation
% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
% Models: {', '.join(models_to_process)}

"""

        # Generate comparison tables if multiple models
        if len(model_data_list) > 1:
            print("Generating comparison tables...")

            comparison_table = generate_model_comparison_table(model_data_list, models_to_process)
            classification_comparison_table = generate_comparison_classification_metrics_table(
                model_data_list, models_to_process)

            all_latex += f"""% === COMPARISON TABLES ===

{comparison_table}

{classification_comparison_table}

"""

        # Generate individual tables unless comparison-only mode
        if not args.compare_only:
            for i, (model_data, model_name) in enumerate(zip(model_data_list, models_to_process)):
                print(f"Generating individual tables for model: {model_name}")

                evaluation_summary_table = generate_evaluation_summary_table(
                    model_data['summary_data'], model_name)
                confusion_table = generate_confusion_matrix_table(
                    model_data['metrics_data'], model_name)
                classification_table = generate_classification_metrics_table(
                    model_data['metrics_data'], model_data['summary_data'], model_name)

                all_latex += f"""% === INDIVIDUAL TABLES FOR {model_name.upper()} ===

{evaluation_summary_table}

{confusion_table}

{classification_table}

"""

        # Save to file
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            if len(models_to_process) == 1:
                output_filename = f"latex_tables_{models_to_process[0]}_{timestamp}.tex"
            else:
                output_filename = f"latex_tables_comparison_{timestamp}.tex"
            output_file = os.path.join(args.input_results, output_filename)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(all_latex)

        print(f"\nLaTeX tables generated and saved to: {output_file}")

        # Also print summary to console
        print("\n" + "=" * 80)
        print("LATEX TABLES GENERATED:")
        print("=" * 80)
        if len(model_data_list) > 1:
            print("✓ Model comparison tables")
            print("✓ Classification comparison tables")
        if not args.compare_only:
            for model_name in models_to_process:
                print(f"✓ Individual tables for {model_name}")

    except Exception as e:
        print(f"Error generating LaTeX tables: {str(e)}")


if __name__ == "__main__":
    main()
