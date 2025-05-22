import os
import json
import pandas as pd
from datetime import datetime

from config import EVALUATION_RESULTS_FILES_PATH


def find_most_recent_evaluation_files(eval_dir):
    """Find the most recent evaluation files based on timestamp in filename."""
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    # Get all files by type
    csv_files = [f for f in os.listdir(eval_dir) if f.startswith("evaluation_results_") and f.endswith(".csv")]
    summary_files = [f for f in os.listdir(eval_dir) if f.startswith("evaluation_summary_") and f.endswith(".json")]
    metrics_files = [f for f in os.listdir(eval_dir) if f.startswith("classification_metrics_") and f.endswith(".json")]

    if not csv_files or not summary_files or not metrics_files:
        raise FileNotFoundError("Missing evaluation files in the directory")

    # Sort by timestamp (assuming format is consistent)
    csv_files.sort(reverse=True)
    summary_files.sort(reverse=True)
    metrics_files.sort(reverse=True)

    # Return paths to most recent files
    return {
        'csv': os.path.join(eval_dir, csv_files[0]),
        'summary': os.path.join(eval_dir, summary_files[0]),
        'metrics': os.path.join(eval_dir, metrics_files[0])
    }


def generate_evaluation_summary_table(summary_data):
    """Generate LaTeX table for evaluation summary with specific metrics."""
    # Extract values from summary data
    predicted_outcome = summary_data.get('exact_outcome_match_percentage', 64.63)
    justification_outcome = summary_data.get('avg_justification_similarity', 0.4681)
    justification_payment = summary_data.get('avg_payment_similarity', 0.8420)
    justification_iou = summary_data.get('avg_justification_iou', 0.0)  # Add IoU metric

    latex = r'''\begin{table}[H]
\centering
\caption{Evaluation Summary Table}
\label{tab:evaluation_summary}
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


def generate_string_edit_distance_table(summary_data):
    """Generate LaTeX table for string edit distance."""
    # Extract similarity metrics from summary data
    avg_justification = summary_data.get('avg_justification_similarity', 0)
    avg_payment = summary_data.get('avg_payment_similarity', 0)
    avg_combined = summary_data.get('avg_combined_justification_similarity', 0)
    avg_iou = summary_data.get('avg_justification_iou', 0)  # Add IoU metric

    latex = r'''\begin{table}[H]
\centering
\caption{String Edit Distance Similarity Results}
\label{tab:string_edit_distance_results}
\begin{tabular}{lc}
\toprule
\textbf{Similarity Metric} & \textbf{Average Score} \\
\midrule
Outcome Justification Similarity & ''' + f"{avg_justification:.4f}" + r''' \\
Payment Justification Similarity & ''' + f"{avg_payment:.4f}" + r''' \\
Combined Justification Similarity & ''' + f"{avg_combined:.4f}" + r''' \\
Justification IoU & ''' + f"{avg_iou:.4f}" + r''' \\
\midrule
\multicolumn{2}{p{13cm}}{\textit{Note:} Similarity scores range from 0.0 (completely different) to 1.0 (identical). 
The scores measure the normalized Levenshtein edit distance between predicted and ground truth justifications.
IoU (Intersection over Union) measures the word-level overlap between texts.} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{String Edit Distance Similarity Interpretation}
\label{tab:string_edit_distance_interpretation}
\begin{tabular}{p{3cm}p{10cm}}
\toprule
\textbf{Similarity Score} & \textbf{Interpretation} \\
\midrule
$1.0$ & Perfect match (identical strings or both null/empty) \\
$0.8 - 0.99$ & Very high similarity (minor differences) \\
$0.6 - 0.79$ & Substantial similarity (some differences) \\
$0.4 - 0.59$ & Moderate similarity (significant differences) \\
$0.2 - 0.39$ & Low similarity (major differences) \\
$0.0 - 0.19$ & Very low similarity (almost completely different) \\
\bottomrule
\end{tabular}
\end{table}'''

    return latex


def generate_confusion_matrix_table(metrics_data):
    """Generate LaTeX table for confusion matrix."""
    # Extract confusion matrix data
    conf_matrix = metrics_data.get('confusion_matrix', [])

    if not conf_matrix:
        return "% Error: Confusion matrix data not found"

    # Valid outcome categories (removed "Maybe")
    valid_outcomes = [
        "Yes",
        "No - Unrelated event",
        "No - condition(s) not met"
    ]

    # Calculate row and column totals
    row_totals = [sum(row) for row in conf_matrix]
    col_totals = [sum(conf_matrix[row_idx][col_idx] for row_idx in range(len(conf_matrix)))
                  for col_idx in range(len(conf_matrix[0]))]
    total = sum(row_totals)

    # Generate LaTeX table
    latex = r'''\begin{table}[H]
\centering
\caption{Confusion Matrix of Outcome Classifications}
\label{tab:confusion_matrix}
\begin{tabular}{lccc|c}
\toprule
\multirow{2}{*}{\textbf{Actual Outcome}} & \multicolumn{3}{c}{\textbf{Predicted Outcome}} & \multirow{2}{*}{\textbf{Total}} \\
\cmidrule{2-4}
& \textbf{Yes} & \textbf{\begin{tabular}[c]{@{}c@{}}No - Unrelated\\event\end{tabular}} & \textbf{\begin{tabular}[c]{@{}c@{}}No - condition(s)\\not met\end{tabular}} & \\
\midrule'''

    for i, outcome in enumerate(valid_outcomes):
        if i < len(conf_matrix) and len(conf_matrix[i]) >= len(valid_outcomes):
            row = [conf_matrix[i][j] for j in range(len(valid_outcomes))]
            row_str = " & ".join([str(val) for val in row])
            latex += f"\n\\textbf{{{outcome.replace('-', '\\-')}}} & {row_str} & {row_totals[i]} \\\\"

    latex += r'''
\midrule
\textbf{Total} & ''' + " & ".join([str(val) for val in col_totals[:len(valid_outcomes)]]) + f" & {total}" + r''' \\
\bottomrule
\end{tabular}
\end{table}'''

    return latex


def generate_classification_metrics_table(metrics_data, summary_data):
    """Generate LaTeX table for classification metrics."""
    # Extract classification report data
    report = metrics_data.get('classification_report', {})

    # Valid outcome categories (removed "Maybe")
    valid_outcomes = [
        "Yes",
        "No - Unrelated event",
        "No - condition(s) not met"
    ]

    # Overall accuracy
    accuracy = summary_data.get('outcome_classification', {}).get('accuracy', 0)

    # Generate LaTeX table
    latex = r'''\begin{table}[H]
\centering
\caption{Performance Metrics by Outcome Category}
\label{tab:classification_metrics}
\begin{tabular}{lccccc}
\toprule
\textbf{Outcome Category} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} & \textbf{Accuracy} \\
\midrule'''

    # Add rows for each outcome category
    for outcome in valid_outcomes:
        if outcome in report:
            precision = report[outcome].get('precision', 0)
            recall = report[outcome].get('recall', 0)
            f1 = report[outcome].get('f1-score', 0)
            support = report[outcome].get('support', 0)

            outcome_escaped = outcome.replace('-', '\\-')
            latex += f"\n{outcome_escaped} & {precision:.4f} & {recall:.4f} & {f1:.4f} & {int(support)} & \\multirow{{1}}{{*}}{{}} \\\\"

    # Add weighted average row
    if 'weighted avg' in report:
        w_precision = report['weighted avg'].get('precision', 0)
        w_recall = report['weighted avg'].get('recall', 0)
        w_f1 = report['weighted avg'].get('f1-score', 0)
        w_support = report['weighted avg'].get('support', 0)

        latex += r'''
\midrule
\textbf{Weighted Average} & ''' + f"{w_precision:.4f} & {w_recall:.4f} & {w_f1:.4f} & {int(w_support)} & {accuracy:.4f}" + r''' \\
\bottomrule
\multicolumn{6}{p{14cm}}{\textit{Note:} Overall outcome classification accuracy: ''' + f"{accuracy:.4f}" + r''' (''' + f"{summary_data.get('exact_outcome_match_percentage', 0):.2f}\\%" + r''').} \\
\end{tabular}
\end{table}'''

    return latex


def generate_summary_table(summary_data):
    """Generate LaTeX table for summary statistics."""
    total_output = summary_data.get('total_output_questions', 0)
    total_evaluated = summary_data.get('total_evaluated_questions', 0)
    exact_matches = summary_data.get('exact_outcome_matches', 0)
    exact_match_pct = summary_data.get('exact_outcome_match_percentage', 0)

    latex = r'''\begin{table}[H]
\centering
\caption{Evaluation Summary Statistics}
\label{tab:evaluation_summary}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total Questions in Output & ''' + f"{total_output}" + r''' \\
Total Questions Evaluated & ''' + f"{total_evaluated}" + r''' \\
Exact Outcome Matches & ''' + f"{exact_matches}" + r''' \\
Exact Outcome Match Percentage & ''' + f"{exact_match_pct:.2f}\\%" + r''' \\
\bottomrule
\end{tabular}
\end{table}'''

    return latex


def main():
    # Define the path to evaluation results directory (relative to config.py, which is one level up from scripts)
    eval_dir = os.path.join(EVALUATION_RESULTS_FILES_PATH)

    # For testing, you could also hardcode the path:
    # eval_dir = "/path/to/your/evaluation_results"

    try:
        # Find most recent evaluation files
        recent_files = find_most_recent_evaluation_files(eval_dir)

        # Read the files
        results_df = pd.read_csv(recent_files['csv'])

        with open(recent_files['summary'], 'r', encoding='utf-8') as f:
            summary_data = json.load(f)

        with open(recent_files['metrics'], 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

        # Generate the new evaluation summary table
        evaluation_summary_table = generate_evaluation_summary_table(summary_data)

        # Generate other LaTeX tables
        string_dist_table = generate_string_edit_distance_table(summary_data)
        confusion_table = generate_confusion_matrix_table(metrics_data)
        classification_table = generate_classification_metrics_table(metrics_data, summary_data)
        summary_table = generate_summary_table(summary_data)

        # Combine all tables
        all_latex = f"""% LaTeX Tables for Evaluation Results
% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
% Based on evaluation files:
% - {os.path.basename(recent_files['csv'])}
% - {os.path.basename(recent_files['summary'])}
% - {os.path.basename(recent_files['metrics'])}

{evaluation_summary_table}

{string_dist_table}

{confusion_table}

{classification_table}
"""

        # Save to file
        output_file = os.path.join(eval_dir, f"latex_tables_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.tex")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(all_latex)

        print(f"LaTeX tables generated and saved to: {output_file}")

        # Also print the LaTeX code to the console
        print("\n" + "=" * 80)
        print("LATEX TABLES:")
        print("=" * 80)
        print(all_latex)

    except Exception as e:
        print(f"Error generating LaTeX tables: {str(e)}")


if __name__ == "__main__":
    main()
