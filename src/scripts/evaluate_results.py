# scripts/evaluate_results.py

import os
import json
import re

import editdistance
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import JSON_PATH, GT_PATH, EVALUATION_RESULTS_FILES_PATH


def normalize_field(value):
    """Normalize a field value to handle None, empty strings, and whitespace consistently."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ["null", "none"]:
            return None
    return value


def calculate_similarity_score(str1, str2):
    """Calculate similarity score between two strings (1 = identical, 0 = completely different)."""
    str1 = normalize_field(str1)
    str2 = normalize_field(str2)

    if str1 is None and str2 is None:
        return 1.0
    if str1 is None:
        str1 = ""
    if str2 is None:
        str2 = ""

    str1 = str(str1)
    str2 = str(str2)

    # Calculate edit distance
    distance = editdistance.eval(str1, str2)

    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0

    return 1.0 - (distance / max_len)


def calculate_text_iou(text1, text2):
    """Calculate Intersection over Union (IoU) for text strings based on word sets."""
    text1 = normalize_field(text1)
    text2 = normalize_field(text2)

    if text1 is None:
        text1 = ""
    if text2 is None:
        text2 = ""

    if text1 == "" and text2 == "":
        return 1.0

    if (text1 == "" and text2 != "") or (text1 != "" and text2 == ""):
        return 0.0

    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)

    tokens1 = tokenize(str(text1))
    tokens2 = tokenize(str(text2))

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if len(union) == 0:
        return 1.0

    return len(intersection) / len(union)


def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj


def get_model_directories(json_path: str) -> List[str]:
    """Get all model directories from the JSON output path."""
    if not os.path.exists(json_path):
        return []

    model_dirs = []
    for item in os.listdir(json_path):
        item_path = os.path.join(json_path, item)
        if os.path.isdir(item_path):
            # Check if directory contains policy output files
            files = os.listdir(item_path)
            if any(f.startswith("policy_id-") and f.endswith(".json") for f in files):
                model_dirs.append(item)

    return sorted(model_dirs)


def get_latest_output_files(model_dir_path: str) -> List[str]:
    """Find the most recent output file for each policy ID in a model directory."""
    if not os.path.exists(model_dir_path):
        return []

    all_files = [f for f in os.listdir(model_dir_path)
                 if f.startswith("policy_id-") and f.endswith(".json")]

    # Group files by policy ID
    policy_files = {}
    for file in all_files:
        # Extract policy ID from filename: policy_id-{id}__{model}__{timestamp}.json
        parts = file.split("__")
        if len(parts) >= 2:
            policy_id_part = parts[0]
            policy_id = policy_id_part.replace("policy_id-", "")

            if policy_id not in policy_files:
                policy_files[policy_id] = []
            policy_files[policy_id].append(file)

    # For each policy ID, find the most recent file
    latest_files = []
    for policy_id, files in policy_files.items():
        # Sort files by timestamp (after the double underscore)
        sorted_files = sorted(files, key=lambda x: x.split("__")[-1].split(".")[0], reverse=True)
        if sorted_files:
            latest_files.append(sorted_files[0])

    return latest_files


def evaluate_model_outputs(model_name: str, model_dir_path: str, gt_path: str) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate outputs for a single model against ground truth files."""
    print(f"\n=== Evaluating Model: {model_name} ===")

    # Lists to store evaluation results
    all_results = []
    y_true = []
    y_pred = []

    # Valid outcome categories
    valid_outcomes = [
        "Yes",
        "No - Unrelated event",
        "No - condition(s) not met"
    ]

    # Get the latest output file for each policy ID
    output_files = get_latest_output_files(model_dir_path)

    # Get all ground truth files
    gt_files = [f for f in os.listdir(gt_path) if f.startswith("GT_policy_") and f.endswith(".json")]

    if not output_files:
        print(f"Error: No output files found in {model_dir_path}")
        return None, None

    if not gt_files:
        print(f"Error: No ground truth files found in {gt_path}")
        return None, None

    # Map policy IDs to GT files for easier lookup
    gt_file_map = {f.split("_")[2].split(".")[0]: f for f in gt_files}

    print(f"Found {len(output_files)} output files and {len(gt_files)} ground truth files")

    total_output_questions = 0
    total_evaluated_questions = 0

    for output_file in output_files:
        # Extract policy ID from filename
        policy_id = output_file.split("__")[0].replace("policy_id-", "")

        # Find corresponding ground truth file
        if policy_id not in gt_file_map:
            print(f"Warning: No ground truth file found for policy {policy_id}")
            continue

        gt_file_name = gt_file_map[policy_id]
        gt_file_path = os.path.join(gt_path, gt_file_name)

        # Load files
        try:
            with open(os.path.join(model_dir_path, output_file), 'r', encoding='utf-8') as f:
                output_data = json.load(f)

            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error loading files for policy {policy_id}: {e}")
            continue

        # Create lookup dictionaries for faster access
        output_questions = output_data.get("questions", [])
        gt_questions_dict = {q["request_id"]: q for q in gt_data.get("questions", [])}

        questions_in_file = len(output_questions)
        total_output_questions += questions_in_file

        print(f"Policy {policy_id}: Found {questions_in_file} questions in output")

        # Compare each output question with ground truth
        for output_question in output_questions:
            request_id = output_question.get("request_id")

            if request_id not in gt_questions_dict:
                print(f"Warning: Policy {policy_id}, Request {request_id} not found in ground truth")
                continue

            gt_question = gt_questions_dict[request_id]
            total_evaluated_questions += 1

            # Get the outcome values and normalize them
            output_outcome = normalize_field(output_question.get("outcome", ""))
            gt_outcome = normalize_field(gt_question.get("outcome", ""))

            # Convert None to empty string for outcome comparison
            output_outcome = output_outcome if output_outcome is not None else ""
            gt_outcome = gt_outcome if gt_outcome is not None else ""

            # Add to lists for classification metrics
            y_true.append(gt_outcome)
            y_pred.append(output_outcome)

            # Calculate exact match for outcome
            outcome_match = 0 if output_outcome == gt_outcome else 1

            # Get and normalize justification fields
            output_justification = normalize_field(output_question.get("outcome_justification"))
            gt_justification = normalize_field(gt_question.get("outcome_justification"))

            # Calculate similarity scores
            justification_similarity = calculate_similarity_score(output_justification, gt_justification)
            justification_iou = calculate_text_iou(output_justification, gt_justification)

            # Get payment justification fields
            output_payment = normalize_field(output_question.get("payment_justification"))
            gt_payment = normalize_field(gt_question.get("payment_justification"))
            payment_similarity = calculate_similarity_score(output_payment, gt_payment)

            # Add to results
            all_results.append({
                "model_name": model_name,
                "policy_id": policy_id,
                "request_id": request_id,
                "question": gt_question.get("question", ""),
                "output_outcome": output_outcome,
                "gt_outcome": gt_outcome,
                "outcome_match": outcome_match,
                "output_justification": output_justification if output_justification is not None else "",
                "gt_justification": gt_justification if gt_justification is not None else "",
                "output_payment": output_payment if output_payment is not None else "",
                "gt_payment": gt_payment if gt_payment is not None else "",
                "justification_similarity": justification_similarity,
                "justification_iou": justification_iou,
                "payment_similarity": payment_similarity,
                "avg_justification_similarity": (justification_similarity + payment_similarity) / 2
                if justification_similarity is not None and payment_similarity is not None else None
            })

    if not all_results:
        print(f"No matching questions found for evaluation of model {model_name}")
        return None, None

    # Create DataFrame for results
    results_df = pd.DataFrame(all_results)

    # Calculate classification metrics
    outcome_accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=valid_outcomes)
    class_report = classification_report(y_true, y_pred, labels=valid_outcomes, output_dict=True, zero_division=0)

    # Calculate per-category accuracy
    category_metrics = {}
    for category in valid_outcomes:
        if category in class_report:
            category_metrics[category] = {
                "precision": class_report[category]["precision"],
                "recall": class_report[category]["recall"],
                "f1-score": class_report[category]["f1-score"],
                "support": class_report[category]["support"]
            }

    # Calculate summary statistics
    summary = {
        "model_name": model_name,
        "total_output_questions": total_output_questions,
        "total_evaluated_questions": total_evaluated_questions,
        "outcome_classification": {
            "accuracy": float(outcome_accuracy),
            "category_metrics": category_metrics
        },
        "exact_outcome_matches": int((results_df["outcome_match"] == 0).sum()),
        "exact_outcome_match_percentage": float((results_df["outcome_match"] == 0).mean() * 100),
        "avg_justification_similarity": float(results_df["justification_similarity"].mean()),
        "avg_justification_iou": float(results_df["justification_iou"].mean()),
        "avg_payment_similarity": float(results_df["payment_similarity"].mean()),
        "avg_combined_justification_similarity": float(
            results_df["avg_justification_similarity"].dropna().mean()
            if not results_df["avg_justification_similarity"].dropna().empty else 0)
    }

    # Convert numpy types to native Python types for JSON serialization
    summary = numpy_to_python(summary)

    return results_df, summary


def save_evaluation_results(results_df: pd.DataFrame, summary: Dict,
                            model_name: str, output_dir: str) -> Dict[str, str]:
    """Save evaluation results to files and return file paths."""
    # Create model-specific directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # Save detailed results to CSV
    csv_filename = os.path.join(model_output_dir, f"evaluation_results_{timestamp}.csv")
    results_df.to_csv(csv_filename, index=False, na_rep='')

    # Save summary statistics
    summary_filename = os.path.join(model_output_dir, f"evaluation_summary_{timestamp}.json")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Save classification metrics
    metrics_filename = os.path.join(model_output_dir, f"classification_metrics_{timestamp}.json")

    # Extract classification metrics from summary
    conf_matrix = confusion_matrix(
        results_df["gt_outcome"].tolist(),
        results_df["output_outcome"].tolist(),
        labels=["Yes", "No - Unrelated event", "No - condition(s) not met"]
    )

    class_report = classification_report(
        results_df["gt_outcome"].tolist(),
        results_df["output_outcome"].tolist(),
        labels=["Yes", "No - Unrelated event", "No - condition(s) not met"],
        output_dict=True,
        zero_division=0
    )

    metrics_data = {
        "confusion_matrix": numpy_to_python(conf_matrix),
        "classification_report": class_report
    }

    with open(metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)

    return {
        'csv': csv_filename,
        'summary': summary_filename,
        'metrics': metrics_filename
    }


def print_evaluation_summary(summary: Dict):
    """Print evaluation summary for a model."""
    model_name = summary.get('model_name', 'Unknown')

    print(f"\n=== EVALUATION RESULTS FOR {model_name.upper()} ===")
    print(f"Total Questions in Output: {summary['total_output_questions']}")
    print(f"Total Questions Evaluated: {summary['total_evaluated_questions']}")
    print(f"Outcome Classification Accuracy: {summary['outcome_classification']['accuracy']:.4f} "
          f"({summary['exact_outcome_match_percentage']:.2f}%)")
    # print(f"Average Justification Similarity: {summary['avg_justification_similarity']:.4f}")
    print(f"Average Justification IoU: {summary['avg_justification_iou']:.4f}")
    # print(f"Average Payment Similarity: {summary['avg_payment_similarity']:.4f}")
    # print(f"Average Combined Justification Similarity: {summary['avg_combined_justification_similarity']:.4f}")


def compare_models(summaries: List[Dict]):
    """Print comparison between multiple models."""
    if len(summaries) < 2:
        return

    print(f"\n=== MODEL COMPARISON ===")
    print(f"{'Metric':<35} | " + " | ".join([f"{s['model_name']:<15}" for s in summaries]))
    print("-" * (35 + len(summaries) * 18))

    metrics = [
        ("Outcome Accuracy (%)", "exact_outcome_match_percentage", ".2f"),
        # ("Justification Similarity", "avg_justification_similarity", ".4f"),
        ("Justification IoU", "avg_justification_iou", ".4f"),
        # ("Payment Similarity", "avg_payment_similarity", ".4f"),
        # ("Combined Similarity", "avg_combined_justification_similarity", ".4f")
    ]

    for metric_name, key, fmt in metrics:
        values = [f"{s[key]:{fmt}}" for s in summaries]
        print(f"{metric_name:<35} | " + " | ".join([f"{v:<15}" for v in values]))


def main():
    parser = argparse.ArgumentParser(description='Evaluate insurance policy analysis results')
    parser.add_argument('--models', nargs='+',
                        help='Model names to evaluate (e.g., phi-4 qwen-2-5-72b-instruct). '
                             'If not specified, all available models will be evaluated.')
    parser.add_argument('--json-path', default=JSON_PATH,
                        help='Path to JSON output directory')
    parser.add_argument('--gt-path', default=GT_PATH,
                        help='Path to ground truth directory')
    parser.add_argument('--output-dir', default=EVALUATION_RESULTS_FILES_PATH,
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Install required libraries if missing
    try:
        import editdistance
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    except ImportError as e:
        print(f"Installing missing library: {e.name}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", e.name])
        if e.name == "editdistance":
            import editdistance
        elif e.name == "scikit-learn":
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Make global for use in other functions
    global accuracy_score, confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Get available model directories
    available_models = get_model_directories(args.json_path)

    if not available_models:
        print(f"No model directories found in {args.json_path}")
        return

    # Determine which models to evaluate
    if args.models:
        models_to_evaluate = []
        for model in args.models:
            if model in available_models:
                models_to_evaluate.append(model)
            else:
                print(f"Warning: Model '{model}' not found in available models: {available_models}")

        if not models_to_evaluate:
            print("No valid models specified for evaluation")
            return
    else:
        models_to_evaluate = available_models
        print(f"No specific models specified. Evaluating all available models: {models_to_evaluate}")

    # Evaluate each model
    all_summaries = []
    all_results_dfs = []

    for model_name in models_to_evaluate:
        model_dir_path = os.path.join(args.json_path, model_name)

        print(f"\nEvaluating model: {model_name}")
        results_df, summary = evaluate_model_outputs(model_name, model_dir_path, args.gt_path)

        if results_df is not None and summary is not None:
            # Save results
            saved_files = save_evaluation_results(results_df, summary, model_name, args.output_dir)

            # Print summary
            print_evaluation_summary(summary)

            print(f"\nResults saved:")
            print(f"  CSV: {saved_files['csv']}")
            print(f"  Summary: {saved_files['summary']}")
            print(f"  Metrics: {saved_files['metrics']}")

            all_summaries.append(summary)
            all_results_dfs.append(results_df)
        else:
            print(f"Failed to evaluate model: {model_name}")

    # Print comparison if multiple models were evaluated
    if len(all_summaries) > 1:
        compare_models(all_summaries)

    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Evaluated {len(all_summaries)} models successfully")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
