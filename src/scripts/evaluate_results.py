import os
import json
import re

import pandas as pd
import numpy as np
from config import JSON_PATH, GT_PATH
from datetime import datetime


def normalize_field(value):
    """Normalize a field value to handle None, empty strings, and whitespace consistently."""
    if value is None:
        return None
    if isinstance(value, str):
        # Strip whitespace
        value = value.strip()
        # Convert empty strings to None
        if value == "" or value.lower() in ["null", "none"]:
            return None
    return value


def calculate_similarity_score(str1, str2):
    """Calculate similarity score between two strings (1 = identical, 0 = completely different)."""
    # Normalize inputs
    str1 = normalize_field(str1)
    str2 = normalize_field(str2)

    if str1 is None and str2 is None:
        return 1.0  # Both are None or empty, so perfect match
    if str1 is None:
        str1 = ""
    if str2 is None:
        str2 = ""

    # Convert to strings if they aren't already
    str1 = str(str1)
    str2 = str(str2)

    # Calculate edit distance
    distance = editdistance.eval(str1, str2)

    # Normalize by maximum length to get a value between 0 and 1
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0  # Both empty, so perfect match

    # Convert to similarity score (1 - normalized distance)
    return 1.0 - (distance / max_len)


def calculate_text_iou(text1, text2):
    """Calculate Intersection over Union (IoU) for text strings based on word sets."""
    # Normalize inputs
    text1 = normalize_field(text1)
    text2 = normalize_field(text2)

    # Handle None and empty cases
    if text1 is None:
        text1 = ""
    if text2 is None:
        text2 = ""

    # If both are empty, return 1.0 (perfect match)
    if text1 == "" and text2 == "":
        return 1.0

    # If one is empty and the other is not, return 0.0
    if (text1 == "" and text2 != "") or (text1 != "" and text2 == ""):
        return 0.0

    # Simple tokenization: split by whitespace and remove punctuation, convert to lowercase
    def tokenize(text):
        # Remove punctuation and convert to lowercase, then split
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)

    tokens1 = tokenize(str(text1))
    tokens2 = tokenize(str(text2))

    # Calculate intersection and union
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    # Calculate IoU
    if len(union) == 0:
        return 1.0  # Both have no tokens after processing

    iou = len(intersection) / len(union)
    return iou


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


def get_latest_output_files(json_path):
    """Find the most recent output file for each policy ID."""
    # Get all output files
    all_files = [f for f in os.listdir(json_path) if f.startswith("policy_id-") and f.endswith(".json")]

    # Group files by policy ID
    policy_files = {}
    for file in all_files:
        # Extract policy ID from the new naming pattern
        parts = file.split("__")
        if len(parts) >= 1:
            policy_id_part = parts[0]
            policy_id = policy_id_part.replace("policy_id-", "")

            # Add to dictionary, keeping track of all files for each policy ID
            if policy_id not in policy_files:
                policy_files[policy_id] = []
            policy_files[policy_id].append(file)

    # For each policy ID, find the most recent file
    latest_files = []
    for policy_id, files in policy_files.items():
        # Sort files by timestamp (which is after the double underscore)
        sorted_files = sorted(files, key=lambda x: x.split("__")[1].split(".")[0], reverse=True)
        if sorted_files:
            latest_files.append(sorted_files[0])

    print(f"Found {len(latest_files)} latest output files")
    return latest_files


def evaluate_outputs():
    """Evaluate system outputs against ground truth files."""
    # Lists to store evaluation results
    all_results = []

    # Lists to store outcome classifications for confusion matrix
    y_true = []
    y_pred = []

    # Valid outcome categories (removed "Maybe")
    valid_outcomes = [
        "Yes",
        "No - Unrelated event",
        "No - condition(s) not met"
    ]

    # Get the latest output file for each policy ID
    output_files = get_latest_output_files(JSON_PATH)

    # Get all ground truth files
    gt_files = [f for f in os.listdir(GT_PATH) if f.startswith("GT_policy_") and f.endswith(".json")]

    if not output_files:
        print("Error: No output files found in", JSON_PATH)
        return None, None

    if not gt_files:
        print("Error: No ground truth files found in", GT_PATH)
        return None, None

    # Map policy IDs to GT files for easier lookup
    gt_file_map = {f.split("_")[2].split(".")[0]: f for f in gt_files}

    print(f"Found {len(output_files)} output files and {len(gt_files)} ground truth files")

    total_output_questions = 0
    total_evaluated_questions = 0

    for output_file in output_files:
        # Extract policy ID from the new filename pattern
        policy_id = output_file.split("__")[0].replace("policy_id-", "")

        # Find corresponding ground truth file
        if policy_id not in gt_file_map:
            print(f"Warning: No ground truth file found for policy {policy_id}")
            continue

        gt_file_name = gt_file_map[policy_id]
        gt_file_path = os.path.join(GT_PATH, gt_file_name)

        # Load files
        try:
            with open(os.path.join(JSON_PATH, output_file), 'r', encoding='utf-8') as f:
                output_data = json.load(f)

            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            continue
        except UnicodeDecodeError as e:
            print(f"Error decoding file: {e}")
            continue

        # Create lookup dictionaries for faster access
        output_questions = output_data.get("questions", [])
        gt_questions_dict = {q["request_id"]: q for q in gt_data.get("questions", [])}

        # Count total questions in this output file
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

            # Check if outcomes are in the valid list - standardize if needed
            if output_outcome not in valid_outcomes:
                # Standardize the output outcome if possible or log warning
                print(
                    f"Warning: Policy {policy_id}, Request {request_id} has non-standard output outcome: {output_outcome}")

            if gt_outcome not in valid_outcomes:
                # Standardize the ground truth outcome if possible or log warning
                print(
                    f"Warning: Policy {policy_id}, Request {request_id} has non-standard ground truth outcome: {gt_outcome}")

            # Add to lists for classification metrics
            y_true.append(gt_outcome)
            y_pred.append(output_outcome)

            # Calculate exact match for outcome (0 = match, 1 = no match)
            outcome_match = 0 if output_outcome == gt_outcome else 1

            # Get and normalize justification fields
            output_justification = normalize_field(output_question.get("outcome_justification"))
            gt_justification = normalize_field(gt_question.get("outcome_justification"))

            # Calculate similarity scores for justifications
            justification_similarity = calculate_similarity_score(
                output_justification,
                gt_justification
            )

            # Calculate IoU for outcome_justification
            justification_iou = calculate_text_iou(
                output_justification,
                gt_justification
            )

            # Get and normalize payment justification fields
            output_payment = normalize_field(output_question.get("payment_justification"))
            gt_payment = normalize_field(gt_question.get("payment_justification"))

            payment_similarity = calculate_similarity_score(
                output_payment,
                gt_payment
            )

            # Add to results with actual justification and payment texts
            # Convert None to empty string for CSV output
            all_results.append({
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
                "avg_justification_similarity": (
                                                        justification_similarity + payment_similarity) / 2 if justification_similarity is not None and payment_similarity is not None else None
            })

    if not all_results:
        print("No matching questions found for evaluation")
        return None, None

    # Check if we have any data to analyze
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: No data to calculate classification metrics")
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

    # Calculate summary statistics with NaN handling
    summary = {
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
            results_df["avg_justification_similarity"].dropna().mean() if not results_df[
                "avg_justification_similarity"].dropna().empty else 0)
    }

    # Convert numpy types to native Python types for JSON serialization
    summary = numpy_to_python(summary)

    # Print results
    print("\n=== EVALUATION RESULTS ===\n")

    # Print summary statistics
    print("Summary Statistics:")
    print(f"Total Questions in Output: {summary['total_output_questions']}")
    print(f"Total Questions Evaluated: {summary['total_evaluated_questions']}")
    print(
        f"Outcome Classification Accuracy: {summary['outcome_classification']['accuracy']:.4f} ({summary['exact_outcome_match_percentage']:.2f}%)")
    print(f"Average Justification Similarity: {summary['avg_justification_similarity']:.4f}")
    print(f"Average Justification IoU: {summary['avg_justification_iou']:.4f}")
    print(f"Average Payment Similarity: {summary['avg_payment_similarity']:.4f}")
    print(f"Average Combined Justification Similarity: {summary['avg_combined_justification_similarity']:.4f}\n")

    # Print confusion matrix
    print("Confusion Matrix:")
    conf_df = pd.DataFrame(conf_matrix, index=valid_outcomes, columns=valid_outcomes)
    print(conf_df)
    print()

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=valid_outcomes))

    # Print results table with nice formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)

    print("Detailed Results:")
    detailed_columns = ["policy_id", "request_id", "output_outcome", "gt_outcome",
                        "outcome_match", "output_justification", "gt_justification",
                        "output_payment", "gt_payment", "justification_similarity",
                        "justification_iou", "payment_similarity"]

    print(results_df.sort_values(["policy_id", "request_id"])[detailed_columns].to_string(index=False))

    # Save results to CSV
    results_dir = os.path.join(os.path.dirname(JSON_PATH), "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate timestamped filename in European format
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    csv_filename = os.path.join(results_dir, f"evaluation_results_{timestamp}.csv")
    results_df.to_csv(csv_filename, index=False, na_rep='')  # Use empty string for NaN values in CSV

    # Also save summary statistics
    summary_filename = os.path.join(results_dir, f"evaluation_summary_{timestamp}.json")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Save confusion matrix and classification report
    metrics_filename = os.path.join(results_dir, f"classification_metrics_{timestamp}.json")
    metrics_data = {
        "confusion_matrix": numpy_to_python(conf_matrix),
        "classification_report": class_report
    }
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"\nResults saved to {csv_filename}")
    print(f"Summary saved to {summary_filename}")
    print(f"Classification metrics saved to {metrics_filename}")

    return results_df, summary


if __name__ == "__main__":
    # Make sure required libraries are installed
    try:
        import editdistance
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    except ImportError as e:
        print(f"Installing missing library: {e.name}...")
        import pip

        pip.main(["install", e.name])
        if e.name == "editdistance":
            import editdistance
        elif e.name == "scikit-learn":
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Run evaluation
    evaluate_outputs()
