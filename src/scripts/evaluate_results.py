# scripts/evaluate_results.py

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import editdistance
import pandas as pd
import numpy as np
import argparse
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


def parse_timestamp_dirname(dirname: str) -> Optional[datetime]:
    """
    Parse timestamp from directory name in format DD-MM-YY--HH-MM-SS.

    Returns:
        datetime object if parsing successful, None otherwise
    """
    try:
        # Expected format: DD-MM-YY--HH-MM-SS
        return datetime.strptime(dirname, "%d-%m-%y--%H-%M-%S")
    except ValueError:
        return None


def get_model_directories(json_path: str) -> List[str]:
    """Get all model directories from the JSON output path."""
    if not os.path.exists(json_path):
        return []

    model_dirs = []
    for item in os.listdir(json_path):
        item_path = os.path.join(json_path, item)
        if os.path.isdir(item_path):
            # Check if directory contains policy output files (either directly or in k/complete-policy subdirectories)
            has_policy_files = False

            # Check for direct policy files (old structure)
            files = os.listdir(item_path)
            if any(f.startswith("policy_") and f.endswith(".json") for f in files):
                has_policy_files = True

            # Check for k subdirectories and complete-policy directory (new structure)
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and (subitem.startswith("k=") or subitem == "complete-policy"):
                    # Check for timestamp directories
                    for timestamp_dir in os.listdir(subitem_path):
                        timestamp_path = os.path.join(subitem_path, timestamp_dir)
                        if os.path.isdir(timestamp_path):
                            subfiles = os.listdir(timestamp_path)
                            if any(f.startswith("policy_") and f.endswith(".json") for f in subfiles):
                                has_policy_files = True
                                break

                    # Also check old structure (files directly in k/complete-policy directory)
                    if not has_policy_files:
                        subfiles = os.listdir(subitem_path)
                        if any(f.startswith("policy_") and f.endswith(".json") for f in subfiles):
                            has_policy_files = True

            if has_policy_files:
                model_dirs.append(item)

    return sorted(model_dirs)

def get_latest_timestamp_dir(k_dir_path: str) -> Optional[str]:
    """
    Find the latest timestamp directory in a k directory.

    Args:
        k_dir_path: Path to k directory (e.g., /path/to/model/k=3)

    Returns:
        Path to latest timestamp directory or None if not found
    """
    if not os.path.exists(k_dir_path):
        return None

    timestamp_dirs = []
    for item in os.listdir(k_dir_path):
        item_path = os.path.join(k_dir_path, item)
        if os.path.isdir(item_path):
            timestamp = parse_timestamp_dirname(item)
            if timestamp:
                timestamp_dirs.append((timestamp, item_path))

    if not timestamp_dirs:
        return None

    # Sort by timestamp and return the latest
    timestamp_dirs.sort(key=lambda x: x[0], reverse=True)
    return timestamp_dirs[0][1]


def get_specific_timestamp_dir(k_dir_path: str, date_str: str) -> Optional[str]:
    """
    Find a specific timestamp directory matching the given date.

    Args:
        k_dir_path: Path to k directory
        date_str: Date string to match (can be partial, e.g., "24-07-25" or full "24-07-25--10-30-45")

    Returns:
        Path to matching timestamp directory or None if not found
    """
    if not os.path.exists(k_dir_path):
        return None

    for item in os.listdir(k_dir_path):
        item_path = os.path.join(k_dir_path, item)
        if os.path.isdir(item_path) and item.startswith(date_str):
            return item_path

    return None


def get_latest_output_files(model_dir_path: str, k_value: str = None,
                            use_latest: bool = False, date_str: str = None,
                            complete_policy: bool = False) -> List[Tuple[str, str]]:
    """
    Find output files for each policy ID in a model directory.

    Args:
        model_dir_path: Path to the model directory
        k_value: Specific k value to look for (e.g., "3" for k=3), or None to search all
        use_latest: If True, use the latest timestamp directory
        date_str: Specific date string to match timestamp directories
        complete_policy: If True, look in complete-policy directory instead of k directories

    Returns:
        List of tuples: (file_path, file_name)
    """
    if not os.path.exists(model_dir_path):
        return []

    all_files = []

    if complete_policy:
        # Look for complete-policy directory
        complete_policy_path = os.path.join(model_dir_path, "complete-policy")
        if os.path.exists(complete_policy_path):
            print(f"  📁 Checking complete-policy directory...")

            # Check for timestamp directories
            has_timestamp_dirs = any(
                parse_timestamp_dirname(d) is not None
                for d in os.listdir(complete_policy_path)
                if os.path.isdir(os.path.join(complete_policy_path, d))
            )

            if has_timestamp_dirs:
                # New structure with timestamp directories
                if use_latest:
                    # Find the latest timestamp directory
                    timestamp_dir = get_latest_timestamp_dir(complete_policy_path)
                    if timestamp_dir:
                        timestamp_name = os.path.basename(timestamp_dir)
                        print(f"    📅 Using latest timestamp: {timestamp_name}")
                        for file in os.listdir(timestamp_dir):
                            if file.startswith("policy_") and file.endswith(".json"):
                                all_files.append((os.path.join(timestamp_dir, file), file))
                elif date_str:
                    # Find specific timestamp directory
                    timestamp_dir = get_specific_timestamp_dir(complete_policy_path, date_str)
                    if timestamp_dir:
                        timestamp_name = os.path.basename(timestamp_dir)
                        print(f"    📅 Using timestamp: {timestamp_name}")
                        for file in os.listdir(timestamp_dir):
                            if file.startswith("policy_") and file.endswith(".json"):
                                all_files.append((os.path.join(timestamp_dir, file), file))
                    else:
                        print(f"    ⚠️  No timestamp directory found matching: {date_str}")
                else:
                    # Process all timestamp directories
                    for timestamp_item in sorted(os.listdir(complete_policy_path)):
                        timestamp_path = os.path.join(complete_policy_path, timestamp_item)
                        if os.path.isdir(timestamp_path) and parse_timestamp_dirname(timestamp_item):
                            print(f"    📅 Processing timestamp: {timestamp_item}")
                            for file in os.listdir(timestamp_path):
                                if file.startswith("policy_") and file.endswith(".json"):
                                    all_files.append((os.path.join(timestamp_path, file), file))
            else:
                # Old structure - files directly in complete-policy directory
                print(f"    📁 No timestamp directories found, checking main complete-policy directory...")
                for file in os.listdir(complete_policy_path):
                    if file.startswith("policy_") and file.endswith(".json"):
                        all_files.append((os.path.join(complete_policy_path, file), file))
        else:
            print(f"  ⚠️  No complete-policy directory found")
            return []
    else:
        # Original k-based directory structure code
        # Check for k subdirectories (new structure)
        k_dirs_found = False
        for item in os.listdir(model_dir_path):
            item_path = os.path.join(model_dir_path, item)
            if os.path.isdir(item_path) and item.startswith("k="):
                k_dirs_found = True
                # Extract k value from directory name
                dir_k_value = item.split("=")[1]

                # If specific k_value requested, only look in that directory
                if k_value is not None and dir_k_value != k_value:
                    continue

                print(f"  📁 Checking k={dir_k_value} subdirectory...")

                # Check for timestamp directories
                has_timestamp_dirs = any(
                    parse_timestamp_dirname(d) is not None
                    for d in os.listdir(item_path)
                    if os.path.isdir(os.path.join(item_path, d))
                )

                if has_timestamp_dirs:
                    # New structure with timestamp directories
                    if use_latest:
                        # Find the latest timestamp directory
                        timestamp_dir = get_latest_timestamp_dir(item_path)
                        if timestamp_dir:
                            timestamp_name = os.path.basename(timestamp_dir)
                            print(f"    📅 Using latest timestamp: {timestamp_name}")
                            for file in os.listdir(timestamp_dir):
                                if file.startswith("policy_") and file.endswith(".json"):
                                    all_files.append((os.path.join(timestamp_dir, file), file))
                    elif date_str:
                        # Find specific timestamp directory
                        timestamp_dir = get_specific_timestamp_dir(item_path, date_str)
                        if timestamp_dir:
                            timestamp_name = os.path.basename(timestamp_dir)
                            print(f"    📅 Using timestamp: {timestamp_name}")
                            for file in os.listdir(timestamp_dir):
                                if file.startswith("policy_") and file.endswith(".json"):
                                    all_files.append((os.path.join(timestamp_dir, file), file))
                        else:
                            print(f"    ⚠️  No timestamp directory found matching: {date_str}")
                    else:
                        # Process all timestamp directories
                        for timestamp_item in sorted(os.listdir(item_path)):
                            timestamp_path = os.path.join(item_path, timestamp_item)
                            if os.path.isdir(timestamp_path) and parse_timestamp_dirname(timestamp_item):
                                print(f"    📅 Processing timestamp: {timestamp_item}")
                                for file in os.listdir(timestamp_path):
                                    if file.startswith("policy_") and file.endswith(".json"):
                                        all_files.append((os.path.join(timestamp_path, file), file))
                else:
                    # Old structure - files directly in k directory
                    print(f"    📁 No timestamp directories found, checking main k directory...")
                    for file in os.listdir(item_path):
                        if file.startswith("policy_") and file.endswith(".json"):
                            all_files.append((os.path.join(item_path, file), file))

        # If no k directories found, check for direct files (old structure)
        if not k_dirs_found:
            print(f"  📁 No k subdirectories found, checking main directory...")
            for file in os.listdir(model_dir_path):
                if file.startswith("policy_") and file.endswith(".json"):
                    all_files.append((os.path.join(model_dir_path, file), file))

    print(f"  📁 Found {len(all_files)} policy output files total")

    # Group files by policy ID
    policy_files = {}
    for file_path, file_name in all_files:
        # Extract policy ID from filename: policy_{id}_results.json
        match = re.match(r'policy_(\d+[-\d]*)_results\.json', file_name)
        if match:
            policy_id = match.group(1)
            if policy_id not in policy_files:
                policy_files[policy_id] = []
            policy_files[policy_id].append((file_path, file_name))

    # For each policy ID, select the file (if multiple, take the first one)
    latest_files = []
    for policy_id, files in sorted(policy_files.items()):
        if files:
            # If multiple files for same policy, take the first one
            selected_file = files[0]
            latest_files.append(selected_file)

            if len(files) > 1:
                print(f"  ⚠️  Policy {policy_id}: Found {len(files)} files, using {selected_file[1]}")

    print(f"  ✅ Selected {len(latest_files)} files for evaluation:")
    for file_path, file_name in sorted(latest_files, key=lambda x: x[1]):
        policy_id = re.match(r'policy_(\d+[-\d]*)_results\.json', file_name).group(1)
        # Extract k value, complete-policy, and timestamp from path if present
        k_match = re.search(r'/k=(\d+)/', file_path)
        complete_match = re.search(r'/complete-policy/', file_path)
        timestamp_match = re.search(r'/(\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})/', file_path)

        info_parts = []
        if k_match:
            info_parts.append(f"k={k_match.group(1)}")
        elif complete_match:
            info_parts.append("complete-policy")
        if timestamp_match:
            info_parts.append(f"timestamp={timestamp_match.group(1)}")

        info_str = f" ({', '.join(info_parts)})" if info_parts else ""
        print(f"    - Policy {policy_id}: {file_name}{info_str}")

    return latest_files


def evaluate_model_outputs(model_name: str, model_dir_path: str, gt_path: str,
                           k_value: str = None, use_latest: bool = False,
                           date_str: str = None, complete_policy: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate outputs for a single model against ground truth files."""
    print(f"\n=== Evaluating Model: {model_name} ===")
    if complete_policy:
        print(f"  📄 Using complete policy mode")
    elif k_value:
        print(f"  🎯 Filtering for k={k_value}")
    if use_latest:
        print(f"  📅 Using latest experiment")
    if date_str:
        print(f"  📅 Using experiments from date: {date_str}")

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

    # Get the output files based on criteria
    output_files = get_latest_output_files(model_dir_path, k_value, use_latest, date_str, complete_policy)

    # Get all ground truth files
    gt_files = [f for f in os.listdir(gt_path) if f.startswith("GT_policy_") and f.endswith(".json")]

    print(f"\n  📋 Ground Truth Files Found ({len(gt_files)}):")
    for gt_file in sorted(gt_files):
        print(f"    - {gt_file}")

    if not output_files:
        print(f"Error: No output files found in {model_dir_path}")
        if complete_policy:
            print(f"       (Looking for complete-policy results)")
        elif k_value:
            print(f"       (Looking specifically for k={k_value})")
        if use_latest:
            print(f"       (Looking for latest experiment)")
        if date_str:
            print(f"       (Looking for date: {date_str})")
        return None, None

    if not gt_files:
        print(f"Error: No ground truth files found in {gt_path}")
        return None, None

    # Map policy IDs to GT files for easier lookup
    gt_file_map = {f.split("_")[2].split(".")[0]: f for f in gt_files}

    print(f"\n  🔗 Policy ID to Ground Truth File Mapping:")
    for policy_id, gt_file in sorted(gt_file_map.items()):
        print(f"    - Policy {policy_id}: {gt_file}")

    print(f"\nFound {len(output_files)} output files and {len(gt_files)} ground truth files")

    total_output_questions = 0
    total_evaluated_questions = 0

    print(f"\n  🔍 Processing File Pairs:")
    for output_file_path, output_file_name in output_files:
        # Extract policy ID from filename
        match = re.match(r'policy_(\d+[-\d]*)_results\.json', output_file_name)
        if not match:
            print(f"    ❌ Could not extract policy ID from {output_file_name}")
            continue

        policy_id = match.group(1)

        # Find corresponding ground truth file
        if policy_id not in gt_file_map:
            print(f"    ❌ Policy {policy_id}: No ground truth file found for output {output_file_name}")
            continue

        gt_file_name = gt_file_map[policy_id]
        gt_file_path = os.path.join(gt_path, gt_file_name)

        # Extract k value, complete-policy, and timestamp from path if present
        k_match = re.search(r'/k=(\d+)/', output_file_path)
        complete_match = re.search(r'/complete-policy/', output_file_path)
        timestamp_match = re.search(r'/(\d{2}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})/', output_file_path)

        info_parts = []
        if k_match:
            info_parts.append(f"k={k_match.group(1)}")
        elif complete_match:
            info_parts.append("complete-policy")
        if timestamp_match:
            info_parts.append(f"timestamp={timestamp_match.group(1)}")

        info_str = f" ({', '.join(info_parts)})" if info_parts else ""

        print(f"    ✅ Policy {policy_id}{info_str}:")
        print(f"       Output: {output_file_name}")
        print(f"       Ground Truth: {gt_file_name}")

        # Load files
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)

            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"    ❌ Error loading files for policy {policy_id}: {e}")
            continue

        # Create lookup dictionaries for faster access
        output_questions = output_data.get("questions", [])
        gt_questions_dict = {q["request_id"]: q for q in gt_data.get("questions", [])}

        questions_in_file = len(output_questions)
        total_output_questions += questions_in_file

        print(f"       Questions: {questions_in_file} in output, {len(gt_questions_dict)} in ground truth")

        # Compare each output question with ground truth
        for output_question in output_questions:
            request_id = output_question.get("request_id")

            if request_id not in gt_questions_dict:
                print(f"       ⚠️  Request {request_id} not found in ground truth")
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

            # Add k value or "complete" and timestamp to results if available
            if complete_match:
                k_value_extracted = "complete"
            elif k_match:
                k_value_extracted = k_match.group(1)
            else:
                k_value_extracted = None

            timestamp_extracted = timestamp_match.group(1) if timestamp_match else None

            # Add to results
            all_results.append({
                "model_name": model_name,
                "k_value": k_value_extracted,
                "timestamp": timestamp_extracted,
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

    # Extract k value or complete-policy info and timestamp info for summary
    if complete_policy:
        k_info = "complete-policy"
    else:
        k_values = results_df['k_value'].dropna().unique()
        # Filter out "complete" values when not in complete_policy mode
        k_values = [v for v in k_values if v != "complete"]
        k_info = f"k={','.join(k_values)}" if len(k_values) > 0 else "k=unknown"

    timestamps = results_df['timestamp'].dropna().unique() if 'timestamp' in results_df else []
    timestamp_info = timestamps[0] if len(timestamps) == 1 else "multiple" if len(timestamps) > 1 else "unknown"

    # Calculate summary statistics
    summary = {
        "model_name": model_name,
        "k_configuration": k_info,
        "experiment_timestamp": timestamp_info,
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

    # Add k configuration and experiment timestamp to filename if available
    k_config = summary.get('k_configuration', '')
    exp_timestamp = summary.get('experiment_timestamp', '')

    suffix_parts = []
    if k_config and k_config != "k=unknown":
        suffix_parts.append(k_config)
    if exp_timestamp and exp_timestamp != "unknown":
        suffix_parts.append(f"exp_{exp_timestamp}")

    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""

    # Save detailed results to CSV
    csv_filename = os.path.join(model_output_dir, f"evaluation_results{suffix}_{timestamp}.csv")
    results_df.to_csv(csv_filename, index=False, na_rep='')

    # Save summary statistics
    summary_filename = os.path.join(model_output_dir, f"evaluation_summary{suffix}_{timestamp}.json")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Save classification metrics
    metrics_filename = os.path.join(model_output_dir, f"classification_metrics{suffix}_{timestamp}.json")

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
        "classification_report": class_report,
        "k_configuration": summary.get('k_configuration', 'unknown'),
        "experiment_timestamp": summary.get('experiment_timestamp', 'unknown')
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
    k_config = summary.get('k_configuration', '')
    exp_timestamp = summary.get('experiment_timestamp', 'unknown')

    print(f"\n=== EVALUATION RESULTS FOR {model_name.upper()} ({k_config}) ===")
    print(f"Experiment Timestamp: {exp_timestamp}")
    print(f"Total Questions in Output: {summary['total_output_questions']}")
    print(f"Total Questions Evaluated: {summary['total_evaluated_questions']}")
    print(f"Outcome Classification Accuracy: {summary['outcome_classification']['accuracy']:.4f} "
          f"({summary['exact_outcome_match_percentage']:.2f}%)")
    print(f"Average Justification IoU: {summary['avg_justification_iou']:.4f}")


def compare_models(summaries: List[Dict]):
    """Print comparison between multiple models."""
    if len(summaries) < 2:
        return

    print(f"\n=== MODEL COMPARISON ===")

    # Include k configuration and timestamp in model names
    model_headers = []
    for s in summaries:
        k_config = s.get('k_configuration', '')
        exp_timestamp = s.get('experiment_timestamp', 'unknown')
        model_name = s['model_name']

        header_parts = [model_name]
        if k_config and k_config != 'k=unknown':
            header_parts.append(k_config)
        if exp_timestamp != 'unknown' and exp_timestamp != 'multiple':
            # Shorten timestamp for display
            if len(exp_timestamp) > 10:
                header_parts.append(exp_timestamp[:8] + "...")
            else:
                header_parts.append(exp_timestamp)

        model_headers.append(" ".join(header_parts))

    print(f"{'Metric':<35} | " + " | ".join([f"{h:<30}" for h in model_headers]))
    print("-" * (35 + len(summaries) * 33))

    metrics = [
        ("Outcome Accuracy (%)", "exact_outcome_match_percentage", ".2f"),
        ("Justification IoU", "avg_justification_iou", ".4f"),
    ]

    for metric_name, key, fmt in metrics:
        values = [f"{s[key]:{fmt}}" for s in summaries]
        print(f"{metric_name:<35} | " + " | ".join([f"{v:<30}" for v in values]))


def main():
    parser = argparse.ArgumentParser(description='Evaluate insurance policy analysis results')
    parser.add_argument('--models', nargs='+',
                        help='Model names to evaluate (e.g., microsoft_phi-4 qwen_qwen-2.5-72b-instruct). '
                             'If not specified, all available models will be evaluated.')
    parser.add_argument('--k', type=str,
                        help='Specific k value to evaluate (e.g., 3 for k=3). '
                             'If not specified, all k values will be evaluated.')
    parser.add_argument('--complete-policy', action='store_true',
                        help='Evaluate complete-policy results instead of RAG results')
    parser.add_argument('--latest', action='store_true',
                        help='Use the latest experiment for each model/k combination')
    parser.add_argument('--date', type=str,
                        help='Specific date to evaluate (e.g., "24-07-25" or "24-07-25--10-30-45"). '
                             'Can be partial date.')
    parser.add_argument('--json-path', default=JSON_PATH,
                        help='Path to JSON output directory')
    parser.add_argument('--gt-path', default=GT_PATH,
                        help='Path to ground truth directory')
    parser.add_argument('--output-dir', default=EVALUATION_RESULTS_FILES_PATH,
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Validate arguments
    if args.latest and args.date:
        print("Error: Cannot use both --latest and --date options simultaneously")
        return

    if args.complete_policy and args.k:
        print("Error: Cannot use both --complete-policy and --k options simultaneously")
        return

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

    print(f"\n🔍 Available Model Directories Found:")
    for model in available_models:
        model_path = os.path.join(args.json_path, model)
        # Check for k subdirectories and complete-policy directory
        k_dirs = [d for d in os.listdir(model_path) if
                  os.path.isdir(os.path.join(model_path, d)) and d.startswith("k=")]
        complete_policy_dir = os.path.join(model_path, "complete-policy")
        has_complete_policy = os.path.exists(complete_policy_dir) and os.path.isdir(complete_policy_dir)

        info_parts = []
        if k_dirs:
            # Check for timestamp directories in each k directory
            k_info = []
            for k_dir in sorted(k_dirs):
                k_path = os.path.join(model_path, k_dir)
                timestamp_dirs = [d for d in os.listdir(k_path)
                                  if os.path.isdir(os.path.join(k_path, d))
                                  and parse_timestamp_dirname(d) is not None]
                if timestamp_dirs:
                    k_info.append(f"{k_dir} ({len(timestamp_dirs)} experiments)")
                else:
                    k_info.append(k_dir)
            info_parts.extend(k_info)

        if has_complete_policy:
            # Check for timestamp directories in complete-policy
            timestamp_dirs = [d for d in os.listdir(complete_policy_dir)
                              if os.path.isdir(os.path.join(complete_policy_dir, d))
                              and parse_timestamp_dirname(d) is not None]
            if timestamp_dirs:
                info_parts.append(f"complete-policy ({len(timestamp_dirs)} experiments)")
            else:
                info_parts.append("complete-policy")

        if info_parts:
            print(f"  - {model} (with {', '.join(info_parts)})")
        else:
            print(f"  - {model}")

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

        print(f"\n🚀 Evaluating model: {model_name}")
        print(f"📂 Model directory: {model_dir_path}")

        if args.complete_policy:
            print(f"📄 Mode: Complete policy")
        elif args.k:
            print(f"🎯 Mode: k={args.k}")

        if args.latest:
            print(f"📅 Mode: Latest experiment")
        elif args.date:
            print(f"📅 Mode: Specific date - {args.date}")
        else:
            print(f"📅 Mode: All experiments")

        results_df, summary = evaluate_model_outputs(
            model_name, model_dir_path, args.gt_path,
            k_value=args.k, use_latest=args.latest, date_str=args.date,
            complete_policy=args.complete_policy
        )

        if results_df is not None and summary is not None:
            # Save results
            saved_files = save_evaluation_results(results_df, summary, model_name, args.output_dir)

            # Print summary
            print_evaluation_summary(summary)

            print(f"\n💾 Results saved:")
            print(f"  CSV: {saved_files['csv']}")
            print(f"  Summary: {saved_files['summary']}")
            print(f"  Metrics: {saved_files['metrics']}")

            all_summaries.append(summary)
            all_results_dfs.append(results_df)
        else:
            print(f"❌ Failed to evaluate model: {model_name}")

    # Print comparison if multiple models were evaluated
    if len(all_summaries) > 1:
        compare_models(all_summaries)

    print(f"\n🎉 === EVALUATION COMPLETE ===")
    print(f"Evaluated {len(all_summaries)} models successfully")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

