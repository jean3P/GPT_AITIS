import re
import json
import argparse
import os
import sys

from pathlib import Path

# Fix import error by using absolute import
# Add the parent directory to sys.path to make the import work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import JSON_PATH, GT_PATH, base_dir


class TextIoU:
    def __init__(self):
        pass

    def normalize_field(self, text):
        """Placeholder for text normalization function."""
        # Replace this with your actual normalize_field implementation
        if text is None:
            return ""
        return str(text).strip()

    def calculate(self, text1, text2):
        """Calculate Intersection over Union (IoU) for text strings based on word sets."""
        # Normalize inputs
        text1 = self.normalize_field(text1)
        text2 = self.normalize_field(text2)

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


def load_json_file(file_path):
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def compare_outcome_justifications(llm_json, gt_json):
    """Compare outcome_justifications between LLM output and ground truth."""
    text_iou = TextIoU()
    results = []

    # Create a dictionary of GT questions by request_id for easy lookup
    gt_questions_dict = {q["request_id"]: q for q in gt_json["questions"]}

    for llm_question in llm_json["questions"]:
        request_id = llm_question["request_id"]

        if request_id in gt_questions_dict:
            gt_question = gt_questions_dict[request_id]

            # Extract outcome_justification fields
            llm_justification = llm_question.get("outcome_justification", "")
            gt_justification = gt_question.get("outcome_justification", "")

            # Calculate IoU score
            iou_score = text_iou.calculate(llm_justification, gt_justification)

            # Store results
            results.append({
                "request_id": request_id,
                "llm_justification": llm_justification,
                "gt_justification": gt_justification,
                "iou_score": iou_score
            })
        else:
            print(f"Warning: Request ID {request_id} not found in ground truth data")

    return results


def calculate_average_iou(comparison_results):
    """Calculate the average IoU score from the comparison results."""
    if not comparison_results:
        return 0.0

    total_score = sum(result["iou_score"] for result in comparison_results)
    return total_score / len(comparison_results)


def get_file_path(file_arg, default_dir):
    """Get the file path, checking if the file exists in default location or as a direct path."""
    # First try the default directory
    default_path = os.path.join(default_dir, file_arg)
    if os.path.isfile(default_path):
        return default_path

    # If not found, check if the argument is a direct path to an existing file
    if os.path.isfile(file_arg):
        return file_arg

    # Look in the current directory
    current_dir_path = os.path.join(os.getcwd(), file_arg)
    if os.path.isfile(current_dir_path):
        return current_dir_path

    # If not found anywhere, return the default path (will cause error later)
    print(f"WARNING: Could not find file '{file_arg}' in default directory or as direct path.")
    print(f"Tried: \n- {default_path}\n- {file_arg}\n- {current_dir_path}")
    return default_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare outcome justifications between LLM and ground truth JSON files')
    parser.add_argument('llm_file', help='Filename or path to the LLM-generated JSON file')
    parser.add_argument('gt_file', help='Filename or path to the ground truth JSON file')
    parser.add_argument('--output', help='Filename or path for saving detailed comparison results')

    args = parser.parse_args()

    # Get file paths, checking multiple locations
    llm_file_path = get_file_path(args.llm_file, JSON_PATH)
    gt_file_path = get_file_path(args.gt_file, GT_PATH)

    # Load JSON files
    print(f"Loading LLM file: {llm_file_path}")
    try:
        llm_json = load_json_file(llm_file_path)
    except FileNotFoundError:
        print(f"ERROR: LLM file not found at '{llm_file_path}'")
        print("Please provide a valid path to the LLM output JSON file.")
        return

    print(f"Loading ground truth file: {gt_file_path}")
    try:
        gt_json = load_json_file(gt_file_path)
    except FileNotFoundError:
        print(f"ERROR: Ground truth file not found at '{gt_file_path}'")
        print("Please provide a valid path to the ground truth JSON file.")
        return

    # Compare outcome justifications
    print("Comparing outcome justifications...")
    comparison_results = compare_outcome_justifications(llm_json, gt_json)

    # Calculate and display average IoU
    avg_iou = calculate_average_iou(comparison_results)
    print(f"\nAverage IoU Score: {avg_iou:.4f}")

    # Output detailed results if requested
    if args.output:
        # Determine output path
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            output_path = os.path.join(JSON_PATH, args.output)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump({
                "average_iou": avg_iou,
                "detailed_results": comparison_results
            }, file, indent=2)
        print(f"Detailed results saved to {output_path}")

    print("\nSample of comparison results:")
    for result in comparison_results[:3]:  # Show first 3 results
        print(f"Request ID: {result['request_id']}")
        print(f"LLM justification: {result['llm_justification'][:50]}..." if len(
            result['llm_justification']) > 50 else f"LLM justification: {result['llm_justification']}")
        print(f"GT justification: {result['gt_justification'][:50]}..." if len(
            result['gt_justification']) > 50 else f"GT justification: {result['gt_justification']}")
        print(f"IoU Score: {result['iou_score']:.4f}\n")


if __name__ == "__main__":
    main()
