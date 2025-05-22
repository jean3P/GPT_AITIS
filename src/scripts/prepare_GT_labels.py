import os
import json
import datetime

from config import RAW_GT_PATH, GT_PATH


def convert_ground_truth_files(input_dir, output_dir):
    """
    Convert raw ground truth JSON files to evaluation format.
    Args:
        input_dir: Directory containing raw ground truth JSON files
        output_dir: Directory to save formatted JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define outcome mapping
    outcome_mapping = {
        "APPROVED": "Yes",
        "DENIED_EXCLUDED": "No - condition(s) not met",
        "DENIED_UNRELATED": "No - Unrelated event",
        "UNDECIDED": "Maybe"
    }

    # Process each JSON file in the input directory
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_dir, filename)

        # Read the input JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Extract policy_id
        policy_id = str(raw_data.get('policy_id', ''))

        # Create output structure
        output_data = {
            "policy_id": policy_id,
            "questions": []
        }

        # Process each policy request
        for policy_request in raw_data.get('policy_requests', []):
            # Get the request data
            request = policy_request.get('request', {})
            request_id = str(request.get('request_id', ''))
            question = request.get('content', '')

            # Get the latest response (highest version number)
            responses = policy_request.get('responses', [])
            if not responses:
                continue

            # Sort responses by version and get the latest
            latest_response = max(responses, key=lambda r: r.get('updated_at', ''))

            # Extract outcome and justifications
            raw_outcome = latest_response.get('response_outcome', '')
            # Map the outcome to the desired format
            outcome = outcome_mapping.get(raw_outcome, raw_outcome)

            outcome_justification = latest_response.get('response_outcome_justification', '')
            payment_justification = latest_response.get('payout_justification', '')

            # Add to output data
            output_data['questions'].append({
                "request_id": request_id,
                "question": question,
                "outcome": outcome,
                "outcome_justification": outcome_justification,
                "payment_justification": payment_justification if payment_justification else None
            })

        # Sort questions by request_id
        output_data['questions'].sort(
            key=lambda q: int(q['request_id']) if q['request_id'].isdigit() else q['request_id'])

        # Write output file
        output_filename = f"GT_policy_{policy_id}.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Converted {filename} -> {output_filename}")


# Example usage
if __name__ == "__main__":
    # Define source and target directories
    raw_gt_dir = RAW_GT_PATH
    formatted_gt_dir = GT_PATH

    convert_ground_truth_files(raw_gt_dir, formatted_gt_dir)