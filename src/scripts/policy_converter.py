import json
import os.path

import pandas as pd

from config import GT_PATH


def convert_policy_to_excel(json_data, output_file='policy_output.xlsx'):
    """
    Convert JSON policy data to Excel format

    Parameters:
    json_data (str): Path to the JSON file or JSON string
    output_file (str): Path for the output Excel file
    """
    # Load the JSON data
    if isinstance(json_data, str) and json_data.endswith('.json'):
        # It's a file path
        with open(json_data, 'r') as f:
            policy = json.load(f)
    else:
        # It's a JSON string
        policy = json.loads(json_data)

    # Extract the policy ID and questions
    policy_id = policy['policy_id']
    questions = policy['questions']

    # Prepare data for DataFrame
    data = []
    for question in questions:
        # Map outcome to status
        outcome = question['outcome']
        if outcome == 'Yes':
            status = 'validated by saverio'
        elif outcome == 'Maybe':
            status = 'unsure'
        else:  # No cases
            status = 'not done'

        # Create row with required data
        row = {
            'policy_id': policy_id,
            'request_id': question['request_id'],
            'status': status,
            'question': question['question'],
            'outcome': outcome,
            'outcome_justification': question['outcome_justification'],
            'payment_justification': question.get('payment_justification', '')
        }
        data.append(row)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Successfully converted policy {policy_id} to Excel: {output_file}")
    return df


if __name__ == "__main__":
    json_file = os.path.join(GT_PATH, 'GT_policy_26.json')
    convert_policy_to_excel(json_file)
