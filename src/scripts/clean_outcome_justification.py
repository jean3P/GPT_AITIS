import fitz
import json
import os
import pandas as pd
import re

from config import DOCUMENT_DIR, JSON_PATH

NB_CONSECUTIVE_TOKENS = 5
AITIS_DELIMITER = '[AITIS-DELIMITER]'


def extract_policy_text(pdf_path):
    text = ""
    document = fitz.open(pdf_path)

    for page in document:
        page_text = page.get_text('text')
        if page_text:  # Avoid appending None if extraction fails
            text += page_text

    return text


def sanitize_text(text: str, is_llm_generated: bool = False):
    sanitized_text = text.replace("\n", " ")              # Replace the end of line by a space
    sanitized_text = re.sub(r'\s+', ' ', sanitized_text)    # Remove unnecessary white spaces

    if(is_llm_generated):
        sanitized_text = sanitized_text.replace('"*', '')
        sanitized_text = sanitized_text.replace('*"', '')
        sanitized_text = sanitized_text.replace('(**', '')
        sanitized_text = sanitized_text.replace('**)', '')
        # sanitized_text = sanitized_text.replace('> **', '')
        # sanitized_text = sanitized_text.replace('> -', '-')
        # sanitized_text = sanitized_text.replace('"', '')
        sanitized_text = sanitized_text.replace('**', '')
        # sanitized_text = sanitized_text.replace('*', ' ')
        # sanitized_text = sanitized_text.replace('"', ' ')
        # sanitized_text = sanitized_text.replace('(', ' ')
        # sanitized_text = sanitized_text.replace(')', ' ')
        # sanitized_text = sanitized_text.replace('>', ' ')
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text)    # Remove unnecessary white spaces

    return sanitized_text


def tokenize_text(text: str):
    return text.split()


def escape_postgres_text(value):
    if isinstance(value, str):
        value = value.replace("'", "''")  # escape single quotes
        return f"E'{value}'"  # use PostgreSQL's E'' notation for special characters
    return value


def extract_ground_truth_text(reference_text: str, mixed_text: str):
    reference_text_sanitized  = sanitize_text(reference_text)
    mixed_text_sanitized      = sanitize_text(mixed_text, is_llm_generated=True)

    reference_tokens          = tokenize_text(reference_text_sanitized)
    mixed_tokens              = tokenize_text(mixed_text_sanitized)

    truth_text                = ' '.join(reference_tokens)

    column_names              = []
    for i in range(NB_CONSECUTIVE_TOKENS):
        column_names.append('Token_' + str(i))

    sliding_window            = []
    for i in range(len(mixed_tokens)):
        if i >= len(mixed_tokens) - NB_CONSECUTIVE_TOKENS + 1:
            break

        else:
            sliding_window.append(mixed_tokens[i:i+NB_CONSECUTIVE_TOKENS])

    mixed_tokens_df = pd.DataFrame(sliding_window, columns=[column_names])
    mixed_tokens_df['Consecutive_Tokens_Found'] = False

    for index, row in mixed_tokens_df.iterrows():
        searched_text = ''

        for i in range(NB_CONSECUTIVE_TOKENS):
            searched_text = ' '.join([searched_text, row['Token_' + str(i)]])

        matches = truth_text.count(searched_text)
        if (matches):
            mixed_tokens_df.at[index, 'Consecutive_Tokens_Found'] = True

    # Detect consecutive sequences in the mixed_text
    sequences = []
    in_sequence = False
    current_tokens = []

    for i, row in mixed_tokens_df.iterrows():
        if row["Consecutive_Tokens_Found"]:
            if not in_sequence:
                # Start of a new valid sequence
                in_sequence = True
                current_tokens = list(row[f'Token_{j}'] for j in range(NB_CONSECUTIVE_TOKENS))
            else:
                # Continue sequence: add only the last token to avoid overlap
                current_tokens.append(row[f'Token_{NB_CONSECUTIVE_TOKENS - 1}'])
        else:
            if in_sequence:
                # End of current sequence
                sequences.append(' '.join(current_tokens))
                current_tokens = []
                in_sequence = False

    # Catch final sequence if it ends at the last row
    if in_sequence:
        sequences.append(' '.join(current_tokens))

    # Return found sequences separated with the delimiter
    corrected_text = AITIS_DELIMITER.join(sequences)
    return corrected_text


def main():
    # Resolve the path relative to this script file
    POLICY_ID =                 18
    POLICY_PDF_DIRECTORY_PATH = DOCUMENT_DIR
    POLICY_PDF_FILENAME =       '18_Nobis - Baggage loss EN.pdf'
    llm_responses_file_path =   os.path.join(JSON_PATH,f'policy_id-{POLICY_ID}__21-05-2025_15-03-07.json')
    output_path =               os.path.join(JSON_PATH, f'policy_{POLICY_ID}_cleanup_21-05-2025_15-03-07.json')
    policy_pdf_path =           os.path.join(POLICY_PDF_DIRECTORY_PATH, POLICY_PDF_FILENAME)

    # Load the JSON file
    with open(llm_responses_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    new_data = {
        'policy_id': data['policy_id'],
        'questions': []
    }

    # Extract the text from the pdf file associated to the specified policy
    policy_text = extract_policy_text(policy_pdf_path)

    for question in data['questions']:
        new_question = question.copy()
        outcome_justification = question.get('outcome_justification')
        payment_justification = question.get('payment_justification')
        request_id = question.get('request_id')
        if outcome_justification:
            new_question['outcome_justification'] = extract_ground_truth_text(policy_text, outcome_justification)
        if payment_justification:
            new_question['payment_justification'] = extract_ground_truth_text(policy_text, payment_justification)
        new_data['questions'].append(new_question)

    # Write to new JSON file with the cleanup justifications
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_data, outfile, indent=2, ensure_ascii=False)

    # with open(f'policy_{POLICY_ID}_pdf_text.txt', 'w', encoding='utf-8') as file:
    #     file.write(sanitize_text(policy_text))


if __name__ == '__main__':
    main()
