import csv
import textwrap

from config import RESULT_PATH

# Path to your result file
file_path = RESULT_PATH

# Field labels (in the order they appear in the TSV file)
field_labels = [
    "Vector Store",
    "Question ID",
    "Question",
    "Eligibility",
    "Eligibility Policy",
    "Amount Policy",
    "Amount Policy Line"
]

# Max line width before wrapping
wrap_width = 80

def display_result_table(row):
    print("\n" + "-" * wrap_width*2)
    print("{:<20}{}".format("Field", "Value"))
    print("-" * wrap_width*2)
    for label, value in zip(field_labels, row):
        wrapped_value = textwrap.fill(value, width=wrap_width, subsequent_indent=" " * 22)
        print(f"{label:<20}{wrapped_value}")
    print("-" * wrap_width*2)

# Read and display each row from the TSV file
with open(file_path, "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        if row:  # skip empty lines
            display_result_table(row)
