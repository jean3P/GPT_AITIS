import fitz  # PyMuPDF
import re
import os
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_sections_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Extract sections from a PDF file based on numbered headers.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary mapping section headers to their content
    """
    # Validate file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logging.info(f"Processing PDF: {pdf_path}")

    try:
        # Open the PDF
        doc = fitz.open(pdf_path)

        # Extract text with page numbers for debugging
        text_by_pages = []
        for i, page in enumerate(doc):
            page_text = page.get_text()
            text_by_pages.append(page_text)
            logging.info(f"Page {i + 1} extracted: {len(page_text)} characters")

        full_text = "\n".join(text_by_pages)
        logging.info(f"Total text extracted: {len(full_text)} characters")

        # More flexible regex pattern for section headers
        # This pattern looks for:
        # 1. Optional newlines
        # 2. One or more digits followed by a period
        # 3. One or more whitespace characters
        # 4. A word starting with uppercase letter followed by any characters (not just letters)
        # 5. The entire match must not exceed a reasonable length for a header
        pattern = r'(?:\n|\r\n?)?(\d+\.\s+[A-Z][^\n\r]{1,60})'

        # Find all matches
        matches = list(re.finditer(pattern, full_text))
        logging.info(f"Found {len(matches)} potential section headers")

        if not matches:
            # If no matches, try an alternative pattern without numbers
            alt_pattern = r'(?:\n|\r\n?)([A-Z][A-Z\s]{2,50}:?)'
            matches = list(re.finditer(alt_pattern, full_text))
            logging.info(f"Tried alternative pattern, found {len(matches)} potential headers")

        # Print found headers for debugging
        for match in matches:
            logging.info(f"Found header: '{match.group(1).strip()}'")

        # Group text into sections
        sections = {}
        for i, match in enumerate(matches):
            header = match.group(1).strip()
            start = match.end()

            # Determine end of this section (start of next section or end of document)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

            # Extract content
            content = full_text[start:end].strip()

            # Skip empty sections
            if not content:
                logging.warning(f"Empty content for section '{header}', skipping")
                continue

            sections[header] = content
            logging.info(f"Extracted section '{header}': {len(content)} characters")

        return sections

    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise
    finally:
        # Close the document
        if 'doc' in locals():
            doc.close()


def print_sections(sections: Dict[str, str], preview_length: int = 300):
    """
    Print sections with preview of content

    Args:
        sections: Dictionary of sections
        preview_length: Number of characters to preview for each section
    """
    if not sections:
        print("No sections found!")
        return

    print(f"\nFound {len(sections)} sections:\n")

    for section_title, section_content in sections.items():
        print(f"\n{'=' * 80}")
        print(f"=== {section_title} ===")
        print(f"{'=' * 80}\n")

        preview = section_content[:preview_length]
        print(f"{preview}...")
        print(f"\n[Total: {len(section_content)} characters]")


def main():
    # Update this path to your PDF location
    from config import DOCUMENT_DIR
    pdf_path = os.path.join(DOCUMENT_DIR, "18_Nobis - Baggage loss EN.pdf")

    try:
        sections = extract_sections_from_pdf(pdf_path)
        print_sections(sections)

        # Optionally save to individual files
        # output_dir = "extracted_sections"
        # os.makedirs(output_dir, exist_ok=True)
        # for title, content in sections.items():
        #     safe_title = "".join(c if c.isalnum() or c.isspace() else "_" for c in title)
        #     with open(os.path.join(output_dir, f"{safe_title}.txt"), "w", encoding="utf-8") as f:
        #         f.write(content)

    except Exception as e:
        logging.error(f"Failed to process PDF: {str(e)}")


if __name__ == "__main__":
    main()
