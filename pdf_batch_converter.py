#!/usr/bin/env python3
"""
PDF Batch Converter - Extract text from all PDFs in a directory
A lean, all-Python pipeline for PDFs with embedded text layers (no OCR).

Usage:
    python pdf_batch_converter.py --input-dir ./pdfs --output-dir ./texts
    python pdf_batch_converter.py -i ./input -o ./output --verbose
    python pdf_batch_converter.py --help
"""

import argparse
import pathlib
import sys
from typing import Optional, List
import logging

try:
    import pdfplumber
except ImportError:
    print("Error: pdfplumber not installed. Run: pip install pdfplumber", file=sys.stderr)
    sys.exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def clean_text(text: str) -> str:
    """Clean extracted text by removing common artifacts."""
    if not text:
        return ""

    # Remove null characters and normalize whitespace
    text = text.replace('\x00', '').replace('\ufeff', '')

    # Normalize line endings and excessive whitespace
    lines = [line.strip() for line in text.split('\n')]

    # Remove empty lines at start/end, but preserve internal structure
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    return '\n'.join(lines)


def extract_tables_as_text(page, min_table_size: int = 2) -> str:
    """Extract tables from page and format as readable text."""
    tables_text = ""

    try:
        tables = page.extract_tables()
        for i, table in enumerate(tables):
            if not table or len(table) < min_table_size:
                continue

            tables_text += f"\n\n=== TABLE {i + 1} ===\n"

            # Process each row
            for row_idx, row in enumerate(table):
                if not row:
                    continue

                # Clean cells and join with tabs
                cleaned_row = [str(cell or "").strip() for cell in row]
                if any(cleaned_row):  # Only add non-empty rows
                    tables_text += "\t".join(cleaned_row) + "\n"

    except Exception as e:
        logging.warning(f"Table extraction failed: {e}")

    return tables_text


def is_header_footer(word_obj, page_height: float, margin: float = 50) -> bool:
    """Check if a word is likely in header or footer area."""
    y_pos = word_obj.get('top', 0)
    return y_pos < margin or y_pos > (page_height - margin)


def pdf_to_txt(
        pdf_path: pathlib.Path,
        output_path: pathlib.Path,
        extract_tables: bool = True,
        remove_headers_footers: bool = False,
        encoding: str = "utf-8"
) -> bool:
    """
    Convert a single PDF to text file.

    Args:
        pdf_path: Source PDF file path
        output_path: Output text file path
        extract_tables: Whether to extract table data
        remove_headers_footers: Whether to filter out headers/footers
        encoding: Text encoding for output file

    Returns:
        True if successful, False otherwise
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            with output_path.open("w", encoding=encoding, errors='ignore') as out:
                # Write file header
                out.write(f"Extracted from: {pdf_path.name}\n")
                out.write(f"Total pages: {len(pdf.pages)}\n")
                out.write("=" * 60 + "\n\n")

                for page_num, page in enumerate(pdf.pages, 1):
                    logging.debug(f"Processing page {page_num}/{len(pdf.pages)}")

                    # Page header
                    out.write(f"\n=== PAGE {page_num} ===\n\n")

                    # Extract main text
                    if remove_headers_footers:
                        # Extract words and filter headers/footers
                        words = page.extract_words()
                        filtered_words = [
                            w for w in words
                            if not is_header_footer(w, page.height)
                        ]

                        # Reconstruct text from filtered words
                        if filtered_words:
                            # Sort by position (top to bottom, left to right)
                            filtered_words.sort(key=lambda w: (w['top'], w['x0']))
                            text = ' '.join(w['text'] for w in filtered_words)
                        else:
                            text = ""
                    else:
                        # Standard text extraction
                        text = page.extract_text(
                            x_tolerance=2,  # Helps join words split by spacing
                            y_tolerance=2,
                            layout=True,  # Preserve layout
                            x_density=7.25,  # Fine-tune character spacing
                            y_density=13  # Fine-tune line spacing
                        )

                    # Clean and write main text
                    cleaned_text = clean_text(text)
                    if cleaned_text:
                        out.write(cleaned_text)
                        out.write("\n")

                    # Extract tables if requested
                    if extract_tables:
                        tables_text = extract_tables_as_text(page)
                        if tables_text:
                            out.write(tables_text)

                # Write file footer
                out.write(f"\n\n{'=' * 60}\n")
                out.write(f"Extraction completed: {len(pdf.pages)} pages processed\n")

        return True

    except Exception as e:
        logging.error(f"Failed to process {pdf_path.name}: {e}")
        return False


def process_directory(
        input_dir: pathlib.Path,
        output_dir: pathlib.Path,
        extract_tables: bool = True,
        remove_headers_footers: bool = False,
        overwrite: bool = False
) -> tuple[int, int]:
    """
    Process all PDF files in input directory.

    Returns:
        Tuple of (successful_count, total_count)
    """
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    pdf_files.extend(input_dir.glob("*.PDF"))  # Case-insensitive

    if not pdf_files:
        logging.warning(f"No PDF files found in {input_dir}")
        return 0, 0

    logging.info(f"Found {len(pdf_files)} PDF files to process")

    successful = 0
    for pdf_path in pdf_files:
        output_path = output_dir / (pdf_path.stem + ".txt")

        # Skip if output exists and not overwriting
        if output_path.exists() and not overwrite:
            logging.info(f"⚠️  Skipping {pdf_path.name} (output exists)")
            continue

        logging.info(f"🔄 Processing: {pdf_path.name}")

        if pdf_to_txt(
                pdf_path,
                output_path,
                extract_tables=extract_tables,
                remove_headers_footers=remove_headers_footers
        ):
            logging.info(f"✅ Completed: {pdf_path.name} → {output_path.name}")
            successful += 1
        else:
            logging.error(f"❌ Failed: {pdf_path.name}")

    return successful, len(pdf_files)


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to text files in batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_batch_converter.py -i ./pdfs -o ./texts
  python pdf_batch_converter.py --input-dir ./input --output-dir ./output --verbose
  python pdf_batch_converter.py -i ./docs -o ./texts --no-tables --overwrite
        """
    )

    parser.add_argument(
        "-i", "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing PDF files to convert"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where text files will be saved"
    )

    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table extraction (faster processing)"
    )

    parser.add_argument(
        "--remove-headers-footers",
        action="store_true",
        help="Attempt to remove headers and footers"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing text files"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate input directory
    if not args.input_dir.exists():
        logging.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        logging.error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {args.output_dir.absolute()}")
    except Exception as e:
        logging.error(f"Cannot create output directory: {e}")
        sys.exit(1)

    # Process files
    logging.info(f"Starting batch conversion from {args.input_dir}")

    successful, total = process_directory(
        args.input_dir,
        args.output_dir,
        extract_tables=not args.no_tables,
        remove_headers_footers=args.remove_headers_footers,
        overwrite=args.overwrite
    )

    # Summary
    if total == 0:
        logging.warning("No PDF files found to process")
        sys.exit(1)
    elif successful == total:
        logging.info(f"🎉 Successfully converted all {total} PDF files")
    else:
        logging.warning(f"⚠️  Converted {successful}/{total} PDF files")
        sys.exit(1)


if __name__ == "__main__":
    main()
