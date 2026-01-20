#!/usr/bin/env python3
"""
Copy Latest Invoices Script

This script copies the latest M invoice files from a source folder to the data/raw/invoices folder.
Supports PDF, PNG, JPG, JPEG, and TIFF formats.

Usage:
    python copy_latest_invoices.py <source_folder> <number_of_invoices>

Example:
    python copy_latest_invoices.py "C:\invoices\2024" 1000
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime


# Supported invoice file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}


def get_latest_invoices(source_folder, count):
    """
    Get the latest 'count' invoice files from source_folder sorted by modification time.

    Args:
        source_folder: Path to the source folder containing invoices
        count: Number of latest invoices to retrieve

    Returns:
        List of Path objects for the latest invoices
    """
    source_path = Path(source_folder)

    if not source_path.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_folder}")

    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_folder}")

    # Get all invoice files with supported extensions
    invoice_files = []
    for ext in SUPPORTED_EXTENSIONS:
        invoice_files.extend(source_path.glob(f'*{ext}'))
        invoice_files.extend(source_path.glob(f'*{ext.upper()}'))

    if not invoice_files:
        raise FileNotFoundError(f"No invoice files found in: {source_folder}")

    # Sort by modification time (newest first)
    invoice_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Return the latest 'count' files
    return invoice_files[:count]


def copy_invoices(source_files, destination_folder, clear_destination=False):
    """
    Copy invoice files to the destination folder.

    Args:
        source_files: List of Path objects to copy
        destination_folder: Destination folder path
        clear_destination: If True, clear the destination folder before copying

    Returns:
        Number of files successfully copied
    """
    dest_path = Path(destination_folder)

    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Optionally clear destination folder
    if clear_destination:
        for file in dest_path.glob('*'):
            if file.is_file():
                file.unlink()
                print(f"Deleted: {file.name}")

    # Copy files
    copied_count = 0
    for source_file in source_files:
        try:
            destination_file = dest_path / source_file.name

            # Handle duplicate filenames
            if destination_file.exists():
                base_name = source_file.stem
                extension = source_file.suffix
                counter = 1
                while destination_file.exists():
                    destination_file = dest_path / f"{base_name}_{counter}{extension}"
                    counter += 1

            shutil.copy2(source_file, destination_file)
            copied_count += 1
            print(f"Copied: {source_file.name} -> {destination_file.name}")

        except Exception as e:
            print(f"Error copying {source_file.name}: {e}", file=sys.stderr)

    return copied_count


def main():
    """Main function to handle command-line arguments and execute the copy operation."""

    # Check arguments
    if len(sys.argv) < 3:
        print("Usage: python copy_latest_invoices.py <source_folder> <number_of_invoices> [--clear]")
        print("\nExample:")
        print('  python copy_latest_invoices.py "C:\\invoices\\2024" 1000')
        print('  python copy_latest_invoices.py "C:\\invoices\\2024" 1000 --clear')
        print("\nOptions:")
        print("  --clear    Clear the destination folder before copying")
        sys.exit(1)

    source_folder = sys.argv[1]

    try:
        invoice_count = int(sys.argv[2])
        if invoice_count <= 0:
            raise ValueError("Number of invoices must be positive")
    except ValueError as e:
        print(f"Error: Invalid number of invoices: {sys.argv[2]}")
        print(f"Details: {e}")
        sys.exit(1)

    # Check for --clear flag
    clear_destination = '--clear' in sys.argv

    # Get script directory and construct destination path
    script_dir = Path(__file__).parent.parent
    destination_folder = script_dir / 'data' / 'raw' / 'invoices'

    print("=" * 70)
    print("Copy Latest Invoices")
    print("=" * 70)
    print(f"Source folder:      {source_folder}")
    print(f"Destination folder: {destination_folder}")
    print(f"Invoices to copy:   {invoice_count}")
    print(f"Clear destination:  {clear_destination}")
    print(f"Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    try:
        # Get latest invoice files
        print(f"Scanning source folder for invoice files...")
        latest_invoices = get_latest_invoices(source_folder, invoice_count)

        print(f"Found {len(latest_invoices)} invoice files")
        print(f"Copying {len(latest_invoices)} invoices...\n")

        # Copy files
        copied = copy_invoices(latest_invoices, destination_folder, clear_destination)

        print()
        print("=" * 70)
        print(f"SUCCESS: Copied {copied} out of {len(latest_invoices)} invoice files")
        print("=" * 70)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        sys.exit(1)


if __name__ == '__main__':
    main()
