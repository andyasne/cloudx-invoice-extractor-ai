"""
Cloudx Invoice AI - Data Preparation Script
Prepare invoice dataset for training
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import InvoicePreprocessor
from src.data.dataset import split_dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Prepare invoice dataset for training"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing invoice files (PDFs/images)"
    )

    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth JSON file or directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )

    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply image enhancement"
    )

    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=[1280, 960],
        help="Target image size (width height)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PDF conversion"
    )

    return parser.parse_args()


def load_ground_truths(ground_truth_path: str) -> dict:
    """
    Load ground truth data

    Args:
        ground_truth_path: Path to JSON file or directory

    Returns:
        Dictionary mapping filename -> ground truth
    """
    gt_path = Path(ground_truth_path)

    if gt_path.is_file():
        # Single JSON file with all ground truths
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    elif gt_path.is_dir():
        # Directory with individual JSON files
        ground_truths = {}
        for json_file in gt_path.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Use filename without extension as key
                key = json_file.stem
                ground_truths[key] = data
        return ground_truths

    else:
        raise ValueError(f"Ground truth path not found: {ground_truth_path}")


def main():
    """Main function"""
    args = parse_args()

    print("="*60)
    print("CLOUDX INVOICE AI - DATA PREPARATION")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Enhancement: {args.enhance}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
    print("="*60 + "\n")

    # Load ground truths
    print("Loading ground truth data...")
    ground_truth_map = load_ground_truths(args.ground_truth)
    print(f"Loaded {len(ground_truth_map)} ground truth entries")

    # Get invoice files
    input_dir = Path(args.input_dir)
    invoice_files = []
    for ext in ['*.pdf', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
        invoice_files.extend(input_dir.glob(ext))

    print(f"Found {len(invoice_files)} invoice files")

    # Match files with ground truths
    document_paths = []
    ground_truths = []

    for invoice_file in invoice_files:
        # Try to match with ground truth
        key = invoice_file.stem

        if key in ground_truth_map:
            document_paths.append(str(invoice_file))
            ground_truths.append(ground_truth_map[key])
        else:
            print(f"Warning: No ground truth for {invoice_file.name}, skipping...")

    print(f"\nMatched {len(document_paths)} invoices with ground truth")

    if len(document_paths) == 0:
        print("Error: No invoices matched with ground truth data!")
        return

    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = InvoicePreprocessor(
        input_size=tuple(args.image_size),
        dpi=args.dpi
    )

    # Process dataset
    print("\nProcessing dataset...")
    metadata_file = preprocessor.process_dataset(
        document_paths=document_paths,
        ground_truths=ground_truths,
        output_dir=args.output_dir,
        metadata_file="full_dataset.jsonl",
        enhance=args.enhance
    )

    # Split dataset
    print("\nSplitting dataset...")
    train_path, val_path, test_path = split_dataset(
        metadata_file=metadata_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_dir=args.output_dir
    )

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    print(f"Test data: {test_path}")
    print("\nYou can now start training with:")
    print("  python train.py --config configs/train_config.yaml")
    print("="*60)


if __name__ == "__main__":
    main()
