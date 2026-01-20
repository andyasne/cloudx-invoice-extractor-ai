"""
Cloudx Invoice AI - Single Invoice Training Helper
Helps set up training with a single invoice + generated samples
"""
import json
import argparse
from pathlib import Path
import shutil


def setup_single_invoice_training(
    invoice_path: str,
    num_samples: int = 99
):
    """
    Set up training with one real invoice + generated samples

    Args:
        invoice_path: Path to your invoice file
        num_samples: Number of sample invoices to generate (default: 99)
    """
    print("="*60)
    print("CLOUDX INVOICE AI - SINGLE INVOICE TRAINING SETUP")
    print("="*60)

    invoice_path = Path(invoice_path)

    if not invoice_path.exists():
        print(f"Error: Invoice file not found: {invoice_path}")
        return

    print(f"\nYour invoice: {invoice_path}")
    print(f"Generating {num_samples} sample invoices to supplement training...")

    # Step 1: Get ground truth for user's invoice
    print("\n" + "="*60)
    print("STEP 1: Create Ground Truth for Your Invoice")
    print("="*60)

    invoice_name = invoice_path.stem

    print(f"\nPlease provide the following information from your invoice:")
    print("(Press Enter to skip optional fields)")

    ground_truth = {}

    # Required fields
    ground_truth["invoice_number"] = input("Invoice Number: ").strip()
    ground_truth["invoice_date"] = input("Invoice Date (YYYY-MM-DD): ").strip()
    ground_truth["vendor_name"] = input("Vendor/Seller Name: ").strip()
    ground_truth["total"] = input("Total Amount (numbers only, e.g., 1500.00): ").strip()

    # Optional fields
    due_date = input("Due Date (YYYY-MM-DD) [Optional]: ").strip()
    if due_date:
        ground_truth["due_date"] = due_date

    vendor_address = input("Vendor Address [Optional]: ").strip()
    if vendor_address:
        ground_truth["vendor_address"] = vendor_address

    customer_name = input("Customer Name [Optional]: ").strip()
    if customer_name:
        ground_truth["customer_name"] = customer_name

    customer_address = input("Customer Address [Optional]: ").strip()
    if customer_address:
        ground_truth["customer_address"] = customer_address

    subtotal = input("Subtotal [Optional]: ").strip()
    if subtotal:
        ground_truth["subtotal"] = subtotal

    tax = input("Tax Amount [Optional]: ").strip()
    if tax:
        ground_truth["tax"] = tax

    currency = input("Currency (e.g., USD, EUR) [Optional]: ").strip()
    if currency:
        ground_truth["currency"] = currency
    else:
        ground_truth["currency"] = "USD"

    payment_terms = input("Payment Terms (e.g., Net 30) [Optional]: ").strip()
    if payment_terms:
        ground_truth["payment_terms"] = payment_terms

    print("\n" + "="*60)
    print("STEP 2: Generate Sample Invoices")
    print("="*60)

    # Generate sample data
    import os
    os.system(f"python scripts/create_sample_data.py --num_samples {num_samples}")

    print("\n" + "="*60)
    print("STEP 3: Copy Your Invoice and Create Combined Ground Truth")
    print("="*60)

    # Copy user's invoice to invoices folder
    invoices_dir = Path("data/raw/invoices")
    invoices_dir.mkdir(parents=True, exist_ok=True)

    dest_path = invoices_dir / f"my_invoice.{invoice_path.suffix}"
    shutil.copy(invoice_path, dest_path)
    print(f"✓ Copied invoice to: {dest_path}")

    # Load generated ground truth
    with open("data/raw/ground_truth.json", 'r') as f:
        sample_gt = json.load(f)

    # Add user's ground truth
    sample_gt["my_invoice"] = ground_truth

    # Save combined
    combined_path = Path("data/raw/ground_truth_combined.json")
    with open(combined_path, 'w') as f:
        json.dump(sample_gt, f, indent=2)

    print(f"✓ Created combined ground truth: {combined_path}")
    print(f"✓ Total invoices: {len(sample_gt)} ({num_samples} samples + 1 real)")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Preprocess the data:")
    print("   python scripts/prepare_data.py \\")
    print("       --input_dir data/raw/invoices \\")
    print("       --ground_truth data/raw/ground_truth_combined.json \\")
    print("       --output_dir data/processed")
    print("\n2. Start training:")
    print("   python train.py --config configs/train_config.yaml --epochs 10")
    print("\n3. After training, test with your invoice:")
    print("   python scripts/inference_example.py \\")
    print(f"       --invoice {dest_path} \\")
    print("       --checkpoint models/checkpoints/best_model.ckpt")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set up training with a single invoice"
    )
    parser.add_argument(
        "--invoice",
        type=str,
        required=True,
        help="Path to your invoice file (PDF or image)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=99,
        help="Number of sample invoices to generate (default: 99)"
    )

    args = parser.parse_args()

    setup_single_invoice_training(args.invoice, args.num_samples)
