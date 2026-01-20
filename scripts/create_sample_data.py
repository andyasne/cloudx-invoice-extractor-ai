"""
Cloudx Invoice AI - Sample Data Generator
Creates sample invoices and ground truth for testing the training pipeline
"""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime, timedelta


def generate_sample_invoice_image(invoice_data: dict, output_path: str):
    """Generate a simple invoice image from data"""
    # Create image
    width, height = 800, 1000
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Try to use a font, fallback to default
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    y_position = 50

    # Title
    draw.text((width//2 - 50, y_position), "INVOICE", fill='black', font=font_large)
    y_position += 60

    # Vendor info
    draw.text((50, y_position), f"From: {invoice_data['vendor_name']}", fill='black', font=font_medium)
    y_position += 25
    draw.text((50, y_position), invoice_data['vendor_address'], fill='black', font=font_small)
    y_position += 40

    # Customer info
    draw.text((50, y_position), f"To: {invoice_data['customer_name']}", fill='black', font=font_medium)
    y_position += 25
    draw.text((50, y_position), invoice_data['customer_address'], fill='black', font=font_small)
    y_position += 60

    # Invoice details
    draw.text((50, y_position), f"Invoice #: {invoice_data['invoice_number']}", fill='black', font=font_medium)
    y_position += 30
    draw.text((50, y_position), f"Date: {invoice_data['invoice_date']}", fill='black', font=font_small)
    y_position += 25
    draw.text((50, y_position), f"Due Date: {invoice_data['due_date']}", fill='black', font=font_small)
    y_position += 25
    draw.text((50, y_position), f"Terms: {invoice_data['payment_terms']}", fill='black', font=font_small)
    y_position += 60

    # Line separator
    draw.line([(50, y_position), (width-50, y_position)], fill='black', width=2)
    y_position += 40

    # Amounts
    draw.text((50, y_position), "Description", fill='black', font=font_medium)
    draw.text((width-200, y_position), "Amount", fill='black', font=font_medium)
    y_position += 30

    draw.text((50, y_position), "Services Rendered", fill='black', font=font_small)
    draw.text((width-200, y_position), f"{invoice_data['currency']} {invoice_data['subtotal']}", fill='black', font=font_small)
    y_position += 60

    # Totals
    draw.line([(50, y_position), (width-50, y_position)], fill='black', width=1)
    y_position += 30

    draw.text((width-400, y_position), "Subtotal:", fill='black', font=font_medium)
    draw.text((width-200, y_position), f"{invoice_data['currency']} {invoice_data['subtotal']}", fill='black', font=font_medium)
    y_position += 30

    draw.text((width-400, y_position), "Tax:", fill='black', font=font_medium)
    draw.text((width-200, y_position), f"{invoice_data['currency']} {invoice_data['tax']}", fill='black', font=font_medium)
    y_position += 30

    draw.line([(width-400, y_position), (width-50, y_position)], fill='black', width=2)
    y_position += 30

    draw.text((width-400, y_position), "Total:", fill='black', font=font_large)
    draw.text((width-200, y_position), f"{invoice_data['currency']} {invoice_data['total']}", fill='black', font=font_large)

    # Save image
    img.save(output_path)


def generate_sample_dataset(num_samples: int = 50, output_dir: str = "data/raw"):
    """Generate sample invoice dataset"""
    output_dir = Path(output_dir)
    invoices_dir = output_dir / "invoices"
    invoices_dir.mkdir(parents=True, exist_ok=True)

    vendors = [
        ("Acme Corporation", "123 Main Street, New York, NY 10001"),
        ("Tech Solutions Ltd", "456 Silicon Valley, San Jose, CA 95110"),
        ("Global Services Inc", "789 Business Ave, Chicago, IL 60601"),
        ("Innovation Group", "321 Tech Park, Austin, TX 78701"),
        ("Enterprise Systems", "654 Corporate Blvd, Seattle, WA 98101")
    ]

    customers = [
        ("Cloudx Inc", "456 Business Ave, Tech City, ST 67890"),
        ("Digital Solutions", "789 Innovation Drive, Metro, ST 12345"),
        ("Future Tech Corp", "321 Modern Street, Urban, ST 54321"),
    ]

    ground_truth = {}

    print(f"Generating {num_samples} sample invoices...")

    for i in range(num_samples):
        # Generate random invoice data
        vendor = random.choice(vendors)
        customer = random.choice(customers)

        base_date = datetime(2024, 1, 1) + timedelta(days=i)
        due_date = base_date + timedelta(days=30)

        subtotal = round(random.uniform(500, 5000), 2)
        tax_rate = 0.1
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)

        invoice_data = {
            "invoice_number": f"INV-2024-{i+1:04d}",
            "invoice_date": base_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "vendor_name": vendor[0],
            "vendor_address": vendor[1],
            "customer_name": customer[0],
            "customer_address": customer[1],
            "subtotal": f"{subtotal:.2f}",
            "tax": f"{tax:.2f}",
            "total": f"{total:.2f}",
            "currency": "USD",
            "payment_terms": "Net 30"
        }

        # Generate invoice image
        invoice_filename = f"invoice_{i+1:04d}.png"
        invoice_path = invoices_dir / invoice_filename
        generate_sample_invoice_image(invoice_data, str(invoice_path))

        # Add to ground truth
        ground_truth[f"invoice_{i+1:04d}"] = invoice_data

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} invoices...")

    # Save ground truth
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✓ Generated {num_samples} sample invoices")
    print(f"✓ Invoice images saved to: {invoices_dir}")
    print(f"✓ Ground truth saved to: {gt_path}")
    print(f"\nNext steps:")
    print(f"1. Run preprocessing:")
    print(f"   python scripts/prepare_data.py \\")
    print(f"       --input_dir {invoices_dir} \\")
    print(f"       --ground_truth {gt_path} \\")
    print(f"       --output_dir data/processed")
    print(f"\n2. Start training:")
    print(f"   python train.py --config configs/train_config.yaml")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample invoice dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of sample invoices to generate")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")

    args = parser.parse_args()

    generate_sample_dataset(args.num_samples, args.output_dir)
