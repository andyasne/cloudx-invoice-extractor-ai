"""
Cloudx Invoice AI - Inference Example
Example script showing how to use the trained model for inference
"""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import load_config, DonutInvoiceModel
from src.evaluation.metrics import token2json
from src.data.preprocessor import InvoicePreprocessor
from transformers import DonutProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run inference on a single invoice"
    )

    parser.add_argument(
        "--invoice",
        type=str,
        required=True,
        help="Path to invoice file (PDF or image)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output JSON (optional)"
    )

    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()

    print("="*60)
    print("CLOUDX INVOICE AI - INFERENCE")
    print("="*60)
    print(f"Invoice: {args.invoice}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Load processor
    print("Loading processor...")
    processor = DonutProcessor.from_pretrained(config["model"]["pretrained_model"])

    # Add special tokens
    special_tokens = []
    for field in config["task"]["fields"]:
        special_tokens.extend([f"<s_{field}>", f"</s_{field}>"])

    special_tokens.extend([
        config["task"]["task_start_token"],
        config["task"]["prompt_end_token"],
        "<sep>"
    ])

    processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Load model
    print("Loading model...")
    model = DonutInvoiceModel.load_from_checkpoint(
        args.checkpoint,
        config=config,
        processor=processor
    )
    model.eval()
    model.to(args.device)

    # Initialize preprocessor
    preprocessor = InvoicePreprocessor(
        input_size=tuple(config["model"]["image_size"]),
        max_length=config["model"]["max_length"]
    )

    # Load and preprocess invoice
    print("Loading invoice...")
    invoice_path = Path(args.invoice)
    ext = invoice_path.suffix.lower()

    if ext == '.pdf':
        # Convert PDF to image (first page)
        images = preprocessor.pdf_to_images(str(invoice_path))
        if not images:
            print("Error: Failed to convert PDF to image")
            return
        image = images[0]
    else:
        # Load image
        image = preprocessor.load_image(str(invoice_path))
        if image is None:
            print("Error: Failed to load image")
            return

    # Preprocess image
    print("Preprocessing image...")
    processed_image = preprocessor.preprocess_image(image)

    # Process with model
    print("Running inference...")
    pixel_values = processor(processed_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(args.device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_length=config["model"]["max_length"]
        )

    # Decode
    prediction_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    prediction_dict = token2json(prediction_text, processor)

    # Print results
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)

    for field, value in prediction_dict.items():
        print(f"{field:.<30} {value}")

    print("="*60)

    # Save to file if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(prediction_dict, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
