"""
Cloudx Invoice AI - Evaluation Script
Evaluate trained model on test dataset
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.trainer import load_config, DonutInvoiceModel
from evaluation.metrics import InvoiceMetrics, token2json
from transformers import DonutProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate Cloudx Invoice AI model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test metadata file (overrides config)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions.jsonl",
        help="Path to save predictions"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    return parser.parse_args()


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override test data if provided
    if args.test_data:
        test_metadata = args.test_data
    else:
        test_metadata = config["data"]["test_metadata"]

    print(f"Test data: {test_metadata}")

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
    print(f"Loading model from {args.checkpoint}")
    model = DonutInvoiceModel.load_from_checkpoint(
        args.checkpoint,
        config=config,
        processor=processor
    )
    model.eval()
    model.to(args.device)

    # Load test data
    print("Loading test data...")
    test_samples = []
    with open(test_metadata, 'r', encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))

    print(f"Loaded {len(test_samples)} test samples")

    # Initialize metrics
    metrics_calculator = InvoiceMetrics(config["task"]["fields"])

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run inference
    print("\nRunning inference...")
    predictions = []

    with torch.no_grad():
        for sample in tqdm(test_samples):
            # Load image
            image = Image.open(sample["image_path"]).convert("RGB")

            # Process image
            pixel_values = processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(args.device)

            # Generate
            outputs = model.generate(
                pixel_values,
                max_length=config["model"]["max_length"]
            )

            # Decode
            prediction_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Convert to JSON
            prediction_dict = token2json(prediction_text, processor)

            # Get ground truth
            ground_truth = sample["ground_truth"]

            # Update metrics
            metrics_calculator.update(prediction_dict, ground_truth)

            # Save prediction
            predictions.append({
                "image_path": sample["image_path"],
                "prediction": prediction_dict,
                "ground_truth": ground_truth,
                "raw_prediction": prediction_text
            })

    # Save predictions
    print(f"\nSaving predictions to {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    # Print metrics
    metrics_calculator.print_metrics()

    # Save metrics
    metrics_file = output_path.parent / "metrics.json"
    metrics = metrics_calculator.compute_all_metrics()
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
