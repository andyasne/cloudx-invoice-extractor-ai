"""
Cloudx Invoice AI - Training Script
Main entry point for model training
"""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.trainer import InvoiceTrainer, load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Cloudx Invoice AI model using Donut"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (overrides config)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.resume:
        config["resume_from_checkpoint"] = args.resume

    if args.gpus is not None:
        config["hardware"]["gpus"] = args.gpus

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if args.epochs is not None:
        config["training"]["max_epochs"] = args.epochs

    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    # Print configuration
    print("\n" + "="*50)
    print("CLOUDX INVOICE AI - TRAINING")
    print("="*50)
    print(f"Model: {config['model']['pretrained_model']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['max_epochs']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"GPUs: {config['hardware']['gpus']}")
    print(f"Precision: {config['hardware']['precision']}")
    print("="*50 + "\n")

    # Initialize trainer
    print("Initializing trainer...")
    trainer_obj = InvoiceTrainer(config)

    # Start training
    print("Starting training...")
    trainer = trainer_obj.train()

    # Run testing if test data available
    if config["data"].get("test_metadata"):
        print("\nRunning evaluation on test set...")
        trainer_obj.test(trainer)

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Checkpoints saved to: {config['checkpoint']['save_dir']}")
    print(f"Logs saved to: {config['logging']['log_dir']}")


if __name__ == "__main__":
    main()
