"""
Cloudx Invoice AI - API Runner
Simple script to run the API server
"""
import argparse
import os

import uvicorn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Cloudx Invoice AI API Server"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number (default: 8000)"
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
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Set environment variables for API
    os.environ["CONFIG_PATH"] = args.config
    os.environ["CHECKPOINT_PATH"] = args.checkpoint

    print("="*60)
    print("CLOUDX INVOICE AI - API SERVER")
    print("="*60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Workers: {args.workers}")
    print("="*60)
    print("\nStarting server...\n")

    # Run server
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
