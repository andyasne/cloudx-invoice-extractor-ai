# Cloudx Invoice AI

AI-powered invoice processing system using Donut (Document Understanding Transformer) for extracting structured data from invoice documents (PDFs and images).

## Overview

Cloudx Invoice AI is an end-to-end solution for training and deploying AI models that extract text and structured information from invoices. Built on the Donut architecture, it provides:

- Document understanding without OCR
- Support for millions of invoice documents
- REST API for invoice processing
- High accuracy text extraction
- Structured JSON output

## Features

- **Multi-format Support**: Process PDFs and images (PNG, JPG, TIFF, etc.)
- **Transformer-based**: Uses state-of-the-art Donut architecture
- **Scalable**: Handles millions of training samples
- **Production-ready**: FastAPI server with Docker support
- **Comprehensive**: Training, evaluation, and inference pipelines
- **Flexible**: Configurable for different invoice formats

## Project Structure

```
Cloudx Invoice AI/
├── src/
│   ├── data/
│   │   ├── preprocessor.py      # Data preprocessing
│   │   └── dataset.py           # Dataset loaders
│   ├── training/
│   │   └── trainer.py           # Training module
│   ├── evaluation/
│   │   └── metrics.py           # Evaluation metrics
│   ├── api/
│   │   └── app.py               # FastAPI server
│   └── utils/                    # Utilities
├── configs/
│   └── train_config.yaml        # Training configuration
├── data/
│   ├── raw/                     # Raw invoice files
│   ├── processed/               # Processed data
│   ├── train/                   # Training split
│   ├── val/                     # Validation split
│   └── test/                    # Test split
├── models/
│   └── checkpoints/             # Model checkpoints
├── logs/                        # Training logs
├── donut_base/                  # Donut base repository
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── run_api.py                   # API server runner
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose config
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.11+
- CUDA 11.x (for GPU training)
- Docker (optional)

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Donut
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch (adjust for your CUDA version):
```bash
# CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

5. Install package:
```bash
pip install -e .
```

### Docker Installation

```bash
# Build images
docker-compose build

# For GPU support
docker-compose --profile gpu build training-gpu
```

## Quick Start

### 1. Prepare Data

Place your invoice files and ground truth data in the `data/raw/` directory.

```python
from src.data.preprocessor import InvoicePreprocessor

# Initialize preprocessor
preprocessor = InvoicePreprocessor()

# Process dataset
metadata_file = preprocessor.process_dataset(
    document_paths=["path/to/invoice1.pdf", "path/to/invoice2.pdf"],
    ground_truths=[{"invoice_number": "INV-001", ...}, {...}],
    output_dir="data/processed"
)
```

### 2. Train Model

```bash
# Local training
python train.py --config configs/train_config.yaml

# Docker training
docker-compose up training

# GPU training
docker-compose --profile gpu up training-gpu
```

### 3. Evaluate Model

```bash
python evaluate.py --checkpoint models/checkpoints/best_model.ckpt
```

### 4. Run API Server

```bash
# Local
python run_api.py --checkpoint models/checkpoints/best_model.ckpt

# Docker
docker-compose up api
```

## Usage

### Training

Basic training:
```bash
python train.py --config configs/train_config.yaml
```

With custom parameters:
```bash
python train.py \
    --config configs/train_config.yaml \
    --gpus 2 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 5e-5
```

Resume from checkpoint:
```bash
python train.py \
    --config configs/train_config.yaml \
    --resume models/checkpoints/last.ckpt
```

### Evaluation

```bash
python evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint models/checkpoints/best_model.ckpt \
    --test_data data/processed/test_metadata.jsonl \
    --output results/predictions.jsonl
```

### API Usage

Start the server:
```bash
python run_api.py --checkpoint models/checkpoints/best_model.ckpt --port 8000
```

Process single invoice:
```bash
curl -X POST "http://localhost:8000/api/v1/extract-invoice" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@invoice.pdf"
```

Response:
```json
{
  "status": "success",
  "invoice_data": {
    "invoice_number": "INV-001",
    "invoice_date": "2024-01-15",
    "vendor_name": "Acme Corp",
    "total": "1500.00",
    ...
  }
}
```

Process multiple invoices:
```bash
curl -X POST "http://localhost:8000/api/v1/extract-invoice-batch" \
    -F "files=@invoice1.pdf" \
    -F "files=@invoice2.pdf"
```

Health check:
```bash
curl http://localhost:8000/health
```

## Configuration

Edit `configs/train_config.yaml` to customize:

- **Model settings**: Image size, max length, pretrained model
- **Training parameters**: Batch size, learning rate, epochs
- **Data paths**: Training/validation/test datasets
- **Hardware**: GPU count, precision, strategy
- **Invoice fields**: Fields to extract

Example configuration:
```yaml
model:
  pretrained_model: "naver-clova-ix/donut-base"
  image_size: [1280, 960]
  max_length: 768

training:
  batch_size: 4
  learning_rate: 3e-5
  max_epochs: 30

task:
  fields:
    - invoice_number
    - invoice_date
    - vendor_name
    - total
```

## Data Format

### Input Data

Invoice files in `data/raw/`:
- PDFs: `.pdf`
- Images: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`

Ground truth JSON format:
```json
{
  "invoice_number": "INV-001",
  "invoice_date": "2024-01-15",
  "vendor_name": "Acme Corp",
  "vendor_address": "123 Main St",
  "total": "1500.00",
  "currency": "USD"
}
```

### Processed Data

Metadata JSONL format (one sample per line):
```json
{"image_path": "data/processed/images/invoice_001_page0.png", "ground_truth": {...}, "original_path": "data/raw/invoice_001.pdf", "page_number": 0}
```

## API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint

Returns:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### `POST /api/v1/extract-invoice`
Extract data from single invoice

Request:
- `file`: Invoice file (multipart/form-data)

Response:
```json
{
  "status": "success",
  "invoice_data": {...},
  "confidence": null
}
```

### `POST /api/v1/extract-invoice-batch`
Extract data from multiple invoices

Request:
- `files`: List of invoice files

Response:
```json
{
  "results": [
    {
      "filename": "invoice1.pdf",
      "status": "success",
      "invoice_data": {...}
    }
  ]
}
```

## Docker Commands

```bash
# Build all services
docker-compose build

# Run training
docker-compose up training

# Run API server
docker-compose up api

# Run with GPU
docker-compose --profile gpu up training-gpu

# Stop all services
docker-compose down

# View logs
docker-compose logs -f api
```

## Performance

Expected metrics on well-formatted invoices:
- **Exact Match Accuracy**: 85-95%
- **Field Accuracy**: 90-98%
- **Inference Speed**: 1-2 seconds per invoice (CPU), 0.3-0.5s (GPU)

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 2
  accumulate_grad_batches: 4
```

### CUDA Out of Memory

Use mixed precision:
```yaml
hardware:
  precision: 16
```

### Model Not Loading

Check checkpoint path:
```bash
ls -l models/checkpoints/
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ train.py evaluate.py run_api.py
```

Lint code:
```bash
flake8 src/
```

## Contributing

Cloudx internal project. Contact the AI team for contributions.

## License

Proprietary - Cloudx

## Credits

Built on [Donut](https://github.com/clovaai/donut) by Clova AI Research.

## Support

For issues and questions, contact:
- Email: ai-team@cloudx.com
- Internal Slack: #cloudx-invoice-ai
