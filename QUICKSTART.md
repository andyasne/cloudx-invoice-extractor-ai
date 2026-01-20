# Cloudx Invoice AI - Quick Start Guide

This guide will help you get started with training and deploying the Cloudx Invoice AI system.

## Table of Contents

1. [Setup](#setup)
2. [Prepare Your Data](#prepare-your-data)
3. [Train the Model](#train-the-model)
4. [Evaluate the Model](#evaluate-the-model)
5. [Deploy the API](#deploy-the-api)
6. [Test the API](#test-the-api)

## Setup

### Option 1: Local Setup

1. **Install Python 3.8+**

2. **Clone and setup**:
```bash
cd Donut
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Option 2: Docker Setup

```bash
docker-compose build
```

## Prepare Your Data

### Step 1: Organize Your Invoice Files

Place your invoice files (PDFs or images) in `data/raw/invoices/`:

```
data/raw/invoices/
├── invoice_001.pdf
├── invoice_002.pdf
└── invoice_003.jpg
```

### Step 2: Create Ground Truth Data

Create a JSON file `data/raw/ground_truth.json` with your invoice data:

```json
{
  "invoice_001": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "vendor_name": "Acme Corporation",
    "vendor_address": "123 Main Street, City, State 12345",
    "customer_name": "Cloudx Inc",
    "customer_address": "456 Business Ave, Tech City, ST 67890",
    "subtotal": "1250.00",
    "tax": "125.00",
    "total": "1375.00",
    "currency": "USD",
    "payment_terms": "Net 30"
  },
  "invoice_002": {
    "invoice_number": "INV-2024-002",
    ...
  }
}
```

### Step 3: Process the Data

```bash
# Local
python scripts/prepare_data.py \
    --input_dir data/raw/invoices \
    --ground_truth data/raw/ground_truth.json \
    --output_dir data/processed

# Docker
docker-compose run --rm preprocessing python scripts/prepare_data.py \
    --input_dir /app/data/raw/invoices \
    --ground_truth /app/data/raw/ground_truth.json \
    --output_dir /app/data/processed
```

This will:
- Convert PDFs to images
- Preprocess and normalize images
- Split into train/val/test sets
- Create metadata files

## Train the Model

### Step 1: Configure Training

Edit `configs/train_config.yaml`:

```yaml
training:
  batch_size: 4          # Adjust based on GPU memory
  max_epochs: 30
  learning_rate: 3e-5

hardware:
  gpus: 1                # Set to 0 for CPU
  precision: 16          # Mixed precision for faster training
```

### Step 2: Start Training

**Local (CPU)**:
```bash
python train.py --config configs/train_config.yaml
```

**Local (GPU)**:
```bash
python train.py --config configs/train_config.yaml --gpus 1
```

**Docker (CPU)**:
```bash
docker-compose up training
```

**Docker (GPU)**:
```bash
docker-compose --profile gpu up training-gpu
```

### Step 3: Monitor Training

Training logs are saved to `logs/`. View with TensorBoard:

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 in your browser.

### Step 4: Find Best Checkpoint

Checkpoints are saved to `models/checkpoints/`:
- `best_model.ckpt` - Best model based on validation loss
- `last.ckpt` - Latest checkpoint

## Evaluate the Model

Test your trained model:

```bash
python evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint models/checkpoints/best_model.ckpt \
    --output results/predictions.jsonl
```

This will:
- Run inference on test set
- Calculate accuracy metrics
- Save predictions to `results/predictions.jsonl`
- Save metrics to `results/metrics.json`

## Deploy the API

### Option 1: Local Deployment

```bash
python run_api.py \
    --checkpoint models/checkpoints/best_model.ckpt \
    --host 0.0.0.0 \
    --port 8000
```

### Option 2: Docker Deployment

```bash
docker-compose up -d api
```

The API will be available at http://localhost:8000

## Test the API

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Process Single Invoice

**Using curl**:
```bash
curl -X POST "http://localhost:8000/api/v1/extract-invoice" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@data/raw/invoices/invoice_001.pdf"
```

**Using Python**:
```python
import requests

url = "http://localhost:8000/api/v1/extract-invoice"
files = {"file": open("data/raw/invoices/invoice_001.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

Expected response:
```json
{
  "status": "success",
  "invoice_data": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "vendor_name": "Acme Corporation",
    "total": "1375.00",
    ...
  }
}
```

### Process Multiple Invoices

```bash
curl -X POST "http://localhost:8000/api/v1/extract-invoice-batch" \
    -F "files=@invoice1.pdf" \
    -F "files=@invoice2.pdf" \
    -F "files=@invoice3.pdf"
```

## Next Steps

### Improve Accuracy

1. **Add more training data**: More examples improve performance
2. **Adjust hyperparameters**: Experiment with learning rate, batch size
3. **Fine-tune longer**: Train for more epochs
4. **Data augmentation**: Enable in config for better generalization

### Production Deployment

1. **Add authentication**: Implement API key or OAuth
2. **Load balancing**: Use nginx or similar
3. **Monitoring**: Add logging, metrics, alerts
4. **Rate limiting**: Prevent abuse
5. **Caching**: Cache common requests

### Integration

Integrate the API into your existing systems:

```python
# Example integration
import requests

def process_invoice(invoice_path):
    url = "http://your-api-server:8000/api/v1/extract-invoice"
    with open(invoice_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    if response.json()["status"] == "success":
        return response.json()["invoice_data"]
    else:
        raise Exception(f"Failed to process invoice: {response.json()['error']}")

# Use it
invoice_data = process_invoice("invoice.pdf")
print(f"Invoice #: {invoice_data['invoice_number']}")
print(f"Total: ${invoice_data['total']}")
```

## Troubleshooting

### Training is slow
- Reduce batch size
- Use GPU instead of CPU
- Enable mixed precision (`precision: 16`)

### Out of memory
- Reduce `batch_size` in config
- Reduce `image_size` in config
- Use `accumulate_grad_batches` for effective larger batches

### Poor accuracy
- Add more training data
- Train for more epochs
- Check data quality
- Verify ground truth alignment

### API not starting
- Check checkpoint path is correct
- Verify model loaded successfully
- Check logs for errors

## Support

For issues:
1. Check logs in `logs/`
2. Review error messages
3. Consult README.md
4. Contact Cloudx AI team

## Resources

- [Full Documentation](README.md)
- [Configuration Guide](configs/train_config.yaml)
- [Donut Paper](https://arxiv.org/abs/2111.15664)
- [Donut GitHub](https://github.com/clovaai/donut)
