# Cloudx Invoice AI - Step-by-Step Training Guide

Complete guide to train your invoice AI model from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Environment Setup](#step-1-environment-setup)
3. [Step 2: Prepare Your Invoice Data](#step-2-prepare-your-invoice-data)
4. [Step 3: Create Ground Truth Annotations](#step-3-create-ground-truth-annotations)
5. [Step 4: Preprocess the Dataset](#step-4-preprocess-the-dataset)
6. [Step 5: Configure Training Parameters](#step-5-configure-training-parameters)
7. [Step 6: Start Training](#step-6-start-training)
8. [Step 7: Monitor Training Progress](#step-7-monitor-training-progress)
9. [Step 8: Evaluate the Model](#step-8-evaluate-the-model)
10. [Step 9: Deploy the Model](#step-9-deploy-the-model)
11. [Step 10: Test the API](#step-10-test-the-api)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Minimum (CPU Training)**:
- CPU: 4+ cores
- RAM: 16GB
- Storage: 50GB free space
- Training time: Very slow (days/weeks)

**Recommended (GPU Training)**:
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060, V100, A100)
- CPU: 8+ cores
- RAM: 32GB
- Storage: 100GB free space
- Training time: Hours to days

### Software Requirements

- Python 3.8 or higher
- CUDA 11.x (for GPU training)
- Git
- Docker (optional, recommended)

---

## Step 1: Environment Setup

### Option A: Local Setup (Recommended for Development)

#### 1.1. Clone the Repository

```bash
cd C:/w/cloudex
git clone https://github.com/andyasne/Donut.git
cd Donut
```

#### 1.2. Create Virtual Environment

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 1.3. Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OR Install PyTorch (GPU version - CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install package in development mode
pip install -e .
```

#### 1.4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from transformers import DonutProcessor; print('Transformers: OK')"
```

### Option B: Docker Setup (Recommended for Production)

```bash
# Build training image (CPU)
docker-compose build training

# Build training image (GPU)
docker-compose --profile gpu build training-gpu
```

---

## Step 2: Prepare Your Invoice Data

### 2.1. Collect Invoice Documents

Gather your invoice files in PDF or image format.

**Supported formats**:
- PDF (.pdf)
- Images (.png, .jpg, .jpeg, .tiff, .bmp)

### 2.2. Organize Files

Create directory structure:

```bash
mkdir -p data/raw/invoices
```

Place all invoice files in `data/raw/invoices/`:

```
data/raw/invoices/
├── invoice_001.pdf
├── invoice_002.pdf
├── invoice_003.jpg
├── invoice_004.png
└── ... (all your invoices)
```

**Important**: Name files consistently (e.g., invoice_001, invoice_002, etc.)

### 2.3. Verify Files

```bash
# Count invoices
ls data/raw/invoices/ | wc -l

# Check file types
ls data/raw/invoices/*.pdf | wc -l
ls data/raw/invoices/*.{png,jpg} | wc -l
```

---

## Step 3: Create Ground Truth Annotations

### 3.1. Understand Invoice Fields

The system can extract these fields (configurable in `configs/train_config.yaml`):

- `invoice_number`: Invoice number/ID
- `invoice_date`: Invoice issue date
- `due_date`: Payment due date
- `vendor_name`: Seller/vendor name
- `vendor_address`: Vendor address
- `customer_name`: Buyer/customer name
- `customer_address`: Customer address
- `subtotal`: Subtotal amount
- `tax`: Tax amount
- `total`: Total amount
- `currency`: Currency (USD, EUR, etc.)
- `payment_terms`: Payment terms (Net 30, etc.)
- `line_items`: Individual items (optional)

### 3.2. Create Ground Truth JSON

Create file: `data/raw/ground_truth.json`

**Format**: Single JSON object with invoice filenames (without extension) as keys

```json
{
  "invoice_001": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
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
    "invoice_date": "2024-01-16",
    "due_date": "2024-02-16",
    "vendor_name": "Tech Solutions Ltd",
    "vendor_address": "789 Innovation Blvd, Tech Park, TP 11111",
    "customer_name": "Cloudx Inc",
    "customer_address": "456 Business Ave, Tech City, ST 67890",
    "subtotal": "2500.00",
    "tax": "250.00",
    "total": "2750.00",
    "currency": "USD",
    "payment_terms": "Net 30"
  },
  "invoice_003": {
    ...
  }
}
```

### 3.3. Alternative: Individual JSON Files

Instead of one large file, you can create individual JSON files:

```
data/raw/ground_truth/
├── invoice_001.json
├── invoice_002.json
├── invoice_003.json
└── ...
```

Each file contains the ground truth for one invoice:

```json
{
  "invoice_number": "INV-2024-001",
  "invoice_date": "2024-01-15",
  ...
}
```

### 3.4. Annotation Tips

1. **Be Consistent**: Use same date format (YYYY-MM-DD)
2. **No Extra Spaces**: Trim whitespace from values
3. **Standardize**: Use consistent formatting for addresses, names
4. **Exact Match**: Text should match invoice exactly
5. **Missing Fields**: Omit fields not present on invoice

### 3.5. Validate Ground Truth

```bash
# Verify JSON syntax
python -c "import json; json.load(open('data/raw/ground_truth.json'))"

# Count entries
python -c "import json; data = json.load(open('data/raw/ground_truth.json')); print(f'Entries: {len(data)}')"
```

---

## Step 4: Preprocess the Dataset

### 4.1. Run Preprocessing Script

**Basic usage**:
```bash
python scripts/prepare_data.py \
    --input_dir data/raw/invoices \
    --ground_truth data/raw/ground_truth.json \
    --output_dir data/processed
```

**With options**:
```bash
python scripts/prepare_data.py \
    --input_dir data/raw/invoices \
    --ground_truth data/raw/ground_truth.json \
    --output_dir data/processed \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --enhance \
    --image_size 1280 960 \
    --dpi 150
```

**Parameters explained**:
- `--input_dir`: Directory with invoice files
- `--ground_truth`: Path to ground truth JSON
- `--output_dir`: Where to save processed data
- `--train_ratio`: Training set percentage (0.8 = 80%)
- `--val_ratio`: Validation set percentage (0.1 = 10%)
- `--test_ratio`: Test set percentage (0.1 = 10%)
- `--enhance`: Apply image enhancement (slower but better quality)
- `--image_size`: Target image size (width height in pixels)
- `--dpi`: DPI for PDF conversion (higher = better quality, slower)

### 4.2. Verify Output

Check created files:

```bash
# List processed data
ls data/processed/

# Check metadata files
ls data/processed/*.jsonl

# Count samples
wc -l data/processed/train_metadata.jsonl
wc -l data/processed/val_metadata.jsonl
wc -l data/processed/test_metadata.jsonl

# View sample
head -1 data/processed/train_metadata.jsonl | python -m json.tool
```

Expected output structure:
```
data/processed/
├── images/                      # Processed invoice images
│   ├── invoice_001_page0.png
│   ├── invoice_002_page0.png
│   └── ...
├── full_dataset.jsonl          # Complete dataset
├── train_metadata.jsonl        # Training split
├── val_metadata.jsonl          # Validation split
└── test_metadata.jsonl         # Test split
```

---

## Step 5: Configure Training Parameters

### 5.1. Edit Configuration File

Open `configs/train_config.yaml` in your editor.

### 5.2. Key Settings to Adjust

#### Model Settings
```yaml
model:
  pretrained_model: "naver-clova-ix/donut-base"  # Base model
  image_size: [1280, 960]                        # Match preprocessing
  max_length: 768                                # Max sequence length
```

#### Training Settings
```yaml
training:
  batch_size: 4              # Reduce if out of memory (try 2 or 1)
  num_workers: 4             # CPU cores for data loading
  max_epochs: 30             # Training epochs (try 10-50)
  learning_rate: 3e-5        # Learning rate (try 1e-5 to 5e-5)
  weight_decay: 0.01         # Regularization
  warmup_steps: 300          # Learning rate warmup
  gradient_clip_val: 1.0     # Gradient clipping
  accumulate_grad_batches: 2 # Effective batch size multiplier
```

**Effective batch size** = batch_size × accumulate_grad_batches × num_gpus
- Example: 4 × 2 × 1 = 8

#### Hardware Settings
```yaml
hardware:
  gpus: 1        # Number of GPUs (0 for CPU)
  precision: 16  # 16 for mixed precision (faster), 32 for full precision
  strategy: null # null for single GPU, "ddp" for multi-GPU
```

#### Data Paths
```yaml
data:
  train_metadata: "data/processed/train_metadata.jsonl"
  val_metadata: "data/processed/val_metadata.jsonl"
  test_metadata: "data/processed/test_metadata.jsonl"
```

#### Invoice Fields
```yaml
task:
  fields:
    - invoice_number
    - invoice_date
    - due_date
    - vendor_name
    - vendor_address
    - customer_name
    - customer_address
    - subtotal
    - tax
    - total
    - currency
    - payment_terms
```

**Important**: Fields must match your ground truth JSON keys!

### 5.3. Memory Optimization

If you get "Out of Memory" errors:

```yaml
training:
  batch_size: 2              # Reduce from 4 to 2
  accumulate_grad_batches: 4 # Increase to maintain effective batch size

hardware:
  precision: 16              # Use mixed precision
```

---

## Step 6: Start Training

### 6.1. Training Commands

#### Local Training (CPU)
```bash
python train.py --config configs/train_config.yaml --gpus 0
```

#### Local Training (GPU)
```bash
python train.py --config configs/train_config.yaml --gpus 1
```

#### Docker Training (CPU)
```bash
docker-compose up training
```

#### Docker Training (GPU)
```bash
docker-compose --profile gpu up training-gpu
```

### 6.2. Override Config Parameters

You can override config settings via command line:

```bash
python train.py \
    --config configs/train_config.yaml \
    --gpus 1 \
    --batch_size 8 \
    --epochs 50 \
    --learning_rate 5e-5
```

### 6.3. Resume Training

If training stops, resume from last checkpoint:

```bash
python train.py \
    --config configs/train_config.yaml \
    --resume models/checkpoints/last.ckpt
```

### 6.4. What to Expect

Training will output:

```
=================================================
CLOUDX INVOICE AI - TRAINING
==================================================
Model: naver-clova-ix/donut-base
Batch size: 4
Epochs: 30
Learning rate: 3e-05
GPUs: 1
Precision: 16
==================================================

Initializing trainer...
Loaded 800 samples for train split
Loaded 100 samples for val split
Starting training...

Epoch 0:  10%|█         | 20/200 [00:45<06:45,  2.25s/it, loss=2.34, v_num=0]
...
```

### 6.5. Training Time Estimates

**CPU Training**:
- Small dataset (100-1000 invoices): Hours to days
- Large dataset (10000+ invoices): Days to weeks

**GPU Training (single GPU)**:
- Small dataset (100-1000 invoices): 1-4 hours
- Medium dataset (1000-10000 invoices): 4-24 hours
- Large dataset (10000+ invoices): 1-5 days

**Factors affecting speed**:
- Dataset size
- Image size
- Batch size
- GPU memory
- Number of epochs

---

## Step 7: Monitor Training Progress

### 7.1. Watch Console Output

Monitor training in real-time:

```
Epoch 5:  45%|████▌     | 90/200 [03:22<04:07,  2.25s/it, loss=1.42, v_num=0]
```

Key metrics:
- `loss`: Training loss (should decrease)
- `v_num`: Version number
- Time per iteration

### 7.2. View TensorBoard Logs

Start TensorBoard:

```bash
tensorboard --logdir logs/
```

Open in browser: http://localhost:6006

View:
- Training loss curve
- Validation loss curve
- Learning rate schedule
- Sample predictions

### 7.3. Check Saved Checkpoints

```bash
# List checkpoints
ls -lh models/checkpoints/

# Should see:
# donut-invoice-epoch=00-val_loss=1.234.ckpt
# donut-invoice-epoch=01-val_loss=1.123.ckpt
# last.ckpt
```

### 7.4. Early Stopping

Training automatically stops if validation loss doesn't improve for 5 epochs (configurable in `train_config.yaml`).

```yaml
training:
  early_stopping:
    patience: 5
    monitor: "val_loss"
    mode: "min"
```

---

## Step 8: Evaluate the Model

### 8.1. Run Evaluation

Evaluate on test set:

```bash
python evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint models/checkpoints/best_model.ckpt \
    --output results/predictions.jsonl
```

Or specify checkpoint manually:

```bash
python evaluate.py \
    --config configs/train_config.yaml \
    --checkpoint models/checkpoints/donut-invoice-epoch=10-val_loss=0.234.ckpt \
    --test_data data/processed/test_metadata.jsonl \
    --output results/test_predictions.jsonl
```

### 8.2. Review Metrics

Evaluation will print:

```
============================================================
CLOUDX INVOICE AI - EVALUATION METRICS
============================================================
Total Samples: 100

Exact Match Accuracy: 0.8750
Average Field Accuracy: 0.9420

------------------------------------------------------------
Per-Field Accuracy:
------------------------------------------------------------
  invoice_number.................... 0.9800
  invoice_date...................... 0.9600
  vendor_name....................... 0.9400
  total............................. 0.9200
  ...

------------------------------------------------------------
Per-Field Levenshtein Similarity:
------------------------------------------------------------
  invoice_number.................... 0.9950
  invoice_date...................... 0.9850
  vendor_name....................... 0.9650
  total............................. 0.9500
  ...
============================================================
```

**Metrics explained**:
- **Exact Match Accuracy**: Percentage of invoices with all fields correct
- **Average Field Accuracy**: Average accuracy across all fields
- **Per-Field Accuracy**: Exact match rate for each field
- **Levenshtein Similarity**: String similarity (accounts for small errors)

### 8.3. Check Results Files

```bash
# View predictions
head results/predictions.jsonl | python -m json.tool

# View metrics
cat results/metrics.json | python -m json.tool
```

### 8.4. Analyze Errors

Check predictions with errors:

```python
import json

with open('results/predictions.jsonl', 'r') as f:
    for line in f:
        pred = json.loads(line)
        if pred['prediction'] != pred['ground_truth']:
            print(f"Error in: {pred['image_path']}")
            print(f"Predicted: {pred['prediction']}")
            print(f"Expected: {pred['ground_truth']}")
            print("---")
```

---

## Step 9: Deploy the Model

### 9.1. Prepare for Deployment

Identify your best checkpoint:

```bash
ls -lh models/checkpoints/
```

Use the one with lowest validation loss or `best_model.ckpt`.

### 9.2. Start API Server

#### Local Deployment

```bash
python run_api.py \
    --checkpoint models/checkpoints/best_model.ckpt \
    --host 0.0.0.0 \
    --port 8000
```

#### Docker Deployment

```bash
# Make sure checkpoint exists
cp models/checkpoints/best_model.ckpt models/checkpoints/best_model.ckpt

# Start API
docker-compose up -d api
```

### 9.3. Verify API Started

Check API health:

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

---

## Step 10: Test the API

### 10.1. Process Single Invoice

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
  },
  "confidence": null
}
```

### 10.2. Process Multiple Invoices

```bash
curl -X POST "http://localhost:8000/api/v1/extract-invoice-batch" \
    -F "files=@invoice1.pdf" \
    -F "files=@invoice2.pdf" \
    -F "files=@invoice3.pdf"
```

### 10.3. Test with Different Formats

```bash
# Test PDF
curl -X POST "http://localhost:8000/api/v1/extract-invoice" -F "file=@invoice.pdf"

# Test PNG
curl -X POST "http://localhost:8000/api/v1/extract-invoice" -F "file=@invoice.png"

# Test JPG
curl -X POST "http://localhost:8000/api/v1/extract-invoice" -F "file=@invoice.jpg"
```

---

## Troubleshooting

### Problem: Out of Memory During Training

**Solutions**:
1. Reduce batch size:
   ```yaml
   training:
     batch_size: 2  # or 1
   ```

2. Use mixed precision:
   ```yaml
   hardware:
     precision: 16
   ```

3. Reduce image size:
   ```yaml
   model:
     image_size: [960, 720]  # Smaller than [1280, 960]
   ```

4. Use gradient accumulation:
   ```yaml
   training:
     batch_size: 2
     accumulate_grad_batches: 4  # Effective batch size = 8
   ```

### Problem: Training is Very Slow

**Solutions**:
1. Use GPU instead of CPU
2. Increase num_workers:
   ```yaml
   training:
     num_workers: 8
   ```
3. Enable mixed precision
4. Reduce validation frequency

### Problem: Poor Accuracy

**Solutions**:
1. **More data**: Collect more training samples
2. **Better annotations**: Improve ground truth quality
3. **Longer training**: Increase max_epochs
4. **Lower learning rate**: Try 1e-5 instead of 3e-5
5. **Data quality**: Check for annotation errors

### Problem: Model Not Loading in API

**Check**:
1. Checkpoint path is correct
2. Checkpoint file exists
3. Config file matches training config
4. All dependencies installed

```bash
# Verify checkpoint
ls -lh models/checkpoints/best_model.ckpt

# Test loading
python -c "from src.training.trainer import DonutInvoiceModel; model = DonutInvoiceModel.load_from_checkpoint('models/checkpoints/best_model.ckpt')"
```

### Problem: Ground Truth Not Matching

**Solutions**:
1. Check filename alignment (without extensions)
2. Verify JSON syntax
3. Ensure consistent field names
4. Check for extra whitespace

```python
# Debug script
import json

gt = json.load(open('data/raw/ground_truth.json'))
print(f"Ground truth entries: {len(gt)}")
print(f"Sample keys: {list(gt.keys())[:5]}")

import os
invoices = [f.split('.')[0] for f in os.listdir('data/raw/invoices/')]
print(f"Invoice files: {len(invoices)}")
print(f"Sample names: {invoices[:5]}")

# Check overlap
gt_keys = set(gt.keys())
inv_keys = set(invoices)
print(f"Matching: {len(gt_keys & inv_keys)}")
print(f"Missing GT: {inv_keys - gt_keys}")
print(f"Missing invoices: {gt_keys - inv_keys}")
```

### Problem: Docker Build Fails

**Solutions**:
1. Check Docker is running
2. Update Docker to latest version
3. Clear Docker cache:
   ```bash
   docker system prune -a
   docker-compose build --no-cache
   ```

---

## Summary Checklist

- [ ] Environment setup complete
- [ ] Invoice files collected and organized
- [ ] Ground truth annotations created
- [ ] Data preprocessing completed
- [ ] Training configuration adjusted
- [ ] Model training started
- [ ] Training monitored via TensorBoard
- [ ] Model evaluation completed
- [ ] API deployed successfully
- [ ] API tested with sample invoices

---

## Next Steps After Training

1. **Integrate into production systems**
2. **Monitor performance on real data**
3. **Collect feedback and edge cases**
4. **Retrain periodically with new data**
5. **Optimize inference speed if needed**
6. **Add authentication to API**
7. **Set up monitoring and logging**

---

## Support

If you encounter issues:
1. Check this guide
2. Review README.md and QUICKSTART.md
3. Check logs in `logs/` directory
4. Verify configuration in `configs/train_config.yaml`
5. Contact Cloudx AI team

---

**Built for Cloudx | Invoice AI Training Guide**
