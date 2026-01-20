# Installation Guide - Cloudx Invoice AI

Complete step-by-step installation guide for setting up Cloudx Invoice AI on a new machine.

## Server Specifications (Recommended)

This guide is optimized for high-spec servers:
- **RAM**: 256GB (minimum 16GB)
- **CPU**: Intel Xeon Platinum 8259CL @ 2.50GHz (2 processors)
- **GPU**: NVIDIA GPU with CUDA support (optional but highly recommended)
- **Storage**: 500GB+ for large datasets
- **OS**: Windows Server 2019+ or Linux (Ubuntu 20.04+)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Verify Installation](#verify-installation)
4. [GPU Setup (Optional)](#gpu-setup-optional)
5. [Data Preparation](#data-preparation)
6. [Next Steps](#next-steps)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Git**
   - Windows: Download from https://git-scm.com/download/win
   - Linux: `sudo apt-get install git`

2. **Python 3.8 - 3.11**
   - Windows: Download from https://www.python.org/downloads/
   - Linux: `sudo apt-get install python3.11 python3.11-venv python3-pip`
   - **Important**: Check "Add Python to PATH" during installation

3. **Visual Studio Build Tools** (Windows only)
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Select "Desktop development with C++"
   - Required for compiling some Python packages

4. **CUDA Toolkit** (For GPU training - recommended)
   - Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - Or CUDA 12.1: https://developer.nvidia.com/cuda-downloads
   - Verify GPU: `nvidia-smi`

---

## Installation Steps

### Step 1: Clone the Repository

Open Command Prompt (Windows) or Terminal (Linux):

```bash
# Navigate to your workspace
cd C:\workspace
# Or on Linux: cd /home/username/workspace

# Clone the repository
git clone https://github.com/andyasne/cloudx-invoice-extractor-ai.git

# Navigate into the project
cd cloudx-invoice-extractor-ai
```

### Step 2: Create Python Virtual Environment

**Windows:**
```batch
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (you should see (venv) in prompt)
```

**Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in prompt)
```

### Step 3: Upgrade pip and setuptools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install PyTorch

Choose the appropriate command based on your hardware:

**Option A: GPU with CUDA 11.8 (Recommended for high-spec servers)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option B: GPU with CUDA 12.1**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Option C: CPU Only (Not recommended for large-scale training)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify PyTorch installation:**
```python
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}')"
```

Expected output (for GPU):
```
PyTorch version: 2.x.x+cu118
CUDA available: True
CUDA version: 11.8
```

### Step 5: Install Project Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

This will install:
- Transformers (Hugging Face)
- PyTorch Lightning
- FastAPI and Uvicorn
- PDF processing libraries (pdf2image, PyPDF2)
- Image processing (Pillow, OpenCV)
- Data processing (pandas, numpy)
- And all other dependencies

### Step 6: Install System Dependencies (PDF Processing)

**Windows:**
```batch
# Download poppler for Windows
# Visit: https://github.com/oschwartz10612/poppler-windows/releases
# Download the latest release (e.g., poppler-23.08.0.zip)
# Extract to C:\Program Files\poppler
# Add C:\Program Files\poppler\Library\bin to PATH

# Or use Chocolatey (if installed)
choco install poppler
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

### Step 7: Verify PDF Processing

```python
python -c "from pdf2image import convert_from_path; print('PDF processing ready!')"
```

### Step 8: Download Donut Base Model (Optional)

The model will be downloaded automatically on first training, but you can pre-download:

```python
python -c "from transformers import VisionEncoderDecoderModel; model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base'); print('Model downloaded successfully!')"
```

---

## Verify Installation

Run this verification script to check all components:

```python
# Save as verify_install.py and run: python verify_install.py

import sys
import torch
import transformers
import PIL
import cv2
import fastapi
import pytorch_lightning as pl

print("=" * 70)
print("Cloudx Invoice AI - Installation Verification")
print("=" * 70)

# Python version
print(f"✓ Python version: {sys.version.split()[0]}")

# PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

# Transformers
print(f"✓ Transformers version: {transformers.__version__}")

# PyTorch Lightning
print(f"✓ PyTorch Lightning version: {pl.__version__}")

# FastAPI
print(f"✓ FastAPI version: {fastapi.__version__}")

# PIL
print(f"✓ Pillow version: {PIL.__version__}")

# OpenCV
print(f"✓ OpenCV version: {cv2.__version__}")

# Test PDF processing
try:
    from pdf2image import convert_from_path
    print("✓ PDF processing (pdf2image): Ready")
except Exception as e:
    print(f"✗ PDF processing: {e}")

print("=" * 70)
print("Installation verification complete!")
print("=" * 70)
```

---

## GPU Setup (Optional)

### Check GPU Status

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Configure GPU Usage

Edit `configs/train_config.yaml`:

```yaml
hardware:
  gpus: 1              # Use 1 GPU (or 2, 4, etc. if you have multiple)
  precision: 16        # Use mixed precision for faster training
  strategy: "auto"     # Or "ddp" for multi-GPU
```

### GPU Memory Optimization

For large datasets on your 256GB RAM server:

```yaml
data:
  num_workers: 16      # Utilize multiple CPU cores
  prefetch_factor: 4   # Pre-load batches

training:
  batch_size: 8        # Adjust based on GPU memory (16GB GPU = 4-8, 24GB GPU = 8-16)
  accumulate_grad_batches: 4  # Effective batch size = 8 * 4 = 32
```

---

## Data Preparation

### Step 1: Create Data Directories

```bash
# Windows
mkdir data\raw\invoices
mkdir data\raw\myInvoices
mkdir data\processed
mkdir data\train
mkdir data\val
mkdir data\test

# Linux
mkdir -p data/raw/invoices data/raw/myInvoices data/processed data/train data/val data/test
```

### Step 2: Copy Your Invoice Files

**Option A: Manual Copy**
```bash
# Copy your invoices to data\raw\invoices
xcopy /s "D:\your_invoices\*" "data\raw\invoices\"
```

**Option B: Use the Copy Script**
```batch
# Copy latest 10,000 invoices from your source folder
scripts\copy_latest_invoices.bat "D:\invoice_storage" 10000

# Or copy and clear destination first
scripts\copy_latest_invoices.bat "D:\invoice_storage" 10000 --clear
```

### Step 3: Prepare Ground Truth (If Available)

Create `data/raw/ground_truth.json`:

```json
{
  "invoice_001": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "vendor_name": "Acme Corp",
    "total": "1500.00"
  },
  "invoice_002": {
    ...
  }
}
```

### Step 4: Generate Sample Data (For Testing)

If you don't have real data yet:

```bash
# Generate 100 sample invoices
python scripts/create_sample_data.py --num_samples 100
```

---

## Next Steps

### 1. Quick Test Training

Run a quick training test with sample data:

```batch
# Windows
START_TRAINING.bat

# Or
scripts\quick_train.bat
```

### 2. Full Training Workflow

For production training with millions of invoices:

```bash
# 1. Copy your invoices
scripts\copy_latest_invoices.bat "D:\invoices" 100000 --clear

# 2. Configure training
# Edit configs/train_config.yaml

# 3. Start training
python train.py --config configs/train_config.yaml

# 4. Monitor with TensorBoard
tensorboard --logdir logs/
```

### 3. Evaluate Model

```bash
python evaluate.py --checkpoint models/checkpoints/best_model.ckpt
```

### 4. Deploy API

```bash
python run_api.py --checkpoint models/checkpoints/best_model.ckpt --host 0.0.0.0 --port 8000
```

### 5. Test API

```bash
curl -X POST "http://localhost:8000/api/v1/extract-invoice" -F "file=@test_invoice.pdf"
```

---

## Troubleshooting

### Issue: "Python not found"

**Solution:**
- Reinstall Python and check "Add to PATH"
- Or add manually: `C:\Users\YourUser\AppData\Local\Programs\Python\Python311`

### Issue: "CUDA not available" but you have GPU

**Solution:**
1. Install NVIDIA drivers: https://www.nvidia.com/download/index.aspx
2. Install CUDA Toolkit
3. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: "No module named 'pdf2image'"

**Solution:**
```bash
pip install pdf2image
# Also install poppler (see Step 6 above)
```

### Issue: Out of memory during training

**Solution:**
Edit `configs/train_config.yaml`:
```yaml
training:
  batch_size: 2        # Reduce batch size
  accumulate_grad_batches: 8  # Maintain effective batch size
```

### Issue: Training is very slow on CPU

**Solution:**
- Install GPU drivers and CUDA
- Or reduce dataset size for testing
- Or use cloud GPU (AWS, GCP, Azure)

### Issue: "ImportError: DLL load failed" (Windows)

**Solution:**
Install Visual C++ Redistributable:
- https://aka.ms/vs/17/release/vc_redist.x64.exe

### Issue: Permission denied when copying files

**Solution:**
Run Command Prompt as Administrator:
- Right-click Command Prompt → "Run as administrator"

---

## System Requirements Summary

### Minimum Requirements
- Python 3.8+
- 16GB RAM
- 100GB storage
- CPU training (slow)

### Recommended for Production
- Python 3.9 or 3.10
- 64GB+ RAM
- NVIDIA GPU (16GB+ VRAM)
- 500GB+ SSD storage
- CUDA 11.8 or 12.1

### Your High-Spec Server (Optimal)
- 256GB RAM ✓
- Intel Xeon Platinum (2 processors) ✓
- Add NVIDIA GPU for faster training
- Large SSD for invoice storage

---

## Quick Reference Commands

```bash
# Activate environment
venv\Scripts\activate              # Windows
source venv/bin/activate          # Linux

# Copy invoices
scripts\copy_latest_invoices.bat "D:\invoices" 10000

# Train model
python train.py --config configs/train_config.yaml

# Evaluate model
python evaluate.py --checkpoint models/checkpoints/best_model.ckpt

# Run API
python run_api.py --checkpoint models/checkpoints/best_model.ckpt

# Test API
curl -X POST http://localhost:8000/api/v1/extract-invoice -F "file=@invoice.pdf"
```

---

## Additional Resources

- **Full Documentation**: [README.md](README.md)
- **Quick Start Guide**: [QUICKSTART.md](QUICKSTART.md)
- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Project Plan**: [claude.md](claude.md)
- **Donut Paper**: https://arxiv.org/abs/2111.15664
- **Donut GitHub**: https://github.com/clovaai/donut

---

## Support

For installation issues:
1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Check project logs in `logs/`
4. Contact Cloudx AI team

---

**Installation Complete!** 🚀

You're now ready to train your invoice AI model with millions of invoices on your high-spec server.
