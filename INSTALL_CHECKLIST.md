# Installation Checklist - Cloudx Invoice AI

Quick checklist for installing on a new server. Check off each item as you complete it.

---

## Pre-Installation Checklist

- [ ] Server has 16GB+ RAM (yours: 256GB ✓)
- [ ] Server has 100GB+ storage available
- [ ] Internet connection available
- [ ] Administrator/sudo access available

---

## Step-by-Step Installation

### 1. Install Prerequisites (15-30 minutes)

- [ ] Install Git
  - Windows: https://git-scm.com/download/win
  - Linux: `sudo apt-get install git`
  - Verify: `git --version`

- [ ] Install Python 3.8-3.11
  - Windows: https://www.python.org/downloads/ (Check "Add to PATH")
  - Linux: `sudo apt-get install python3.11 python3-pip`
  - Verify: `python --version`

- [ ] Install Visual Studio Build Tools (Windows only)
  - https://visualstudio.microsoft.com/downloads/
  - Select "Desktop development with C++"

- [ ] Install NVIDIA drivers (if you have GPU)
  - https://www.nvidia.com/download/index.aspx
  - Verify: `nvidia-smi`

- [ ] Install CUDA Toolkit (if you have GPU)
  - CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
  - Verify: `nvcc --version`

### 2. Clone Repository (2 minutes)

- [ ] Navigate to workspace: `cd C:\workspace` or `cd /home/user/workspace`
- [ ] Clone repo: `git clone https://github.com/andyasne/cloudx-invoice-extractor-ai.git`
- [ ] Enter directory: `cd cloudx-invoice-extractor-ai`

### 3. Setup Python Environment (5 minutes)

- [ ] Create venv: `python -m venv venv`
- [ ] Activate venv:
  - Windows: `venv\Scripts\activate`
  - Linux: `source venv/bin/activate`
- [ ] Verify: You should see `(venv)` in prompt
- [ ] Upgrade pip: `python -m pip install --upgrade pip setuptools wheel`

### 4. Install PyTorch (5 minutes)

Choose ONE option:

- [ ] GPU with CUDA 11.8 (Recommended):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- [ ] GPU with CUDA 12.1:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- [ ] CPU only:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- [ ] Verify PyTorch:
  ```python
  python -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"
  ```

### 5. Install Dependencies (10 minutes)

- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Install project: `pip install -e .`
- [ ] Wait for installation to complete (may take 5-10 minutes)

### 6. Install Poppler (PDF Processing)

- [ ] Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
  - Extract to `C:\Program Files\poppler`
  - Add `C:\Program Files\poppler\Library\bin` to PATH

- [ ] Linux: `sudo apt-get install poppler-utils`

- [ ] Verify: `python -c "from pdf2image import convert_from_path; print('OK')"`

### 7. Create Data Directories (1 minute)

- [ ] Windows:
  ```batch
  mkdir data\raw\invoices
  mkdir data\processed
  mkdir data\train
  mkdir data\val
  mkdir data\test
  ```

- [ ] Linux:
  ```bash
  mkdir -p data/raw/invoices data/processed data/train data/val data/test
  ```

### 8. Verify Installation (2 minutes)

Create and run `verify_install.py`:

```python
import sys, torch, transformers, PIL, cv2, fastapi
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Installation verified! ✓")
```

- [ ] Run: `python verify_install.py`
- [ ] All checks passed

---

## Post-Installation Setup

### 9. Download Model (Optional - 5 minutes)

- [ ] Pre-download Donut base model:
  ```python
  python -c "from transformers import VisionEncoderDecoderModel; VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base')"
  ```

### 10. Prepare Your Data

- [ ] Copy invoices to server
  - If from another machine: Use SCP/FTP/USB
  - If already on server: Note the path

- [ ] Use copy script:
  ```batch
  scripts\copy_latest_invoices.bat "D:\your_invoices" 10000
  ```

- [ ] Verify files copied:
  ```bash
  dir data\raw\invoices    # Windows
  ls data/raw/invoices     # Linux
  ```

### 11. Test Installation (Optional - 5 minutes)

- [ ] Generate sample data:
  ```bash
  python scripts/create_sample_data.py --num_samples 50
  ```

- [ ] Run quick training test:
  ```batch
  START_TRAINING.bat           # Windows
  ./quick_train.sh             # Linux
  ```

- [ ] Training starts without errors

---

## Configuration

### 12. Configure for Your Server (2 minutes)

Edit `configs/train_config.yaml`:

- [ ] Set GPU count: `gpus: 1` (or 0 for CPU)
- [ ] Set batch size: `batch_size: 8` (adjust for GPU memory)
- [ ] Set workers: `num_workers: 16` (for 256GB RAM, use 8-16)
- [ ] Enable mixed precision: `precision: 16`

---

## Ready to Train!

You're all set! Next steps:

- [ ] Copy your million invoices:
  ```batch
  scripts\copy_latest_invoices.bat "D:\invoices" 1000000 --clear
  ```

- [ ] Start training:
  ```bash
  python train.py --config configs/train_config.yaml
  ```

- [ ] Monitor progress:
  ```bash
  tensorboard --logdir logs/
  ```

---

## Quick Commands Reference

```bash
# Activate environment (always do this first!)
venv\Scripts\activate                                    # Windows
source venv/bin/activate                                # Linux

# Copy invoices
scripts\copy_latest_invoices.bat "source_path" 10000

# Train
python train.py --config configs/train_config.yaml

# Evaluate
python evaluate.py --checkpoint models/checkpoints/best_model.ckpt

# Run API
python run_api.py --checkpoint models/checkpoints/best_model.ckpt

# Test API
curl -X POST http://localhost:8000/api/v1/extract-invoice -F "file=@invoice.pdf"
```

---

## Troubleshooting Common Issues

**"Python not found"**
- Reinstall Python with "Add to PATH" checked

**"CUDA not available"**
- Install NVIDIA drivers
- Install CUDA Toolkit
- Reinstall PyTorch with correct CUDA version

**"pip install fails"**
- Run as Administrator (Windows)
- Install Visual Studio Build Tools

**"pdf2image error"**
- Install poppler and add to PATH

**Out of memory**
- Reduce `batch_size` in config
- Use `accumulate_grad_batches: 4`

---

## Installation Time Estimate

| Task | Time |
|------|------|
| Prerequisites | 15-30 min |
| Clone repo | 2 min |
| Python setup | 5 min |
| Install PyTorch | 5 min |
| Install dependencies | 10 min |
| System dependencies | 5 min |
| Verification | 2 min |
| **Total** | **45-60 min** |

---

## Success Criteria

Installation is successful when:

- [x] `python --version` shows 3.8+
- [x] `git --version` works
- [x] `nvidia-smi` shows GPU (if applicable)
- [x] `python -c "import torch; print(torch.cuda.is_available())"` returns True (if GPU)
- [x] `verify_install.py` runs without errors
- [x] Data directories exist
- [x] Can generate sample data
- [x] Quick training test runs

---

**Last Updated**: 2026-01-20

For detailed instructions, see [INSTALL.md](INSTALL.md)
