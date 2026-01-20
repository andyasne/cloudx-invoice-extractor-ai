@echo off
REM Cloudx Invoice AI - Start Training with Your Invoice
REM This script completes the setup and starts training

echo ==================================================
echo CLOUDX INVOICE EXTRACTOR AI - TRAINING STARTER
echo ==================================================
echo.
echo Your invoice: data/raw/myInvoices/my_invoice_001.pdf
echo Dataset: 100 invoices (99 samples + 1 real)
echo.
echo ==================================================

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import torch; print('PyTorch: OK')" 2>nul
if errorlevel 1 (
    echo Installing PyTorch... This may take 5-10 minutes.
    python -m pip install torch torchvision --timeout=300
)

python -c "import transformers; print('Transformers: OK')" 2>nul
if errorlevel 1 (
    echo Installing ML dependencies... This may take 5-10 minutes.
    python -m pip install transformers timm datasets pytorch-lightning nltk sentencepiece zss sconf --quiet
)

python -c "import fastapi; print('FastAPI: OK')" 2>nul
if errorlevel 1 (
    echo Installing API dependencies...
    python -m pip install fastapi uvicorn python-multipart pydantic --quiet
)

python -c "import pandas; print('Data tools: OK')" 2>nul
if errorlevel 1 (
    echo Installing data processing tools...
    python -m pip install pandas pyyaml jsonlines scikit-learn --quiet
)

echo.
echo ==================================================
echo STEP 1: Preprocessing Data
echo ==================================================
echo This will convert all invoices to the format needed for training.
echo Estimated time: 5-10 minutes
echo.

python scripts/prepare_data.py --input_dir data/raw/invoices --ground_truth data/raw/ground_truth_combined.json --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

if errorlevel 1 (
    echo ERROR: Preprocessing failed!
    pause
    exit /b 1
)

echo.
echo ==================================================
echo STEP 2: Training Model
echo ==================================================
echo This will train the AI on your invoices.
echo Estimated time: 1-3 hours on CPU, 20-60 minutes on GPU
echo.
echo Training with 10 epochs, batch size 2
echo You can stop training anytime with Ctrl+C
echo.

python train.py --config configs/train_config.yaml --epochs 10 --batch_size 2

if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ==================================================
echo TRAINING COMPLETE!
echo ==================================================
echo.
echo Your model is saved in: models/checkpoints/
echo.
echo Next steps:
echo 1. Test your model:
echo    python scripts/inference_example.py --invoice data/raw/myInvoices/my_invoice_001.pdf --checkpoint models/checkpoints/best_model.ckpt
echo.
echo 2. Start the API server:
echo    python run_api.py --checkpoint models/checkpoints/best_model.ckpt
echo.
echo 3. Process invoices via API:
echo    curl -X POST "http://localhost:8000/api/v1/extract-invoice" -F "file=@your_invoice.pdf"
echo.
echo ==================================================
pause
