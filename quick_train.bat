@echo off
REM Cloudx Invoice AI - Quick Training Script (Windows)
REM Automated workflow from sample data generation to training

echo ==================================================
echo CLOUDX INVOICE AI - QUICK TRAINING WORKFLOW
echo ==================================================

REM Step 1: Generate sample data
echo.
echo Step 1: Generating sample invoice data...
python scripts/create_sample_data.py --num_samples 100
if errorlevel 1 goto error

REM Step 2: Preprocess data
echo.
echo Step 2: Preprocessing dataset...
python scripts/prepare_data.py --input_dir data/raw/invoices --ground_truth data/raw/ground_truth.json --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
if errorlevel 1 goto error

REM Step 3: Start training
echo.
echo Step 3: Starting model training...
echo This may take several hours depending on your hardware.
echo.
python train.py --config configs/train_config.yaml --epochs 10 --batch_size 4
if errorlevel 1 goto error

echo.
echo ==================================================
echo TRAINING COMPLETE!
echo ==================================================
echo Next steps:
echo 1. Evaluate model: python evaluate.py --checkpoint models/checkpoints/best_model.ckpt
echo 2. Start API: python run_api.py --checkpoint models/checkpoints/best_model.ckpt
echo ==================================================
goto end

:error
echo.
echo ERROR: Training workflow failed!
echo Please check the error messages above.
exit /b 1

:end
