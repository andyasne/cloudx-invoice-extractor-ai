"""
Cloudx Invoice AI - API Server
FastAPI application for invoice processing
"""
import io
import os
from pathlib import Path
from typing import Optional, Dict, Any
import sys

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.trainer import load_config, DonutInvoiceModel
from src.evaluation.metrics import token2json
from src.data.preprocessor import InvoicePreprocessor
from transformers import DonutProcessor
import pdf2image


class InvoiceResponse(BaseModel):
    """Response model for invoice extraction"""
    status: str
    invoice_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    confidence: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str


class InvoiceAPI:
    """Invoice processing API"""

    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize API

        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
        """
        self.config = load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor
        print("Loading processor...")
        self.processor = DonutProcessor.from_pretrained(
            self.config["model"]["pretrained_model"]
        )

        # Add special tokens
        special_tokens = []
        for field in self.config["task"]["fields"]:
            special_tokens.extend([f"<s_{field}>", f"</s_{field}>"])

        special_tokens.extend([
            self.config["task"]["task_start_token"],
            self.config["task"]["prompt_end_token"],
            "<sep>"
        ])

        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = DonutInvoiceModel.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
            processor=self.processor
        )
        self.model.eval()
        self.model.to(self.device)

        # Initialize preprocessor
        self.preprocessor = InvoicePreprocessor(
            input_size=tuple(self.config["model"]["image_size"]),
            max_length=self.config["model"]["max_length"]
        )

        print(f"Model loaded successfully on {self.device}")

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process a single invoice image

        Args:
            image: PIL Image

        Returns:
            Dictionary of extracted invoice fields
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image)

        # Process with model processor
        pixel_values = self.processor(processed_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=self.config["model"]["max_length"]
            )

        # Decode
        prediction_text = self.processor.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Convert to JSON
        prediction_dict = token2json(prediction_text, self.processor)

        return prediction_dict

    def process_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Process a PDF invoice (extracts first page)

        Args:
            pdf_bytes: PDF file bytes

        Returns:
            Dictionary of extracted invoice fields
        """
        # Convert PDF to images
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=150)

        if not images:
            raise ValueError("Failed to convert PDF to image")

        # Process first page
        return self.process_image(images[0])


# Initialize FastAPI app
app = FastAPI(
    title="Cloudx Invoice AI API",
    description="AI-powered invoice text extraction and structuring using Donut transformer",
    version="0.1.0"
)

# Global API instance
api_instance: Optional[InvoiceAPI] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global api_instance

    # Get config and checkpoint from environment variables
    config_path = os.getenv("CONFIG_PATH", "configs/train_config.yaml")
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "models/checkpoints/best_model.ckpt")

    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("API will start but inference endpoints will fail")
        return

    try:
        api_instance = InvoiceAPI(config_path, checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but inference endpoints will fail")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Cloudx Invoice AI API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "process_invoice": "/api/v1/extract-invoice"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=api_instance is not None,
        device=api_instance.device if api_instance else "none"
    )


@app.post("/api/v1/extract-invoice", response_model=InvoiceResponse)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Extract structured data from invoice

    Args:
        file: Invoice file (PDF or image)

    Returns:
        Extracted invoice data
    """
    if api_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Read file
        contents = await file.read()

        # Determine file type
        file_ext = Path(file.filename).suffix.lower()

        # Process based on file type
        if file_ext == '.pdf':
            invoice_data = api_instance.process_pdf(contents)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            invoice_data = api_instance.process_image(image)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: PDF, PNG, JPG, JPEG, TIFF, BMP"
            )

        return InvoiceResponse(
            status="success",
            invoice_data=invoice_data,
            confidence=None  # TODO: Add confidence scores
        )

    except Exception as e:
        return InvoiceResponse(
            status="error",
            error=str(e)
        )


@app.post("/api/v1/extract-invoice-batch")
async def extract_invoice_batch(files: list[UploadFile] = File(...)):
    """
    Extract structured data from multiple invoices

    Args:
        files: List of invoice files

    Returns:
        List of extracted invoice data
    """
    if api_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )

    results = []

    for file in files:
        try:
            contents = await file.read()
            file_ext = Path(file.filename).suffix.lower()

            if file_ext == '.pdf':
                invoice_data = api_instance.process_pdf(contents)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                invoice_data = api_instance.process_image(image)
            else:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": f"Unsupported file type: {file_ext}"
                })
                continue

            results.append({
                "filename": file.filename,
                "status": "success",
                "invoice_data": invoice_data
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    return {"results": results}


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server

    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cloudx Invoice AI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=args.reload)
