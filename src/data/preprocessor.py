"""
Data preprocessing module for invoice documents
Handles PDF and image conversion, normalization, and preparation for Donut model
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pdf2image
import cv2
import numpy as np
from tqdm import tqdm


class InvoicePreprocessor:
    """Preprocesses invoice documents (PDF/images) for Donut training"""

    def __init__(
        self,
        input_size: Tuple[int, int] = (1280, 960),
        max_length: int = 768,
        dpi: int = 150
    ):
        """
        Initialize preprocessor

        Args:
            input_size: Target image size (width, height) for model input
            max_length: Maximum sequence length for text
            dpi: DPI for PDF to image conversion
        """
        self.input_size = input_size
        self.max_length = max_length
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Images (one per page)
        """
        try:
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='PNG'
            )
            return images
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            return []

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image from file

        Args:
            image_path: Path to image file

        Returns:
            PIL Image or None if error
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for model input
        - Resize to target size
        - Normalize

        Args:
            image: PIL Image

        Returns:
            Preprocessed PIL Image
        """
        # Resize maintaining aspect ratio
        image.thumbnail(self.input_size, Image.Resampling.LANCZOS)

        # Create new image with padding
        new_image = Image.new("RGB", self.input_size, (255, 255, 255))

        # Paste resized image centered
        paste_x = (self.input_size[0] - image.size[0]) // 2
        paste_y = (self.input_size[1] - image.size[1]) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image

    def enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality (optional preprocessing)
        - Denoise
        - Sharpen
        - Adjust contrast

        Args:
            image: PIL Image

        Returns:
            Enhanced PIL Image
        """
        # Convert to numpy array
        img_array = np.array(image)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

        # Convert back to PIL
        enhanced = Image.fromarray(denoised)

        return enhanced

    def process_invoice_pair(
        self,
        document_path: str,
        ground_truth: Dict,
        output_dir: str,
        enhance: bool = False
    ) -> List[Dict]:
        """
        Process a single invoice document with its ground truth

        Args:
            document_path: Path to PDF or image file
            ground_truth: Dictionary containing ground truth text/structure
            output_dir: Directory to save processed images
            enhance: Whether to apply image enhancement

        Returns:
            List of processed samples (one per page if PDF)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file type
        ext = Path(document_path).suffix.lower()

        # Load images
        if ext == '.pdf':
            images = self.pdf_to_images(document_path)
        else:
            image = self.load_image(document_path)
            images = [image] if image else []

        # Process each image
        processed_samples = []
        for idx, image in enumerate(images):
            if image is None:
                continue

            # Enhance if requested
            if enhance:
                image = self.enhance_image(image)

            # Preprocess
            processed_image = self.preprocess_image(image)

            # Save processed image
            doc_name = Path(document_path).stem
            image_filename = f"{doc_name}_page{idx}.png"
            image_path = output_dir / image_filename
            processed_image.save(image_path)

            # Create sample entry
            sample = {
                "image_path": str(image_path),
                "ground_truth": ground_truth,
                "original_path": document_path,
                "page_number": idx
            }
            processed_samples.append(sample)

        return processed_samples

    def process_dataset(
        self,
        document_paths: List[str],
        ground_truths: List[Dict],
        output_dir: str,
        metadata_file: str = "dataset_metadata.jsonl",
        enhance: bool = False
    ) -> str:
        """
        Process entire dataset of invoices

        Args:
            document_paths: List of paths to invoice documents
            ground_truths: List of ground truth dictionaries (aligned with document_paths)
            output_dir: Directory to save processed data
            metadata_file: Name of metadata file to create
            enhance: Whether to apply image enhancement

        Returns:
            Path to metadata file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = output_dir / metadata_file

        all_samples = []

        print(f"Processing {len(document_paths)} documents...")
        for doc_path, ground_truth in tqdm(zip(document_paths, ground_truths), total=len(document_paths)):
            samples = self.process_invoice_pair(
                doc_path,
                ground_truth,
                str(output_dir / "images"),
                enhance=enhance
            )
            all_samples.extend(samples)

        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + '\n')

        print(f"Processed {len(all_samples)} samples")
        print(f"Metadata saved to {metadata_path}")

        return str(metadata_path)


def load_ground_truth_from_json(json_path: str) -> Dict:
    """
    Load ground truth from JSON file

    Args:
        json_path: Path to JSON file containing ground truth

    Returns:
        Dictionary with ground truth data
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_donut_format(ground_truth: Dict) -> str:
    """
    Convert ground truth to Donut's expected format
    Donut expects a string in a specific format for training

    Args:
        ground_truth: Dictionary containing invoice fields

    Returns:
        Formatted string for Donut training
    """
    # Donut uses a special token format
    # Example: "<s_invoice><s_invoice_number>INV-001</s_invoice_number><s_date>2024-01-01</s_date>...</s_invoice>"

    output = "<s_invoice>"

    for key, value in ground_truth.items():
        output += f"<s_{key}>{value}</s_{key}>"

    output += "</s_invoice>"

    return output
