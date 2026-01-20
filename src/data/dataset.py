"""
Dataset loader for invoice documents
Compatible with Donut model and PyTorch Lightning
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import DonutProcessor


class InvoiceDataset(Dataset):
    """Dataset for invoice documents with Donut model"""

    def __init__(
        self,
        metadata_file: str,
        processor: DonutProcessor,
        max_length: int = 768,
        split: str = "train",
        task_start_token: str = "<s_invoice>",
        prompt_end_token: str = "<s_invoice>",
        sort_json_key: bool = True
    ):
        """
        Initialize dataset

        Args:
            metadata_file: Path to JSONL metadata file
            processor: Donut processor for image and text processing
            max_length: Maximum sequence length
            split: Dataset split (train/val/test)
            task_start_token: Token to start task
            prompt_end_token: Token to end prompt
            sort_json_key: Whether to sort JSON keys
        """
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token
        self.sort_json_key = sort_json_key

        # Load metadata
        self.samples = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self) -> int:
        return len(self.samples)

    def json2token(self, obj: Dict, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True) -> str:
        """
        Convert JSON object to token sequence

        Args:
            obj: JSON object (dictionary)
            update_special_tokens_for_json_key: Whether to use special tokens for keys
            sort_json_key: Whether to sort keys

        Returns:
            Token sequence string
        """
        if sort_json_key:
            obj = {k: obj[k] for k in sorted(obj.keys())}

        output = ""
        for key, value in obj.items():
            if update_special_tokens_for_json_key:
                output += f"<s_{key}>"

            if isinstance(value, dict):
                output += self.json2token(value, update_special_tokens_for_json_key, sort_json_key)
            elif isinstance(value, list):
                output += "<sep>".join([str(item) for item in value])
            else:
                output += str(value)

            if update_special_tokens_for_json_key:
                output += f"</s_{key}>"

        return output

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Tuple of (pixel_values, labels, attention_mask)
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Prepare ground truth text
        ground_truth = sample["ground_truth"]
        target_sequence = self.task_start_token + self.json2token(ground_truth, sort_json_key=self.sort_json_key) + self.processor.tokenizer.eos_token

        # Tokenize target
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Create labels (same as input_ids for teacher forcing)
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return pixel_values, labels, input_ids


class InvoiceDataModule:
    """Data module for managing train/val/test datasets"""

    def __init__(
        self,
        train_metadata: str,
        val_metadata: str,
        test_metadata: Optional[str],
        processor: DonutProcessor,
        batch_size: int = 4,
        num_workers: int = 4,
        max_length: int = 768
    ):
        """
        Initialize data module

        Args:
            train_metadata: Path to training metadata file
            val_metadata: Path to validation metadata file
            test_metadata: Path to test metadata file (optional)
            processor: Donut processor
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            max_length: Maximum sequence length
        """
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.test_metadata = test_metadata
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Setup datasets"""
        self.train_dataset = InvoiceDataset(
            self.train_metadata,
            self.processor,
            max_length=self.max_length,
            split="train"
        )

        self.val_dataset = InvoiceDataset(
            self.val_metadata,
            self.processor,
            max_length=self.max_length,
            split="val"
        )

        if self.test_metadata:
            self.test_dataset = InvoiceDataset(
                self.test_metadata,
                self.processor,
                max_length=self.max_length,
                split="test"
            )

    def train_dataloader(self):
        """Get training dataloader"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """Get validation dataloader"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """Get test dataloader"""
        if self.test_dataset:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None


def split_dataset(
    metadata_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = "data/processed",
    seed: int = 42
) -> Tuple[str, str, str]:
    """
    Split dataset into train/val/test

    Args:
        metadata_file: Path to full dataset metadata
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        output_dir: Directory to save split metadata files
        seed: Random seed

    Returns:
        Tuple of (train_metadata_path, val_metadata_path, test_metadata_path)
    """
    # Load all samples
    samples = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    # Shuffle
    random.seed(seed)
    random.shuffle(samples)

    # Calculate splits
    total = len(samples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]

    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_metadata.jsonl"
    val_path = output_dir / "val_metadata.jsonl"
    test_path = output_dir / "test_metadata.jsonl"

    # Write train
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')

    # Write val
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')

    # Write test
    with open(test_path, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Dataset split complete:")
    print(f"  Train: {len(train_samples)} samples -> {train_path}")
    print(f"  Val: {len(val_samples)} samples -> {val_path}")
    print(f"  Test: {len(test_samples)} samples -> {test_path}")

    return str(train_path), str(val_path), str(test_path)
