"""
Training module for Cloudx Invoice AI
Uses PyTorch Lightning for training orchestration
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import InvoiceDataModule


class DonutInvoiceModel(pl.LightningModule):
    """PyTorch Lightning module for Donut invoice model"""

    def __init__(
        self,
        config: Dict[str, Any],
        processor: DonutProcessor,
        model: Optional[VisionEncoderDecoderModel] = None
    ):
        """
        Initialize model

        Args:
            config: Configuration dictionary
            processor: Donut processor
            model: Pre-initialized model (optional)
        """
        super().__init__()
        self.config = config
        self.processor = processor

        # Initialize or load model
        if model is None:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                config["model"]["pretrained_model"]
            )
        else:
            self.model = model

        # Configure model
        self.model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]
        self.model.config.pad_token_id = processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = processor.tokenizer.eos_token_id

        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'processor'])

        # Training config
        self.learning_rate = config["training"]["learning_rate"]
        self.warmup_steps = config["training"]["warmup_steps"]

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def forward(self, pixel_values, labels=None):
        """Forward pass"""
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        """Training step"""
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.item())

        return loss

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        avg_val_loss = torch.tensor(self.val_losses).mean()
        self.log("avg_val_loss", avg_val_loss)
        self.val_losses.clear()

    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config["training"]["weight_decay"]
        )

        # Calculate total steps
        # Note: This is approximate, actual steps depend on dataset size
        total_steps = self.trainer.estimated_stepping_batches

        # Scheduler
        scheduler_type = self.config["training"]["scheduler"]["type"]

        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps,
                num_cycles=self.config["training"]["scheduler"]["num_cycles"]
            )
        elif scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            # Constant learning rate
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def generate(self, pixel_values, max_length: int = 768):
        """
        Generate predictions

        Args:
            pixel_values: Input image tensor
            max_length: Maximum generation length

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            pixel_values,
            max_length=max_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=self.config["inference"]["num_beams"],
            temperature=self.config["inference"]["temperature"] if self.config["inference"]["temperature"] > 0 else None,
            do_sample=self.config["inference"]["temperature"] > 0
        )


class InvoiceTrainer:
    """Trainer class for managing training process"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Set seed
        pl.seed_everything(config["seed"])

        # Initialize processor
        self.processor = DonutProcessor.from_pretrained(
            config["model"]["pretrained_model"]
        )

        # Add special tokens for invoice fields
        special_tokens = []
        for field in config["task"]["fields"]:
            special_tokens.extend([f"<s_{field}>", f"</s_{field}>"])

        # Add task tokens
        special_tokens.extend([
            config["task"]["task_start_token"],
            config["task"]["prompt_end_token"],
            "<sep>"
        ])

        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        # Initialize model
        self.model = DonutInvoiceModel(config, self.processor)

        # Resize token embeddings
        self.model.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))

        # Initialize data module
        self.data_module = InvoiceDataModule(
            train_metadata=config["data"]["train_metadata"],
            val_metadata=config["data"]["val_metadata"],
            test_metadata=config["data"].get("test_metadata"),
            processor=self.processor,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            max_length=config["model"]["max_length"]
        )
        self.data_module.setup()

        # Setup callbacks
        self.callbacks = self._setup_callbacks()

        # Setup logger
        self.logger = self._setup_logger()

    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config["checkpoint"]["save_dir"],
            filename="donut-invoice-{epoch:02d}-{val_loss:.4f}",
            save_top_k=self.config["checkpoint"]["save_top_k"],
            monitor=self.config["checkpoint"]["monitor"],
            mode=self.config["checkpoint"]["mode"],
            every_n_epochs=self.config["checkpoint"]["every_n_epochs"],
            save_last=True
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor=self.config["training"]["early_stopping"]["monitor"],
            patience=self.config["training"]["early_stopping"]["patience"],
            mode=self.config["training"]["early_stopping"]["mode"],
            verbose=True
        )
        callbacks.append(early_stop_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        return callbacks

    def _setup_logger(self):
        """Setup experiment logger"""
        if self.config["logging"]["wandb"]["enabled"]:
            logger = WandbLogger(
                project=self.config["logging"]["wandb"]["project"],
                entity=self.config["logging"]["wandb"]["entity"],
                name=self.config["logging"]["experiment_name"],
                save_dir=self.config["logging"]["log_dir"]
            )
        else:
            logger = TensorBoardLogger(
                save_dir=self.config["logging"]["log_dir"],
                name=self.config["logging"]["experiment_name"]
            )

        return logger

    def train(self):
        """Start training"""
        # Create directories
        Path(self.config["checkpoint"]["save_dir"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["max_epochs"],
            callbacks=self.callbacks,
            logger=self.logger,
            accelerator="gpu" if self.config["hardware"]["gpus"] > 0 else "cpu",
            devices=self.config["hardware"]["gpus"],
            precision=self.config["hardware"]["precision"],
            strategy=self.config["hardware"]["strategy"],
            gradient_clip_val=self.config["training"]["gradient_clip_val"],
            accumulate_grad_batches=self.config["training"]["accumulate_grad_batches"],
            log_every_n_steps=self.config["logging"]["log_every_n_steps"],
            deterministic=True
        )

        # Train
        trainer.fit(
            self.model,
            train_dataloaders=self.data_module.train_dataloader(),
            val_dataloaders=self.data_module.val_dataloader(),
            ckpt_path=self.config["resume_from_checkpoint"]
        )

        return trainer

    def test(self, trainer: pl.Trainer):
        """Run testing"""
        if self.data_module.test_dataloader():
            trainer.test(
                self.model,
                dataloaders=self.data_module.test_dataloader()
            )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
