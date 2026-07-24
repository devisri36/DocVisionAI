import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from configs.config_loader import ConfigManager
from utils.logger import setup_logger
from model import save_model_checkpoint
from metrics import compute_batch_metrics

logger = setup_logger("trainer")

class FlorenceTrainer:
    """Trainer class for managing the Florence-2 PEFT training and evaluation loops."""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        device: torch.device = None,
        epochs: int = 10,
        grad_accumulation_steps: int = 1,
        use_amp: bool = True,
        early_stopping_patience: int = 3,
        output_dir: str = "outputs/checkpoints",
        resume_from: Optional[str] = None
    ):
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")
        
        self.epochs = epochs
        self.grad_accumulation_steps = grad_accumulation_steps
        
        # AMP scaler (only active on CUDA)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = GradScaler() if self.use_amp else None
        
        self.early_stopping_patience = early_stopping_patience
        self.output_dir = Path(output_dir)
        self.resume_from = resume_from
        
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Adjust model parameters device
        self.model.to(self.device)

        if self.resume_from:
            self._resume_training()

    def _resume_training(self):
        """Loads weights and state from a previous LoRA adapter checkpoint directory."""
        checkpoint_path = Path(self.resume_from)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint path '{checkpoint_path}' does not exist. Starting from scratch.")
            return

        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        # PEFT has built-in support to load adapter weights
        from peft import PeftModel
        try:
            self.model = PeftModel.from_pretrained(
                self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model,
                checkpoint_path,
                is_trainable=True
            ).to(self.device)
            
            # Load metadata
            meta_file = checkpoint_path / "checkpoint_meta.json"
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    self.start_epoch = meta.get("step_or_epoch", 0) + 1
                    logger.info(f"Resumed from epoch {meta.get('step_or_epoch')}. Resuming at epoch {self.start_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint state: {e}. Starting training from scratch.")

    def train(self) -> Dict[str, Any]:
        """Main training loop across epochs."""
        logger.info(f"Starting training on device: {self.device} (AMP: {self.use_amp})")
        logger.info(f"Total epochs: {self.epochs} | Batch size: {self.train_loader.batch_size} | Accumulation: {self.grad_accumulation_steps}")

        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")
            
            # 1. Training Phase
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # 2. Validation Phase
            val_loss, val_metrics = self._validate_epoch(epoch)
            history["val_loss"].append(val_loss)
            
            val_f1 = val_metrics.get("f1", 0.0)
            history["val_f1"].append(val_f1)
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Log metrics
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(f"Epoch {epoch + 1} results - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {metric_str}")

            # 3. Checkpointing & Early Stopping
            checkpoint_dir = self.output_dir / f"epoch_{epoch}"
            save_model_checkpoint(self.model, self.processor, checkpoint_dir, step_or_epoch=epoch)
            
            if val_loss < self.best_val_loss:
                logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving best model.")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_dir = self.output_dir / "best_model"
                save_model_checkpoint(self.model, self.processor, best_dir, step_or_epoch=epoch)
            else:
                self.patience_counter += 1
                logger.info(f"Validation loss did not improve. Early stopping counter: {self.patience_counter}/{self.early_stopping_patience}")

            if self.patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping triggered. Halting training.")
                break

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Runs one epoch of training over the loader."""
        self.model.train()
        total_loss = 0.0
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc="Training Batches", leave=False)
        for step, batch in enumerate(progress_bar):
            # Move inputs to target device
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Autocast mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.grad_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.grad_accumulation_steps
                loss.backward()

            total_loss += loss.item() * self.grad_accumulation_steps
            
            # Step optimizer after accumulation steps
            if (step + 1) % self.grad_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item() * self.grad_accumulation_steps:.4f}"})

        return total_loss / len(self.train_loader)

    def _validate_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        """Runs evaluation and generates text predictions on validation data."""
        self.model.eval()
        total_loss = 0.0
        
        all_predictions = []
        all_references = []
        
        logger.info("Evaluating validation loss and metrics...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation Batches", leave=False):
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Validation forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                total_loss += outputs.loss.item()
                
                # Autoregressive text generation check (evaluate F1 / CER / EM)
                # To prevent slow generations during validation, we run on a limit
                if len(all_predictions) < 50:
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        max_new_tokens=128,
                        early_stopping=True
                    )
                    
                    predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Decode labels (ignore pad token -100)
                    # Replace -100 in labels back to tokenizer pad token
                    clean_labels = labels.clone()
                    clean_labels[clean_labels == -100] = self.processor.tokenizer.pad_token_id
                    references = self.processor.batch_decode(clean_labels, skip_special_tokens=True)
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)

        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metric scores
        metrics = compute_batch_metrics(all_predictions, all_references)
        
        return avg_loss, metrics
