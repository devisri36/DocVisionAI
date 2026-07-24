import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from configs.config_loader import ConfigManager
from utils.logger import setup_logger
from backend.downloader import DatasetDownloader
from backend.preprocessor import DocumentPreprocessor
from dataset import FlorenceDocumentDataset, FlorenceDataCollator
from model import load_florence_model, get_device
from trainer import FlorenceTrainer

logger = setup_logger("training_cli")

def parse_args():
    parser = argparse.ArgumentParser(description="DocVision AI - Florence-2 Fine-Tuning Script")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sroie",
        choices=["funsd", "cord", "sroie", "docvqa"],
        help="Target dataset to use for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training and validation loaders"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA scaling alpha factor"
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (AMP)"
    )
    parser.add_argument(
        "--no_amp",
        action="store_false",
        dest="use_amp",
        help="Disable Automatic Mixed Precision"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="Florence-2 base model identifier"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory to save checkpoint directories"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = ConfigManager().config
    device = get_device()
    
    logger.info(f"Preparing training job for dataset '{args.dataset}'...")

    # 1. Verify and download datasets if needed
    downloader = DatasetDownloader()
    dataset_key = args.dataset.lower()
    dataset_config = config.datasets.get(dataset_key)
    if not dataset_config:
        logger.error(f"Dataset '{dataset_key}' is not configured in configs/config.yaml")
        return
        
    raw_dir = Path(dataset_config.raw_dir)
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        logger.info(f"Local dataset files for '{dataset_key}' not found at '{raw_dir}'. Triggering auto-downloader...")
        if dataset_key == "funsd":
            downloader.download_funsd()
        elif dataset_key == "cord":
            downloader.download_cord()
        elif dataset_key == "sroie":
            downloader.download_sroie()
        elif dataset_key == "docvqa":
            downloader.download_docvqa()

    # 2. Load model and processor
    model, processor = load_florence_model(
        model_name=args.model_name,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )

    # 3. Create datasets and loader collators
    # Instantiate custom DocumentPreprocessor
    preprocessor = DocumentPreprocessor(augment=True)
    
    train_dataset = FlorenceDocumentDataset(
        dataset_name=args.dataset,
        split="train",
        processor=processor,
        preprocessor=preprocessor
    )
    
    # Validation dataset (augment=False)
    val_preprocessor = DocumentPreprocessor(augment=False)
    val_dataset = FlorenceDocumentDataset(
        dataset_name=args.dataset,
        split="test" if args.dataset == "sroie" else "validation",
        processor=processor,
        preprocessor=val_preprocessor
    )

    collator = FlorenceDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        label_pad_token_id=-100
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # Avoid Windows multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    # 4. Set up optimizer and scheduler
    # Ensure only trainable parameters are passed to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=0.5, 
        patience=1, 
        verbose=True
    )

    # 5. Initialize and run trainer
    trainer = FlorenceTrainer(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        grad_accumulation_steps=args.grad_accum,
        use_amp=args.use_amp,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )

    history = trainer.train()
    logger.info("Training process completed successfully.")
    logger.info(f"Training History: {history}")

if __name__ == "__main__":
    main()
