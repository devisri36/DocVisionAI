import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_loader import ConfigManager
from utils.logger import setup_logger
from backend.preprocessor import DocumentPreprocessor
from dataset import FlorenceDocumentDataset, FlorenceDataCollator
from model import get_device
from metrics import compute_batch_metrics

logger = setup_logger("validation_cli")

def parse_args():
    parser = argparse.ArgumentParser(description="DocVision AI - Florence-2 Checkpoint Validation Script")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved LoRA checkpoint directory (containing adapter_model.bin)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sroie",
        choices=["funsd", "cord", "sroie", "docvqa"],
        help="Target dataset to use for validation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for loader"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="Base Florence-2 model identifier"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device()
    
    logger.info(f"Validating checkpoint '{args.checkpoint}' on dataset '{args.dataset}'...")

    # 1. Load base model and processor
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    logger.info(f"Loading base model '{args.model_name}'...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    # 2. Wrap with PEFT LoRA adapter
    logger.info(f"Loading LoRA adapters from '{args.checkpoint}'...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.to(device)
    model.eval()

    # 3. Create dataset and loader
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    # 4. Evaluation Loop
    total_loss = 0.0
    all_predictions = []
    all_references = []

    logger.info("Running inference across validation dataset...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Compute loss
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()

            # Autoregressively generate targets
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=128,
                early_stopping=True
            )
            
            predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode labels
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = processor.tokenizer.pad_token_id
            references = processor.batch_decode(clean_labels, skip_special_tokens=True)

            all_predictions.extend(predictions)
            all_references.extend(references)

    # 5. Compute and print metrics
    avg_loss = total_loss / len(val_loader)
    metrics = compute_batch_metrics(all_predictions, all_references)
    
    logger.info("=== Validation Summary ===")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    for k, v in metrics.items():
        logger.info(f"{k.upper()}: {v:.4f}")
    
    # Save validation metrics summary to disk
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "val_results.json", "w", encoding="utf-8") as f:
        import json
        json.dump({"loss": avg_loss, "metrics": metrics}, f, indent=4)
    logger.info(f"Saved evaluation results to outputs/val_results.json")

if __name__ == "__main__":
    main()
