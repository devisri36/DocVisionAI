# Model Training & Fine-Tuning Guide

This guide details how to fine-tune the Vision-Language Model (Florence-2) using PEFT LoRA (Low-Rank Adaptation) on visual document datasets.

---

## 1. PEFT LoRA Setup

We load Florence-2 modules and apply LoRA layers to key linear layers in the transformer attention blocks to reduce trainable parameters:
- **`target_modules`**: `["q_proj", "k_proj", "v_proj", "out_proj"]`
- **`r` (rank)**: 8
- **`lora_alpha`**: 16
- **`lora_dropout`**: 0.05

---

## 2. Command-Line Training Execution

Run fine-tuning using `training.py`:

```bash
python training.py \
    --dataset funsd \
    --epochs 5 \
    --batch_size 4 \
    --lr 5e-5 \
    --resume_from_checkpoint ""
```

### Configurable Hyperparameters
- `--dataset`: Target dataset keys (`funsd`, `cord`, `sroie`, `docvqa`).
- `--epochs`: Total training cycles.
- `--batch_size`: Batch dimensions.
- `--lr`: Learning rate optimizer value.
- `--resume_from_checkpoint`: Path to previous LoRA adapter weights folder if resuming.

---

## 3. Training Architecture Features
- **Mixed Precision**: Uses FP16/BF16 (`torch.cuda.amp.autocast`) to speed up GPU backpropagation.
- **Gradient Accumulation**: Aggregates batch inputs to simulate higher batch dimensions when running on lower GPU RAM constraints.
- **Early Stopping**: Halts execution when validation loss stagnates over 3 sequential periods to prevent model overfitting.
