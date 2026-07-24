import os
from pathlib import Path
from typing import Tuple, Any
import torch

from configs.config_loader import ConfigManager
from utils.logger import setup_logger

logger = setup_logger("model_loader")

def get_device() -> torch.device:
    """Detects and returns the best available hardware device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using execution device: {device}")
    return device

def load_florence_model(
    model_name: str = "microsoft/Florence-2-base",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05
) -> Tuple[Any, Any]:
    """Loads Florence-2 model and processor, wrapping with LoRA if specified.
    
    Args:
        model_name: Hugging Face model hub identifier.
        use_lora: If True, wraps model with LoraConfig for PEFT.
        lora_r: Rank dimension for LoRA projection matrices.
        lora_alpha: Scaling factor for LoRA updates.
        lora_dropout: Dropout probability for LoRA layers.
        
    Returns:
        Tuple of (model, processor)
    """
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    logger.info(f"Loading processor and model configuration for '{model_name}'...")
    
    # Florence-2 models require trust_remote_code=True
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model weights
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Handle device placement (PEFT preparation needs to be done prior to placing or after depending on setup)
    if use_lora:
        logger.info("Initializing LoRA adapters via PEFT...")
        from peft import LoraConfig, get_peft_model
        
        # Florence-2 attention modules use projection matrices: q_proj, k_proj, v_proj, out_proj
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    return model, processor

def save_model_checkpoint(model: Any, processor: Any, output_dir: Path, step_or_epoch: int = 0):
    """Saves checkpoint weights (LoRA adapters only if LoRA is active, full model otherwise)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving checkpoint to {output_dir}...")
    
    # Save the base/LoRA model
    model.save_pretrained(output_dir)
    
    # Save processor
    if processor is not None:
        processor.save_pretrained(output_dir)
        
    # Save a small helper indicating step/epoch
    with open(output_dir / "checkpoint_meta.json", "w") as f:
        import json
        json.dump({"step_or_epoch": step_or_epoch}, f)
        
    logger.info("Checkpoint saved successfully.")
