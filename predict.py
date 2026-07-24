import argparse
import re
from pathlib import Path
from PIL import Image, ImageDraw
import torch

from configs.config_loader import ConfigManager
from utils.logger import setup_logger
from model import get_device

logger = setup_logger("predict_cli")

def parse_args():
    parser = argparse.ArgumentParser(description="DocVision AI - Florence-2 Document Inference Script")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the document image to analyze"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved LoRA checkpoint directory (optional, runs base model if omitted)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sroie",
        help="Task type: 'sroie', 'docvqa', 'funsd', 'cord', or a direct Florence-2 task prompt (e.g. '<OCR_WITH_REGION>')"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question text (required for 'docvqa' task)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/Florence-2-base",
        help="Base Florence-2 model identifier"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/prediction.png",
        help="Output path to save the annotated image"
    )
    return parser.parse_args()

def parse_florence_locations(text: str, img_width: int, img_height: int) -> tuple[list[list[int]], list[str], list[str]]:
    """Parses text containing location tokens and extracts absolute pixel bounding boxes.
    
    Florence-2 location tokens format: text <loc_y1><loc_x1><loc_y2><loc_x2> (label)
    Coordinates are scaled to [0, 1000].
    """
    # Regex matching text, followed by 4 location tags, and optional labels
    # Format: text <loc_y0><loc_x0><loc_y1><loc_x1> (label) or label: text <loc_y0><loc_x0><loc_y1><loc_x1>
    pattern = r'([^|<]+)\s*<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>(?:\s*\(([^)]+)\))?'
    matches = re.finditer(pattern, text)
    
    bboxes = []
    words = []
    labels = []
    
    for m in matches:
        word = m.group(1).strip()
        y0_val = int(m.group(2))
        x0_val = int(m.group(3))
        y1_val = int(m.group(4))
        x1_val = int(m.group(5))
        label = m.group(6) or "text"
        
        # SROIE/Florence standard: coordinates are normalized [0, 1000] where y0, x0, y1, x1 is the order
        # Translate coordinates back to image pixel space
        x0 = int((x0_val / 1000.0) * img_width)
        y0 = int((y0_val / 1000.0) * img_height)
        x1 = int((x1_val / 1000.0) * img_width)
        y1 = int((y1_val / 1000.0) * img_height)
        
        bboxes.append([x0, y0, x1, y1])
        words.append(word)
        labels.append(label)
        
    return bboxes, words, labels

def main():
    args = parse_args()
    device = get_device()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Input image file does not exist: {image_path}")
        return

    logger.info(f"Loading image '{image_path}'...")
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # 1. Setup Florence-2 Task Prompts
    task = args.task.lower().strip()
    if task == "sroie":
        prompt = "<KIE>"
    elif task == "docvqa":
        if not args.question:
            logger.error("Error: --question is required for 'docvqa' task.")
            return
        prompt = f"<DocVQA> Question: {args.question}"
    elif task in ["funsd", "cord"]:
        prompt = "<OCR_WITH_REGION>"
    else:
        # Custom task token directly
        prompt = args.task

    logger.info(f"Formulated task prompt: '{prompt}'")

    # 2. Load model and processor
    from transformers import AutoProcessor, AutoModelForCausalLM
    from peft import PeftModel

    logger.info(f"Loading base model '{args.model_name}'...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    if args.checkpoint:
        logger.info(f"Loading PEFT adapter weights from '{args.checkpoint}'...")
        model = PeftModel.from_pretrained(model, args.checkpoint)

    model.to(device)
    model.eval()

    # 3. Preprocess and generate
    logger.info("Running document inference...")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            early_stopping=True
        )

    # 4. Decode output
    prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info(f"Raw Prediction: {prediction}")

    # 5. Parse bounding boxes and draw if locations exist
    bboxes, words, labels = parse_florence_locations(prediction, img_w, img_h)
    
    if bboxes:
        logger.info(f"Extracted {len(bboxes)} bounding boxes. Drawing overlays...")
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Color palette for entity labels
        colors = {"header": "blue", "date": "green", "address": "orange", "total": "red", "question": "purple", "answer": "teal"}
        
        for box, word, label in zip(bboxes, words, labels):
            color = colors.get(label.lower(), "red")
            draw.rectangle(box, outline=color, width=2)
            # Label
            draw.text((box[0], max(0, box[1] - 12)), f"{label}: {word}", fill=color)
            
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_image.save(output_path)
        logger.info(f"Saved annotated prediction image to: {output_path.absolute()}")
    else:
        logger.info("No location tags found in the generated output text. No annotations drawn.")
        
    print(f"\n--- Inference Result ---")
    print(prediction)

if __name__ == "__main__":
    main()
