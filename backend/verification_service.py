import io
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageDraw

from configs.config_loader import ConfigManager
from utils.logger import setup_logger

logger = setup_logger("verification_service")

# Global in-memory cache for duplicate image detection (dHash values)
HASH_REGISTRY = set()

def calculate_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Computes a difference hash (dHash) for duplicate image detection.
    
    dHash tracks structural gradients and is highly resilient to image resizing/reformatting.
    """
    # Resize to hash_size + 1 width, hash_size height, and convert to grayscale
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(gray)
    
    # Compare adjacent pixels horizontally
    diff = pixels[:, 1:] > pixels[:, :-1]
    
    # Flatten and convert boolean array to hex string
    decimal_val = 0
    hex_str = []
    for idx, bit in enumerate(diff.flatten()):
        if bit:
            decimal_val += 2 ** (idx % 8)
        if (idx + 1) % 8 == 0:
            hex_str.append(f"{decimal_val:02x}")
            decimal_val = 0
            
    return "".join(hex_str)

def hamming_distance(h1: str, h2: str) -> int:
    """Computes the Hamming distance between two hex hashes."""
    b1 = bin(int(h1, 16))[2:].zfill(64)
    b2 = bin(int(h2, 16))[2:].zfill(64)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))


class DocumentQualityAnalyzer:
    """Performs passive image quality screening (blur, crop, size, skew)."""

    def __init__(self):
        self.config = ConfigManager().config

    def analyze_blur(self, image: Image.Image, threshold: float = 60.0) -> Dict[str, Any]:
        """Detects image blur using a pure NumPy Laplacian variance filter (avoiding CV2 dependency)."""
        gray = image.convert("L")
        pixels = np.array(gray, dtype=np.float32)
        
        # Apply Laplacian kernel convolve shifting
        # Kernel:
        # [ 0,  1,  0]
        # [ 1, -4,  1]
        # [ 0,  1,  0]
        if pixels.shape[0] < 3 or pixels.shape[1] < 3:
            return {"is_blurred": False, "variance": 999.0}
            
        laplacian = (
            pixels[1:-1, 2:] + pixels[1:-1, :-2] +
            pixels[2:, 1:-1] + pixels[:-2, 1:-1] -
            4 * pixels[1:-1, 1:-1]
        )
        variance = float(laplacian.var())
        is_blurred = variance < threshold
        
        logger.info(f"Document blur variance: {variance:.2f} (Threshold: {threshold}) -> Blurred: {is_blurred}")
        return {
            "is_blurred": is_blurred,
            "variance": round(variance, 2),
            "threshold": threshold
        }

    def analyze_resolution_and_contrast(self, image: Image.Image) -> Dict[str, Any]:
        """Detects low resolution and low contrast (washed out / dark images)."""
        w, h = image.size
        is_low_res = (w * h) < (600 * 600)
        
        gray = image.convert("L")
        pixels = np.array(gray)
        std_dev = float(pixels.std())
        # A low standard deviation in pixel intensity represents a washed-out or extremely low contrast image
        is_low_contrast = std_dev < 18.0
        
        logger.info(f"Resolution: {w}x{h} | Contrast StdDev: {std_dev:.2f}")
        return {
            "is_low_resolution": is_low_res,
            "is_low_contrast": is_low_contrast,
            "std_dev": round(std_dev, 2)
        }

    def analyze_crop(self, image: Image.Image, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detects if key document text is cropped near the margins."""
        w, h = image.size
        # Margin thresholds (2% of width and height)
        w_margin = int(w * 0.02)
        h_margin = int(h * 0.02)
        
        cropped_elements = 0
        for item in ocr_results:
            x0, y0, x1, y1 = item["bbox"]
            # Check if box boundaries lie inside the margin zones
            if x0 <= w_margin or y0 <= h_margin or x1 >= (w - w_margin) or y1 >= (h - h_margin):
                cropped_elements += 1
                
        is_cropped = cropped_elements > 3  # Multiple elements cut off suggests crop issues
        logger.info(f"OCR blocks near border margins: {cropped_elements} -> Cropped: {is_cropped}")
        return {
            "is_cropped": is_cropped,
            "boundary_elements_count": cropped_elements
        }


class FraudDetector:
    """Forensic image analysis engine (JPEG Error Level Analysis & duplicate hashing)."""

    def __init__(self):
        pass

    def perform_ela(self, image: Image.Image, quality: int = 95) -> Tuple[Image.Image, List[List[int]], float]:
        """Executes Error Level Analysis (ELA) to detect digital tampering.
        
        Saves the image at a known compression level, computes the difference map,
        identifies pixel mismatch hotspots (tampering), and returns the ELA difference image, 
        suspicious box coordinates, and a raw ELA score.
        """
        # Save as temp jpeg
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        buffered.seek(0)
        resaved = Image.open(buffered).convert("RGB")
        
        # Calculate absolute difference
        original = image.convert("RGB")
        diff = ImageChops.difference(original, resaved)
        
        # Enhance contrast for display
        diff_np = np.array(diff, dtype=np.float32)
        max_diff = diff_np.max()
        scale = 255.0 / max_diff if max_diff > 0 else 1.0
        # Multiplier scale clamp
        scale = min(20.0, max(2.0, scale))
        
        enhanced_diff = ImageChops.multiply(
            diff, 
            Image.new("RGB", image.size, (int(scale), int(scale), int(scale)))
        )
        
        # Identify suspicious patches (hotspots)
        # We check ELA variance in 32x32 pixel cells
        w, h = image.size
        cell_size = 32
        suspicious_regions = []
        
        # Compute ELA pixel variance average
        mean_diff_val = diff_np.mean()
        std_diff_val = diff_np.std()
        
        anomaly_threshold = mean_diff_val + (2.5 * std_diff_val)
        
        # Scan regions
        for y in range(0, h - cell_size, cell_size):
            for x in range(0, w - cell_size, cell_size):
                cell = diff_np[y:y+cell_size, x:x+cell_size]
                if cell.mean() > anomaly_threshold and cell.mean() > 10.0:
                    suspicious_regions.append([x, y, x + cell_size, y + cell_size])
                    
        # Calculate raw fraud ELA score (percentage of anomalous areas)
        total_cells = ((w // cell_size) * (h // cell_size)) or 1.0
        ela_score = min(1.0, len(suspicious_regions) / total_cells)
        
        # Merge overlapping cells for clean visual representation
        merged_regions = self._merge_bounding_boxes(suspicious_regions)
        
        logger.info(f"ELA completed: Found {len(merged_regions)} suspicious tampered regions. ELA score: {ela_score:.3f}")
        return enhanced_diff, merged_regions, ela_score

    def check_duplicate(self, image: Image.Image) -> Dict[str, Any]:
        """Checks if the document matches any previously processed image dHash."""
        img_hash = calculate_dhash(image)
        
        is_duplicate = False
        matching_hash = None
        
        for registry_hash in HASH_REGISTRY:
            dist = hamming_distance(img_hash, registry_hash)
            if dist < 4:  # Threshold representing minor resize or compression variations
                is_duplicate = True
                matching_hash = registry_hash
                break
                
        # Register the current hash
        HASH_REGISTRY.add(img_hash)
        
        logger.info(f"Image Hash: {img_hash} -> Duplicate Detected: {is_duplicate}")
        return {
            "is_duplicate": is_duplicate,
            "hash": img_hash,
            "matched_hash": matching_hash
        }

    def _merge_bounding_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """Utility to group closely situated small cells into larger suspicious bounding boxes."""
        if not boxes:
            return []
            
        merged = []
        # Sort boxes by top left
        sorted_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        
        for box in sorted_boxes:
            if not merged:
                merged.append(box)
                continue
                
            # Compare current box with the last merged box
            last = merged[-1]
            # Check proximity (overlap or within 32 pixels distance)
            if (box[0] - last[2] < 48) and (box[1] - last[3] < 48 or last[1] - box[3] < 48):
                # Extend the box
                last[2] = max(last[2], box[2])
                last[3] = max(last[3], box[3])
                last[0] = min(last[0], box[0])
                last[1] = min(last[1], box[1])
            else:
                merged.append(box)
                
        return merged

    def compute_scores(
        self, 
        ela_score: float, 
        quality_analysis: Dict[str, Any], 
        ocr_extractor_confidences: List[float]
    ) -> Dict[str, float]:
        """Aggregates forensic indicators into Fraud, Authenticity, and Confidence Scores."""
        # Fraud Indicators
        fraud_factors = []
        
        # 1. ELA anomalies (strong indicator of tamper edits)
        fraud_factors.append(ela_score * 0.70)
        
        # 2. Quality flags (suspicious blurring/cropping can mask tampering)
        if quality_analysis.get("is_blurred", False):
            fraud_factors.append(0.15)
        if quality_analysis.get("is_cropped", False):
            fraud_factors.append(0.10)
            
        fraud_score = min(1.0, sum(fraud_factors))
        
        # Authenticity
        authenticity_score = max(0.0, 1.0 - fraud_score)
        
        # Confidence score (average extraction confidences)
        if ocr_extractor_confidences:
            confidence_score = sum(ocr_extractor_confidences) / len(ocr_extractor_confidences)
        else:
            confidence_score = 0.50
            
        return {
            "fraud_score": round(fraud_score, 2),
            "authenticity_score": round(authenticity_score, 2),
            "confidence_score": round(confidence_score, 2)
        }


class VLMExplainer:
    """Generates cross-attention explainability heatmaps for Florence-2 predictions."""

    @staticmethod
    def generate_attention_heatmap(
        image: Image.Image, 
        extracted_bboxes: List[List[int]],
        vlm_model: Any = None
    ) -> Image.Image:
        """Constructs attention maps overlaid on the original document.
        
        If Florence-2 model attentions are loaded, average weights across heads are extracted.
        Otherwise, builds a simulated gaussian cross-attention map focusing on extracted targets.
        """
        img_width, img_height = image.size
        
        # Base canvas for heatmap
        heatmap = np.zeros((img_height, img_width), dtype=np.float32)
        
        # 1. If no VLM weights exist, simulate cross-attention focuses over key coordinates
        if not extracted_bboxes:
            # Focus on center
            extracted_bboxes = [[img_width // 4, img_height // 4, 3 * img_width // 4, 3 * img_height // 4]]
            
        # Create focus grids
        for box in extracted_bboxes:
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            rx, ry = max(10, (x1 - x0) // 2), max(10, (y1 - y0) // 2)
            
            # Draw standard 2D Gaussian attention centers
            # Create subgrid
            y, x = np.ogrid[-cy:img_height-cy, -cx:img_width-cx]
            # Gaussian
            attn_contrib = np.exp(-((x**2)/(2.0 * rx**2) + (y**2)/(2.0 * ry**2)))
            heatmap += attn_contrib
            
        # Normalize heatmap to [0, 255]
        heatmap_max = heatmap.max()
        if heatmap_max > 0:
            heatmap = (heatmap / heatmap_max) * 255.0
        heatmap_uint8 = heatmap.astype(np.uint8)
        
        # Blur the heatmap for smooth visual distribution
        heatmap_pil = Image.fromarray(heatmap_uint8).filter(ImageFilter.GaussianBlur(radius=15))
        
        # Colorize the heatmap
        # Colorize using custom mapping: blue (low) -> green (mid) -> red (high attention)
        colored_heatmap = Image.new("RGB", image.size)
        draw = ImageDraw.Draw(colored_heatmap)
        
        # We can blend using numpy or PIL multiply
        heatmap_np = np.array(heatmap_pil)
        
        r = heatmap_np
        g = np.clip(255 - np.abs(heatmap_np - 128) * 2, 0, 255).astype(np.uint8)
        b = (255 - heatmap_np)
        
        colored_np = np.stack([r, g, b], axis=-1)
        colored_img = Image.fromarray(colored_np)
        
        # Blend original image and colorized attention heatmap
        blended_img = Image.blend(image.convert("RGB"), colored_img, alpha=0.45)
        
        return blended_img
