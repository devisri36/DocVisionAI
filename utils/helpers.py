import time
import functools
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from utils.logger import setup_logger

logger = setup_logger("utils")

def time_it(func):
    """Decorator to measure execution time of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        logger.debug(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

def normalize_bbox(
    bbox: List[float], 
    width: int, 
    height: int
) -> List[int]:
    """Normalizes bounding box coordinates to LayoutLM scale [0, 1000].
    
    Args:
        bbox: Coordinate list [x_min, y_min, x_max, y_max].
        width: Image width.
        height: Image height.
        
    Returns:
        A list of normalized integers [x_min_norm, y_min_norm, x_max_norm, y_max_norm] scaled to [0, 1000].
    """
    if not (width > 0 and height > 0):
        return [0, 0, 0, 0]
        
    x_min, y_min, x_max, y_max = bbox
    
    x_min_norm = int(max(0, min(1000, (x_min / width) * 1000)))
    y_min_norm = int(max(0, min(1000, (y_min / height) * 1000)))
    x_max_norm = int(max(0, min(1000, (x_max / width) * 1000)))
    y_max_norm = int(max(0, min(1000, (y_max / height) * 1000)))
    
    return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]

def unnormalize_bbox(
    bbox: List[int], 
    width: int, 
    height: int
) -> List[int]:
    """Unnormalizes LayoutLM scale [0, 1000] bounding boxes back to pixel dimensions.
    
    Args:
        bbox: Normalized coordinate list [x_min_norm, y_min_norm, x_max_norm, y_max_norm].
        width: Original image width.
        height: Original image height.
        
    Returns:
        A list of pixel coordinate integers [x_min, y_min, x_max, y_max].
    """
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox
    
    x_min = int((x_min_norm / 1000) * width)
    y_min = int((y_min_norm / 1000) * height)
    x_max = int((x_max_norm / 1000) * width)
    y_max = int((y_max_norm / 1000) * height)
    
    return [x_min, y_min, x_max, y_max]

def draw_bboxes_on_image(
    image: Image.Image, 
    bboxes: List[List[float]], 
    labels: List[str] = None, 
    color: str = "red", 
    width: int = 2
) -> Image.Image:
    """Draws bounding boxes and labels onto a PIL Image for debugging and visualization.
    
    Args:
        image: PIL Image object.
        bboxes: List of bounding box coordinates [x_min, y_min, x_max, y_max].
        labels: Optional labels matching the boxes.
        color: Color to draw the boxes.
        width: Thickness of box lines.
        
    Returns:
        PIL Image with drawn annotations.
    """
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    for i, bbox in enumerate(bboxes):
        draw.rectangle(bbox, outline=color, width=width)
        if labels and i < len(labels):
            label = labels[i]
            # Draw simple text label near the top-left of the box
            draw.text((bbox[0], max(0, bbox[1] - 10)), label, fill=color)
            
    return annotated_img
