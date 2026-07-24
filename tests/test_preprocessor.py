import unittest
from PIL import Image
import torch
import numpy as np
from backend.preprocessor import DocumentPreprocessor

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Create a dummy image (width=100, height=200)
        self.image = Image.new("RGB", (100, 200), color="white")
        # Define mock absolute bounding boxes
        self.bboxes = [
            [10, 20, 50, 80],
            [90, 150, 110, 210]  # Exceeds bounds, needs clamping
        ]
        self.words = ["Hello", "World"]
        self.labels = ["header", "text"]
        self.preprocessor = DocumentPreprocessor(augment=False)

    def test_preprocessing_pipeline(self):
        processed = self.preprocessor.preprocess(
            image=self.image,
            bboxes=self.bboxes,
            words=self.words,
            labels=self.labels
        )
        
        # Test image tensor shape
        img_tensor = processed["image"]
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.shape, (3, 384, 384))
        
        # Test coordinates resizing and scaling
        final_boxes = processed["bboxes"]
        self.assertTrue(len(final_boxes) > 0)
        
        # Test normalized boxes [0, 1000]
        norm_boxes = processed["normalized_bboxes"]
        self.assertEqual(len(norm_boxes), len(final_boxes))
        for box in norm_boxes:
            self.assertEqual(len(box), 4)
            for coord in box:
                self.assertTrue(0 <= coord <= 1000)

    def test_clamping_logic(self):
        # Box 2 [90, 150, 110, 210] in 100x200 image should clamp to [90, 150, 100, 200]
        cleaned_boxes, cleaned_labels, cleaned_words = self.preprocessor._clean_and_clamp_bboxes(
            self.bboxes, 100, 200, self.labels, self.words
        )
        
        self.assertEqual(cleaned_boxes[1], [90.0, 150.0, 100.0, 200.0])
        self.assertEqual(len(cleaned_boxes), 2)
        
    def test_train_val_split(self):
        items = list(range(10))
        train, val = DocumentPreprocessor.train_validation_split(items, split_ratio=0.8, seed=42)
        
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 2)
        # Check no duplicates
        self.assertEqual(len(set(train + val)), 10)
