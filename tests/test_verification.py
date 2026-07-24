import unittest
from PIL import Image, ImageFilter
import numpy as np

from backend.verification_service import (
    calculate_dhash,
    hamming_distance,
    DocumentQualityAnalyzer,
    FraudDetector,
    VLMExplainer
)

class TestVerification(unittest.TestCase):
    def setUp(self):
        # Create a sharp image (high contrast checkerboard)
        self.image = Image.new("RGB", (300, 300), color="white")
        draw_array = np.array(self.image)
        draw_array[100:200, 100:200] = 0  # Black square in the middle
        self.image = Image.fromarray(draw_array)
        
        # Create a blurred image
        self.blurred_image = self.image.filter(ImageFilter.GaussianBlur(radius=8))

    def test_dhash_and_hamming(self):
        h1 = calculate_dhash(self.image)
        h2 = calculate_dhash(self.image)
        h3 = calculate_dhash(self.blurred_image)
        
        # Exact same image must have 0 Hamming distance
        self.assertEqual(hamming_distance(h1, h2), 0)
        
        # Similar images should have a low Hamming distance
        self.assertTrue(len(h1) == 16)  # 64-bit in hex is 16 chars
        dist = hamming_distance(h1, h3)
        self.assertTrue(dist >= 0)

    def test_blur_detection(self):
        analyzer = DocumentQualityAnalyzer()
        
        # Sharp image variance should be higher
        res_sharp = analyzer.analyze_blur(self.image, threshold=50.0)
        res_blur = analyzer.analyze_blur(self.blurred_image, threshold=50.0)
        
        self.assertFalse(res_sharp["is_blurred"])
        self.assertTrue(res_blur["is_blurred"])
        self.assertTrue(res_sharp["variance"] > res_blur["variance"])

    def test_resolution_and_contrast(self):
        analyzer = DocumentQualityAnalyzer()
        res = analyzer.analyze_resolution_and_contrast(self.image)
        
        # 300x300 = 90k pixels which is < 360k (600x600)
        self.assertTrue(res["is_low_resolution"])
        self.assertFalse(res["is_low_contrast"])  # High contrast due to black/white square

    def test_crop_detection(self):
        analyzer = DocumentQualityAnalyzer()
        
        # Coordinates way inside boundaries
        ocr_inside = [
            {"bbox": [100, 100, 150, 120]}
        ]
        res_inside = analyzer.analyze_crop(self.image, ocr_inside)
        self.assertFalse(res_inside["is_cropped"])
        
        # Coordinates cut off near border margins (2% of 300 is 6 pixels)
        ocr_outside = [
            {"bbox": [2, 10, 50, 30]},
            {"bbox": [100, 2, 150, 20]},
            {"bbox": [200, 100, 299, 120]},
            {"bbox": [10, 298, 50, 300]}
        ]
        res_outside = analyzer.analyze_crop(self.image, ocr_outside)
        self.assertTrue(res_outside["is_cropped"])

    def test_ela_detection(self):
        detector = FraudDetector()
        ela_img, boxes, score = detector.perform_ela(self.image, quality=90)
        
        self.assertIsInstance(ela_img, Image.Image)
        self.assertEqual(ela_img.size, self.image.size)
        self.assertTrue(0.0 <= score <= 1.0)
        self.assertIsInstance(boxes, list)

    def test_scoring_engine(self):
        detector = FraudDetector()
        
        # High quality, no ELA, high OCR conf -> Fraud=0, Authenticity=1, Conf=0.98
        scores_good = detector.compute_scores(
            ela_score=0.0,
            quality_analysis={"is_blurred": False, "is_cropped": False},
            ocr_extractor_confidences=[0.98, 0.97, 0.99]
        )
        self.assertEqual(scores_good["fraud_score"], 0.0)
        self.assertEqual(scores_good["authenticity_score"], 1.0)
        self.assertEqual(scores_good["confidence_score"], 0.98)
        
        # Poor quality + high ELA -> Fraud should increase
        scores_bad = detector.compute_scores(
            ela_score=0.5,
            quality_analysis={"is_blurred": True, "is_cropped": True},
            ocr_extractor_confidences=[0.8, 0.7]
        )
        self.assertTrue(scores_bad["fraud_score"] > 0.3)
        self.assertTrue(scores_bad["authenticity_score"] < 0.7)

    def test_explainer_heatmap(self):
        # Generate simulated attention overlay
        explainer = VLMExplainer()
        boxes = [[50, 50, 150, 150]]
        overlay = explainer.generate_attention_heatmap(self.image, boxes)
        
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, self.image.size)
