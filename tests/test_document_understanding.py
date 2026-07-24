import unittest
from PIL import Image
import numpy as np

from backend.ocr_engine import DocVisionOCREngine
from backend.document_understanding import DocumentClassifier, InformationExtractor, VLMQAEngine

class TestDocumentUnderstanding(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (400, 500), color="white")
        self.ocr_engine = DocVisionOCREngine()

    def test_mock_ocr_extraction(self):
        # Triggering mock extractor fallback by default
        results = self.ocr_engine._generate_mock_ocr_results(400, 500)
        self.assertTrue(len(results) > 0)
        
        # Verify schema keys
        first = results[0]
        self.assertIn("text", first)
        self.assertIn("bbox", first)
        self.assertIn("confidence", first)
        self.assertEqual(len(first["bbox"]), 4)

    def test_document_classification_rules(self):
        classifier = DocumentClassifier()
        
        # PAN Heuristic Test
        pan_text = "INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT CARD ABCDE1234F"
        res_pan = classifier.classify(self.image, pan_text)
        self.assertEqual(res_pan["category"], "PAN")
        self.assertEqual(res_pan["method"], "Rule-based")
        self.assertTrue(res_pan["confidence"] >= 0.8)
        
        # Aadhaar Heuristic Test
        aadhaar_text = "Government of India Unique Identification Authority of India UIDAI Aadhaar Card"
        res_aadhaar = classifier.classify(self.image, aadhaar_text)
        self.assertEqual(res_aadhaar["category"], "Aadhaar")
        self.assertTrue(res_aadhaar["confidence"] >= 0.8)
        
        # Invoice Heuristic Test
        invoice_text = "Tax Invoice Bill To Amount Due $150.00 PO Number"
        res_inv = classifier.classify(self.image, invoice_text)
        self.assertEqual(res_inv["category"], "Invoice")

    def test_information_extraction_regex(self):
        extractor = InformationExtractor()
        
        # OCR records mockup containing PAN and Aadhaar patterns
        ocr_results = [
            {"text": "NAME:", "bbox": [10, 10, 40, 20], "confidence": 0.95},
            {"text": "JOHN DOE", "bbox": [50, 10, 100, 20], "confidence": 0.96},
            {"text": "PAN CARD", "bbox": [10, 30, 40, 40], "confidence": 0.94},
            {"text": "ABCDE1234F", "bbox": [50, 30, 100, 40], "confidence": 0.98},
            {"text": "DOB:", "bbox": [10, 50, 40, 60], "confidence": 0.91},
            {"text": "01/01/1990", "bbox": [50, 50, 100, 60], "confidence": 0.96},
            {"text": "AADHAAR:", "bbox": [10, 70, 40, 80], "confidence": 0.90},
            {"text": "1234 5678 9012", "bbox": [50, 70, 120, 80], "confidence": 0.99},
            {"text": "GENDER:", "bbox": [10, 90, 40, 100], "confidence": 0.95},
            {"text": "MALE", "bbox": [50, 90, 80, 100], "confidence": 0.97}
        ]
        
        extracted_data = extractor.extract_fields(self.image, ocr_results)
        
        # Test PAN Number extraction
        self.assertIn("PAN Number", extracted_data)
        self.assertEqual(extracted_data["PAN Number"]["value"], "ABCDE1234F")
        self.assertEqual(extracted_data["PAN Number"]["extraction_method"], "Regex")
        self.assertTrue(extracted_data["PAN Number"]["confidence"] >= 0.90)
        
        # Test Aadhaar Number extraction
        self.assertIn("Aadhaar Number", extracted_data)
        self.assertEqual(extracted_data["Aadhaar Number"]["value"], "1234 5678 9012")
        
        # Test DOB extraction
        self.assertIn("Date of Birth", extracted_data)
        self.assertEqual(extracted_data["Date of Birth"]["value"], "01/01/1990")

        # Test Gender normalization
        self.assertIn("Gender", extracted_data)
        self.assertEqual(extracted_data["Gender"]["value"], "Male")

    def test_confidence_calculations(self):
        extractor = InformationExtractor()
        ocr_results = [
            {"text": "ABCDE1234F", "bbox": [10, 10, 100, 20], "confidence": 0.90}
        ]
        # Regex extraction confidence calculation
        conf_regex = extractor._calculate_confidence("ABCDE1234F", "PAN Number", ocr_results, is_regex=True)
        self.assertTrue(0.0 <= conf_regex <= 1.0)
        self.assertTrue(conf_regex >= 0.90)

        # VLM alignment checks
        conf_vlm_aligned = extractor._calculate_confidence("ABCDE1234F", "PAN Number", ocr_results, is_regex=False)
        self.assertEqual(conf_vlm_aligned, 0.95)
        
        conf_vlm_unaligned = extractor._calculate_confidence("Different Text", "PAN Number", ocr_results, is_regex=False)
        self.assertEqual(conf_vlm_unaligned, 0.78)
