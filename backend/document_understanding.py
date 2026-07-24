import re
from typing import Dict, Any, List, Optional
from PIL import Image
import torch
import numpy as np

from configs.config_loader import ConfigManager
from utils.logger import setup_logger
from backend.ocr_engine import DocVisionOCREngine
from model import get_device

logger = setup_logger("document_understanding")

class DocumentClassifier:
    """Classifies document images using a hybrid rule-based and VLM zero-shot approach."""

    def __init__(self, florence_model: Any = None, florence_processor: Any = None):
        self.model = florence_model
        self.processor = florence_processor
        self.device = get_device()
        
        # Define keywords for heuristic checks
        self.rules = {
            "Aadhaar": [r"aadhaar", r"unique identification", r"government of india", r"uidai", r"enrollment"],
            "PAN": [r"permanent account number", r"income tax department", r"pan card", r"govt of india"],
            "Passport": [r"passport", r"republic of india", r"code of issuing state", r"nationality"],
            "Driving License": [r"driving license", r"driving licence", r"licence to drive", r"transport department", r"authorization to drive"],
            "Invoice": [r"invoice", r"bill to", r"tax invoice", r"invoice number", r"purchase order", r"po number", r"amount due"],
            "Bank Statement": [r"bank statement", r"statement of account", r"account number", r"transaction details", r"balance summary", r"withdrawal", r"deposit"],
            "College ID": [r"college", r"university", r"student id", r"student card", r"roll no", r"regd\.? no", r"b\.tech", r"engineering"]
        }

    def classify(self, image: Image.Image, ocr_text: str) -> Dict[str, Any]:
        """Classifies the image and returns the category along with a confidence score.
        
        Returns:
            Dict: {"category": "PAN", "confidence": 0.95, "method": "Rule-based"}
        """
        ocr_text_lower = ocr_text.lower()
        
        # 1. Try Rule-based / Keyword Heuristic Check
        counts = {}
        for category, patterns in self.rules.items():
            count = 0
            for pattern in patterns:
                if re.search(pattern, ocr_text_lower):
                    count += 1
            if count > 0:
                counts[category] = count
                
        if counts:
            # Sort categories by matches count
            best_cat = max(counts, key=counts.get)
            match_percentage = counts[best_cat] / len(self.rules[best_cat])
            # Scale confidence to [0.8, 0.99] range based on keyword overlaps
            confidence = min(0.99, 0.75 + (match_percentage * 0.25))
            
            logger.info(f"Heuristics classified document as '{best_cat}' with confidence {confidence:.2f}")
            return {
                "category": best_cat,
                "confidence": round(confidence, 2),
                "method": "Rule-based"
            }

        # 2. Try VLM Zero-Shot Classification (if model is loaded)
        if self.model is not None and self.processor is not None:
            try:
                prompt = (
                    "<DocVQA> Question: What type of document is this: "
                    "Aadhaar, PAN, Passport, Driving License, Invoice, or Bank Statement?"
                )
                
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=inputs["pixel_values"],
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=32,
                        early_stopping=True
                    )
                
                prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                logger.info(f"VLM Classification Raw Output: '{prediction}'")
                
                # Check for category match in prediction string
                for category in self.rules.keys():
                    if category.lower() in prediction.lower():
                        return {
                            "category": category,
                            "confidence": 0.85,
                            "method": "VLM Zero-Shot"
                        }
            except Exception as e:
                logger.error(f"Error during VLM classification: {e}")

        # 3. Default Fallback
        logger.warning("Document classification was inconclusive. Defaulting to 'Invoice'.")
        return {
            "category": "Invoice",
            "confidence": 0.50,
            "method": "Default Fallback"
        }


class InformationExtractor:
    """Extracts key identity and financial fields from documents using Regex + VLM QA queries."""

    def __init__(self, florence_model: Any = None, florence_processor: Any = None):
        self.model = florence_model
        self.processor = florence_processor
        self.device = get_device()
        
        # Regex mappings for identity/receipt patterns
        self.regex_patterns = {
            "Student ID / Reg Number": r"\b[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9A-Z]{4}\b",
            "Aadhaar Number": r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b",
            "PAN Number": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
            "Date of Birth": r"\b\d{2}[-/]\d{2}[-/]\d{4}\b|\b\d{4}[-/]\d{2}[-/]\d{2}\b",
            "Gender": r"\b(male|female|transgender|m\b|f\b)\b",
            "Expiry Date": r"\b(?:expiry|exp|valid\s+thru|valid\s+up\s+to|valid\s+till|till)\b.*?\b(\d{2}[-/]\d{2}[-/]\d{4}|\d{2}[-/]\d{2})\b"
        }

    def _query_vlm(self, image: Image.Image, question: str) -> str:
        """Queries the VLM for a specific fact about the document."""
        if self.model is None or self.processor is None:
            return ""
            
        try:
            prompt = f"<DocVQA> Question: {question}"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=128,
                    early_stopping=True
                )
                
            prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return prediction
        except Exception as e:
            logger.error(f"VLM QA execution failed for question '{question}': {e}")
            return ""

    def extract_fields(self, image: Image.Image, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts target structured fields from document using OCR-Regex parsing and VLM lookups."""
        ocr_text = " ".join([item["text"] for item in ocr_results])
        ocr_text_lower = ocr_text.lower()
        
        extracted_data = {}
        
        # 1. Extract Aadhaar, PAN, Gender, DOB and Expiry Date via Regex Heuristics
        for field, pattern in self.regex_patterns.items():
            match = re.search(pattern, ocr_text, re.IGNORECASE)
            if match:
                extracted_value = match.group(0).strip()
                # Clean nested groups if necessary (e.g. Expiry Date)
                if field == "Expiry Date" and len(match.groups()) > 0 and match.group(1):
                    extracted_value = match.group(1).strip()
                    
                # Capitalize codes
                if field in ["PAN Number", "Aadhaar Number"]:
                    extracted_value = extracted_value.upper()
                elif field == "Gender":
                    val_lower = extracted_value.lower()
                    extracted_value = "Female" if val_lower in ["female", "f"] else "Male"
                    
                extracted_data[field] = {
                    "value": extracted_value,
                    "confidence": self._calculate_confidence(extracted_value, field, ocr_results, is_regex=True),
                    "extraction_method": "Regex"
                }

        # 2. Extract missing fields using VLM (Aadhaar/PAN/DOB if regex failed, plus Name and Address)
        vlm_fields = {
            "Name": "What is the full name of the person?",
            "Address": "What is the complete address?",
            "Aadhaar Number": "What is the Aadhaar card number?",
            "PAN Number": "What is the Permanent Account Number (PAN)?",
            "Date of Birth": "What is the Date of Birth (DOB)?",
            "Gender": "What is the gender of the card holder?",
            "Expiry Date": "What is the expiry date of this document?"
        }

        for field, question in vlm_fields.items():
            # Skip if already extracted with high confidence using Regex
            if field in extracted_data and extracted_data[field]["confidence"] >= 0.90:
                continue
                
            vlm_value = self._query_vlm(image, question)
            if vlm_value and vlm_value.lower() not in ["not found", "unknown", "n/a", "none"]:
                # If regex had a low confidence match, or if VLM provides a new value
                extracted_data[field] = {
                    "value": vlm_value,
                    "confidence": self._calculate_confidence(vlm_value, field, ocr_results, is_regex=False),
                    "extraction_method": "VLM"
                }
            elif field not in extracted_data:
                # Store null-placeholder if no extraction worked
                extracted_data[field] = {
                    "value": None,
                    "confidence": 0.0,
                    "extraction_method": "Inconclusive"
                }

        # Regex fallback for Name when VLM is offline/unavailable
        if "Name" not in extracted_data or extracted_data["Name"]["value"] is None:
            name_match = re.search(r"Name:?\s+([A-Za-z\s]+?)(?=\s+(?:Date|DOB|Father|Permanent|Aadhaar|PAN|Gender|Expiry|$))", ocr_text, re.IGNORECASE)
            if name_match:
                name_val = name_match.group(1).strip()
                extracted_data["Name"] = {
                    "value": name_val,
                    "confidence": 0.92,
                    "extraction_method": "Regex Fallback"
                }

        return extracted_data

    def _calculate_confidence(
        self, 
        value: str, 
        field: str, 
        ocr_results: List[Dict[str, Any]], 
        is_regex: bool = True
    ) -> float:
        """Heuristically calculates confidence score [0.0, 1.0] for the extracted value."""
        if not value:
            return 0.0
            
        # If extracted via Regex matching on raw OCR, we base confidence on matching character confidences
        if is_regex:
            # Find which OCR segments matched the extracted value
            matched_confidences = []
            val_clean = value.replace(" ", "").lower()
            
            for item in ocr_results:
                item_clean = item["text"].replace(" ", "").lower()
                if item_clean in val_clean or val_clean in item_clean:
                    matched_confidences.append(item["confidence"])
                    
            if matched_confidences:
                # Average OCR confidence + Regex bonus (Regex structure provides high layout confirmation)
                avg_ocr_conf = sum(matched_confidences) / len(matched_confidences)
                return round(min(0.99, avg_ocr_conf * 0.95 + 0.05), 2)
            return 0.92  # Default high score for Regex confirmation

        # If extracted via VLM, we verify if the extracted text string matches or exists inside our OCR output
        val_clean = value.lower().replace(" ", "")
        ocr_text = " ".join([item["text"] for item in ocr_results]).lower().replace(" ", "")
        
        if val_clean in ocr_text:
            # If VLM output aligns perfectly with raw OCR characters, high confidence
            return 0.95
        else:
            # If VLM generates text not directly found in raw OCR, we give a moderate confidence score
            # representing model generation uncertainty
            return 0.78


class VLMQAEngine:
    """Wrapper that enables general visual question-answering on documents."""

    def __init__(self, florence_model: Any, florence_processor: Any):
        self.model = florence_model
        self.processor = florence_processor
        self.device = get_device()

    def answer_question(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Runs Florence-2 DocVQA task to answer arbitrary document queries.
        
        Returns:
            Dict: {"question": "...", "answer": "...", "confidence": ...}
        """
        if self.model is None or self.processor is None:
            return {
                "question": question,
                "answer": "Backend VLM is currently unavailable.",
                "confidence": 0.0
            }

        try:
            prompt = f"<DocVQA> Question: {question}"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    early_stopping=True
                )
                
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Estimate a basic confidence score
            confidence = 0.88 if answer and answer.lower() not in ["not found", "n/a"] else 0.0
            
            return {
                "question": question,
                "answer": answer,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Inference QA failed for: '{question}': {e}")
            return {
                "question": question,
                "answer": f"Processing failed: {str(e)}",
                "confidence": 0.0
            }
