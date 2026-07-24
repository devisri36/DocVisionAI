import unittest
from pathlib import Path
import torch

from metrics import (
    levenshtein_distance,
    compute_exact_match,
    compute_f1_score,
    compute_cer,
    compute_wer,
    compute_anls,
    compute_batch_metrics
)
from backend.loaders.base import DocumentItem
from dataset import FlorenceDocumentDataset

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        
    def __call__(self, text, **kwargs):
        class MockIds:
            def squeeze(self, dim):
                return torch.tensor([1, 2, 3])
        return {"input_ids": MockIds()}

class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        
    def __call__(self, text, images, **kwargs):
        class MockInputs:
            def squeeze(self, dim):
                if dim == 0:
                    return torch.zeros((3, 384, 384)) if "pixel_values" in self.data else torch.tensor([1, 2, 3])
            def __getitem__(self, key):
                if key == "pixel_values":
                    return torch.zeros((1, 3, 384, 384))
                return torch.tensor([[1, 2, 3]])
        mock_in = MockInputs()
        mock_in.data = ["pixel_values", "input_ids"]
        return mock_in

class TestDLPipeline(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("cat", "cat"), 0)
        self.assertEqual(levenshtein_distance("cat", "bat"), 1)
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)

    def test_metrics_computation(self):
        pred = "total amount is 100.0"
        ref = "total amount is 100.0"
        self.assertEqual(compute_exact_match(pred, ref), 1.0)
        self.assertEqual(compute_f1_score(pred, ref), 1.0)
        self.assertEqual(compute_cer(pred, ref), 0.0)
        self.assertEqual(compute_wer(pred, ref), 0.0)
        
        pred_err = "total amount is 90.0"
        self.assertEqual(compute_exact_match(pred_err, ref), 0.0)
        self.assertTrue(compute_cer(pred_err, ref) > 0.0)
        self.assertTrue(compute_wer(pred_err, ref) > 0.0)

    def test_anls_metric(self):
        # ANLS for exact matches should be 1.0
        self.assertEqual(compute_anls("total: $15.50", "total: $15.50"), 1.0)
        # Below threshold (0.5) similarity is set to 0.0
        self.assertEqual(compute_anls("total: $15.50", "completely different"), 0.0)
        # Near matches
        anls_score = compute_anls("total: 15.50", "total: $15.50")
        self.assertTrue(0.5 < anls_score < 1.0)

    def test_dataset_prompt_generation(self):
        # Create a mock document item
        item = DocumentItem(
            image_path=Path("mock.png"),
            words=["Burger", "King"],
            bboxes=[[10, 20, 30, 40], [50, 60, 70, 80]],
            labels=["company", "company"],
            metadata={"company": "Burger King", "total": "$15"}
        )
        
        # Instantiate dataset with mock loader data
        dataset = FlorenceDocumentDataset.__new__(FlorenceDocumentDataset)
        dataset.dataset_name = "sroie"
        dataset.split = "train"
        dataset.processor = MockProcessor()
        dataset.preprocessor = None
        
        prompt, target = dataset._generate_florence_target(item)
        self.assertEqual(prompt, "<KIE>")
        self.assertIn("company: Burger King", target)
        self.assertIn("total: $15", target)

        # FUNSD
        dataset.dataset_name = "funsd"
        prompt, target = dataset._generate_florence_target(item)
        self.assertEqual(prompt, "<OCR_WITH_REGION>")
        self.assertIn("Burger <loc_20><loc_10><loc_40><loc_30> (company)", target)
        
        # DocVQA
        item_vqa = DocumentItem(
            image_path=Path("mock.png"),
            words=["text"],
            bboxes=[[0,0,1,1]],
            metadata={"question": "What is the total?", "answers": ["$15"]}
        )
        dataset.dataset_name = "docvqa"
        prompt, target = dataset._generate_florence_target(item_vqa)
        self.assertEqual(prompt, "<DocVQA> Question: What is the total?")
        self.assertEqual(target, "$15")
        
    def test_batch_metrics(self):
        preds = ["hello", "world"]
        refs = ["hello", "world"]
        res = compute_batch_metrics(preds, refs, task="vqa")
        self.assertEqual(res["exact_match"], 1.0)
        self.assertEqual(res["f1"], 1.0)
        self.assertEqual(res["anls"], 1.0)
