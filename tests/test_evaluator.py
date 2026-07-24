import unittest
from PIL import Image
from pathlib import Path
import os
import csv

from evaluation.evaluator import (
    ImagePerturbationEngine,
    PerformanceEvaluator,
    ResearchBenchmarkRunner
)

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 200), "white")
        self.runner = ResearchBenchmarkRunner()

    def test_perturbations(self):
        # 1. Blur
        blurred = ImagePerturbationEngine.apply_blur(self.image, radius=2)
        self.assertEqual(blurred.size, self.image.size)
        
        # 2. Noise
        noisy = ImagePerturbationEngine.apply_noise(self.image, std_dev=10.0)
        self.assertEqual(noisy.size, self.image.size)
        
        # 3. Rotation
        rotated = ImagePerturbationEngine.apply_rotation(self.image, angle=45.0)
        self.assertTrue(rotated.size[0] >= self.image.size[0])
        
        # 4. Adversarial
        adversarial = ImagePerturbationEngine.apply_adversarial(self.image)
        self.assertEqual(adversarial.size, self.image.size)

    def test_nlp_metrics(self):
        # Equal strings
        acc, prec, rec, f1 = PerformanceEvaluator.calculate_text_metrics("John Doe", "John Doe")
        self.assertEqual(acc, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)
        
        # Partial match
        acc, prec, rec, f1 = PerformanceEvaluator.calculate_text_metrics("John", "John Doe")
        self.assertEqual(acc, 0.5)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 0.5)
        self.assertTrue(f1 > 0.0)
        
        # Mismatched strings
        acc, prec, rec, f1 = PerformanceEvaluator.calculate_text_metrics("Smith", "John Doe")
        self.assertEqual(acc, 0.0)

    def test_memory_measurement(self):
        mem = PerformanceEvaluator.measure_memory_mb()
        self.assertTrue(mem > 0.0)

    def test_benchmark_runner_exports(self):
        res = self.runner.run_comparative_benchmark(self.image)
        
        csv_path = Path(res["csv_path"])
        report_path = Path(res["report_path"])
        
        self.assertTrue(csv_path.exists())
        self.assertTrue(report_path.exists())
        
        # Validate CSV columns
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertIn("Model", headers)
            self.assertIn("Condition", headers)
            self.assertIn("Accuracy", headers)
            self.assertIn("Latency_ms", headers)
            self.assertIn("Memory_MB", headers)

        # Clean up files
        try:
            csv_path.unlink()
            report_path.unlink()
        except OSError:
            pass
