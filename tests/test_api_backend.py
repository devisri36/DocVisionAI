import unittest
import io
import os
import sqlite3
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

# Mock base directory setup before importing database/app
os.environ["outputs_dir"] = "outputs_test"

from api.app import app
from backend.database import get_db_path, init_db

class TestAPIBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Patch VLM loading to return None for offline tests
        import backend.services.document_service
        backend.services.document_service.get_cached_vlm = lambda: (None, None)

        # Enforce clean test DB init
        db_path = get_db_path()
        if db_path.exists():
            try:
                db_path.unlink()
            except PermissionError:
                pass
        init_db()
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "Healthy")
        self.assertIn("database", data)

    def test_auth_and_restricted_endpoints(self):
        # 1. Attempt upload without token -> 401 Unauthorized
        img_io = io.BytesIO()
        Image.new("RGB", (100, 100), "white").save(img_io, "PNG")
        img_io.seek(0)
        
        response = self.client.post(
            "/upload",
            files={"file": ("test.png", img_io, "image/png")}
        )
        self.assertEqual(response.status_code, 401)

        # 2. Register user
        reg_response = self.client.post(
            "/auth/register",
            json={"username": "testuser", "password": "testpassword"}
        )
        self.assertEqual(reg_response.status_code, 201)

        # 3. Duplicate registration -> 400 Bad Request
        dup_response = self.client.post(
            "/auth/register",
            json={"username": "testuser", "password": "testpassword"}
        )
        self.assertEqual(dup_response.status_code, 400)

        # 4. Login -> Receive token
        login_response = self.client.post(
            "/auth/login",
            json={"username": "testuser", "password": "testpassword"}
        )
        self.assertEqual(login_response.status_code, 200)
        token_data = login_response.json()
        self.assertIn("access_token", token_data)
        token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 5. Upload document with token
        img_io.seek(0)
        upload_response = self.client.post(
            "/upload",
            headers=headers,
            files={"file": ("pan_card.png", img_io, "image/png")}
        )
        self.assertEqual(upload_response.status_code, 200)
        upload_data = upload_response.json()
        self.assertIn("file_id", upload_data)
        file_id = upload_data["file_id"]

        # 6. Verify Document Quality
        verify_response = self.client.post(
            "/verify",
            headers=headers,
            json={"file_id": file_id}
        )
        self.assertEqual(verify_response.status_code, 200)
        verify_data = verify_response.json()
        self.assertEqual(verify_data["file_id"], file_id)
        self.assertIn("quality_summary", verify_data)

        # 7. Detect Document Fraud Forensics
        fraud_response = self.client.post(
            "/detect-fraud",
            headers=headers,
            json={"file_id": file_id}
        )
        self.assertEqual(fraud_response.status_code, 200)
        fraud_data = fraud_response.json()
        self.assertEqual(fraud_data["file_id"], file_id)
        self.assertIn("ela_image_base64", fraud_data)
        self.assertIn("is_duplicate", fraud_data)

        # 8. Document Information Extraction and SQLite logging
        extract_response = self.client.post(
            "/extract",
            headers=headers,
            json={"file_id": file_id}
        )
        self.assertEqual(extract_response.status_code, 200)
        extract_data = extract_response.json()
        self.assertEqual(extract_data["file_id"], file_id)
        self.assertIn("extracted_fields", extract_data)
        self.assertIn("scores", extract_data)
        
        # 9. VLM QA
        ask_response = self.client.post(
            "/ask",
            headers=headers,
            json={"file_id": file_id, "question": "What is the name?"}
        )
        self.assertEqual(ask_response.status_code, 200)
        ask_data = ask_response.json()
        self.assertEqual(ask_data["file_id"], file_id)
        self.assertIn("answer", ask_data)

        # 10. Metrics endpoint check
        metrics_response = self.client.get("/metrics", headers=headers)
        self.assertEqual(metrics_response.status_code, 200)
        metrics_data = metrics_response.json()
        self.assertTrue(metrics_data["total_verifications"] >= 1)
        self.assertIn("average_authenticity", metrics_data)

        # 11. History logging check
        history_response = self.client.get("/history", headers=headers)
        self.assertEqual(history_response.status_code, 200)
        history_data = history_response.json()
        self.assertTrue(len(history_data) >= 1)
        self.assertEqual(history_data[0]["filename"], "pan_card.png")
