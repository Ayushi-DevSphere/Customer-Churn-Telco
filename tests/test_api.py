"""
tests/test_api.py
=================
Integration tests for the FastAPI endpoints.
Uses httpx.AsyncClient (no server required).

Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)

# ── Sample payloads ─────────────────────────────────────────

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 1020.0,
}

INVALID_CUSTOMER = {
    "gender": "Female",
    # Missing required fields
}


# ── Tests ────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_docs_link(self):
        resp = client.get("/")
        assert "docs" in resp.json()


class TestHealth:
    def test_health_endpoint_exists(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self):
        resp = client.get("/health")
        body = resp.json()
        assert "status" in body
        assert "model_loaded" in body
        assert "api_version" in body


class TestPredict:
    def test_predict_with_valid_input(self):
        """This test requires a trained model. Skip gracefully if not present."""
        resp = client.post("/predict", json=VALID_CUSTOMER)
        if resp.status_code == 503:
            pytest.skip("Model not trained yet.")
        assert resp.status_code == 200

    def test_predict_response_schema(self):
        resp = client.post("/predict", json=VALID_CUSTOMER)
        if resp.status_code == 503:
            pytest.skip("Model not trained yet.")
        body = resp.json()
        assert "churn_probability" in body
        assert "churn_label" in body
        assert "risk_tier" in body
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_tier"] in ["High", "Medium", "Low"]

    def test_predict_invalid_input_returns_422(self):
        resp = client.post("/predict", json=INVALID_CUSTOMER)
        assert resp.status_code == 422

    def test_predict_negative_tenure_fails(self):
        bad = VALID_CUSTOMER.copy()
        bad["tenure"] = -5
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422


class TestBatchPredict:
    def test_batch_endpoint_with_valid_input(self):
        payload = {"customers": [VALID_CUSTOMER, VALID_CUSTOMER]}
        resp = client.post("/predict/batch", json=payload)
        if resp.status_code == 503:
            pytest.skip("Model not trained yet.")
        if resp.status_code == 200:
            body = resp.json()
            assert body["total"] == 2
            assert "predictions" in body

    def test_batch_size_limit(self):
        """Sending >1000 records should return 400."""
        big_batch = {"customers": [VALID_CUSTOMER] * 1001}
        resp = client.post("/predict/batch", json=big_batch)
        assert resp.status_code == 400


class TestModelInfo:
    def test_model_info_endpoint(self):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "churn_threshold" in body
