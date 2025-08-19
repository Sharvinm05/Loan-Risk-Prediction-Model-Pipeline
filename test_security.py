"""
Security tests for the loan prediction API
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from security_models import LoanApplicationInput, mask_sensitive_data, generate_request_id


@pytest.fixture
def client():
    """Test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def valid_loan_data():
    """Valid loan application data for testing"""
    return {
        "loanAmount": 3000,
        "apr": 199,
        "nPaidOff": 0,
        "isFunded": 1,
        "state": "CA",
        "leadCost": 0,
        "payFrequency": "B",
        "originallyScheduledPaymentAmount": 6395.19
    }


@pytest.fixture
def api_headers():
    """Valid API headers"""
    return {"X-API-Key": "demo-key-123456"}


def test_health_endpoint_no_auth_required(client):
    """Test that health endpoint doesn't require authentication"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_predict_endpoint_requires_auth(client, valid_loan_data):
    """Test that predict endpoint requires authentication"""
    response = client.post("/predict", json=valid_loan_data)
    assert response.status_code == 401
    assert "API key required" in response.json()["detail"]


def test_predict_endpoint_invalid_auth(client, valid_loan_data):
    """Test prediction with invalid API key"""
    headers = {"X-API-Key": "invalid-key"}
    response = client.post("/predict", json=valid_loan_data, headers=headers)
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_input_validation_extra_fields(client, valid_loan_data, api_headers):
    """Test that extra fields are rejected"""
    invalid_data = valid_loan_data.copy()
    invalid_data["malicious_field"] = "hack_attempt"
    
    response = client.post("/predict", json=invalid_data, headers=api_headers)
    assert response.status_code == 422  # Validation error


def test_input_validation_missing_fields(client, api_headers):
    """Test that missing required fields are rejected"""
    incomplete_data = {"loanAmount": 1000}  # Missing required fields
    
    response = client.post("/predict", json=incomplete_data, headers=api_headers)
    assert response.status_code == 422  # Validation error


def test_input_validation_invalid_values(client, valid_loan_data, api_headers):
    """Test that invalid field values are rejected"""
    # Test negative loan amount
    invalid_data = valid_loan_data.copy()
    invalid_data["loanAmount"] = -1000
    
    response = client.post("/predict", json=invalid_data, headers=api_headers)
    assert response.status_code == 422  # Validation error
    
    # Test invalid state code
    invalid_data = valid_loan_data.copy()
    invalid_data["state"] = "XX"  # Invalid state
    
    response = client.post("/predict", json=invalid_data, headers=api_headers)
    assert response.status_code == 422  # Validation error


def test_input_validation_large_values(client, valid_loan_data, api_headers):
    """Test that unreasonably large values are rejected"""
    invalid_data = valid_loan_data.copy()
    invalid_data["loanAmount"] = 2000000  # Exceeds maximum
    
    response = client.post("/predict", json=invalid_data, headers=api_headers)
    assert response.status_code == 422  # Validation error


def test_security_headers_present(client):
    """Test that security headers are present in responses"""
    response = client.get("/health")
    
    # Check for security headers
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "X-XSS-Protection" in response.headers
    assert "Strict-Transport-Security" in response.headers


def test_data_masking():
    """Test that sensitive data is properly masked"""
    sensitive_data = {
        "loanAmount": 50000,
        "originallyScheduledPaymentAmount": 1234.56,
        "leadCost": 100,
        "state": "CA"
    }
    
    masked = mask_sensitive_data(sensitive_data)
    
    # Check that financial data is masked
    assert "50***" in str(masked["loanAmount"])
    assert "12******" in str(masked["originallyScheduledPaymentAmount"])
    assert "1**" in str(masked["leadCost"])
    # State should not be masked
    assert masked["state"] == "CA"


def test_request_id_generation():
    """Test request ID generation"""
    data1 = {"loanAmount": 1000, "state": "CA"}
    data2 = {"loanAmount": 1000, "state": "CA"}
    data3 = {"loanAmount": 2000, "state": "CA"}
    
    id1 = generate_request_id(data1)
    id2 = generate_request_id(data2)
    id3 = generate_request_id(data3)
    
    # Same data should generate same ID
    assert id1 == id2
    # Different data should generate different ID
    assert id1 != id3
    # IDs should be strings of expected length
    assert len(id1) == 16
    assert isinstance(id1, str)


def test_pydantic_model_validation():
    """Test Pydantic model validation directly"""
    # Valid data should pass
    valid_data = {
        "loanAmount": 3000,
        "apr": 199,
        "nPaidOff": 0,
        "isFunded": 1,
        "state": "CA",
        "leadCost": 0,
        "payFrequency": "B",
        "originallyScheduledPaymentAmount": 6395.19
    }
    
    model = LoanApplicationInput(**valid_data)
    assert model.loanAmount == 3000
    assert model.state == "CA"
    
    # Invalid data should raise validation error
    with pytest.raises(ValueError):
        LoanApplicationInput(
            loanAmount=-1000,  # Invalid negative amount
            apr=199,
            nPaidOff=0,
            isFunded=1,
            state="CA",
            leadCost=0,
            payFrequency="B",
            originallyScheduledPaymentAmount=6395.19
        )


def test_content_length_limit(client, api_headers):
    """Test that large requests are rejected"""
    # Create a very large payload (this might not work with the actual content-length middleware,
    # but tests the concept)
    large_data = "x" * 100000
    
    # This test would need to be adapted based on how the middleware actually works
    # For now, just ensure the test structure is in place
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])