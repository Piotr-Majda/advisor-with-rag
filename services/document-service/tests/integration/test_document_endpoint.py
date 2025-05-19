import pytest
from fastapi.testclient import TestClient
import os
import io
from unittest.mock import patch, MagicMock
from app import app

client = TestClient(app)

@pytest.fixture
def sample_pdf():
    # Create a more valid PDF file in memory
    return io.BytesIO(
        b"%PDF-1.7\n"
        b"1 0 obj\n"
        b"<</Type/Catalog/Pages 2 0 R>>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<<>>>>\n"
        b"endobj\n"
        b"xref\n"
        b"0 4\n"
        b"0000000000 65535 f\n"
        b"0000000010 00000 n\n"
        b"0000000053 00000 n\n"
        b"0000000102 00000 n\n"
        b"trailer\n"
        b"<</Size 4/Root 1 0 R>>\n"
        b"startxref\n"
        b"183\n"
        b"%%EOF\n"
    )

@pytest.fixture
def mock_vector_service():
    with patch('requests.post') as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"success": True, "doc_ids": ["doc_1", "doc_2"]}
        )
        yield mock_post

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup: Set test environment variables
    os.environ['VECTOR_SERVICE_URL'] = 'http://test-vector-service:8004'
    os.environ['OPENAI_API_KEY'] = 'test-key'
    
    yield
    
    # Teardown: Clean up environment
    if 'VECTOR_SERVICE_URL' in os.environ:
        del os.environ['VECTOR_SERVICE_URL']
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']

class TestDocumentEndpoint:
    def test_process_single_pdf(self, sample_pdf, mock_vector_service):
        files = [
            ("files", ("test.pdf", sample_pdf, "application/pdf"))
        ]
        
        response = client.post("/process", files=files)
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["message"] == "Documents processed successfully"
        
        # Verify vector service was called
        mock_vector_service.assert_called_once()
        call_args = mock_vector_service.call_args[1]
        assert "documents" in call_args["json"]

    def test_process_multiple_pdfs(self, sample_pdf, mock_vector_service):
        files = [
            ("files", (f"test{i}.pdf", sample_pdf, "application/pdf"))
            for i in range(2)
        ]
        
        response = client.post("/process", files=files)
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["message"] == "Documents processed successfully"
        
        # Verify vector service was called with multiple documents
        mock_vector_service.assert_called_once()
        call_args = mock_vector_service.call_args[1]
        assert "documents" in call_args["json"]

    def test_process_empty_pdf(self):
        files = [
            ("files", ("empty.pdf", io.BytesIO(b""), "application/pdf"))
        ]
        
        response = client.post("/process", files=files)
        assert response.status_code == 200
        assert response.json()["success"] is False
        assert "error" in response.json()

    def test_process_invalid_pdf(self):
        files = [
            ("files", ("invalid.pdf", io.BytesIO(b"not a pdf"), "application/pdf"))
        ]
        
        response = client.post("/process", files=files)
        assert response.status_code == 200
        assert response.json()["success"] is False
        assert "error" in response.json()

    def test_process_no_files(self):
        response = client.post("/process", files=[])
        assert response.status_code == 422  # FastAPI validation error

    def test_vector_service_error(self, sample_pdf):
        with patch('requests.post') as mock_post:
            mock_post.return_value = MagicMock(
                status_code=500,
                json=lambda: {"error": "Internal server error"}
            )
            
            files = [("files", ("test.pdf", sample_pdf, "application/pdf"))]
            response = client.post("/process", files=files)
            
            assert response.status_code == 200
            assert response.json()["success"] is False
            assert "Failed to store vectors" in response.json()["error"]

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
