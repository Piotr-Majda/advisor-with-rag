import pytest
from unittest.mock import Mock, patch, mock_open
import os
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

@pytest.fixture
def mock_pdf_file():
    return Mock(spec=UploadFile, filename="test.pdf")

@pytest.fixture
def mock_empty_pdf_file():
    mock_file = Mock(spec=UploadFile, filename="empty.pdf")
    mock_file.read = Mock(return_value=b"")
    return mock_file

@pytest.fixture
def mock_pdf_loader():
    return Mock(spec=PyPDFLoader)

class TestDocumentProcessing:
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.getsize")
    async def test_empty_file_detection(self, mock_getsize, mock_temp_file, mock_empty_pdf_file):
        from app import process_documents
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "temp.pdf"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Mock file size as 0
        mock_getsize.return_value = 0
        
        result = await process_documents([mock_empty_pdf_file])
        
        assert result == {
            "success": False,
            "error": "Empty file: empty.pdf"
        }

    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.getsize")
    @patch("langchain_community.document_loaders.PyPDFLoader")
    async def test_successful_file_processing(
        self, mock_pdf_loader_class, mock_getsize, mock_temp_file, mock_pdf_file
    ):
        from app import process_documents
        
        # Mock file content
        mock_pdf_file.read = Mock(return_value=b"%PDF-1.4 content")
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "temp.pdf"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Mock file size as non-zero
        mock_getsize.return_value = 1000
        
        # Mock PDF loader
        mock_loader = Mock()
        mock_loader.load.return_value = [
            Document(page_content="Page 1 content", metadata={"page": 1}),
            Document(page_content="Page 2 content", metadata={"page": 2})
        ]
        mock_pdf_loader_class.return_value = mock_loader
        
        with patch("os.unlink") as mock_unlink:
            result = await process_documents([mock_pdf_file])
            
            # Verify temporary file was cleaned up
            mock_unlink.assert_called_once_with(mock_temp.name)
        
        # Verify PDF was processed
        mock_loader.load.assert_called_once()

    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.getsize")
    @patch("langchain_community.document_loaders.PyPDFLoader")
    async def test_multiple_files_processing(
        self, mock_pdf_loader_class, mock_getsize, mock_temp_file
    ):
        from app import process_documents
        
        # Create multiple mock files
        mock_files = [
            Mock(spec=UploadFile, filename=f"test{i}.pdf")
            for i in range(2)
        ]
        for mock_file in mock_files:
            mock_file.read = Mock(return_value=b"%PDF-1.4 content")
        
        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "temp.pdf"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        # Mock file size as non-zero
        mock_getsize.return_value = 1000
        
        # Mock PDF loader
        mock_loader = Mock()
        mock_loader.load.return_value = [
            Document(page_content=f"File {i} content", metadata={"page": 1})
            for i in range(2)
        ]
        mock_pdf_loader_class.return_value = mock_loader
        
        with patch("os.unlink") as mock_unlink:
            await process_documents(mock_files)
            
            # Verify temporary file was cleaned up for each file
            assert mock_unlink.call_count == len(mock_files)

    @patch("tempfile.NamedTemporaryFile")
    async def test_file_read_error(self, mock_temp_file, mock_pdf_file):
        from app import process_documents
        
        # Mock file read error
        mock_pdf_file.read = Mock(side_effect=Exception("File read error"))
        
        result = await process_documents([mock_pdf_file])
        
        assert result["success"] is False
        assert "error" in result