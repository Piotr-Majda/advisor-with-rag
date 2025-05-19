import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
import os
import shutil
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch
import logging
from typing import AsyncGenerator
from app.main import app, Document, UpsertRequest
import json


# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        logger.debug(f"Open Client: {client}")
        yield client


@pytest.mark.asyncio
async def test_async_client(async_client):
    """Test the async client"""
    logger.debug(f"Open Client: {async_client}")


@pytest_asyncio.fixture(autouse=True)
async def mock_openai():
    """Mock OpenAI client for all tests"""
    mock_embeddings = MagicMock()

    # Create an async mock for the create method
    async def async_create(*args, **kwargs):
        return MagicMock(
            data=[MagicMock(embedding=np.array([0.1] * 1536, dtype=np.float32))]
        )

    # Assign the async mock to the create method
    mock_embeddings.create = async_create

    mock_client = MagicMock()
    mock_client.embeddings = mock_embeddings

    with patch(
        "app.vector_store.embeddings.openai.AsyncOpenAI", return_value=mock_client
    ):
        logger.debug(f"Mock Client: {mock_client}")
        yield mock_client
        logger.debug(f"Closing Mock Client: {mock_client}")


@pytest.mark.asyncio
async def test_mock_openai(mock_openai):
    """Test the openai client"""
    logger.debug(f"Open Client: {mock_openai}")


@pytest_asyncio.fixture(autouse=True)
async def setup_test_environment():
    """Setup and teardown for each test"""
    # Use temporary directory for tests
    import tempfile

    temp_dir = tempfile.mkdtemp()
    vector_store_dir = os.path.join(temp_dir, "vector_store")
    backup_dir = os.path.join(temp_dir, "backups")
    store_file = os.path.join(vector_store_dir, "store.pkl")

    # Create directories
    os.makedirs(vector_store_dir, exist_ok=True)
    os.makedirs(backup_dir, exist_ok=True)

    # Set environment variables to use temporary paths
    os.environ["VECTOR_STORE_PATH"] = store_file
    os.environ["BACKUP_PATH"] = backup_dir
    os.environ["VECTOR_DB_TYPE"] = "local"

    yield

    # Cleanup
    shutil.rmtree(temp_dir)
    for var in ["VECTOR_STORE_PATH", "BACKUP_PATH", "VECTOR_DB_TYPE"]:
        if var in os.environ:
            del os.environ[var]


class TestVectorServiceIntegration:
    @pytest.mark.asyncio
    async def test_document_lifecycle(self, async_client: AsyncClient, mock_openai):
        """Test complete document lifecycle: insert, query, update, delete"""
        # Insert document
        doc = UpsertRequest(
            documents=[
                Document(content="Test document content", metadata={"source": "test"})
            ]
        )
        request_data = doc.model_dump()
        insert_response = await async_client.post("/upsert", json=request_data)
        assert insert_response.status_code == 200
        doc_id = insert_response.json()["doc_ids"][0]

        # Query document
        query_response = await async_client.post(
            "/query", json={"query": "test", "k": 1}
        )
        assert query_response.status_code == 200
        assert len(query_response.json()["documents"]) == 1

        # Update document
        updated_doc = UpsertRequest(
            documents=[
                Document(
                    content="Updated test content",
                    metadata={"source": "test", "id": doc_id},
                )
            ]
        )
        update_response = await async_client.post(
            "/upsert", json=updated_doc.model_dump()
        )
        assert update_response.status_code == 200

        # Delete document - Fixed this part
        delete_data = {"ids": [doc_id]}
        delete_response = await async_client.request(
            "DELETE",  # Use `request()` to allow JSON body
            "/documents",
            headers={"Content-Type": "application/json"},
            data=json.dumps(delete_data),
        )
        assert delete_response.status_code == 200
        assert delete_response.json()["success"] is True

        # Verify deletion
        query_response = await async_client.post(
            "/query", json={"query": "test", "k": 1}
        )
        assert len(query_response.json()["documents"]) == 0

    @pytest.mark.asyncio
    async def test_backup_restore_workflow(self, async_client, mock_openai):
        """Test backup creation and restoration"""
        # Insert initial data
        doc = UpsertRequest(
            documents=[
                Document(
                    content="Backup test content", metadata={"source": "backup_test"}
                )
            ]
        )
        response = await async_client.post("/upsert", json=doc.model_dump())
        assert response.status_code == 200

        # Create backup
        backup_response = await async_client.post("/backup")
        assert backup_response.status_code == 200
        backup_path = backup_response.json()["backup_path"]
        assert os.path.exists(backup_path)

        # Insert new data
        new_doc = UpsertRequest(
            documents=[
                Document(
                    content="New content after backup",
                    metadata={"source": "post_backup"},
                )
            ]
        )
        response = await async_client.post("/upsert", json=new_doc.model_dump())
        assert response.status_code == 200

        # Restore from backup
        restore_response = await async_client.post(
            "/restore", json={"backup_path": backup_path}
        )
        assert restore_response.status_code == 200

        # Verify restored state
        query_response = await async_client.post(
            "/query", json={"query": "test", "k": 10}
        )
        documents = query_response.json()["documents"]
        assert any(d["content"] == "Backup test content" for d in documents)
        assert not any(d["content"] == "New content after backup" for d in documents)

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, async_client, mock_openai):
        """Test advanced metadata filtering capabilities"""
        # Insert documents with different metadata
        docs = UpsertRequest(
            documents=[
                Document(
                    content="Document A",
                    metadata={"category": "tech", "priority": "high"},
                ),
                Document(
                    content="Document B",
                    metadata={"category": "tech", "priority": "low"},
                ),
                Document(
                    content="Document C",
                    metadata={"category": "finance", "priority": "high"},
                ),
            ]
        )
        response = await async_client.post("/upsert", json=docs.model_dump())
        assert response.status_code == 200

        # Test single filter
        response = await async_client.post(
            "/query",
            json={
                "query": "Document",
                "k": 10,
                "filter_metadata": {"category": "tech"},
            },
        )
        results = response.json()["documents"]
        assert len(results) == 2
        assert all(d["metadata"]["category"] == "tech" for d in results)

        # Test multiple filters
        response = await async_client.post(
            "/query",
            json={
                "query": "Document",
                "k": 10,
                "filter_metadata": {"category": "tech", "priority": "high"},
            },
        )
        results = response.json()["documents"]
        assert len(results) == 1
        assert results[0]["metadata"]["priority"] == "high"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_input,expected_status",
        [
            ({"query": "", "k": 1}, 200),  # Empty query should return empty results
            ({"query": "test", "k": 0}, 422),  # Invalid k value
            ({"query": "test", "k": "invalid"}, 422),  # Wrong type for k
            ({"query": 123, "k": 1}, 422),  # Wrong type for query
        ],
    )
    async def test_input_validation(self, async_client, invalid_input, expected_status):
        """Test API input validation"""
        response = await async_client.post("/query", json=invalid_input)
        assert response.status_code == expected_status

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, async_client, mock_openai):
        """Test concurrent document operations"""
        # Prepare documents
        docs = [
            UpsertRequest(
                documents=[
                    Document(
                        content=f"Concurrent test document {i}", metadata={"test_id": i}
                    )
                ]
            )
            for i in range(10)
        ]

        # Concurrent insertion using asyncio.gather
        tasks = [async_client.post("/upsert", json=doc.model_dump()) for doc in docs]
        responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses)

        # Verify all documents were inserted
        query_response = await async_client.post(
            "/query", json={"query": "Concurrent test", "k": 10}
        )
        assert len(query_response.json()["documents"]) == 10
