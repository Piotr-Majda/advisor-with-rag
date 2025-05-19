import pytest
from unittest.mock import MagicMock, call
from app.services.vector_service import VectorService
from app.vector_store.vector_store import SearchResult, Document
from app.vector_store.protocols import VectorDatabaseProtocol, EmbeddingService
import numpy as np


@pytest.fixture
async def mock_vector_db():
    db = MagicMock(spec=VectorDatabaseProtocol)
    # Set up async return values
    db.add_vectors.return_value = ["doc_id_1"]
    db.search_vectors.return_value = [
        SearchResult(content="test content", metadata={"source": "test"}, score=0.95)
    ]
    db.is_healthy.return_value = True
    # Make async methods return coroutines
    db.initialize = MagicMock(return_value=None)
    db.add_vectors = MagicMock(return_value=["doc_id_1"])
    db.search_vectors = MagicMock(
        return_value=[
            SearchResult(
                content="test content", metadata={"source": "test"}, score=0.95
            )
        ]
    )
    db.delete_vectors = MagicMock(return_value=True)
    db.create_backup = MagicMock(return_value="/path/to/backup")
    db.restore_from_backup = MagicMock(return_value=None)
    return db


@pytest.fixture
async def mock_embedding_service():
    service = MagicMock(spec=EmbeddingService)
    # Set up async return values
    service.get_embedding = MagicMock(return_value=np.array([0.1] * 1536))
    service.get_embeddings = MagicMock(return_value=np.array([[0.1] * 1536]))
    return service


@pytest.fixture
async def vector_service(
    mock_vector_db: VectorDatabaseProtocol, mock_embedding_service: EmbeddingService
) -> VectorService:
    service = VectorService(
        vector_db=mock_vector_db, embedding_service=mock_embedding_service
    )
    await service.initialize()
    return service


class TestAddDocuments:
    """Tests for adding documents to the vector service"""

    @pytest.mark.asyncio
    async def test_add_single_valid_document(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
        mock_embedding_service: EmbeddingService,
    ):
        # Arrange
        document = {"content": "test content", "metadata": {"source": "test"}}

        # Act
        doc_ids = await vector_service.add_documents([document])

        # Assert
        assert doc_ids == ["doc_id_1"]
        mock_embedding_service.get_embeddings.assert_called_once_with(["test content"])
        # Verify the document was properly transformed
        call_args = mock_vector_db.add_vectors.call_args[0]
        vectors, docs = call_args
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content == "test content"
        assert docs[0].metadata["source"] == "test"
        assert "timestamp" in docs[0].metadata

    @pytest.mark.asyncio
    async def test_add_empty_document_list(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
        mock_embedding_service: EmbeddingService,
    ):
        # Arrange
        mock_vector_db.add_vectors = MagicMock(return_value=[])

        # Act
        result = await vector_service.add_documents([])

        # Assert
        assert result == []
        # Get the actual call arguments
        call_args = mock_vector_db.add_vectors.call_args[0]
        vectors, docs = call_args
        # Verify the arguments
        assert vectors.size == 0  # Empty numpy array
        assert docs == []  # Empty list of documents
        mock_embedding_service.get_embeddings.assert_not_called()  # Should not be called for empty list

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_doc, error_msg",
        [
            ({"metadata": {}}, "Document missing required 'content' field"),
            ({"content": 123, "metadata": {}}, "Document 'content' must be a string"),
            (["not a dict"], "Invalid document format: expected dict"),
        ],
    )
    async def test_add_invalid_document(
        self, vector_service: VectorService, invalid_doc, error_msg
    ):
        with pytest.raises(ValueError, match=error_msg):
            await vector_service.add_documents([invalid_doc])


class TestSearchDocuments:
    """Tests for searching documents in the vector service"""

    @pytest.mark.asyncio
    async def test_search_with_default_limit(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
        mock_embedding_service: EmbeddingService,
    ):
        # Arrange
        expected_result = [
            SearchResult(
                content="test content", metadata={"source": "test"}, score=0.95
            )
        ]
        mock_vector_db.search_vectors = MagicMock(return_value=expected_result)

        # Act
        results = await vector_service.search(query="test")

        # Assert
        assert len(results) == 1
        assert results[0].content == "test content"
        assert results[0].metadata["source"] == "test"
        assert results[0].score == 0.95
        # Verify search was called with correct arguments
        mock_vector_db.search_vectors.assert_called_once_with(
            mock_embedding_service.get_embedding.return_value,  # vector
            3,  # default limit
            None,  # no filters
        )
        mock_embedding_service.get_embedding.assert_called_once_with("test")

    def test_search_with_custom_limit(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
        mock_embedding_service: EmbeddingService,
    ):
        # Arrange
        expected_result = [
            SearchResult(
                content="test content", metadata={"source": "test"}, score=0.95
            )
        ]
        mock_vector_db.search_vectors.return_value = expected_result

        # Act
        results = vector_service.search(query="test", limit=5)

        # Assert
        assert len(results) == 1
        assert results[0].content == "test content"
        assert results[0].metadata["source"] == "test"
        assert results[0].score == 0.95
        # Verify search was called with correct arguments
        mock_vector_db.search_vectors.assert_called_once_with(
            mock_embedding_service.get_embedding.return_value,  # vector
            5,  # custom limit
            None,  # no filters
        )
        mock_embedding_service.get_embedding.assert_called_once_with("test")

    def test_search_with_metadata_filter(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
        mock_embedding_service: EmbeddingService,
    ):
        # Arrange
        filters = {"category": "test"}
        expected_result = [
            SearchResult(
                content="test content", metadata={"source": "test"}, score=0.95
            )
        ]
        mock_vector_db.search_vectors.return_value = expected_result

        # Act
        results = vector_service.search(query="test", filters=filters)

        # Assert
        assert len(results) == 1
        assert results[0].content == "test content"
        assert results[0].metadata["source"] == "test"
        assert results[0].score == 0.95
        # Verify search was called with correct arguments
        mock_vector_db.search_vectors.assert_called_once_with(
            mock_embedding_service.get_embedding.return_value,  # vector
            3,  # default limit
            filters,  # filters
        )


class TestDeleteDocuments:
    """Tests for deleting documents from the vector service"""

    def test_delete_existing_documents(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
    ):
        # Arrange
        doc_ids = ["doc1", "doc2"]
        mock_vector_db.delete_vectors.return_value = True
        mock_vector_db.create_backup.return_value = "/path/to/backup"
        mock_vector_db.reset_mock()  # Reset mock to clear the initialize() call from constructor

        # Act
        result = vector_service.delete_documents(doc_ids)

        # Assert
        assert result is True
        # Verify the calls were made with correct arguments
        mock_vector_db.delete_vectors.assert_called_once_with(doc_ids)
        mock_vector_db.create_backup.assert_called_once_with()

        # Verify the order of operations
        assert mock_vector_db.method_calls == [
            call.delete_vectors(doc_ids),
            call.create_backup(),
        ]


class TestBackupOperations:
    """Tests for backup and restore operations"""

    def test_create_backup(
        self, vector_service: VectorService, mock_vector_db: VectorDatabaseProtocol
    ):
        # Arrange
        expected_path = "/app/backups/backup_20231201_120000"
        mock_vector_db.create_backup.return_value = expected_path
        mock_vector_db.reset_mock()  # Reset mock to clear the initialize() call from constructor

        # Act
        backup_path = vector_service.create_backup()

        # Assert
        assert backup_path == expected_path
        mock_vector_db.create_backup.assert_called_once_with()
        # Verify no other methods were called
        assert mock_vector_db.method_calls == [call.create_backup()]

    def test_restore_from_backup(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
        monkeypatch,
    ):
        # Arrange
        backup_path = "/app/backups/backup_20231201_120000"
        safety_backup_path = "/app/backups/safety_backup_20231201_120000"
        monkeypatch.setattr("os.path.exists", lambda x: True)
        mock_vector_db.create_backup.return_value = safety_backup_path
        mock_vector_db.reset_mock()  # Reset mock to clear the initialize() call from constructor

        # Act
        vector_service.restore_from_backup(backup_path)

        # Assert
        # Verify each method was called with correct arguments
        mock_vector_db.create_backup.assert_called_once_with()  # Safety backup is created
        mock_vector_db.restore_from_backup.assert_called_once_with(backup_path)
        mock_vector_db.initialize.assert_called_once_with()  # Verify reinitialization

        # Verify the order of operations
        assert mock_vector_db.method_calls == [
            call.create_backup(),  # First create safety backup
            call.restore_from_backup(backup_path),  # Then restore from specified backup
            call.initialize(),  # Finally reinitialize
        ]

    def test_restore_from_nonexistent_backup(
        self, vector_service: VectorService, monkeypatch
    ):
        # Arrange
        monkeypatch.setattr("os.path.exists", lambda x: False)

        # Act & Assert
        with pytest.raises(ValueError, match="Backup path does not exist"):
            vector_service.restore_from_backup("/nonexistent/path")

    def test_restore_from_backup_handles_errors(
        self,
        vector_service: VectorService,
        mock_vector_db: VectorDatabaseProtocol,
        monkeypatch,
    ):
        # Arrange
        backup_path = "/app/backups/backup_20231201_120000"
        monkeypatch.setattr("os.path.exists", lambda x: True)
        mock_vector_db.restore_from_backup.side_effect = Exception("Restore failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Failed to restore from backup"):
            vector_service.restore_from_backup(backup_path)


class TestHealthCheck:
    """Tests for health check functionality"""

    def test_is_healthy_returns_true_when_db_healthy(
        self, vector_service: VectorService, mock_vector_db: VectorDatabaseProtocol
    ):
        # Arrange
        mock_vector_db.is_healthy.return_value = True

        # Act
        result = vector_service.is_healthy()

        # Assert
        assert result is True
        mock_vector_db.is_healthy.assert_called_once()

    def test_is_healthy_returns_false_when_db_unhealthy(
        self, vector_service: VectorService, mock_vector_db: VectorDatabaseProtocol
    ):
        # Arrange
        mock_vector_db.is_healthy.return_value = False

        # Act
        result = vector_service.is_healthy()

        # Assert
        assert result is False
        mock_vector_db.is_healthy.assert_called_once()
