import pytest
import numpy as np
from unittest.mock import MagicMock
from app.vector_store.implementations.local_faiss_db import LocalFAISSDatabase
from app.vector_store.vector_store import Document, SearchResult
from app.vector_store.storage import FileSystemStorage
from copy import deepcopy


class TestFAISSVectorDatabase:
    @pytest.fixture
    def mock_storage(self):
        mock = MagicMock(spec=FileSystemStorage)
        # Mock the required storage methods
        mock.save.return_value = True
        mock.load.return_value = None
        mock.exists.return_value = False
        mock.get_store_path.return_value = "test_store"
        mock.data = {}  # For backup testing
        return mock

    @pytest.fixture
    def vector_db(self, mock_storage):
        db = LocalFAISSDatabase(
            dimension=1536,
            storage_service=mock_storage,
        )
        db.initialize()
        return db

    def create_random_vector(self):
        return np.random.rand(1536).astype("float32")

    def create_random_vectors(self, n):
        return np.random.rand(n, 1536).astype("float32")

    def test_initialization(self, vector_db: LocalFAISSDatabase):
        """Test store initialization creates empty store"""
        assert vector_db.is_healthy()
        assert vector_db.dimension == 1536

    def test_upsert_single_document(self, vector_db: LocalFAISSDatabase):
        """Test upserting a single document"""
        doc = Document(content="test", metadata={"source": "test"})
        vector = self.create_random_vector()  # Create proper vector
        doc_ids = vector_db.add_vectors(vector.reshape(1, -1), [doc])
        data = vector_db.export_data()
        assert len(doc_ids) == 1
        assert len(data["documents"]) == 1
        stored_doc = data["documents"][0]
        assert stored_doc["content"] == "test"
        assert "doc_id" in stored_doc["metadata"]
        assert stored_doc["metadata"]["source"] == "test"

    def test_upsert_multiple_documents(self, vector_db: LocalFAISSDatabase):
        """Test upserting multiple documents"""
        docs = [
            Document(content="test1", metadata={"source": "test"}),
            Document(content="test2", metadata={"source": "test"}),
        ]
        vectors = self.create_random_vectors(2)  # Create proper vectors
        doc_ids = vector_db.add_vectors(vectors, docs)
        data = vector_db.export_data()
        assert len(doc_ids) == 2
        assert len(data["documents"]) == 2
        stored_contents = {doc["content"] for doc in data["documents"]}
        assert stored_contents == {"test1", "test2"}

    def test_upsert_empty_list(self, vector_db: LocalFAISSDatabase):
        """Test upserting empty document list"""
        doc_ids = vector_db.add_vectors(np.array([]), [])
        data = vector_db.export_data()
        assert doc_ids == []
        assert len(data["documents"]) == 0

    def test_search_by_similarity(self, vector_db: LocalFAISSDatabase):
        """Test searching returns documents by vector similarity"""
        # Given - two documents with known vectors
        docs = [
            Document(content="doc1", metadata={"source": "test"}),
            Document(content="doc2", metadata={"source": "test"}),
        ]
        # Create two distinct vectors
        vec1 = np.ones(1536).astype("float32")  # Vector of all 1s
        vec2 = np.zeros(1536).astype("float32")  # Vector of all 0s
        vectors = np.stack([vec1, vec2])
        vector_db.add_vectors(vectors, docs)

        # When - search with a query vector more similar to vec1
        query_vector = (
            np.ones(1536).astype("float32") * 0.9
        )  # More similar to vec1 (all 1s)
        results = vector_db.search_vectors(query_vector, limit=2)

        # Then - results should be ordered by vector similarity
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        # First result should be doc1 (closer to query vector)
        assert results[0].content == "doc1"
        assert results[1].content == "doc2"
        # First result should have higher similarity score
        assert results[0].score > results[1].score

    def test_search_with_metadata_filters(self, vector_db: LocalFAISSDatabase):
        """Test searching with metadata filters"""
        docs = [
            Document(content="test1", metadata={"source": "A"}),
            Document(content="test2", metadata={"source": "B"}),
        ]
        vectors = self.create_random_vectors(2)
        vector_db.add_vectors(vectors, docs)

        results = vector_db.search_vectors(
            self.create_random_vector(), limit=2, filters={"source": "A"}
        )
        assert len(results) == 1
        assert results[0].metadata["source"] == "A"

    def test_search_with_nonexistent_filters(self, vector_db: LocalFAISSDatabase):
        """Test searching with filters that match no documents"""
        docs = [Document(content="test", metadata={"source": "A"})]
        vector_db.add_vectors(self.create_random_vectors(1), docs)

        results = vector_db.search_vectors(
            self.create_random_vector(), limit=1, filters={"source": "nonexistent"}
        )
        assert len(results) == 0

    def test_search_empty_store(self, vector_db: LocalFAISSDatabase):
        """Test searching in empty store"""
        results = vector_db.search_vectors(self.create_random_vector(), limit=1)
        assert len(results) == 0

    def test_delete_existing_documents(self, vector_db: LocalFAISSDatabase):
        """Test deleting existing documents"""
        docs = [
            Document(content="test1", metadata={"source": "test"}),
            Document(content="test2", metadata={"source": "test"}),
        ]
        vectors = self.create_random_vectors(2)
        doc_ids = vector_db.add_vectors(vectors, docs)

        success = vector_db.delete_vectors([doc_ids[0]])
        assert success

        results = vector_db.search_vectors(vectors[0], limit=1)
        assert len(results) == 0

    def test_delete_nonexistent_documents(self, vector_db: LocalFAISSDatabase):
        """Test deleting nonexistent documents"""
        success = vector_db.delete_vectors(["nonexistent_id"])
        assert not success

    def test_create_backup(self, vector_db: LocalFAISSDatabase, mock_storage):
        """Test backup creation"""
        docs = [Document(content="test", metadata={"source": "test"})]
        vector = self.create_random_vector()
        vector_db.add_vectors(vector.reshape(1, -1), docs)

        # Mock the storage service to store backup data
        stored_backup = None

        def mock_save(path, data):
            nonlocal stored_backup
            stored_backup = data
            return True

        mock_storage.save.side_effect = mock_save
        mock_storage.create_backup.return_value = "test_backup/backup_123"

        backup_path = vector_db.create_backup()

        assert isinstance(backup_path, str)
        assert backup_path.startswith("test_backup/backup_")
        assert stored_backup is not None
        assert "index" in stored_backup
        assert "documents" in stored_backup

    def test_similarity_score_calculation(self, vector_db: LocalFAISSDatabase):
        """Test similarity score calculation"""
        docs = [
            Document(content="short", metadata={}),
            Document(content="longer text here", metadata={}),
        ]
        vectors = self.create_random_vectors(2)
        vector_db.add_vectors(vectors, docs)

        results = vector_db.search_vectors(vectors[0], limit=2)
        assert len(results) == 2
        assert results[0].score > results[1].score

    def test_document_metadata_preservation(self, vector_db: LocalFAISSDatabase):
        """Test that document metadata is preserved"""
        original_metadata = {"source": "test", "timestamp": "2023", "custom": "value"}
        doc = Document(content="test", metadata=original_metadata)

        vector_db.add_vectors(self.create_random_vectors(1), [doc])
        results = vector_db.search_vectors(self.create_random_vector(), limit=1)

        stored_metadata = results[0].metadata
        for key, value in original_metadata.items():
            assert stored_metadata[key] == value

    def test_store_persistence(self, vector_db: LocalFAISSDatabase, mock_storage):
        """Test that store data persists after save/load"""
        # Add document and create backup
        doc = Document(content="test", metadata={"source": "test"})
        vector = self.create_random_vector()
        vector_db.add_vectors(vector.reshape(1, -1), [doc])

        # Mock the storage service to return the stored data
        stored_data = None

        def mock_save(path, data):
            nonlocal stored_data
            stored_data = data
            return True

        def mock_load(path):
            return stored_data

        mock_storage.save.side_effect = mock_save
        mock_storage.load.side_effect = mock_load
        mock_storage.exists.return_value = True

        # Create backup
        backup_path = vector_db.create_backup()

        # Create new store instance and restore from backup
        new_vector_db = LocalFAISSDatabase(
            dimension=1536,
            storage_service=mock_storage,
        )
        new_vector_db.initialize()
        new_vector_db.restore_from_backup(backup_path)

        # Verify data persisted
        query_vector = self.create_random_vector()
        results = new_vector_db.search_vectors(query_vector.reshape(1, -1), limit=1)
        assert len(results) == 1
        assert results[0].content == "test"
        assert results[0].metadata["source"] == "test"

    def test_backup_and_restore(self, vector_db: LocalFAISSDatabase, mock_storage):
        """Test backup creation and restoration"""
        # Add initial documents
        docs = [
            Document(content="test1", metadata={"source": "A"}),
            Document(content="test2", metadata={"source": "B"}),
        ]
        vectors = self.create_random_vectors(2)
        vector_db.add_vectors(vectors, docs)

        # Mock storage for backup/restore
        store_data = None
        backup_data = None

        def mock_save(path, data):
            nonlocal store_data, backup_data
            if path == "test_store":
                store_data = {"index": data["index"], "documents": data["documents"]}
            else:
                backup_data = {
                    "index": deepcopy(data["index"]),
                    "documents": deepcopy(data["documents"]),
                }
            return True

        def mock_load(path):
            if path == "test_store":
                return store_data
            return backup_data

        def mock_create_backup(path):
            return "backup_path"

        mock_storage.save.side_effect = mock_save
        mock_storage.load.side_effect = mock_load
        mock_storage.create_backup.side_effect = mock_create_backup
        mock_storage.exists.return_value = True

        # Create backup
        backup_path = vector_db.create_backup()

        # Add more documents after backup
        new_doc = Document(content="test3", metadata={"source": "C"})
        new_vector = self.create_random_vector()
        vector_db.add_vectors(new_vector.reshape(1, -1), [new_doc])

        # Restore from backup
        vector_db.restore_from_backup(backup_path)

        # Verify state is restored to backup point
        query_vector = self.create_random_vector()
        results = vector_db.search_vectors(query_vector.reshape(1, -1), limit=3)
        assert len(results) == 2
        contents = {r.content for r in results}
        assert contents == {"test1", "test2"}
        assert "test3" not in contents

    def test_backup_with_empty_store(self, vector_db: LocalFAISSDatabase, mock_storage):
        """Test creating backup of empty store"""
        # Mock storage for backup/restore
        stored_data = None

        def mock_save(path, data):
            nonlocal stored_data
            stored_data = data
            return True

        def mock_load(path):
            return stored_data

        mock_storage.save.side_effect = mock_save
        mock_storage.load.side_effect = mock_load
        mock_storage.exists.return_value = True
        mock_storage.create_backup.return_value = "test_backup/backup_empty"

        backup_path = vector_db.create_backup()
        assert backup_path.startswith("test_backup/backup_")

        # Restore from empty backup
        new_vector_db = LocalFAISSDatabase(
            dimension=1536,
            storage_service=mock_storage,
        )
        new_vector_db.initialize()
        new_vector_db.restore_from_backup(backup_path)

        query_vector = self.create_random_vector()
        results = new_vector_db.search_vectors(query_vector.reshape(1, -1), limit=1)
        assert len(results) == 0
