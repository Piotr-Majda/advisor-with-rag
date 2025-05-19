import pytest
import pytest_asyncio
from app.vector_store.implementations.local_faiss_db import LocalFAISSDatabase
from app.vector_store.embeddings.openai import OpenAIEmbeddingService
from app.vector_store.storage.filesystem import FileSystemStorage
from app.vector_store.protocols import EmbeddingConfig
from app.vector_store.vector_store import Document


class TestFAISSVectorStore:
    @pytest.fixture
    def embedding_service(self):
        config = EmbeddingConfig(model_name="text-embedding-ada-002", dimension=1536)
        return OpenAIEmbeddingService(config)
    
    @pytest.fixture
    def storage_service(self) -> FileSystemStorage:
        storage = FileSystemStorage(
            store_path="/tmp/test_vector_store.pkl",
            backup_path="/tmp/test_backups"
        )
        storage.delete(storage.get_store_path())
        yield storage
        storage.delete(storage.get_store_path())

    @pytest_asyncio.fixture
    async def store(self, storage_service: FileSystemStorage) -> LocalFAISSDatabase:
        store = LocalFAISSDatabase(
            dimension=1536,
            storage_service=storage_service,
        )
        await store.initialize()
        yield store
    
    @pytest.mark.asyncio
    async def test_store_initialization(self, store: LocalFAISSDatabase):
        assert store._index is not None
        assert store._documents is not None
        assert len(store._documents) == 0

    @pytest.mark.asyncio
    async def test_real_vector_similarity(self, store: LocalFAISSDatabase, embedding_service: OpenAIEmbeddingService):
        # Given
        docs = [
            Document(content="The quick brown fox", metadata={"source": "test"}),
            Document(content="The lazy dog", metadata={"source": "test"}),
        ]
        # Get embeddings for documents
        doc_embeddings = await embedding_service.get_embeddings([doc.content for doc in docs])
        # Add vectors to store
        await store.add_vectors(doc_embeddings, docs)

        # When
        query_embedding = await embedding_service.get_embedding("quick fox")
        results = await store.search_vectors(query_embedding, limit=1)

        # Then
        assert len(results) == 1
        assert "fox" in results[0].content
        assert results[0].score > 0.5

    @pytest.mark.asyncio
    async def test_vector_operations(self, store: LocalFAISSDatabase, embedding_service: OpenAIEmbeddingService):
        """Test basic vector operations with actual embeddings"""
        # Given
        docs = [
            Document(content="First test document", metadata={"type": "test"}),
            Document(content="Second test document", metadata={"type": "test"}),
            Document(content="Third test document", metadata={"type": "test"}),
        ]
        embeddings = await embedding_service.get_embeddings([doc.content for doc in docs])

        # When adding vectors
        doc_ids = await store.add_vectors(embeddings, docs)

        # Then
        assert len(doc_ids) == 3

        # When searching
        query_embedding = await embedding_service.get_embedding("first document")
        results = await store.search_vectors(query_embedding, limit=2)

        # Then
        assert len(results) == 2
        assert any("First" in result.content for result in results)

    @pytest.mark.asyncio
    async def test_vector_deletion(self, store: LocalFAISSDatabase, embedding_service: OpenAIEmbeddingService):
        """Test vector deletion with actual embeddings"""
        # Given
        doc = Document(content="Delete test document", metadata={"type": "test"})
        embedding = await embedding_service.get_embedding(doc.content)
        doc_ids = await store.add_vectors(embedding.reshape(1, -1), [doc])

        # When
        success = await store.delete_vectors(doc_ids)

        # Then
        assert success
        query_embedding = await embedding_service.get_embedding("Delete test")
        results = await store.search_vectors(query_embedding, limit=1)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, store: LocalFAISSDatabase, embedding_service: OpenAIEmbeddingService):
        """Test metadata filtering with actual embeddings"""
        # Given
        docs = [
            Document(content="Filter doc A", metadata={"category": "A"}),
            Document(content="Filter doc B", metadata={"category": "B"}),
        ]
        embeddings = await embedding_service.get_embeddings([doc.content for doc in docs])
        await store.add_vectors(embeddings, docs)

        # When
        query_embedding = await embedding_service.get_embedding("Filter doc")
        results = await store.search_vectors(
            query_embedding, limit=2, filters={"category": "A"}
        )

        # Then
        assert len(results) == 1
        assert results[0].metadata["category"] == "A"
