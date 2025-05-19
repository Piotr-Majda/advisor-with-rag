from fastapi import Depends
from functools import lru_cache
from app.vector_store.embeddings import OpenAIEmbeddingService
from app.vector_store.storage import FileSystemStorage
from app.vector_store.factory import VectorDatabaseFactory
from app.vector_store.protocols import EmbeddingConfig, VectorDatabaseProtocol
from app.services.vector_service import VectorService
import os
from typing import AsyncGenerator


@lru_cache()
def get_config():
    return {
        "dimension": 1536,
        "store_path": os.getenv("VECTOR_STORE_PATH", "/app/vector_store/store.pkl"),
        "backup_path": os.getenv("BACKUP_PATH", "/app/backups"),
        "vector_db_type": os.getenv("VECTOR_DB_TYPE", "local"),  # "local" or "pinecone"
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT"),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME"),
    }


@lru_cache()
def get_embedding_config():
    return EmbeddingConfig(model_name="text-embedding-ada-002", dimension=1536)


def get_storage_service(config: dict = Depends(get_config)):
    return FileSystemStorage(
        store_path=config["store_path"], backup_path=config["backup_path"]
    )


def get_vector_db(
    config: dict = Depends(get_config), storage_service=Depends(get_storage_service)
) -> VectorDatabaseProtocol:
    return VectorDatabaseFactory.create(
        db_type=config["vector_db_type"], config=config, storage_service=storage_service
    )


async def get_vector_service(
    vector_db: VectorDatabaseProtocol = Depends(get_vector_db),
    config: EmbeddingConfig = Depends(get_embedding_config),
) -> AsyncGenerator[VectorService, None]:
    embedding_service = OpenAIEmbeddingService(config)
    service = VectorService(
        vector_db=vector_db,
        embedding_service=embedding_service,
    )
    await service.initialize()
    yield service
