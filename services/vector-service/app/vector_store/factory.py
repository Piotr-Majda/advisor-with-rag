from typing import Dict
from .protocols import VectorDatabaseProtocol, StorageService
from .implementations.local_faiss_db import LocalFAISSDatabase
from .implementations.pinecone_db import PineconeDatabase


class VectorDatabaseFactory:
    """Factory for creating vector database instances based on configuration."""

    @staticmethod
    def create(
        db_type: str, config: Dict, storage_service: StorageService = None
    ) -> VectorDatabaseProtocol:
        """
        Create a vector database instance based on the specified type and configuration.

        Args:
            db_type: Type of vector database ("local" or "pinecone")
            config: Configuration dictionary
            storage_service: Optional storage service for local implementations

        Returns:
            VectorDatabaseProtocol: Configured vector database instance

        Raises:
            ValueError: If db_type is not supported
        """
        if db_type == "local":
            if not storage_service:
                raise ValueError(
                    "storage_service is required for local vector database"
                )

            return LocalFAISSDatabase(
                dimension=config["dimension"], storage_service=storage_service
            )

        elif db_type == "pinecone":
            required_fields = [
                "pinecone_api_key",
                "pinecone_environment",
                "pinecone_index_name",
            ]
            if not all(field in config for field in required_fields):
                raise ValueError(
                    f"Missing required Pinecone configuration. Required fields: {required_fields}"
                )

            return PineconeDatabase(
                api_key=config["pinecone_api_key"],
                environment=config["pinecone_environment"],
                index_name=config["pinecone_index_name"],
            )

        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
