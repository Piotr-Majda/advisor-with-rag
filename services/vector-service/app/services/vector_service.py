from datetime import datetime
from typing import List, Dict, Optional
import os
from app.vector_store.protocols import VectorDatabaseProtocol, EmbeddingService
from app.vector_store.vector_store import Document, SearchResult
from shared import ServiceLogger
import numpy as np

logger = ServiceLogger("vector_service")


class VectorService:
    """
    High-level service layer for vector operations.
    Handles configuration, initialization, and API-specific transformations.
    """

    def __init__(
        self,
        vector_db: VectorDatabaseProtocol,
        embedding_service: EmbeddingService,
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service

        logger.info(
            "VectorService initialized",
            vector_db_type=type(vector_db).__name__,
            embedding_service_type=type(embedding_service).__name__,
        )

    async def initialize(self) -> None:
        """Initialize the vector store"""
        try:
            await self.vector_db.initialize()
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize vector store", error=str(e))
            raise RuntimeError(f"Failed to initialize vector database: {str(e)}")

    async def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.info("No documents to add")
                return await self.vector_db.add_vectors(np.array([]), [])

            # Transform and validate documents
            vector_docs = self._prepare_documents(documents)
            logger.info(
                "Documents prepared for embedding", document_count=len(vector_docs)
            )

            # Get embeddings for all documents
            embeddings = await self.embedding_service.get_embeddings(
                [doc.content for doc in vector_docs]
            )
            logger.info("Embeddings generated", embedding_count=len(embeddings))

            # Add to vector store
            doc_ids = await self.vector_db.add_vectors(embeddings, vector_docs)
            logger.info(
                "Documents added to vector store",
                document_count=len(doc_ids),
                doc_ids=doc_ids,
            )
            return doc_ids

        except Exception as e:
            logger.error(
                "Error adding documents", error=str(e), document_count=len(documents)
            )
            raise

    async def search(
        self, query: str, limit: int = 3, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        try:
            logger.info(
                "Starting vector search", query=query, limit=limit, filters=filters
            )

            embedding = await self.embedding_service.get_embedding(query)
            results = await self.vector_db.search_vectors(embedding, limit, filters)

            logger.info(
                "Search completed",
                results_count=len(results),
                top_score=results[0].score if results else None,
            )
            return results

        except Exception as e:
            logger.error(
                "Error during search", error=str(e), query=query, filters=filters
            )
            raise

    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents and create a backup."""
        try:
            logger.info("Deleting documents", doc_ids=doc_ids)
            success = await self.vector_db.delete_vectors(doc_ids)

            if success:
                await self.create_backup()
                logger.info("Documents deleted successfully", doc_ids=doc_ids)
            else:
                logger.warning("Failed to delete documents", doc_ids=doc_ids)

            return success

        except Exception as e:
            logger.error("Error deleting documents", error=str(e), doc_ids=doc_ids)
            raise

    async def create_backup(self) -> Optional[str]:
        """Create a backup of the current vector store state."""
        try:
            backup_path = await self.vector_db.create_backup()
            logger.info("Backup created successfully", backup_path=backup_path)
            return backup_path

        except Exception as e:
            logger.error("Failed to create backup", error=str(e))
            raise

    async def restore_from_backup(self, backup_path: str) -> None:
        """Restore the vector store from a backup."""
        try:
            if not backup_path or not os.path.exists(backup_path):
                logger.error("Invalid backup path", path=backup_path)
                raise ValueError(f"Backup path does not exist: {backup_path}")

            logger.info("Starting restore from backup", backup_path=backup_path)

            # Create a safety backup
            safety_backup = await self.create_backup()
            logger.info("Safety backup created", backup_path=safety_backup)

            # Perform restore
            await self.vector_db.restore_from_backup(backup_path)
            await self.initialize()

            logger.info("Restore completed successfully", backup_path=backup_path)

        except Exception as e:
            logger.error(
                "Failed to restore from backup", error=str(e), backup_path=backup_path
            )
            raise RuntimeError(f"Failed to restore from backup: {str(e)}")

    def _prepare_documents(self, documents: List[Dict]) -> List[Document]:
        """Transform API documents to vector store format."""
        try:
            if not documents:
                return []

            for doc in documents:
                if not isinstance(doc, dict):
                    raise ValueError(
                        f"Invalid document format: expected dict, got {type(doc)}"
                    )
                if "content" not in doc:
                    raise ValueError("Document missing required 'content' field")
                if not isinstance(doc.get("content"), str):
                    raise ValueError("Document 'content' must be a string")

            prepared_docs = [
                Document(
                    content=doc["content"],
                    metadata={
                        **(doc.get("metadata", {})),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                for doc in documents
            ]

            logger.debug("Documents prepared", document_count=len(prepared_docs))
            return prepared_docs

        except Exception as e:
            logger.error(
                "Error preparing documents", error=str(e), document_count=len(documents)
            )
            raise

    def is_healthy(self) -> bool:
        """Check if the service and its dependencies are healthy."""
        is_healthy = self.vector_db is not None and self.vector_db.is_healthy()
        logger.debug(
            "Health check performed",
            is_healthy=is_healthy,
            vector_db_initialized=self.vector_db is not None,
        )
        return is_healthy
