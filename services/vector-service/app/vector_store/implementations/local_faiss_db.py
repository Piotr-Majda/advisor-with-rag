import os
import faiss
import numpy as np
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from ..protocols import VectorDatabaseProtocol, StorageService
from ..vector_store import Document, SearchResult


class LocalFAISSDatabase(VectorDatabaseProtocol):
    def __init__(self, dimension: int, storage_service: StorageService):
        self.dimension = dimension
        self.storage_service = storage_service
        self._index = None
        self._documents: Dict[int, Document] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the vector database.
        If a store exists, load it. Otherwise, create a new one.
        The store path should point to a file, not a directory.
        Parent directories will be created if they don't exist.
        """
        async with self._lock:
            store_path = self.storage_service.get_store_path()

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(store_path), exist_ok=True)

            try:
                if os.path.isfile(store_path):
                    data = self.storage_service.load(store_path)
                    if (
                        isinstance(data, dict)
                        and "index" in data
                        and "documents" in data
                    ):
                        self._index = data["index"]
                        self._documents = data["documents"]
                        # Validate the loaded index
                        if (
                            not isinstance(self._index, faiss.Index)
                            or self._index.d != self.dimension
                        ):
                            raise ValueError(
                                "Invalid index format or dimension mismatch"
                            )
                        return
                    else:
                        raise ValueError("Invalid store format")
            except Exception as e:
                # Log the error but don't fail - we'll create a new store
                print(
                    f"Failed to load existing store at {store_path}: {e}. Creating new store."
                )

            # If we get here, either there was no store or it was invalid
            await self._create_new_store()

    async def _create_new_store(self) -> None:
        """Create a new empty vector store"""
        self._index = faiss.IndexFlatL2(self.dimension)
        self._documents = {}
        # Save the initial empty store
        await self._save_store()

    async def add_vectors(
        self, vectors: np.ndarray, documents: List[Document]
    ) -> List[str]:
        if self._index is None:
            raise RuntimeError("Vector store not initialized")

        print(f"Adding {len(documents)} documents to store")
        print(f"Vectors shape: {vectors.shape}")
        if len(vectors) > 0:
            print(f"First vector shape: {vectors[0].shape}")

        doc_ids = []
        async with self._lock:
            for i, (vector, doc) in enumerate(zip(vectors, documents)):
                print(f"Processing vector {i+1} with shape {vector.shape}")
                doc_id = self._generate_doc_id(doc.content)
                index_id = len(self._documents)
                self._documents[index_id] = Document(
                    content=doc.content,
                    metadata={**doc.metadata, "doc_id": doc_id, "index_id": index_id},
                )
                self._index.add(vector.reshape(1, -1))
                doc_ids.append(doc_id)
                print(
                    f"Added document {i+1}: {doc.content} with metadata {doc.metadata}"
                )

            print(f"Total documents in store after add: {len(self._documents)}")
            await self._save_store()
        return doc_ids

    async def search_vectors(
        self, vector: np.ndarray, limit: int, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        if self._index is None:
            raise RuntimeError("Vector store not initialized")

        print(f"Searching with limit={limit} and filters={filters}")
        print(f"Total documents in store before search: {len(self._documents)}")

        async with self._lock:
            # If filters are applied, search all documents to ensure we don't miss matches
            search_limit = self._index.ntotal if filters else limit
            print(f"Searching through {search_limit} documents")
            scores, indices = self._index.search(vector.reshape(1, -1), search_limit)

            # Apply filters and sort by score
            results = [
                SearchResult(
                    content=self._documents[int(idx)].content,
                    metadata=self._documents[int(idx)].metadata,
                    score=self._calculate_similarity_score(score),
                )
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
                and self._matches_filters(self._documents[int(idx)], filters)
            ]

            print(f"Found {len(results)} results after filtering")
            for r in results:
                print(f"Result: {r.content} with metadata {r.metadata}")

            # Sort by score in descending order and return top k results
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:limit]

    async def delete_vectors(self, doc_ids: List[str]) -> bool:
        async with self._lock:
            documents_to_keep = {
                idx: doc
                for idx, doc in self._documents.items()
                if doc.metadata.get("doc_id") not in doc_ids
            }

            if len(documents_to_keep) == len(self._documents):
                return False

            self._rebuild_index(documents_to_keep)
            await self._save_store()
            return True

    async def create_backup(self) -> str:
        """Create a backup of the current state.

        Returns:
            str: Path to the backup file
        """
        async with self._lock:
            # Create a deep copy of the documents
            backup_documents = {}
            for idx, doc in self._documents.items():
                backup_documents[idx] = Document(
                    content=doc.content, metadata=doc.metadata.copy()
                )

            # Create a new FAISS index with the same data
            backup_index = faiss.IndexFlatL2(self.dimension)
            if self._index.ntotal > 0:
                backup_vectors = self._index.reconstruct_n(0, self._index.ntotal)
                backup_index.add(backup_vectors)

            # Create backup path and save backup data directly
            backup_path = self.storage_service.create_backup(
                self.storage_service.get_store_path()
            )
            backup_data = {"index": backup_index, "documents": backup_documents}
            self.storage_service.save(backup_path, backup_data)
            return backup_path

    async def restore_from_backup(self, backup_path: str) -> None:
        async with self._lock:
            data = self.storage_service.load(backup_path)
            if (
                not isinstance(data, dict)
                or "index" not in data
                or "documents" not in data
            ):
                raise ValueError("Invalid backup data format")

            if data["index"].d != self.dimension:
                raise ValueError(
                    f"Backup index dimension ({data['index'].d}) does not match current configuration ({self.dimension})"
                )

            self._index = data["index"]
            self._documents = data["documents"]
            await self._save_store()

    async def _save_store(self) -> None:
            data = {"index": self._index, "documents": self._documents}
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: self.storage_service.save(self.storage_service.get_store_path(), data)
            )

    def _rebuild_index(self, documents_to_keep: Dict[int, Document]) -> None:
            new_index = faiss.IndexFlatL2(self.dimension)
            new_documents = {}

            for i, (_, doc) in enumerate(documents_to_keep.items()):
                new_documents[i] = Document(
                    content=doc.content, metadata={**doc.metadata, "index_id": i}
                )

            self._index = new_index
            self._documents = new_documents

    def _generate_doc_id(self, content: str) -> str:
        return f"doc_{hash(content)}_{int(datetime.now().timestamp())}"

    def _calculate_similarity_score(self, distance: float) -> float:
        return float(1.0 / (1.0 + distance))

    def _matches_filters(self, doc: Document, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        return all(
            key in doc.metadata and doc.metadata[key] == value
            for key, value in filters.items()
        )

    def export_data(self) -> Dict:
        """Export data in an implementation-agnostic format"""
        return {
            "dimension": self.dimension,
            "documents": [
                {"content": doc.content, "metadata": doc.metadata}
                for doc in self._documents.values()
            ],
        }

    async def import_data(self, data: Dict) -> None:
        """Import data from an implementation-agnostic format"""
        if (
            not isinstance(data, dict)
            or "dimension" not in data
            or "documents" not in data
        ):
            raise ValueError("Invalid data format")

        if data["dimension"] != self.dimension:
            raise ValueError(
                f"Data dimension ({data['dimension']}) does not match current configuration ({self.dimension})"
            )

        # Reset the store
        async with self._lock:
            self._create_new_store()

        # Add documents one by one to rebuild the index
        documents = [Document(**doc) for doc in data["documents"]]
        if documents:
            vectors = np.zeros(
                (len(documents), self.dimension)
            )  # This would normally come from your embedding service
            await self.add_vectors(vectors, documents)

    def is_healthy(self) -> bool:
        return self._index is not None and self._documents is not None
