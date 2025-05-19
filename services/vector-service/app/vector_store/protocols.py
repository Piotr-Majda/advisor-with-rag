from typing import Protocol, List, Dict, Optional
import numpy as np
from dataclasses import dataclass
from .vector_store import Document, SearchResult


class VectorDatabaseProtocol(Protocol):
    async def initialize(self) -> None:
        ...

    async def add_vectors(
        self, vectors: np.ndarray, documents: List[Document]
    ) -> List[str]:
        ...

    async def search_vectors(
        self, vector: np.ndarray, limit: int, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        ...

    async def delete_vectors(self, doc_ids: List[str]) -> bool:
        ...

    def export_data(self) -> Dict:
        ...

    def import_data(self, data: Dict) -> None:
        ...

    async def create_backup(self) -> str:
        ...

    async def restore_from_backup(self, backup_path: str) -> None:
        ...

    def is_healthy(self) -> bool:
        ...


@dataclass
class EmbeddingConfig:
    model_name: str
    dimension: int


class EmbeddingService(Protocol):
    async def get_embedding(self, text: str) -> np.ndarray:
        ...

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        ...


class StorageService(Protocol):
    def save(self, path: str, data: any) -> None:
        ...

    def load(self, path: str) -> any:
        ...

    def create_backup(self, source: str) -> str:
        ...  # Simplified signature

    def delete(self, path: str) -> None:
        ...

    def exists(self, path: str) -> bool:
        ...

    def get_store_path(self) -> str:
        ...  # New method

    def get_backup_path(self) -> str:
        ...  # New method
