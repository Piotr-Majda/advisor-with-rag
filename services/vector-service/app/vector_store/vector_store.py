from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Document:
    content: str
    metadata: Dict[str, any]


@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, any]
    score: float


class VectorStoreService(ABC):
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector store"""
        pass

    @abstractmethod
    def upsert_documents(self, documents: List[Document]) -> List[str]:
        """Add or update documents in the store and return their IDs"""
        pass

    @abstractmethod
    def search(
        self, query: str, limit: int = 3, filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by their IDs"""
        pass

    @abstractmethod
    def create_backup(self) -> str:
        """Create a backup and return the backup path"""
        pass

    @abstractmethod
    def restore_from_backup(self, backup_path: str) -> None:
        """Restore the vector store from a backup"""
        pass
