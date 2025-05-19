import numpy as np
from typing import List, Dict, Optional
from ..protocols import VectorDatabaseProtocol
from ..vector_store import Document, SearchResult
#import pinecone
from datetime import datetime

class PineconeDatabase(VectorDatabaseProtocol):
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        pinecone.init(api_key=api_key, environment=environment)
        
    def initialize(self) -> None:
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=1536)
        self.index = pinecone.Index(self.index_name)

    def add_vectors(self, vectors: np.ndarray, documents: List[Document]) -> List[str]:
        doc_ids = [self._generate_doc_id(doc.content) for doc in documents]
        
        vectors_to_upsert = [
            (doc_id, vector.tolist(), {**doc.metadata, "content": doc.content})
            for doc_id, vector, doc in zip(doc_ids, vectors, documents)
        ]
        
        self.index.upsert(vectors=vectors_to_upsert)
        return doc_ids

    def search_vectors(self, vector: np.ndarray, limit: int, filters: Optional[Dict] = None) -> List[SearchResult]:
        results = self.index.query(
            vector=vector.tolist(),
            top_k=limit,
            filter=filters
        )
        
        return [
            SearchResult(
                content=match.metadata["content"],
                metadata={k: v for k, v in match.metadata.items() if k != "content"},
                score=match.score
            )
            for match in results.matches
        ]

    def delete_vectors(self, doc_ids: List[str]) -> bool:
        self.index.delete(ids=doc_ids)
        return True

    def create_backup(self) -> Optional[str]:
        # Pinecone handles persistence automatically
        return None

    def restore_from_backup(self, backup_path: Optional[str]) -> None:
        # No-op for Pinecone as it handles persistence
        pass

    def _generate_doc_id(self, content: str) -> str:
        return f"doc_{hash(content)}_{int(datetime.now().timestamp())}"