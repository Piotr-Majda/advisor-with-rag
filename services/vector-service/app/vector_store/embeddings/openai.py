import numpy as np
from typing import List
from openai import AsyncOpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse
from ..protocols import EmbeddingService, EmbeddingConfig


class OpenAIEmbeddingService(EmbeddingService):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = AsyncOpenAI()

    async def get_embedding(self, text: str) -> np.ndarray:
        response: CreateEmbeddingResponse = await self.client.embeddings.create(
            model=self.config.model_name, input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        response: CreateEmbeddingResponse = await self.client.embeddings.create(
            model=self.config.model_name, input=texts
        )
        return np.array([r.embedding for r in response.data], dtype=np.float32)
