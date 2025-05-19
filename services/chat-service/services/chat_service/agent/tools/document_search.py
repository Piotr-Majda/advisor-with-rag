
import httpx
from typing import Dict, Any
from pydantic import Field
from shared import ServiceLogger
from services.chat_service.exceptions import ServiceError
from services.chat_service.agent.tools.tool import BaseTool
import os


logger = ServiceLogger("agent_tools")

class DocumentSearchTool(BaseTool):
    vector_service_url: str = Field(description="The URL of the vector service", default_factory=lambda: os.getenv("VECTOR_SERVICE_URL", "http://vector-service:8004"))
    def __init__(self):
        super().__init__(
            name="search_documents",
            description="Search through financial documents for investment options, strategy and historical data",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        )

    async def execute(self, query: str) -> str:
        try:
            logger.info("Querying vector service", query=query)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_service_url}/query",
                    json={"query": query, "k": 3},
                    timeout=10.0
                )
                response.raise_for_status()
                
            docs = response.json().get("documents", [])
            return "\n\n".join(doc["content"] for doc in docs)
        except Exception as e:
            logger.error("Vector service query failed", error=str(e))
            raise ServiceError(f"Document search failed: {str(e)}") from e
