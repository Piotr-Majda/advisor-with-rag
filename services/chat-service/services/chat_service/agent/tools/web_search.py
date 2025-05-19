import httpx
from typing import Dict, Any
from pydantic import Field
from shared import ServiceLogger
from services.chat_service.exceptions import ServiceError
from services.chat_service.agent.tools.tool import BaseTool
import os


logger = ServiceLogger("agent_tools")

class WebSearchTool(BaseTool):
    search_service_url: str = Field(description="The URL of the search service", default_factory=lambda: os.getenv("SEARCH_SERVICE_URL", "http://search-service:8002"))
    
    def __init__(self):
        super().__init__(
            name="search_web",
            description="Get current market data and investment opportunities",
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
            logger.info("Querying search service", query=query)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.search_service_url}/search",
                    json={"query": query},
                    timeout=10.0
                )
                response.raise_for_status()
            return response.json().get("results", "")
        except Exception as e:
            logger.error("Search service query failed", error=str(e))
            raise ServiceError(f"Web search failed: {str(e)}") from e
