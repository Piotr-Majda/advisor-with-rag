from abc import ABC, abstractmethod
from typing import Any, Dict

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict, Field


class BaseTool(BaseModel, ABC):
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="A description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="The parameters schema for the tool",
    )

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters"""
        pass

    def to_openai_function(self) -> ChatCompletionToolParam:
        """Convert tool to OpenAI function format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
