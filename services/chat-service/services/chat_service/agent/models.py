from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    """Type of response from the agent"""

    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    DONE = "done"


class ToolCallData(BaseModel):
    """Data for a tool call"""

    name: str = Field(description="Name of the tool to call")
    arguments: Dict = Field(
        default_factory=dict, description="Arguments for the tool call"
    )
    call_id: str = Field(description="Unique ID for this tool call")


class AgentResponse(BaseModel):
    """Response from the agent"""

    type: ResponseType = Field(description="Type of the response")
    content: Optional[str] = Field(
        default=None, description="Content of the message if type is MESSAGE"
    )
    tool_call: Optional[ToolCallData] = Field(
        default=None, description="Tool call data if type is TOOL_CALL"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if type is ERROR"
    )
    finish_reason: Optional[str] = Field(
        default=None, description="Finish reason if type is DONE (stop, tool_calls, etc)"
    )

    class Config:
        validate_assignment = True

    def __init__(self, **data):
        # Set defaults based on response type
        if "type" in data:
            if data["type"] == ResponseType.MESSAGE and "content" not in data:
                data["content"] = ""
            elif data["type"] == ResponseType.TOOL_CALL and "tool_call" not in data:
                data["tool_call"] = None
            elif data["type"] == ResponseType.ERROR and "error" not in data:
                data["error"] = ""
        super().__init__(**data)


class ToolResponse(BaseModel):
    """Response from a tool"""

    name: str = Field(description="Name of the tool that executed")
    content: str = Field(description="Content of the tool execution")
    error: Optional[str] = Field(
        default=None, description="Error message if the tool execution failed"
    )
    call_id: Optional[str] = Field(
        default=None, description="ID of the tool call that executed"
    )

    class Config:
        validate_assignment = True
