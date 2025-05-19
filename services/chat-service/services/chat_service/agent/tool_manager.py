import json
from typing import List

from .models import ToolCallData, ToolResponse
from .tools.tool import BaseTool


class ToolManager:
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}

    async def execute(self, tool_call: ToolCallData) -> ToolResponse:
        """Execute a tool with the given parameters, handling argument parsing errors."""
        try:
            # Access attributes directly from ToolCallData model
            tool_name = tool_call.name
            arguments = tool_call.arguments  # Access .arguments directly
            tool_call_id = tool_call.call_id
        except AttributeError as e:
            # Handle cases where the ToolCallData structure is not as expected
            return ToolResponse(
                name="unknown_tool",
                content="",
                error=f"Internal error: Invalid ToolCallData structure - {e}",
                call_id=getattr(tool_call, "call_id", None),
            )

        # Get the tool instance
        tool = self.tools.get(tool_name)
        if not tool:
            # Return an error if the requested tool isn't found
            return ToolResponse(
                name=tool_name,
                content="",
                error=f"Tool '{tool_name}' not found.",
                call_id=tool_call_id,
            )

        arguments_dict = {}
        try:
            # Arguments should already be a dict based on ToolCallData model
            if isinstance(arguments, dict):
                arguments_dict = arguments
            elif isinstance(arguments, str):  # Add fallback for string just in case
                try:
                    arguments_dict = json.loads(arguments)
                except json.JSONDecodeError as json_err:
                    error_message = f"Failed to parse tool arguments JSON: {json_err}. Received: {arguments}"
                    return ToolResponse(
                        name=tool_name,
                        content="",
                        error=error_message,
                        call_id=tool_call_id,
                    )
            else:
                return ToolResponse(
                    name=tool_name,
                    content="",
                    error=f"Unexpected tool arguments type: {type(arguments)}",
                    call_id=tool_call_id,
                )

            # Execute the tool with parsed arguments
            result = await tool.execute(**arguments_dict)
            return ToolResponse(name=tool_name, content=result, call_id=tool_call_id)

        except Exception as e:
            # Catch any other exceptions during tool execution
            import traceback

            error_details = (
                f"Error executing tool '{tool_name}': {e}\n{traceback.format_exc()}"
            )
            return ToolResponse(
                name=tool_name, content="", error=error_details, call_id=tool_call_id
            )
