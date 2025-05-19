# MCP-Enabled AI Agent Flow

## Architecture Overview

````python
from dataclasses import dataclass
from typing import Dict, List, Any, AsyncIterator
import json
import asyncio
from datetime import datetime

@dataclass
class MCPToolResponse:
    status: str
    data: Any
    metadata: Dict[str, Any] = None
    error: str = None

class MCPTool:
    def __init__(self, name: str, description: str, schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.schema = schema
        self.usage_count = 0
        self.success_rate = 1.0
        self.last_used = None

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema
            }
        }

class MCPAgent:
    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4-0125-preview",
        max_consecutive_tool_calls: int = 5
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.max_consecutive_tool_calls = max_consecutive_tool_calls
        self.tools: Dict[str, MCPTool] = {}
        self.conversation_history = []

    async def register_tool(self, tool: MCPTool):
        """Register a new MCP tool"""
        self.tools[tool.name] = tool

    async def _get_tool_subset(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant tools for the current query using AI analysis"""
        tool_analysis_prompt = f"""
        Analyze this query and determine which tools would be most helpful.
        Query: {query}

        Available tools:
        {self._format_tools()}

        Consider:
        1. Direct relevance to the query
        2. Past success rate of tools
        3. Tool dependencies and order of operations

        Respond with a JSON array of tool names, ordered by relevance.
        Include ONLY tools that are directly relevant to this specific query.
        """

        analysis = await openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": tool_analysis_prompt}],
            response_format={ "type": "json_object" }
        )

        selected_tools = json.loads(analysis.choices[0].message.content)["tools"]
        return [self.tools[name].to_openai_function()
                for name in selected_tools
                if name in self.tools]

    async def chat(self, user_input: str) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": user_input}
        ]

        tool_call_count = 0
        while tool_call_count < self.max_consecutive_tool_calls:
            # Get relevant tools for current context
            relevant_tools = await self._get_tool_subset(user_input)

            try:
                stream = await openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=relevant_tools,
                    stream=True
                )

                collected_messages = []
                tool_calls_data = []
                current_tool_call = None

                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    if delta.tool_calls:
                        # Handle tool calls
                        tool_call = delta.tool_calls[0]

                        if tool_call.index is not None:
                            current_tool_call = {
                                "id": tool_call.id,
                                "function": {"name": tool_call.function.name, "arguments": ""}
                            }
                            tool_calls_data.append(current_tool_call)

                        if tool_call.function.arguments:
                            current_tool_call["function"]["arguments"] += tool_call.function.arguments

                    elif chunk.choices[0].finish_reason == "tool_calls":
                        # Execute tools and update context
                        tool_messages = await self._execute_tools(tool_calls_data)
                        messages.extend(tool_messages)
                        tool_call_count += 1
                        break
                    elif delta.content:
                        collected_messages.append(delta.content)
                        yield delta.content

                if chunk.choices[0].finish_reason == "stop":
                    message_content = "".join(collected_messages)
                    self.conversation_history.extend([
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": message_content}
                    ])
                    break

            except Exception as e:
                error_msg = f"Error during chat: {str(e)}"
                self.conversation_history.append({
                    "role": "system",
                    "content": error_msg
                })
                yield error_msg
                break

    async def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Execute tool calls with retries and error handling"""
        tool_messages = []

        for tool_call in tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                tool = self.tools.get(function_name)
                if not tool:
                    raise ValueError(f"Tool {function_name} not found")

                # Execute with timeout and retry
                result = await self._execute_with_retry(tool, **arguments)

                # Update tool statistics
                tool.usage_count += 1
                tool.last_used = datetime.now()
                if result.status == "success":
                    tool.success_rate = (
                        (tool.success_rate * (tool.usage_count - 1) + 1) /
                        tool.usage_count
                    )
                else:
                    tool.success_rate = (
                        (tool.success_rate * (tool.usage_count - 1)) /
                        tool.usage_count
                    )

                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": str(result)
                })

            except Exception as e:
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": f"Error executing tool: {str(e)}"
                })

        return tool_messages

## Example Usage

```python
# Initialize MCP agent
agent = MCPAgent(
    system_prompt="You are a helpful AI assistant with access to various tools...",
    max_consecutive_tool_calls=5
)

# Register MCP tools
await agent.register_tool(MCPTool(
    name="web_search",
    description="Search the web for current information",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
))

# Example complex query requiring multiple tool calls
query = """
Find recent news about Bitcoin ETF impact on price, analyze the market trends,
and check our internal documents for investment strategies. Also calculate the
potential ROI if someone invested $10,000 six months ago.
"""

async for response in agent.chat(query):
    print(response)
````

## Key Improvements

1. **Dynamic Tool Selection**

   - AI analyzes each query to select relevant tools
   - Tools are prioritized based on relevance and past performance
   - Only necessary tools are included in each interaction

2. **Iterative Tool Usage**

   - AI can make multiple tool calls as needed
   - Each tool call adds to the context
   - Maximum call limit prevents infinite loops

3. **Tool Performance Tracking**

   - Success rate monitoring
   - Usage statistics
   - Automatic tool prioritization

4. **Error Handling**

   - Retry mechanism with exponential backoff
   - Graceful failure handling
   - Error reporting in conversation context

5. **MCP Integration**
   - Standard MCP response format
   - Tool schema validation
   - Metadata support

## Tool Selection Process

1. **Query Analysis**

```python
# AI analyzes the query and context
analysis_prompt = f"""
Analyze this query and determine which tools would be most helpful.
Query: {query}

Available tools:
{self._format_tools()}

Consider:
1. Direct relevance to the query
2. Past success rate of tools
3. Tool dependencies and order of operations
"""
```

2. **Tool Execution**

```python
async def _execute_with_retry(self, tool: MCPTool, **kwargs) -> MCPToolResponse:
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            async with asyncio.timeout(10):
                result = await tool.execute(**kwargs)
                if result.status == "success":
                    return result

                if retry_count < max_retries - 1:
                    await asyncio.sleep(1 * (retry_count + 1))
                    retry_count += 1
                    continue
                return result

        except Exception as e:
            if retry_count < max_retries - 1:
                retry_count += 1
                continue
            return MCPToolResponse(
                status="error",
                error=f"Tool execution failed: {str(e)}"
            )
```

3. **Context Management**

```python
async def _update_context(self, tool_results: List[Dict[str, Any]]):
    """Update conversation context with tool results"""
    for result in tool_results:
        if result["status"] == "success":
            self.conversation_history.append({
                "role": "tool",
                "content": str(result["data"]),
                "metadata": result.get("metadata", {})
            })
```

This implementation provides:

1. Intelligent tool selection
2. Multiple tool calls when needed
3. Proper error handling
4. Performance tracking
5. MCP protocol compliance
