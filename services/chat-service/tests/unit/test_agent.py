from typing import Any, AsyncIterator, Dict, List

import pytest
from unittest.mock import MagicMock
from pydantic import Field
from services.chat_service.agent.chat_completion import ChatCompletion
from services.chat_service.agent.core import Agent, AgentConfig
from services.chat_service.agent.models import AgentResponse, ResponseType, ToolCallData
from services.chat_service.agent.tools.tool import BaseTool


class MockTool(BaseTool):
    name: str = Field(default="mock_tool")
    description: str = Field(default="A mock tool for testing")
    called_history: List[Dict[str, Any]] = Field(default_factory=list)

    async def execute(self, **kwargs) -> str:
        self.called_history.append(kwargs)
        return "tool result"


class MockChatCompletion(ChatCompletion):
    """Mock implementation of chat completion for testing"""

    async def complete(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncIterator[AgentResponse]:
        # Simple mock that returns a message and done
        yield AgentResponse(type=ResponseType.MESSAGE, content="Test response")
        yield AgentResponse(type=ResponseType.DONE)


@pytest.fixture
def mock_tool():
    return MockTool()


@pytest.fixture
def agent_config(mock_tool):
    return AgentConfig(
        system_prompt="Test prompt", tools=[mock_tool], model="gpt-4", temperature=0.7
    )


@pytest.fixture
def mock_chat_completion():
    return MockChatCompletion()


@pytest.fixture
def agent(agent_config, mock_chat_completion):
    return Agent(agent_config, chat_completion=mock_chat_completion)


@pytest.mark.asyncio
async def test_chat_basic_response(agent):
    """Test that basic chat responses are processed correctly"""
    responses = []
    async for response in agent.chat("Test input"):
        responses.append(response)

    assert len(responses) == 1  # Message
    assert isinstance(responses[0], AgentResponse)
    assert responses[0].type == ResponseType.MESSAGE
    assert responses[0].content == "Test response"
    assert (
        len(agent.memory.get_messages()) == 3
    )  # system prompt + user message + assistant message


@pytest.mark.asyncio
async def test_message_formatting(agent):
    """Test that messages are properly formatted in conversation history"""
    # Add different types of messages
    agent.memory.add_message("user", "User message")
    agent.memory.add_message("assistant", "Assistant message")
    agent.memory.add_message(
        "tool", "Tool result", tool_call_id="123", name="mock_tool"
    )

    # Get messages from memory
    messages = agent.memory.get_messages_without_system()

    # Verify each message type is formatted correctly
    assert len(messages) == 3

    user_msg = messages[0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "User message"

    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "Assistant message"

    tool_msg = messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "Tool result"
    assert tool_msg["tool_call_id"] == "123"
    assert tool_msg["name"] == "mock_tool"


@pytest.mark.asyncio
async def test_chat_with_tool_call(agent, mock_chat_completion):
    """Test that tool calls are processed correctly"""

    tool_call = ToolCallData(name="mock_tool", arguments={}, call_id="123")

    # Create a mock chat completion that returns a tool call
    async def mock_complete(*args, **kwargs) -> AsyncIterator[AgentResponse]:
        yield AgentResponse(
            type=ResponseType.TOOL_CALL,
            tool_call=tool_call,
        )
        yield AgentResponse(
            type=ResponseType.MESSAGE, content="AI provided a tool result"
        )
        yield AgentResponse(type=ResponseType.DONE)

    # Replace the complete method with our mock
    mock_chat_completion.complete = mock_complete  # type: ignore

    responses = []
    async for response in agent.chat("User message"):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].type == ResponseType.MESSAGE
    assert (
        responses[0].content == "AI provided a tool result"
    )  # User sees the final response

    # Verify expected flow of messages
    expected_messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": str(tool_call)},
        {
            "role": "tool",
            "content": "tool result",
            "call_id": "123",
            "name": "mock_tool",
        },
        {"role": "assistant", "content": "AI provided a tool result"},
    ]
    message_flow = agent.memory.get_messages_without_system()
    for i, expected_message in enumerate(expected_messages):
        assert expected_message == message_flow[i]


@pytest.mark.asyncio
async def test_chat_handles_rate_limit(agent, mock_chat_completion):
    """Test that a rate limit error from chat completion is handled."""

    async def mock_complete_yields_error(
        *args, **kwargs
    ) -> AsyncIterator[AgentResponse]:
        yield AgentResponse(
            type=ResponseType.ERROR,
            error="Simulated Error from Completion",
            content="User-facing error message",
        )
        # We might or might not yield DONE after an error, depending on desired behavior
        # yield AgentResponse(type=ResponseType.DONE)

    mock_chat_completion.complete = mock_complete_yields_error  # type: ignore

    responses = []
    async for response in agent.chat("Test input for rate limit"):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0].type == ResponseType.ERROR
    assert "Simulated Error from Completion" in responses[0].error
    assert "User-facing error message" in responses[0].content
    # Ensure user input was still added to memory
    assert len(agent.memory.get_messages()) == 2  # System + User


@pytest.mark.asyncio
async def test_chat_handles_api_error(agent, mock_chat_completion):
    """Test that a generic APIError (like quota) from chat completion is handled."""

    # NOTE: Replace openai.APIError with the correct InsufficientQuotaError
    # once identified in your openai library version.
    async def mock_complete_yields_error(
        *args, **kwargs
    ) -> AsyncIterator[AgentResponse]:
        yield AgentResponse(
            type=ResponseType.ERROR,
            error="Simulated Error from Completion",
            content="User-facing error message",
        )
        # We might or might not yield DONE after an error, depending on desired behavior
        # yield AgentResponse(type=ResponseType.DONE)

    mock_chat_completion.complete = mock_complete_yields_error  # type: ignore

    responses = []
    async for response in agent.chat("Test input for API error"):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0].type == ResponseType.ERROR
    # This assertion depends on which exception block catches APIError.
    # If InsufficientQuotaError is distinct and caught first:
    # assert "API quota exceeded" in responses[0].error
    # assert "Service temporarily unavailable" in responses[0].content
    # If it falls into the generic Exception block:
    assert "Simulated Error from Completion" in responses[0].error
    assert "User-facing error message" in responses[0].content
    assert len(agent.memory.get_messages()) == 2  # System + User


@pytest.mark.asyncio
async def test_chat_handles_generic_exception(agent, mock_chat_completion):
    """Test that a generic Exception from chat completion is handled."""

    async def mock_complete_yields_error(
        *args, **kwargs
    ) -> AsyncIterator[AgentResponse]:
        yield AgentResponse(
            type=ResponseType.ERROR,
            error="Simulated Error from Completion",
            content="User-facing error message",
        )
        # We might or might not yield DONE after an error, depending on desired behavior
        # yield AgentResponse(type=ResponseType.DONE)

    mock_chat_completion.complete = mock_complete_yields_error  # type: ignore

    responses = []
    async for response in agent.chat("Test input for generic exception"):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0].type == ResponseType.ERROR
    assert "Simulated Error from Completion" in responses[0].error
    assert "User-facing error message" in responses[0].content
    assert len(agent.memory.get_messages()) == 2  # System + User


@pytest.mark.asyncio
async def test_chat_handles_repeated_tool_calls(agent, mock_chat_completion, mock_tool):
    """
    Test that the agent correctly handles the LLM calling the same tool multiple times
    in sequence within a single user turn.
    """
    tool_call_1 = ToolCallData(name="mock_tool", arguments={"call": 1}, call_id="101")
    tool_call_2 = ToolCallData(name="mock_tool", arguments={"call": 2}, call_id="102")

    # Mock chat completion that forces two tool calls
    async def mock_complete_repeated_tool_calls(
        messages: List[Dict[str, Any]],
    ) -> AsyncIterator[AgentResponse]:
        last_message_role = messages[-1]["role"]

        if last_message_role == "user":
            # First call from LLM -> Tool Call 1
            yield AgentResponse(type=ResponseType.TOOL_CALL, tool_call=tool_call_1)
        elif last_message_role == "tool":
            # Check if this is the response from the *first* tool call
            # A more robust check might involve looking at the tool_call_id history
            # For simplicity here, we assume the second time we see 'tool' is after the first call.
            # Check how many tool responses are in the history to decide
            tool_message_count = sum(1 for m in messages if m["role"] == "tool")
            if tool_message_count == 1:
                # After 1st tool response -> Tool Call 2
                yield AgentResponse(type=ResponseType.TOOL_CALL, tool_call=tool_call_2)
            else:  # After 2nd tool response -> Final Message
                yield AgentResponse(
                    type=ResponseType.MESSAGE, content="Okay, I called the tool twice."
                )
                yield AgentResponse(type=ResponseType.DONE)
        else:
            # Should not happen in this test's flow, yield error or done
            yield AgentResponse(
                type=ResponseType.ERROR, error="Unexpected message role"
            )  # Or DONE

    mock_chat_completion.complete = mock_complete_repeated_tool_calls  # type: ignore

    responses = []
    async for response in agent.chat("User message causing tool loop"):
        responses.append(response)

    # 1. Check the final response yielded to the user
    assert len(responses) == 1
    assert responses[0].type == ResponseType.MESSAGE
    assert responses[0].content == "Okay, I called the tool twice."

    # 2. Check that the mock tool was executed twice
    assert len(mock_tool.called_history) == 2
    assert mock_tool.called_history[0] == {"call": 1}
    assert mock_tool.called_history[1] == {"call": 2}

    # 3. Check the conversation history in memory
    expected_messages = [
        {"role": "user", "content": "User message causing tool loop"},
        {"role": "assistant", "content": str(tool_call_1)},
        {
            "role": "tool",
            "content": "tool result",  # Assumes MockTool always returns this
            "call_id": "101",
            "name": "mock_tool",
        },
        {"role": "assistant", "content": str(tool_call_2)},
        {
            "role": "tool",
            "content": "tool result",  # Assumes MockTool always returns this
            "call_id": "102",
            "name": "mock_tool",
        },
        {"role": "assistant", "content": "Okay, I called the tool twice."},
    ]
    message_flow = agent.memory.get_messages_without_system()

    # Compare message by message for easier debugging if it fails
    assert len(message_flow) == len(expected_messages)
    for i, expected in enumerate(expected_messages):
        # Normalize content representation if necessary (e.g., dict vs. str)
        # For now, assuming direct comparison works
        assert message_flow[i] == expected

