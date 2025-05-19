from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from openai.types.chat import ChatCompletionToolParam
from services.chat_service.agent.chat_completion import OpenAIChatCompletion
from services.chat_service.agent.models import AgentResponse, ResponseType, ToolCallData

# --- Test Data ---

SAMPLE_MESSAGES: List[Dict[str, Any]] = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
]

SAMPLE_TOOLS_DEFINITION: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

# --- Helper to Create Mock Chunks ---


def create_mock_chat_completion_chunk(
    content: str | None = None,
    tool_call_id: str | None = None,
    tool_function_name: str | None = None,
    tool_function_args_delta: str | None = None,
    finish_reason: str | None = None,
) -> MagicMock:
    """Creates a mock OpenAI ChatCompletionChunk object."""
    mock_chunk = MagicMock()
    mock_choice = MagicMock()
    mock_delta = MagicMock()

    mock_delta.content = content
    mock_delta.tool_calls = None
    mock_choice.finish_reason = finish_reason

    if tool_call_id or tool_function_name or tool_function_args_delta:
        mock_tool_call = MagicMock()
        mock_tool_call.id = tool_call_id
        mock_function = MagicMock()
        mock_function.name = tool_function_name
        mock_function.arguments = tool_function_args_delta
        mock_tool_call.function = mock_function
        mock_delta.tool_calls = [mock_tool_call]
        # Setting content to None if tool_calls are present, mimicking OpenAI behaviour
        mock_delta.content = None

    mock_choice.delta = mock_delta
    mock_chunk.choices = [mock_choice]
    return mock_chunk


async def async_generator(items):
    """Helper to create an async generator from a list."""
    for item in items:
        yield item


# --- Tests ---

@pytest.fixture
def rate_limiter():
    return MagicMock(is_allowed=MagicMock(return_value=True), get_reset_time=MagicMock(return_value=0))


@pytest.fixture
def chat_completion(rate_limiter):
    """Fixture for OpenAIChatCompletion instance."""
    return OpenAIChatCompletion(
        model="test-model",
        temperature=0.5,
        tools=SAMPLE_TOOLS_DEFINITION,
        rate_limiter=rate_limiter,
    )


@pytest.mark.asyncio
@patch("openai.chat.completions.create", new_callable=AsyncMock)
async def test_complete_simple_message(mock_create, chat_completion):
    """Test completion yields a simple message correctly."""
    mock_create.return_value = async_generator(
        [
            create_mock_chat_completion_chunk(content="Hello "),
            create_mock_chat_completion_chunk(content="there!"),
            create_mock_chat_completion_chunk(finish_reason="stop"),
        ]
    )

    responses = [resp async for resp in chat_completion.complete(SAMPLE_MESSAGES)]

    # Check API call arguments
    mock_create.assert_called_once()
    call_args, call_kwargs = mock_create.call_args
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["stream"] is True
    assert call_kwargs["tools"] == SAMPLE_TOOLS_DEFINITION  # Verify tools are passed
    # Check messages formatting (basic check)
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"

    # Check yielded responses
    assert len(responses) == 4  # chunk1, chunk2, complete_msg, done
    assert (
        responses[0].type == ResponseType.MESSAGE and responses[0].content == "Hello "
    )
    assert (
        responses[1].type == ResponseType.MESSAGE and responses[1].content == "there!"
    )
    assert (
        responses[2].type == ResponseType.MESSAGE
        and responses[2].content == "Hello there!"
    )
    assert responses[3].type == ResponseType.DONE


@pytest.mark.asyncio
@patch("openai.chat.completions.create", new_callable=AsyncMock)
async def test_complete_tool_call(mock_create, chat_completion):
    """Test completion yields a tool call correctly."""
    mock_create.return_value = async_generator(
        [
            create_mock_chat_completion_chunk(
                tool_call_id="call_123", tool_function_name="get_weather"
            ),
            create_mock_chat_completion_chunk(
                tool_call_id="call_123", tool_function_args_delta='{"loca'
            ),
            create_mock_chat_completion_chunk(
                tool_call_id="call_123", tool_function_args_delta='tion": "Boston"}'
            ),
            create_mock_chat_completion_chunk(
                finish_reason="tool_calls"
            ),  # Usually stop or tool_calls
        ]
    )

    responses = [resp async for resp in chat_completion.complete(SAMPLE_MESSAGES)]

    mock_create.assert_called_once()
    # Check yielded responses
    assert len(responses) == 2  # Expect TOOL_CALL then DONE
    assert responses[0].type == ResponseType.TOOL_CALL
    assert isinstance(responses[0].tool_call, ToolCallData)
    assert responses[0].tool_call.call_id == "call_123"
    assert responses[0].tool_call.name == "get_weather"
    assert responses[0].tool_call.arguments == {"location": "Boston"}
    assert responses[1].type == ResponseType.DONE  # Check for the DONE response


@pytest.mark.asyncio
@patch("openai.chat.completions.create", new_callable=AsyncMock)
async def test_complete_handles_rate_limit_error(mock_create, chat_completion):
    """Test handling of RateLimitError."""
    # Configure the mock to raise RateLimitError
    mock_create.side_effect = openai.RateLimitError(
        "Rate limit exceeded",
        response=httpx.Response(429, request=httpx.Request("POST", "/")),  # type: ignore
        body=None,
    )

    responses = [resp async for resp in chat_completion.complete(SAMPLE_MESSAGES)]

    mock_create.assert_called_once()
    assert len(responses) == 1
    assert responses[0].type == ResponseType.ERROR
    assert "Rate limit exceeded" in responses[0].error
    assert "high demand" in responses[0].content


@pytest.mark.asyncio
@patch("openai.chat.completions.create", new_callable=AsyncMock)
async def test_complete_handles_generic_exception(mock_create, chat_completion):
    """Test handling of a generic Exception."""
    mock_create.side_effect = ValueError("Something unexpected")

    responses = [resp async for resp in chat_completion.complete(SAMPLE_MESSAGES)]

    mock_create.assert_called_once()
    assert len(responses) == 1
    assert responses[0].type == ResponseType.ERROR
    assert "Unexpected error: Something unexpected" in responses[0].error
    assert "An error occurred" in responses[0].content
