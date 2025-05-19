# services/chat-service/tests/test_chat_completion_int.py

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletionToolParam
from services.chat_service.agent.chat_completion import OpenAIChatCompletion
from services.chat_service.agent.models import AgentResponse, ResponseType, ToolCallData

# Import the actual rate limiter to mock its spec if needed, but we'll use MagicMock
# from shared.shared.rate_limiter import TokenBucketRateLimiter

# --- Test Configuration ---

# Skip these tests unless the environment variable is set
# Use `export RUN_INTEGRATION_TESTS=1` or similar before running pytest
RUN_INTEGRATION_TESTS = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"
API_KEY_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))

# Decorator to skip tests if conditions aren't met
integration_test = pytest.mark.skipif(
    not RUN_INTEGRATION_TESTS or not API_KEY_AVAILABLE,
    reason="Integration tests not enabled or OPENAI_API_KEY not set",
)

MODEL_UNDER_TEST = "gpt-3.5-turbo"  # Or another model you want to test against

SAMPLE_MESSAGES: List[Dict[str, Any]] = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Say hello."},
]

# A valid tool definition using the correct type
SAMPLE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_current_location",
        "description": "Get the user's current location",
        "parameters": {"type": "object", "properties": {}},  # No params needed
    },
}

# --- Fixtures ---


@pytest.fixture(scope="module")  # Scope to module to potentially reuse client if needed
def mock_rate_limiter_int():
    """Fixture for a mock rate limiter that always allows."""
    limiter = MagicMock()  # spec=TokenBucketRateLimiter if needed
    limiter.is_allowed.return_value = True
    limiter.logger = MagicMock()
    return limiter


@pytest.fixture(scope="module")
def chat_completion_service_int(mock_rate_limiter_int):
    """Fixture for the actual OpenAIChatCompletion service instance."""
    if not API_KEY_AVAILABLE:
        pytest.skip("OPENAI_API_KEY not set, skipping integration test setup.")
    # Note: Tools are not included by default, add them per test case if needed
    return OpenAIChatCompletion(
        model=MODEL_UNDER_TEST,
        temperature=0.1,  # Low temp for more predictable (but not guaranteed) results
        tools=[],  # Start with no tools
        rate_limiter=mock_rate_limiter_int,
    )


@pytest.fixture(scope="module")
def chat_completion_service_with_tool_int(mock_rate_limiter_int):
    """Fixture for the service instance configured WITH a tool."""
    if not API_KEY_AVAILABLE:
        pytest.skip("OPENAI_API_KEY not set, skipping integration test setup.")
    return OpenAIChatCompletion(
        model=MODEL_UNDER_TEST,
        temperature=0.1,
        tools=[SAMPLE_TOOL],  # Include the tool definition
        rate_limiter=mock_rate_limiter_int,
    )


# --- Integration Tests ---


@integration_test
@pytest.mark.asyncio
async def test_integration_simple_message(chat_completion_service_int):
    """
    Test getting a simple text response from the actual OpenAI API.
    Verifies basic connectivity and response streaming.
    """
    print(f"\nRunning integration test: Simple Message ({MODEL_UNDER_TEST})")
    final_message_content = None
    message_received = False
    done_received = False

    try:
        async for response in chat_completion_service_int.complete(
            messages=SAMPLE_MESSAGES
        ):
            assert response.type in [ResponseType.MESSAGE, ResponseType.DONE]
            if response.type == ResponseType.MESSAGE:
                assert isinstance(response.content, str)
                # Accumulate for final check if needed, or just check if *any* message received
                message_received = True
                print(
                    f"  Received chunk: {response.content!r}"
                )  # Log output during test
            elif response.type == ResponseType.DONE:
                done_received = True
                print("  Received DONE")

    except Exception as e:
        pytest.fail(f"OpenAI API call failed unexpectedly: {e}")

    assert message_received, "Should have received at least one MESSAGE response"
    assert done_received, "Should have received a DONE response"
    print("Integration test: Simple Message PASSED")


@integration_test
@pytest.mark.asyncio
async def test_integration_call_with_tool_defined(
    chat_completion_service_with_tool_int,
):
    """
    Test making an API call with a tool defined.
    Verifies the API accepts the tool format without error and returns a response
    (doesn't guarantee the tool is called, just that the definition is valid).
    """
    print(f"\nRunning integration test: Call With Tool ({MODEL_UNDER_TEST})")
    response_received = False
    done_received = False

    # Use a prompt that *might* elicit tool use, but don't rely on it
    messages_for_tool = [
        {
            "role": "system",
            "content": "You are an assistant that can get the user location.",
        },
        {"role": "user", "content": "Where am I?"},
    ]

    try:
        async for response in chat_completion_service_with_tool_int.complete(
            messages=messages_for_tool
        ):
            assert response.type in [
                ResponseType.MESSAGE,
                ResponseType.TOOL_CALL,
                ResponseType.DONE,
            ]
            response_received = True
            print(f"  Received response type: {response.type.name}")  # Log output
            if response.type == ResponseType.TOOL_CALL:
                assert response.tool_call is not None
                assert response.tool_call.name == SAMPLE_TOOL["function"]["name"]
                print(f"  Received TOOL_CALL: {response.tool_call.name}")
            elif response.type == ResponseType.MESSAGE:
                print(f"  Received MESSAGE chunk: {response.content!r}")
            elif response.type == ResponseType.DONE:
                done_received = True
                print("  Received DONE")

    except Exception as e:
        pytest.fail(f"OpenAI API call with tool failed unexpectedly: {e}")

    assert (
        response_received
    ), "Should have received at least one response (MESSAGE, TOOL_CALL, or DONE)"
    assert done_received, "Should have received a DONE response"
    print("Integration test: Call With Tool PASSED")


@integration_test
@pytest.mark.asyncio
async def test_integration_summarize_tool_result(chat_completion_service_with_tool_int):
    """
    Test providing a tool result back to the LLM and getting a response.
    """
    print(f"\nRunning integration test: Summarize Tool Result ({MODEL_UNDER_TEST})")
    # Use the service instance that knows about the tool
    service = chat_completion_service_with_tool_int

    message_received = False
    done_received = False

    tool_call_id = "call_abc123"
    tool_name = "get_current_location"
    # Tool result usually comes back as a string (often JSON) from the tool execution
    tool_result_content = '{"location": "San Francisco, CA"}'

    messages_with_tool_result: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When given a location, state it clearly.",
        },
        {"role": "user", "content": "Where am I?"},
        {
            "role": "assistant",
            "content": None,  # Assistant message content is often None when making tool calls
            "tool_calls": [
                {
                    "id": tool_call_id,  # Must match the ID in the tool role message
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        # Arguments the LLM decided to call the tool with (can be empty string if none)
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,  # Must match the ID in the assistant message
            # "name": tool_name, # Name is not part of the tool role message schema
            "content": tool_result_content,  # The result the tool provided
        },
    ]

    final_summary = ""
    try:
        # Call complete using the service that has the tool defined
        async for response in service.complete(messages=messages_with_tool_result):
            assert response.type in [ResponseType.MESSAGE, ResponseType.DONE]
            if response.type == ResponseType.MESSAGE:
                assert isinstance(response.content, str)
                message_received = True
                final_summary += response.content  # Accumulate the final message
                print(f"  Received chunk: {response.content!r}")
            elif response.type == ResponseType.DONE:
                done_received = True
                print("  Received DONE")

    except Exception as e:
        pytest.fail(f"OpenAI API call with tool result failed unexpectedly: {e}")

    assert (
        message_received
    ), "Should have received at least one MESSAGE response after tool result"
    assert done_received, "Should have received a DONE response"
    # Check if the summary contains the location (case-insensitive)
    assert (
        "san francisco" in final_summary.lower()
    ), f"Expected location in summary, got: {final_summary!r}"
    print(f"Integration test: Summarize Tool Result PASSED. Summary: {final_summary!r}")


@integration_test
@pytest.mark.asyncio
async def test_integration_multi_turn_conversation(chat_completion_service_int):
    """
    Test that the LLM uses information from previous turns in the conversation.
    Simulates: User asks -> Assistant asks follow-up -> User provides info -> Assistant responds using info.
    """
    print(f"\nRunning integration test: Multi-turn Conversation ({MODEL_UNDER_TEST})")
    service = chat_completion_service_int
    message_received = False
    done_received = False

    # Simulate the conversation history leading up to the potentially problematic turn
    conversation_history: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are an investment advisor assistant. Be helpful and ask clarifying questions if needed.",
        },
        {
            "role": "user",
            "content": "What kind of investments should I consider?",  # Initial user query
        },
        {
            "role": "assistant",
            "content": "To give you the best advice, I need a little more information. Could you tell me about your investment goals (e.g., long-term growth, income) and your risk tolerance?",  # Simulated assistant follow-up
        },
        {
            "role": "user",
            "content": "My main goal is long-term growth, and I have a moderate risk tolerance. I'm particularly interested in sustainable energy companies.",
            # User provides the requested specifics
        },
    ]

    final_assistant_response = ""
    try:
        # Ask the service to generate the *next* response based on the history
        async for response in service.complete(messages=conversation_history):
            assert response.type in [ResponseType.MESSAGE, ResponseType.DONE]
            if response.type == ResponseType.MESSAGE:
                assert isinstance(response.content, str)
                message_received = True
                final_assistant_response += (
                    response.content
                )  # Accumulate the final message
                print(f"  Received chunk: {response.content!r}")
            elif response.type == ResponseType.DONE:
                done_received = True
                print("  Received DONE")

    except Exception as e:
        pytest.fail(f"OpenAI API call for multi-turn failed unexpectedly: {e}")

    assert (
        message_received
    ), "Should have received at least one MESSAGE response for the final turn"
    assert done_received, "Should have received a DONE response for the final turn"

    # Check if the final response acknowledges the user's specifics (case-insensitive)
    response_lower = final_assistant_response.lower()
    keywords_found = [
        keyword
        for keyword in ["long-term", "growth", "moderate risk", "sustainable", "energy"]
        if keyword in response_lower
    ]

    assert (
        len(keywords_found) > 0
    ), f"Expected final response to mention user specifics (long-term, growth, moderate risk, sustainable, energy), but got: {final_assistant_response!r}"

    print(
        f"Integration test: Multi-turn Conversation PASSED. Final Response: {final_assistant_response!r}"
    )
