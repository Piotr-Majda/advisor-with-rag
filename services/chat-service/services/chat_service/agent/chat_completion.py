import json
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from services.chat_service.agent.models import AgentResponse, ResponseType, ToolCallData

from shared import TokenBucketRateLimiter


class ChatCompletion(ABC):
    """Abstract base class for chat completion providers"""

    @abstractmethod
    async def complete(
        self, messages: List[Dict[str, Any]]
    ) -> AsyncIterator[AgentResponse]:
        """Process chat completion and return structured responses"""
        pass


class OpenAIChatCompletion(ChatCompletion):
    """OpenAI implementation of chat completion"""

    def __init__(
        self,
        model: str,
        temperature: float,
        tools: List[ChatCompletionToolParam],
        rate_limiter: TokenBucketRateLimiter,
    ):
        self.model = model
        self.temperature = temperature
        self.tools = tools
        self.rate_limiter = rate_limiter
        self.client = AsyncOpenAI()

    async def complete(
        self,
        messages: List[Dict[str, Any]],
    ) -> AsyncIterator[AgentResponse]:
        """Process chat completion using OpenAI and return structured responses"""

        limiter_key = "openai_chat_completion"
        # if not self.rate_limiter.is_allowed(limiter_key):
        #     self.rate_limiter.logger.warning(
        #         f"Global rate limit exceeded for key: {limiter_key} in ChatCompletion"
        #     )
        #     yield AgentResponse(
        #         type=ResponseType.ERROR,
        #         error="Rate limit exceeded for chat completion service.",
        #         content="Service is busy, please try again shortly.",
        #     )
        #     return

        formatted_messages = []
        for message in messages:
            role = message["role"]

            # Base message structure
            msg: Dict[str, Any] = {"role": role}

            if role == "tool":
                # Add required fields for tool role
                msg["content"] = str(message.get("content", ""))
                msg["tool_call_id"] = str(message.get("tool_call_id", ""))
            elif role == "assistant":
                # Handle assistant messages: could have content or tool_calls
                if "tool_calls" in message and message["tool_calls"]:
                    # Ensure content is None or omitted if tool_calls is present
                    msg["content"] = message.get("content")  # Could be None
                    # Copy the tool_calls structure
                    msg["tool_calls"] = message["tool_calls"]
                else:
                    # Regular assistant message with content
                    msg["content"] = str(message.get("content", ""))
            else:  # System, User
                msg["content"] = str(message.get("content", ""))

            # Ensure None content is handled correctly (OpenAI API might prefer omitting the key)
            if msg.get("content") is None and "tool_calls" not in msg:
                # If content is None and it's not a tool_calls message,
                # we might need to send empty string or omit content key depending on API strictness.
                # Let's send empty string for now, adjust if needed.
                msg["content"] = ""
            elif msg.get("content") is None and "tool_calls" in msg:
                # If it's a tool_calls message, content *should* be None. Remove the key entirely.
                del msg["content"]

            formatted_messages.append(cast(ChatCompletionMessageParam, msg))

        try:
            # Add safety parameters to help prevent prompt injection
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=self.tools,
                temperature=self.temperature,
                stream=True,
                max_tokens=4000,
                timeout=30,
            )

            current_content = []
            current_tool_call = None

            async for chunk in completion:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Validate and sanitize tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_call = delta.tool_calls[0]
                    if not current_tool_call:
                        # Validate tool name against allowlist
                        func_name = (
                            tool_call.function.name if tool_call.function else ""
                        )
                        if func_name not in [t.get("name") for t in self.tools]:
                            raise ValueError(f"Invalid tool name: {func_name}")

                        func_args = (
                            tool_call.function.arguments if tool_call.function else None
                        )
                        current_tool_call = {
                            "id": tool_call.id,
                            "name": func_name,
                            "arguments": func_args or "",
                        }
                    else:
                        if tool_call.function and tool_call.function.arguments:
                            current_tool_call[
                                "arguments"
                            ] += tool_call.function.arguments

                elif hasattr(delta, "content") and delta.content:
                    current_content.append(delta.content)
                    yield AgentResponse(
                        type=ResponseType.MESSAGE, content=delta.content
                    )

                if finish_reason == "stop":
                    yield AgentResponse(
                        type=ResponseType.DONE,
                        finish_reason=finish_reason,
                    )
                    break

                elif finish_reason == "tool_calls":
                    parsed_arguments = {}
                    arguments_string = ""
                    if (
                        current_tool_call
                        and current_tool_call.get("name")
                        and current_tool_call.get("arguments") is not None
                    ):
                        arguments_string = current_tool_call["arguments"]
                        try:
                            # Validate JSON structure
                            parsed_arguments = json.loads(arguments_string)
                            if not isinstance(parsed_arguments, dict):
                                raise json.JSONDecodeError(
                                    "Arguments JSON is not an object",
                                    arguments_string,
                                    0,
                                )

                            # Additional argument validation
                            for key, value in parsed_arguments.items():
                                if not isinstance(key, str):
                                    raise ValueError("Argument keys must be strings")
                                if len(key) > 100:  # Reasonable key length limit
                                    raise ValueError("Argument key too long")
                                if isinstance(value, str) and len(value) > 1000:
                                    raise ValueError("Argument value too long")

                            yield AgentResponse(
                                type=ResponseType.TOOL_CALL,
                                tool_call=ToolCallData(
                                    name=current_tool_call["name"],
                                    arguments=parsed_arguments,
                                    call_id=current_tool_call["id"],
                                ),
                            )
                        except (json.JSONDecodeError, ValueError) as err:
                            yield AgentResponse(
                                type=ResponseType.ERROR,
                                error=f"Invalid tool arguments: {str(err)}",
                                content="An error occurred validating tool arguments.",
                            )

                        current_tool_call = None

                    yield AgentResponse(
                        type=ResponseType.DONE,
                        finish_reason=finish_reason,
                    )
                    break
        except openai.RateLimitError as e:
            yield AgentResponse(
                type=ResponseType.ERROR,
                error="Rate limit exceeded. Please try again later.",
                content="The service is temporarily unavailable due to high demand.",
            )
        except openai.APIError as e:
            yield AgentResponse(
                type=ResponseType.ERROR,
                error=f"OpenAI API error: {str(e)}",
                content="An error occurred while processing your request.",
            )
