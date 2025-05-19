from ast import Break
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List

from pydantic import BaseModel, Field
from services.chat_service.agent.chat_completion import ChatCompletion
from services.chat_service.agent.memory import AgentMemory
from services.chat_service.agent.models import AgentResponse, ResponseType, ToolCallData
from services.chat_service.agent.tool_manager import ToolManager
from services.chat_service.agent.tools.tool import BaseTool

from shared import ServiceLogger

# Get a logger instance for this module
logger = ServiceLogger("agent_core")


class AgentConfig(BaseModel):
    system_prompt: str = Field(description="The system prompt for the agent")
    tools: List[BaseTool] = Field(description="The tools to use for the agent")
    model: str = Field(
        default="gpt-3.5-turbo", description="The model to use for the agent"
    )
    max_tokens: int = Field(
        default=4000,
        ge=1000,
        le=5000,
        description="The maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=1.0, description="The temperature of the model"
    )
    max_conversation_depth: int = Field(
        default=10,
        ge=1,
        le=100,
        description="The maximum number of iterations the agent will take",
    )

    class Config:
        arbitrary_types_allowed = True


@dataclass
class Agent:
    config: AgentConfig
    memory: AgentMemory
    tool_manager: ToolManager
    chat_completion: ChatCompletion

    def __init__(
        self,
        config: AgentConfig,
        chat_completion: ChatCompletion,
    ):
        self.config = config
        self.memory = AgentMemory()
        self.tool_manager = ToolManager(config.tools)
        self.chat_completion = chat_completion
        self.max_conversation_depth = config.max_conversation_depth

    async def chat(
        self, user_input: str, history: List[Dict[str, Any]]
    ) -> AsyncIterator[AgentResponse]:
        """
        Main chat method that handles the conversation flow, using provided history.
        Only yields meaningful responses to the user.
        """
        try:
            self.memory.clear()
            self.memory.initialize_with_system_prompt(self.config.system_prompt)
            self.memory.load_history(history)

            conversation_depth = 0
            last_response_was_done = False
            while True:
                conversation_depth += 1
                if conversation_depth > self.max_conversation_depth:
                    yield AgentResponse(
                        type=ResponseType.ERROR,
                        error="Conversation depth exceeded",
                        content="The conversation became too long. Please start a new one.",
                    )
                    return

                context = self.memory.get_messages()

                # --- Log the EXACT context being sent to LLM in THIS iteration ---
                try:
                    current_context_log = json.dumps(context, indent=2)
                except TypeError:
                    current_context_log = str(context)
                logger.info(
                    f"--- Sending context to LLM (Iteration {conversation_depth}) ---:\n{current_context_log}"
                )
                # -------------------------------------------------------------------

                async for response in self.chat_completion.complete(messages=context):
                    # --- Tool Call Handling ---
                    if response.type == ResponseType.TOOL_CALL and response.tool_call:
                        llm_tool_call: ToolCallData = response.tool_call

                        # Manually construct assistant message
                        try:
                            arguments_json_str = json.dumps(llm_tool_call.arguments)
                        except TypeError as e:
                            logger.error(
                                f"Failed to serialize tool arguments to JSON: {e}",
                                arguments=llm_tool_call.arguments,
                            )
                            yield AgentResponse(
                                type=ResponseType.ERROR,
                                content=f"Internal error: Could not serialize arguments for tool {llm_tool_call.name}",
                            )
                            return
                        openai_tool_call_struct = {
                            "id": llm_tool_call.call_id,
                            "type": "function",
                            "function": {
                                "name": llm_tool_call.name,
                                "arguments": arguments_json_str,
                            },
                        }
                        assistant_message_with_tool_call = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [openai_tool_call_struct],
                        }
                        self.memory.conversation_history.append(
                            assistant_message_with_tool_call
                        )
                        logger.debug(
                            f"--- Added assistant tool_calls message to memory ---:\n{json.dumps(assistant_message_with_tool_call, indent=2)}"
                        )

                        # Execute Tool
                        tool_response = await self.tool_manager.execute(llm_tool_call)
                        tool_call_id = llm_tool_call.call_id
                        tool_name = llm_tool_call.name

                        # Construct base tool result message
                        tool_result_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                        }

                        if tool_response.error:
                            simplified_error = f"Tool execution failed: {tool_response.error.splitlines()[0]}"
                            tool_result_message["content"] = simplified_error
                            # Add tool error message to history
                            self.memory.conversation_history.append(tool_result_message)
                            logger.debug(
                                f"--- Added tool error message to history (direct append) ---"
                            )
                            # Yield user-facing message about error
                            yield AgentResponse(
                                type=ResponseType.MESSAGE,
                                content=f"I encountered an error trying to use the `{tool_name}` tool. I'll try to answer without it.",
                            )
                            # Break inner loop - outer loop will call LLM again with error context
                            break

                        elif tool_response.content:
                            tool_result_message["content"] = tool_response.content
                            # Add tool success message to history
                            self.memory.conversation_history.append(tool_result_message)
                            logger.debug(
                                f"--- Added tool success message to history (direct append) ---"
                            )
                            # Continue inner loop - let LLM process result immediately
                            continue
                        else:  # Tool success, no content
                            no_content_msg = (
                                "(Tool executed successfully but returned no content)"
                            )
                            tool_result_message["content"] = no_content_msg
                            # Add tool no-content message to history
                            self.memory.conversation_history.append(tool_result_message)
                            logger.debug(
                                f"--- Added tool no-content message to history (direct append) ---"
                            )
                            # Yield user-facing message about no content
                            yield AgentResponse(
                                type=ResponseType.MESSAGE,
                                content=f"I used the `{tool_name}` tool but didn't find specific details. Let me see what I can suggest based on our conversation...",
                            )
                            # Break inner loop - outer loop will call LLM again with no-content context
                            break

                    # --- Regular Message Handling ---
                    elif response.type == ResponseType.MESSAGE and response.content:
                        # DO NOT add assistant message chunks to the agent's internal memory here.
                        # StreamingChatService will add the full accumulated message to history later.
                        # self.memory.add_message("assistant", response.content)
                        yield response  # Yield the chunk to the service/user
                        continue

                    # --- Error Handling ---
                    elif response.type == ResponseType.ERROR:
                        yield response
                        return

                    # --- Completion Handling ---
                    elif response.type == ResponseType.DONE:
                        if response.finish_reason == "stop":
                            last_response_was_done = True
                            # LLM finished its response stream for this inner loop call.
                            # Return control to the outer while loop, which might continue or terminate.
                            break
                        else:
                            # LLM finished its response stream for this inner loop call.
                            # Return control to the outer while loop, which might continue or terminate.
                            continue

                # After breaking from inner loop (tool call handled, or LLM finished stream)
                # Check if the last response yielded was DONE - indicates end of turn.
                # This check might be fragile depending on exact streaming behavior.
                # A potentially more robust approach is to check if finish_reason was 'stop'.
                # For now, relying on the outer loop's depth limit or next iteration.
                # break  # Let the outer while loop continue or hit depth limit
                if last_response_was_done:
                    break

        except Exception as e:
            import traceback
            logger.debug(f"Conversation depth: {conversation_depth} user_input: {user_input} context: {context}")
            logger.error(f"Unexpected error in Agent.chat: {traceback.format_exc()}")
            yield AgentResponse(
                type=ResponseType.ERROR,
                error=f"Unexpected error in Agent.chat: {str(e)}",
                content="An unexpected error occurred while processing your request.",
            )
