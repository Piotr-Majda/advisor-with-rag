import json
from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from redis import Redis
from services.chat_service.agent.core import Agent
from services.chat_service.agent.models import ResponseType

from shared import ServiceLogger

logger = ServiceLogger("chat_service")

# Define a consistent message format
MESSAGE_TYPE = Dict[str, Any]  # e.g., {"role": "user", "content": "hello"}


class StreamingChatService:
    def __init__(self, agent: Agent, session_id: str, redis_client: Redis):
        self.agent = agent
        self.websocket: Optional[WebSocket] = None
        self.connection_open = False
        self.session_id = session_id
        self.redis_client = redis_client
        self.history: List[MESSAGE_TYPE] = []  # Initialize history
        self._redis_key = f"chat_history:{self.session_id}"

    async def initialize(self, websocket: WebSocket) -> None:
        """Initialize WebSocket connection and load history."""
        self.websocket = websocket
        if not self.websocket:
            raise ValueError("WebSocket connection not established")
        await self.websocket.accept()
        self.connection_open = True
        await self._load_history()  # Load history on initialization
        logger.info(
            f"New WebSocket connection established for session {self.session_id}. History loaded."
        )

    async def _load_history(self) -> None:
        """Load chat history from Redis."""
        try:
            # Retrieve all messages from the list
            raw_messages = self.redis_client.lrange(self._redis_key, 0, -1)
            self.history = [json.loads(msg) for msg in raw_messages]
            logger.info(
                f"Loaded {len(self.history)} messages for session {self.session_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load history for session {self.session_id} from Redis",
                error=str(e),
            )
            self.history = []  # Start fresh if loading fails

    async def _save_history(self) -> None:
        """Save chat history to Redis."""
        try:
            # Use a pipeline for atomic delete and push
            pipeline = self.redis_client.pipeline()
            pipeline.delete(self._redis_key)
            if self.history:  # Only push if there's history
                # Serialize messages to JSON strings before saving
                serialized_messages = [json.dumps(msg) for msg in self.history]
                pipeline.rpush(self._redis_key, *serialized_messages)
            # Optionally set an expiration time for the history (e.g., 24 hours)
            # 30 min expiration time
            pipeline.expire(self._redis_key, 1800)  # 30 min = 60 sec * 30 = 1800 sec
            pipeline.execute()
            logger.info(
                f"Saved {len(self.history)} messages for session {self.session_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save history for session {self.session_id} to Redis",
                error=str(e),
            )

    async def handle_question(self, question: str) -> None:
        """Handle incoming question, update history, stream responses, and save history."""
        if not self.websocket or not self.connection_open:
            raise ValueError("WebSocket connection not established")

        if not question.strip():
            await self._send_error("Please provide a valid question.")
            return

        # Append user message to history
        self.history.append({"role": "user", "content": question})

        full_response_content = ""
        try:
            # !!! IMPORTANT: Assumes self.agent.chat accepts history !!!
            # We will need to modify Agent.chat later
            async for response_chunk in self.agent.chat(question, history=self.history):
                if (
                    response_chunk.type == ResponseType.MESSAGE
                    and response_chunk.content
                ):
                    logger.info(
                        f"Sent message to websocket: '{response_chunk.content}'"
                    )
                    await self.websocket.send_text(response_chunk.content)
                    full_response_content += (
                        response_chunk.content
                    )  # Collect full response
                elif (
                    response_chunk.type == ResponseType.ERROR and response_chunk.content
                ):
                    logger.error(
                        "Error processing question",
                        error=response_chunk.error,
                        session_id=self.session_id,
                    )
                    await self._send_error(response_chunk.content)
                    # Don't save history if there was an error mid-stream? Or save up to error? Decide policy.
                    # For now, we break and save what we have before the error response.
                    break

            # Append full assistant response to history *after* successful streaming
            if full_response_content and not (
                response_chunk.type == ResponseType.ERROR
            ):  # Check it wasn't an error response
                self.history.append(
                    {"role": "assistant", "content": full_response_content}
                )

            # Send end marker
            await self.websocket.send_text("[END]")

        except Exception as e:
            logger.error(
                "Error processing question", error=str(e), session_id=self.session_id
            )
            await self._send_error("An error occurred while processing your question.")
            await self.websocket.send_text("[END]")
            # Decide if history should be saved on exception. Let's save for now to capture context.
        finally:
            # Save history after handling the question (even if errors occurred)
            await self._save_history()

    async def _send_error(self, message: str) -> None:
        """Send error message through WebSocket."""
        if self.websocket and self.connection_open:
            # Append error to history? Maybe not, as it's not part of the LLM convo.
            await self.websocket.send_text(f"Error: {message}")

    async def close(self) -> None:
        """Close WebSocket connection. History is saved after each message, so no save needed here typically."""
        if not self.connection_open:
            return

        try:
            if self.websocket:
                await self.websocket.close()
                logger.info(
                    f"WebSocket connection closed by server for session {self.session_id}"
                )
        except Exception as e:
            logger.info(
                f"Error closing WebSocket for session {self.session_id}", error=str(e)
            )
        finally:
            self.connection_open = False
            self.websocket = None
            # History is saved after each message in handle_question.
            # If you needed a final save on disconnect, add _save_history() here.
