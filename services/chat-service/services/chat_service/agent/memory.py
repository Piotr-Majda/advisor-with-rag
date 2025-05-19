from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AgentMemory(BaseModel):
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)

    def initialize_with_system_prompt(self, system_prompt: str) -> None:
        """Initialize conversation history with system prompt"""
        if (
            not self.conversation_history
            or self.conversation_history[0]["role"] != "system"
        ):
            self.conversation_history.insert(
                0, {"role": "system", "content": system_prompt}
            )
        else:
            # If system prompt already exists, maybe update it? Or ignore?
            # For now, let's assume we only set it once after clearing.
            pass

    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation history"""
        if role == "system":
            # Avoid adding multiple system prompts accidentally
            # Initialization should handle the system prompt
            raise ValueError(
                "System messages should only be added during initialization."
            )
        else:
            message = {"role": role, "content": content, **kwargs}
            self.conversation_history.append(message)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation history"""
        return self.conversation_history

    def get_messages_without_system(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation history without system message"""
        return [msg for msg in self.conversation_history if msg["role"] != "system"]

    def load_history(self, history: List[Dict[str, Any]]) -> None:
        """Load messages from an external history list, skipping any system prompts within it"""
        # Assumes self.conversation_history already contains the system prompt (if any)
        for msg in history:
            if msg.get("role") != "system":  # Avoid adding duplicate system prompts
                self.conversation_history.append(msg)

    def clear(self) -> None:
        """Clear entire conversation history"""
        self.conversation_history = []
