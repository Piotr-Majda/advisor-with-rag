from services.chat_service.agent.chat_completion import ChatCompletion
from services.chat_service.agent.core import Agent, AgentConfig


# Singleton pattern
class AgentManager:
    _instance = None
    _initialized = False  # Add an initialization flag

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Correctly call object.__new__ which only takes the class
            cls._instance = super(AgentManager, cls).__new__(cls)
        return cls._instance

    def __init__(
        self, config: AgentConfig, chat_completion: ChatCompletion
    ):  # Corrected typo chat_complition -> chat_completion
        # Ensure __init__ only runs once
        if AgentManager._initialized:
            return
        self.agent = Agent(config, chat_completion)
        AgentManager._initialized = True  # Mark as initialized

    def get_agent(self) -> Agent:
        if not AgentManager._initialized or not hasattr(self, "agent"):
            raise RuntimeError(
                "AgentManager has not been initialized with config and chat_completion yet."
            )
        return self.agent
