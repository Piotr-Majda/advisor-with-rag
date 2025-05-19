
import os
from fastapi import Depends
from typing import List
from services.chat_service.agent.agent_manager import AgentManager
from services.chat_service.agent.core import Agent, AgentConfig
from services.chat_service.agent.chat_completion import OpenAIChatCompletion
from services.chat_service.agent.tools.document_search import DocumentSearchTool
from services.chat_service.agent.tools.web_search import WebSearchTool
from services.chat_service.agent.tools.tool import BaseTool
from services.chat_service.agent.prompts import system_prompt
from shared import TokenBucketRateLimiter, redis_manager
from redis import Redis
from dotenv import load_dotenv

load_dotenv()


def get_redis_client() -> Redis:
    return redis_manager.get_client()

def get_rate_limiter_chat_completion(redis_client: Redis = Depends(get_redis_client)) -> TokenBucketRateLimiter:
    return TokenBucketRateLimiter(int(os.getenv("RATE_LIMIT_CHAT_COMPLETION", 10)), int(os.getenv("WINDOW_CHAT_COMPLETION", 60)), redis_client=redis_client)


def get_tools() -> List[BaseTool]:
    return [DocumentSearchTool(), WebSearchTool()]


def get_chat_completion(tools: List[BaseTool] = Depends(get_tools), rate_limiter: TokenBucketRateLimiter = Depends(get_rate_limiter_chat_completion)) -> OpenAIChatCompletion:
    return OpenAIChatCompletion(model="gpt-3.5-turbo", temperature=0.7, tools=[tool.to_openai_function() for tool in tools], rate_limiter=rate_limiter)


def get_agent(chat_completion: OpenAIChatCompletion = Depends(get_chat_completion)) -> Agent:
    tools = [DocumentSearchTool(), WebSearchTool()]
    agent_manager = AgentManager(AgentConfig(system_prompt=system_prompt, tools=tools), chat_completion)
    return agent_manager.get_agent()
