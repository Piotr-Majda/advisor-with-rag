from dependencies import get_agent, get_redis_client
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from services.chat_service.agent.core import Agent
from services.chat_service.chat_service import StreamingChatService

from shared import ServiceLogger, create_logging_middleware

logger = ServiceLogger("chat_service")

app = FastAPI(title="Chat Service")

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.middleware("http")(create_logging_middleware(logger))


@app.websocket("/chat")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    agent: Agent = Depends(get_agent),
    redis_client: Redis = Depends(get_redis_client),
):
    logger.info(f"Chat connection established for session_id: {session_id}")

    chat_service = StreamingChatService(
        agent=agent, session_id=session_id, redis_client=redis_client
    )

    try:
        await chat_service.initialize(websocket)
        while True:
            try:
                question = await websocket.receive_text()
                logger.info(
                    "Received question", question=question, session_id=session_id
                )
                await chat_service.handle_question(question)
            except WebSocketDisconnect:
                logger.info(
                    "WebSocket connection closed by client", session_id=session_id
                )
                break
    except Exception as e:
        logger.error("Critical WebSocket error", error=str(e), session_id=session_id)
    finally:
        await chat_service.close()
        logger.info("Chat service connection closed", session_id=session_id)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
