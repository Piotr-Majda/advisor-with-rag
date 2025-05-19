import asyncio
import os
from typing import List, Optional

import requests
import websockets
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from rate_limiter_middleware import create_rate_limit_middleware

from shared import ServiceLogger, create_logging_middleware

logger = ServiceLogger("api-gateway")

load_dotenv()

app = FastAPI(title="API Gateway")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.middleware("http")(create_logging_middleware(logger))
app.middleware("http")(create_rate_limit_middleware(logger))

# Service URLs
DOCUMENT_SERVICE_URL = os.getenv("DOCUMENT_SERVICE_URL", "http://localhost:8001")
CHAT_SERVICE_URL = os.getenv("CHAT_SERVICE_URL", "http://localhost:8003")
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8002")


@app.post("/documents/process")
async def process_documents(files: List[UploadFile] = File(...)):
    try:
        files_data = []
        for file in files:
            file_content = await file.read()
            files_data.append(
                ("files", (file.filename, file_content, file.content_type))
            )

        response = requests.post(f"{DOCUMENT_SERVICE_URL}/process", files=files_data)
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket, session_id: Optional[str] = None):
    logger.critical(
        f"!!!!!! Entered websocket_chat. Received session_id: {session_id} !!!!!!"
    )

    if not session_id:
        logger.error(
            "!!!!!! Session ID is missing from query parameters. Closing connection. !!!!!!"
        )
        await websocket.close(code=1008)
        return

    logger.info(
        f"Incoming chat connection accepted from client {websocket.client} for session_id: {session_id}"
    )
    try:
        await websocket.accept()
        logger.info("Client connection accepted")

        chat_service_ws_url = f"ws://{CHAT_SERVICE_URL.replace('http://', '')}/chat?session_id={session_id}"
        logger.info(f"Connecting to chat service at: {chat_service_ws_url}")

        async with websockets.connect(chat_service_ws_url) as chat_ws:
            logger.info("Chat service connection established")

            async def receive_and_forward():
                """Receives messages from client and forwards to chat service"""
                while True:
                    message = await websocket.receive_text()
                    logger.info(
                        f"Forwarding message from client to chat service",
                        session_id=session_id,
                    )
                    await chat_ws.send(message)

            async def receive_and_send_back():
                """Receives messages from chat service and sends to client"""
                async for response in chat_ws:
                    if isinstance(response, bytes):
                        response_text = response.decode("utf-8")
                    else:
                        response_text = str(response)

                    logger.info(
                        f"Forwarding message from chat service to client",
                        session_id=session_id,
                    )
                    await websocket.send_text(response_text)
                    if response_text == "[END]":
                        continue

            await asyncio.gather(receive_and_forward(), receive_and_send_back())

        logger.info("Chat session closed normally", session_id=session_id)

    except WebSocketDisconnect as e:
        logger.info(
            f"WebSocket client disconnected: code={e.code}", session_id=session_id
        )
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(
            f"WebSocket closed unexpectedly: code={e.code}, reason={e.reason}",
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"WebSocket error: {type(e)} {str(e)}", session_id=session_id)
    finally:
        logger.info("Connection closed", session_id=session_id)


@app.get("/health")
async def health_check():
    services_health = {
        "gateway": "healthy",
        "document_service": "unknown",
        "chat_service": "unknown",
        "search_service": "unknown",
    }

    try:
        doc_health = requests.get(f"{DOCUMENT_SERVICE_URL}/health")
        services_health["document_service"] = (
            "healthy" if doc_health.status_code == 200 else "unhealthy"
        )
    except:
        services_health["document_service"] = "unreachable"

    try:
        chat_health = requests.get(f"{CHAT_SERVICE_URL}/health")
        services_health["chat_service"] = (
            "healthy" if chat_health.status_code == 200 else "unhealthy"
        )
    except:
        services_health["chat_service"] = "unreachable"

    try:
        search_health = requests.get(f"{SEARCH_SERVICE_URL}/health")
        services_health["search_service"] = (
            "healthy" if search_health.status_code == 200 else "unhealthy"
        )
    except:
        services_health["search_service"] = "unreachable"

    return services_health
