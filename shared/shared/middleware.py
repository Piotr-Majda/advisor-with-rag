from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from typing import Callable, Awaitable, AsyncGenerator, Any, cast
import json
from .logger import ServiceLogger
import asyncio


def create_logging_middleware(logger: ServiceLogger):
    """Creates a logging middleware that can be used across services"""
    
    async def log_request_body(request: Request) -> dict:
        """Safely read and log request body"""
        body = {}
        if request.headers.get("content-type") == "application/json":
            try:
                body = await request.json()
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON request body")
        return body

    async def wrap_streaming_response(
        response: StreamingResponse,
        request: Request,
        start_time: float
    ) -> AsyncGenerator[bytes, None]:
        """Wrap streaming response and log its completion"""
        try:
            async for chunk in response.body_iterator:
                yield cast(bytes, chunk)
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(
                f"Completed streaming response - {request.url.path}",
                path=request.url.path,
                method=request.method,
                duration=duration,
                status_code=response.status_code,
                headers=dict(response.headers)
            )

    async def logging_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Middleware to log detailed request and response information"""
        start_time = asyncio.get_event_loop().time()

        # Log request details
        request_body = await log_request_body(request)
        query_params = dict(request.query_params)
        path_params = request.path_params
        logger.debug(
            f"Incoming request - {request.url.path}",
            path=request.url.path,
            method=request.method,
            headers=dict(request.headers),
            query_params=query_params,
            path_params=path_params,
            body=request_body,
            client_host=request.client.host if request.client else None,
        )

        # Process request
        response = await call_next(request)
        duration = asyncio.get_event_loop().time() - start_time

        # Handle streaming response
        if isinstance(response, StreamingResponse):
            logger.debug(
                f"Starting streaming response - {request.url.path}",
                path=request.url.path,
                method=request.method,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            return StreamingResponse(
                wrap_streaming_response(response, request, start_time),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

        # Log regular response
        try:
            response_body = None
            if hasattr(response, 'body'):
                try:
                    response_body = json.loads(cast(bytes, response.body).decode())
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    response_body = "<<binary or non-JSON data>>"
        except Exception as e:
            logger.warning(f"Failed to capture response body: {str(e)}")
            response_body = "<<failed to capture>>"

        logger.debug(
            f"Outgoing response - {request.url.path}",
            path=request.url.path,
            method=request.method,
            duration=duration,
            status_code=response.status_code,
            headers=dict(response.headers),
            response_body=response_body
        )

        return response

    return logging_middleware
