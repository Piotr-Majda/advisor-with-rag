import os
from dataclasses import Field
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from redis import Redis
from redis.connection import ConnectionPool

from shared import ServiceLogger, TokenBucketRateLimiter, redis_manager

load_dotenv()


def get_redis_client():
    return redis_manager.get_client()


def get_limiter(logger: ServiceLogger, redis_client: Redis = Depends(get_redis_client)):
    try:
        limit = int(os.getenv("RATE_LIMIT_API_GATEWAY", 10))
        window = int(os.getenv("WINDOW_API_GATEWAY", 60))
        if limit <= 0 or window <= 0:
            logger.warning(
                f"Invalid rate limit config: limit={limit}, window={window}. Using defaults."
            )
            limit = max(limit, 10)
            window = max(window, 60)
        return TokenBucketRateLimiter(limit, window, redis_client)
    except ValueError as e:
        logger.error(f"Error parsing rate limit config: {e}")
        return TokenBucketRateLimiter(10, 60, redis_client)  # fallback to defaults


def create_rate_limit_middleware(logger: ServiceLogger):
    # Retrieve the Redis client dependency correctly
    redis_client = get_redis_client()
    limiter = get_limiter(logger, redis_client=redis_client)

    async def rate_limit(request: Request, call_next):
        # Exclude paths that shouldn't be rate limited by this HTTP middleware
        # Add the WebSocket path to exclusions
        if request.url.path in ["/reset", "/health", "/chat"]:
            logger.debug(f"Skipping rate limit for path: {request.url.path}")
            return await call_next(request)

        # Get client identifier
        # For regular HTTP, prefer X-Client-ID, fallback to host
        client_id = request.headers.get("X-Client-ID") or getattr(
            request.client, "host", None
        )
        if not client_id:
            logger.warning(
                "Could not identify client for rate limiting (missing X-Client-ID or host info)."
            )
            # Return 401 for regular HTTP if client cannot be identified
            return JSONResponse(
                status_code=401, content={"detail": "Client identifier missing"}
            )

        logger.debug(
            f"Rate limiting request from client: {client_id} for path: {request.url.path}"
        )

        # Check rate limit
        if not limiter.is_allowed(client_id):
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return JSONResponse(
                status_code=429, content={"detail": "Too Many Requests"}
            )

        # Process request and add rate limit headers
        response = await call_next(request)
        # Check if response is not None and has headers attribute before adding headers
        # (StreamingResponse might not have headers initially)
        if response and hasattr(response, "headers"):
            try:
                tokens, last_refill = limiter.get_remaining(client_id)
                response.headers["X-RateLimit-Limit"] = str(limiter.limit)
                response.headers["X-RateLimit-Remaining"] = str(
                    int(tokens)
                )  # Ensure tokens is int
                response.headers["X-RateLimit-Reset"] = str(
                    int(last_refill + limiter.window)
                )
            except Exception as e:
                logger.error(f"Error getting remaining tokens or setting headers: {e}")
        return response

    return rate_limit
