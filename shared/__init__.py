"""
Shared utility package for the Advisor microservices.
"""

# Re-export from shared.shared
from shared.shared.logger import ServiceLogger
from shared.shared.middleware import create_logging_middleware
from shared.shared.rate_limiter import TokenBucketRateLimiter
from shared.shared.redis_manager import redis_manager

# Make all these importable directly from 'shared'
__all__ = [
    "ServiceLogger",
    "create_logging_middleware",
    "TokenBucketRateLimiter",
    "redis_manager",
]
