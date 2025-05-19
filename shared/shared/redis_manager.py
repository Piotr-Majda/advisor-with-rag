import os
from redis import Redis
from redis.connection import ConnectionPool


# Singleton pattern
class RedisManager:
    _instance = None
    _initialized = False  # Add an initialization flag

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Correctly call object.__new__ which only takes the class
            cls._instance = super(RedisManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Ensure __init__ only runs once
        if RedisManager._initialized:
            return
        connection_pool = ConnectionPool(
            host=os.getenv("REDIS_HOST", 'redis'),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
        )
        self.redis_client = Redis(connection_pool=connection_pool)
        RedisManager._initialized = True  # Mark as initialized

    def get_client(self) -> Redis:
        if not RedisManager._initialized or not hasattr(self, "redis_client"):
            raise RuntimeError(
                "RedisManager has not been initialized with config and chat_completion yet."
            )
        return self.redis_client


redis_manager = RedisManager()
