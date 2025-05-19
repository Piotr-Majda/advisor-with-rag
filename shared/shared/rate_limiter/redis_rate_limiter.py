import time
from typing import cast
from redis import Redis

from .rate_limiter_base import RateLimiterBase


class TokenBucketRateLimiter(RateLimiterBase):
    """
    Rate limiter using token bucket algorithm
    """

    def __init__(self, limit: int, window: int, redis_client: Redis):
        self.limit = limit
        self.window = window
        self.refill_rate = limit / window
        self.redis_client = redis_client

    def is_allowed(self, client_id: str) -> bool:
        tokens, last_refill = self.get_remaining(client_id)
        current_time = time.time()
        elapsed_time = current_time - last_refill
        tokens_to_add = min(self.limit, int(elapsed_time * self.refill_rate))
        if tokens_to_add >= 1:  # refill bucket
            print(f"Refilling bucket for {client_id} with {tokens_to_add} tokens")
        potential_tokens = min(self.limit, tokens + tokens_to_add)
        if potential_tokens < 1:  # check if bucket is empty
            print(f"Bucket for {client_id} is empty")
            self.set_token(client_id, potential_tokens, current_time)
            return False
        # consume token
        print(f"Consuming token for {client_id}")
        final_tokens = potential_tokens - 1
        self.set_token(client_id, final_tokens, current_time)
        return True

    def set_token(self, client_id: str, tokens: int, timestamp: float):
        self.redis_client.setex(client_id, self.window, f"{tokens},{timestamp}")
        return True

    def get_remaining(self, client_id: str) -> tuple[int, float]:
        tokens_data = self.redis_client.get(client_id)
        if tokens_data is None:
            self.reset_client(client_id)
            return self.limit, 0.0
        tokens_data_str = cast(bytes, tokens_data).decode("utf-8")
        tokens, last_refill = tokens_data_str.split(",")
        print(f"Tokens: {tokens}, Last Refill: {last_refill}")
        return int(tokens), float(last_refill)

    def get_reset_time(self, client_id: str) -> int:
        tokens, last_refill = self.get_remaining(client_id)
        return int(self.window - (time.time() - last_refill))

    def reset_client(self, client_id: str):
        print(f"Resetting bucket for {client_id}")
        current_time = time.time()
        self.redis_client.setex(client_id, self.window, f"{self.limit},{current_time}")
