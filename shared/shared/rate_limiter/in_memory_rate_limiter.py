# In-memory implementation for testing

import time

from .rate_limiter_base import RateLimiterBase


class InMemoryRateLimiter(RateLimiterBase):
    """In-memory rate limiter for testing purposes"""
    
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        # Store request timestamps per user
        self._requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        
        # Initialize if not exists
        if client_id not in self._requests:
            self._requests[client_id] = []
            
        # Clean old requests
        self._requests[client_id] = [t for t in self._requests[client_id] 
                             if t > now - self.window]
        
        # Check limit
        if len(self._requests[client_id]) >= self.limit:
            return False
            
        # Add this request
        self._requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> tuple[int, float]:
        now = time.time()
        
        if client_id not in self._requests:
            return self.limit, 0.0
            
        # Clean old requests
        self._requests[client_id] = [t for t in self._requests[client_id] 
                             if t > now - self.window]
                             
        return max(0, self.limit - len(self._requests[client_id])), 0.0
    
    def get_reset_time(self, client_id: str) -> int:
        now = time.time()
        
        if client_id not in self._requests or not self._requests[client_id]:
            return 0
            
        oldest = min(self._requests[client_id])
        return max(0, int(self.window - (now - oldest)))
