
# Abstract base class
from abc import ABC, abstractmethod


class RateLimiterBase(ABC):

    @abstractmethod
    def is_allowed(self, client_id: str) -> bool:
        pass

    @abstractmethod
    def get_remaining(self, client_id: str) -> tuple[int, float]:
        pass

    @abstractmethod
    def get_reset_time(self, client_id: str) -> int:
        pass
