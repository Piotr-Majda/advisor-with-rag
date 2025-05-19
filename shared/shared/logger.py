"""Centralized logging configuration for microservices."""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any


class CustomJsonFormatter(logging.Formatter):
    """Custom JSON formatter for logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        # Get human readable time
        human_time = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Base log record
        log_record = {
            "time": human_time,
            "level": record.levelname,
            "service": getattr(record, 'service', record.name),
            "message": record.getMessage(),
        }

        # Add any extra fields from record
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        # Add any additional kwargs passed directly
        if hasattr(record, '_extra'):
            log_record.update(record._extra)

        # Add exception info if present
        if record.exc_info:
            log_record["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }

        # Ensure proper JSON formatting with newline
        return json.dumps(log_record, ensure_ascii=False)


class ServiceLogger:
    """Centralized logger for microservices with structured logging support."""

    def __init__(
        self,
        service_name: str,
        log_level: Optional[str] = None,
        env: Optional[str] = None,
    ) -> None:
        """Initialize the service logger."""
        self.service_name = service_name
        self.env = env or os.getenv("APP_ENV", "development")
        
        # Initialize base logger
        self.logger = logging.getLogger(service_name)
        
        # Set log level
        log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Configure handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Configure logging handlers based on environment."""
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = CustomJsonFormatter()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _log(self, level: str, message: str, **kwargs) -> None:
        """Internal logging method that handles extra fields."""
        extra = {
            'service': self.service_name,
            'env': self.env,
            '_extra': kwargs  # Store kwargs separately to avoid conflicts
        }
        getattr(self.logger, level.lower())(message, extra=extra)

    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        self._log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        self._log("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        self._log("WARNING", message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        self._log("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        self._log("CRITICAL", message, **kwargs)
