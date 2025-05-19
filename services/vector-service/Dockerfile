# Base image
FROM python:3.11-slim AS base

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential curl sudo && \
    rm -rf /var/lib/apt/lists/*

# Copy shared package
COPY ./shared/ /app/shared/
RUN pip install -e /app/shared/

# Copy requirements
COPY ./services/vector-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set development environment
ENV PYTHONPATH=/app:/app/shared
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Development stage
FROM base AS development

# Create non-root user and set up permissions
RUN groupadd --gid 1000 vscode && \
    useradd --uid 1000 --gid 1000 -m vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode && \
    mkdir -p /app/vector_store /app/backups /app/logs && \
    chown -R vscode:vscode /app/vector_store /app/backups /app/logs && \
    chmod -R 755 /app/vector_store /app/backups /app/logs

# Install development tools
COPY ./services/vector-service/requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

EXPOSE 8004
EXPOSE 5678

USER vscode

# Default development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]

# Test stage
FROM development AS test

# Copy service code
COPY ./services/vector-service .

# Set PYTHONPATH
ENV PYTHONPATH=/app/shared:${PYTHONPATH}

# Default test command
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-v", "tests/unit"]

# Production stage
FROM base AS production

# Create necessary directories with proper permissions
RUN mkdir -p /app/vector_store /app/backups /app/logs && \
    chmod -R 755 /app/vector_store /app/backups /app/logs

# Copy service code
COPY ./services/vector-service .

# Create necessary directories
RUN mkdir -p /app/vector_store /app/backups

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]

