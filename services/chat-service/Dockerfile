FROM python:3.13-slim AS base

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential curl sudo && \
    rm -rf /var/lib/apt/lists/*

# Copy shared package
COPY ./shared/ /app/shared/
# Use a non-editable install for the base image build
RUN pip install /app/shared/

# Copy requirements
COPY ./services/chat-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set development environment
ENV PYTHONPATH=/app:/app/shared
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Development stage
FROM base AS development

# Create non-root user and set up permissions
RUN mkdir -p /etc/sudoers.d && \
    groupadd --gid 1000 vscode && \
    useradd --uid 1000 --gid 1000 -m vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode && \
    mkdir -p /app/logs && \
    chown -R vscode:vscode /app/logs && \
    chmod -R 755 /app/logs && \
    # Grant ownership of /app to the user for cache/etc
    chown -R vscode:vscode /app

# Install development tools
COPY ./services/chat-service/requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

EXPOSE 8003
EXPOSE 5678

USER vscode

# Default development command with hot reload
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]

# Test stage
FROM development AS test

# Copy service code and tests
COPY ./services/chat-service .

# Run tests with coverage
CMD ["pytest", "-vvs", "tests/unit/"]


# Production stage
FROM base AS production

# Copy service code
COPY ./services/chat-service .

EXPOSE 8003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
