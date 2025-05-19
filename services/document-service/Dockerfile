FROM python:3.11-slim AS base

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential curl sudo && \
    rm -rf /var/lib/apt/lists/*

# Copy shared package
COPY ./shared/ /app/shared/
RUN pip install -e /app/shared/

# Copy requirements
COPY ./services/document-service/requirements.txt .
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
    mkdir -p /app/logs /home/vscode/.vscode-server/bin && \
    chown -R vscode:vscode /app/logs /home/vscode/.vscode-server && \
    chmod -R 755 /app/logs /home/vscode/.vscode-server

# Install development tools
COPY ./services/document-service/requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

EXPOSE 8001
EXPOSE 5678

USER vscode

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# Production stage
FROM base AS production

COPY ./services/document-service .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"] 
