FROM python:3.11-slim AS base

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy shared package
COPY ./shared/ /app/shared/
RUN pip install -e /app/shared/

# Copy requirements and install dependencies
COPY ./api-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY ./api-gateway/ .

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
    chmod -R 755 /app/logs

EXPOSE 8000
EXPOSE 5678

USER vscode

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS production

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 
