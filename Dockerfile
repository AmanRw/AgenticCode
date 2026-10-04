# Hardened Sandbox Base Image for AgenticCode
FROM python:3.11-slim

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/home/sandboxuser/.local/bin:$PATH"

# Install minimal OS dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group (UID 1000:GID 1000)
RUN groupadd -g 1000 sandboxgroup && \
    useradd -u 1000 -g sandboxgroup -m -s /bin/bash sandboxuser

# Pre-install common testing, linting, and coverage dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pytest \
    pytest-xdist \
    pytest-timeout \
    pytest-asyncio \
    coverage \
    ruff \
    mypy

# Setup workdir and permissions
WORKDIR /workspace
RUN chown -R sandboxuser:sandboxgroup /workspace /home/sandboxuser

# Switch to unprivileged user
USER 1000:1000

# Default command
CMD ["bash"]
