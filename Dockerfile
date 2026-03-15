FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Install CPU-only PyTorch separately first
RUN uv pip install --system --no-cache \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match

COPY requirements.txt .

# Install remaining packages from PyPI only
RUN uv pip install --system --no-cache \
    -r requirements.txt \
    --index-strategy unsafe-best-match

COPY src/ ./src/
COPY data/chunks/ ./data/chunks/
COPY data/processed/ ./data/processed/
COPY scripts/startup.sh ./scripts/startup.sh
RUN chmod +x ./scripts/startup.sh

EXPOSE 8000

CMD ["./scripts/startup.sh"]