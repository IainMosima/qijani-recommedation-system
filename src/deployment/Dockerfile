FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    libssl-dev \
    libffi-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

ENV VENV_PATH="/opt/venv"
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

COPY . /app

RUN pip install -e . && pip install "langgraph-cli[inmem]"

EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0"]