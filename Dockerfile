FROM docker:27.5.1-cli AS docker_cli

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    build-essential \
    ca-certificates \
    curl \
    git \
    libhdf5-dev \
    procps \
    ripgrep \
    socat \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts /app/scripts
COPY adapters /app/adapters
RUN chmod +x /app/scripts/container-entrypoint.sh \
    /app/scripts/lib/official.sh \
    /app/scripts/harnesses/*.sh \
    /app/adapters/*/run.sh \
    && ln -s /app/scripts/container-entrypoint.sh /usr/local/bin/kairyu-bench-entrypoint

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/kairyu-bench-entrypoint"]
