FROM docker:27.5.1-cli@sha256:851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c AS docker_cli

FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Security updates intentionally follow the pinned Debian base's current repository.
# hadolint ignore=DL3008
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
COPY --from=docker_cli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose

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
