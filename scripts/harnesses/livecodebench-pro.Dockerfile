# syntax=docker/dockerfile:1

# Mirrors the Dockerfile from LightCPVerifier revision
# 021d121c882e70856b66113142495e31b9fe1d80. The upstream image installs
# npm@latest on Node.js 20, which became incompatible when npm 12 was released.
FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    jq \
    git \
    unzip \
    zip \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    pypy3 \
    openjdk-17-jdk \
    kotlin \
    rustc \
    cargo \
    golang \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# npm 10 supports Node.js 20. Pin the exact version so a future npm release
# cannot make this otherwise pinned verifier image stop building.
RUN npm install -g npm@10.8.2

RUN set -eux; \
  url=$(curl -fsSL https://api.github.com/repos/criyle/go-judge/releases/latest \
    | jq -r '.assets[] | select(.name | test("linux.*amd64.*tar.gz$")) | .browser_download_url' \
    | head -n 1); \
  curl -fsSL "$url" | tar -xz -C /usr/local/bin go-judge; \
  chmod +x /usr/local/bin/go-judge

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --only=production --ignore-scripts

COPY server.js entrypoint.sh ./
COPY src/ ./src/
COPY include/ ./include/
COPY config/ ./config/
COPY include/ /lib/testlib/

RUN chmod +x entrypoint.sh && sed -i 's/\r$//' entrypoint.sh

ENV PORT=8081
ENV GJ_ADDR=http://127.0.0.1:5050
ENV JUDGE_WORKERS=$(nproc)
ENV GJ_PARALLELISM=$(nproc)

ENTRYPOINT ["/app/entrypoint.sh"]
