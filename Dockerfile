FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY scripts/container-entrypoint.sh /usr/local/bin/kairyu-bench-entrypoint
COPY adapters /app/adapters

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/kairyu-bench-entrypoint"]
