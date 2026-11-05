# Deploying optional live observation

Run deployment commands as your normal host user. Live collectors require the
documented vLLM worker and middleware extension points; no framework patch is
needed. Use an immutable source snapshot and a fresh vLLM compilation cache for
each observer implementation/site selection. Existing cached graphs may omit hooks.
Stop benchmark traffic before changing a replica.

## Tracing through Kairyu

Kairyu must enable its existing `server.tracing` option and an OpenTelemetry
exporter so the benchmark trace reaches the model replica. If the production
configuration must remain untouched, the optional recipe
`scripts/observe-trace-gateway.py` creates an internal-only gateway with the same
image, command and configuration, enables tracing, and uses separate writable
directories. It runs as the invoking non-root UID and exposes no host ports.
It supports the standard configuration/placement mounts and rejects other layouts.

```sh
python3 scripts/observe-trace-gateway.py --source serving-kairyu-1 \
  --name wsp-observe-gateway --directory /mnt/nvme/experiment/trace-gateway
# Inspect the dry-run summary, then repeat with --apply.
```

Use `http://wsp-observe-gateway:8000/v1` from the benchmark runner on the same
Docker network. Environment and configuration values are kept private; the
receipt contains only image/container identifiers. The source gateway is never
recreated. The exact package in the source container is copied read-only so its
permissions permit the non-root UID; its contents are unchanged.

After requests finish and worker queues drain, remove only the temporary gateway
named in its receipt with `docker rm -f <name>`. Keep the capture and receipt.
A console exporter writes traces to the temporary gateway's Docker log; treat
those logs as experiment data and configure ordinary Docker retention as needed.
