"""Start an internal-only tracing gateway from an existing Kairyu container.

Run as the invoking non-root user; the source gateway remains untouched.
Configuration and environment may contain credentials and are never printed.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess


def docker_json(*args: str) -> dict:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)[0]


def configuration(source: dict, name: str, directory: Path, uid: int, gid: int) -> dict:
    if uid == 0:
        raise ValueError("run this helper as a non-root user")
    destinations = {m["Destination"] for m in source["Mounts"]}
    if destinations != {"/etc/kairyu/kairyu.yaml", "/var/lib/kairyu/placement"}:
        raise ValueError("unsupported gateway mounts; inspect and adapt the documented recipe")
    keys = ("AttachStdin", "AttachStdout", "AttachStderr", "ExposedPorts", "Tty",
            "OpenStdin", "StdinOnce", "Cmd", "Healthcheck", "WorkingDir", "Entrypoint",
            "StopSignal", "StopTimeout")
    config = {key: copy.deepcopy(source["Config"][key]) for key in keys if key in source["Config"]}
    config.update(Image=source["Image"], User=f"{uid}:{gid}", Hostname=name,
                  Labels={"kairyu.bench.trace-gateway": name})
    env = dict(item.split("=", 1) for item in source["Config"].get("Env", []) if "=" in item)
    env.update(OTEL_TRACES_EXPORTER="console", XDG_CACHE_HOME=str(directory / "cache"),
               TMPDIR=str(directory / "cache/tmp"))
    config["Env"] = [key + "=" + value for key, value in env.items()]
    # Preserve CPU/memory bounds, never inherit published ports, devices, privilege,
    # original writable mounts, capabilities or Docker sockets.
    host = {key: copy.deepcopy(source["HostConfig"][key]) for key in
            ("CpusetCpus", "CpuShares", "NanoCpus", "Memory", "MemorySwap", "PidsLimit")
            if key in source["HostConfig"]}
    host.update(Binds=[
        f"{directory / 'kairyu.yaml'}:/etc/kairyu/kairyu.yaml:ro",
        f"{directory / 'placement'}:/var/lib/kairyu/placement:rw",
        f"{directory / 'source-package'}:/app/kairyu:ro",
        f"{directory / 'cache'}:{directory / 'cache'}:rw",
    ], RestartPolicy={"Name": "no"}, PortBindings={}, PublishAllPorts=False,
        CapDrop=["ALL"], SecurityOpt=["no-new-privileges:true"])
    config["HostConfig"] = host
    networks = source["NetworkSettings"]["Networks"]
    if not networks or "host" in networks:
        raise ValueError("source must use named bridge networks")
    config["NetworkingConfig"] = {"EndpointsConfig": {
        network: {"Aliases": [name]} for network in networks}}
    return config


def api(method: str, path: str, body: dict | None = None) -> dict:
    import http.client
    import socket
    connection = http.client.HTTPConnection("localhost", timeout=120)
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.connect("/var/run/docker.sock")
    try:
        connection.request(method, path, body=None if body is None else json.dumps(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read()
        if response.status >= 300:
            raise RuntimeError(f"Docker operation failed: HTTP {response.status}")
        return json.loads(data) if data else {}
    finally:
        connection.close()


def main() -> int:
    import urllib.parse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    source = docker_json("inspect", args.source)
    config = configuration(source, args.name, directory, os.getuid(), os.getgid())
    existing = subprocess.run(["docker", "container", "inspect", args.name],
                              capture_output=True)
    if existing.returncode == 0:
        parser.error("destination container already exists")
    if directory.exists() and any(directory.iterdir()):
        parser.error("directory must be new or empty")
    if not args.apply:
        print(json.dumps({"operation": "create internal trace gateway", "apply": False,
                          "name": args.name, "image": source["Image"], "uid": os.getuid()}))
        return 0
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    for part in ("placement", "cache/tmp"):
        (directory / part).mkdir(parents=True, exist_ok=True)
    transform = ("import yaml,pathlib; "
                 "c=yaml.safe_load(pathlib.Path('/etc/kairyu/kairyu.yaml').read_text()); "
                 "c['server']['tracing']=True; print(yaml.safe_dump(c,sort_keys=False))")
    result = subprocess.run(["docker", "exec", args.source, "python", "-c", transform],
                            text=True, capture_output=True, check=True)
    fd = os.open(directory / "kairyu.yaml", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(result.stdout)
    # Copy the exact existing package solely to make the same image runnable as
    # this UID. No source transformations or framework fork.
    subprocess.run(["docker", "cp", args.source + ":/app/kairyu",
                    str(directory / "source-package")], check=True, capture_output=True)
    created = api("POST", "/containers/create?name=" + urllib.parse.quote(args.name), config)
    try:
        api("POST", "/containers/" + created["Id"] + "/start")
    except Exception:
        api("DELETE", "/containers/" + created["Id"] + "?force=true")
        raise
    receipt = {"name": args.name, "container": created["Id"], "image": source["Image"],
               "source_container": source["Id"], "uid": os.getuid(),
               "published_ports": False, "source_unchanged": True}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
