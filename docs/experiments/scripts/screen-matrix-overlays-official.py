#!/usr/bin/env python3
"""Bounded exploratory official code90: one empty-parent overlay plus up to six matrix overlays.
Default preview is inert. No checkpoint writes or adoption. Physical finalists require ordinary confirmation.
Source relationship: v3 overlay owner 9158b092 + prospective DP2 lifecycle cdac49af.
"""

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class OwnerCheckError(ValueError):
    pass


def need(value, message):
    if not value:
        raise OwnerCheckError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def canonical(value):
    value = dict(value)
    value["mounts"] = sorted(
        value.get("mounts", []), key=lambda x: json.dumps(x, sort_keys=True)
    )
    return value


def command(argv, timeout=120):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    need(result.returncode == 0, "owned metadata/container command failed")
    return result.stdout.strip()


def pinned(ref):
    need(sha(ref["path"]) == ref["sha256"], "pinned input changed")
    return read(ref["path"])


def saved_identity(ref, path, tree):
    freeze = pinned(ref)
    need(
        freeze["checkpoint_path"] == path and freeze["control_tree_sha256"] == tree,
        "saved checkpoint identity differs",
    )
    from kairyu_bench.wsp.weights.checkpoint import Checkpoint

    rows = Checkpoint(path).identity(hash_files=False)["files"]
    need(
        [(r["path"], r["size"]) for r in rows]
        == [(r["path"], r["size"]) for r in freeze["all_checkpoint_files"]],
        "saved inventory differs",
    )
    for row in rows:
        old = freeze["candidate_inventory_stat"][row["path"]]
        need(
            row["size"] == old["size"]
            and all(old[k] == v for k, v in row["stat"].items()),
            "saved file stat differs",
        )
    return freeze


def snapshot(service):
    return guard.snapshot(guard.inspect(plan["serving_baseline"][service]["id"]))


def inside(path, root):
    path = Path(path)
    return path.is_absolute() and path.resolve() == path and path.is_relative_to(root)


def overlay_metadata(arm):
    op = pinned(arm["overlay_plan"])
    parent = plan["parent"]
    draft = plan["draft"]
    need(
        op["kind"] == "kairyu_sparse_bf16_v1"
        and op["schema_version"] == 1
        and op["tensor_parallel_size"] == 2,
        "overlay format/TP differs",
    )
    need(
        op["physical_source"]
        == {
            "path": parent["checkpoint"],
            "tree_sha256": parent["tree"],
            "proof": parent["freeze"],
        },
        "overlay physical parent differs",
    )
    need(
        op["logical_parent"]
        == {"path": parent["checkpoint"], "tree_sha256": parent["tree"]},
        "overlay chaining is unsupported",
    )
    need(
        op["draft_bypass"]
        == {
            "path": draft["checkpoint"],
            "tree_sha256": draft["tree"],
            "proof": draft["freeze"],
        },
        "pinned draft differs",
    )
    need(
        op["implementation_sha256"] == plan["overlay_implementation_sha256"]
        and op["vllm_source_sha256"] == plan["vllm_source_sha256"],
        "installed overlay source differs",
    )
    spec = pinned(op["spec"])
    need(
        spec.get("schema_version", 1) == 1 and spec.get("kind") is None,
        "unsupported sparse spec",
    )
    need(
        spec["source_checkpoint"] == parent["checkpoint"]
        and spec["base_tree_sha256"] == parent["tree"],
        "spec not current-parent increment",
    )
    pattern = re.compile(
        r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|o)_proj|mlp\.(?:gate|up|down)_proj)\.weight"
    )
    total = 0
    for edit in spec["edits"]:
        need(
            set(edit) == {"tensor", "coordinates", "set_values"}
            and pattern.fullmatch(edit["tensor"]),
            "only Q/K/O/MLP matrix edits supported",
        )
        need(
            len(edit["coordinates"]) == len(edit["set_values"]),
            "coordinate/value mismatch",
        )
        total += len(edit["coordinates"])
    need(
        total == arm["selected_coordinates"]
        and ((total == 0) == (arm["label"] == "control")),
        "selected count/control differs",
    )
    logical = hashlib.sha256(
        json.dumps(
            {
                "kind": op["kind"],
                "parent": op["logical_parent"],
                "spec_sha256": op["spec"]["sha256"],
                "implementation_sha256": op["implementation_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    need(logical == arm["logical_overlay_sha256"], "logical overlay identity differs")
    need(
        inside(op["receipt_directory"], P / "load-receipts"),
        "load receipt path outside owner",
    )
    return op


def validate(serving):
    need(
        sha(args.plan) == args.plan_sha
        and sha(args.authorization) == args.authorization_sha,
        "owner binding changed",
    )
    auth = read(args.authorization)
    need(
        auth["status"] == "authorized"
        and auth["issued_by"] == "root"
        and auth["plan_sha256"] == args.plan_sha
        and auth["runtime_sha256"] == sha(__file__),
        "exact root execution authorization required",
    )
    need(
        plan["binding_complete"] and 1 <= len(plan["candidates"]) <= 6,
        "bound candidate roster required",
    )
    need(
        plan["image_id"]
        == "sha256:c1b5c0a94b7e77e0672346d7c191fb143c3dbf88e6b28c673f4d2b754d288ba8",
        "validated overlay image required",
    )
    need(
        plan["topology"] == {"tp": 2, "dp": 1, "pp": 1, "static_mtp": 4, "slots": 4},
        "reviewed four TP2 slots required",
    )
    need(
        [slot["gpus"] for slot in plan["slots"]] == [[0, 1], [2, 3], [4, 5], [6, 7]],
        "four disjoint TP2 GPU slots required",
    )
    need(
        set(plan["serving_baseline"]) == {"gemma-0", "gemma-1", "gemma-2", "gemma-3"},
        "exact four old services required",
    )
    for path, digest in plan["pinned_inputs"].items():
        need(sha(path) == digest, "source/input pin changed")
    state_file = pinned(plan["incumbent"])
    parent = plan["parent"]
    draft = plan["draft"]
    need(
        all(
            not P.is_relative_to(Path(x["checkpoint"]))
            and not Path(x["checkpoint"]).is_relative_to(P)
            for x in (parent, draft)
        ),
        "owner output overlaps saved source",
    )
    need(
        state_file["checkpoint_path"] == parent["checkpoint"]
        and state_file["control_tree_sha256"] == parent["tree"],
        "parent is not current incumbent",
    )
    saved_identity(parent["freeze"], parent["checkpoint"], parent["tree"])
    saved_identity(draft["freeze"], draft["checkpoint"], draft["tree"])
    closure = pinned(plan["finite_completion"])
    need(closure["status"] == "completed", "finite owner not closed")
    # Exact selection/closure bytes are root-bound; no additional statistical gate.
    pinned(plan["selection_decision"])
    official_validate()
    arms = [plan["control"], *plan["candidates"]]
    need(
        plan["control"]["label"] == "control"
        and len({a["label"] for a in arms}) == len(arms),
        "unique arm labels required",
    )
    for arm in arms:
        need(
            re.fullmatch("[a-z][a-z0-9-]{0,63}", arm["label"])
            and re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", arm["container_name"]),
            "safe arm/container label required",
        )
        overlay_metadata(arm)
    need(
        len({a["container_name"] for a in arms}) == len(arms),
        "duplicate temporary names",
    )
    need(
        len({read(a["overlay_plan"]["path"])["receipt_directory"] for a in arms})
        == len(arms),
        "duplicate load receipts",
    )
    needed = {parent["freeze"]["path"], draft["freeze"]["path"]}
    for arm in arms:
        op = read(arm["overlay_plan"]["path"])
        needed.add(op["spec"]["path"])
    # Read-only external files bind canonical host paths; no /model alias projection.
    for path in needed:
        need(
            inside(path, P) or path in plan["external_readonly_files"],
            "external proof/spec mount missing",
        )
    for path in plan["external_readonly_files"]:
        need(
            Path(path).is_file()
            and Path(path).resolve() == Path(path)
            and path in plan["pinned_inputs"],
            "external mount not pinned regular file",
        )
    for service, want in plan["serving_baseline"].items():
        actual = snapshot(service)
        config = guard.inspect(want["id"])
        fixed = plan["original_configuration"][service]
        need(
            guard.digest(config["Config"]) == fixed["config_sha256"]
            and guard.digest(config["HostConfig"]) == fixed["host_config_sha256"],
            "original full configuration changed",
        )
        if serving:
            need(canonical(actual) == canonical(want), "original service changed")
        else:
            need(not actual["running"], "owned original unexpectedly running")
            need(
                canonical(
                    {k: v for k, v in actual.items() if k not in ("running", "health")}
                )
                == canonical(
                    {k: v for k, v in want.items() if k not in ("running", "health")}
                ),
                "stopped original identity changed",
            )
    for old in plan["embedding_baseline"]:
        item = guard.inspect(old["id"])
        need(
            item["State"]["Running"] == old["running"]
            and item["State"]["StartedAt"] == old["started_at"]
            and guard.digest(item["Config"]) == old["config_sha256"]
            and guard.digest(item["HostConfig"]) == old["host_config_sha256"],
            "embedding changed",
        )


def publish():
    if owned:
        save(P / "execution-status.json", state)


def endpoint(cid):
    item = guard.inspect(cid)
    ips = [
        v["IPAddress"]
        for v in item["NetworkSettings"]["Networks"].values()
        if v.get("IPAddress")
    ]
    need(len(ips) == 1, "single temporary endpoint required")
    return "http://" + ips[0] + ":8000/v1"


def wait_ready(record, deadline):
    while time.monotonic() < deadline:
        need(
            guard.inspect(record["container_id"])["State"]["Running"],
            "temporary server exited",
        )
        ep = endpoint(record["container_id"])
        try:
            with urllib.request.urlopen(
                ep.removesuffix("/v1") + "/health", timeout=5
            ) as response:
                if response.status == 200:
                    return ep
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("temporary readiness deadline")


def counters(ep, cid, path, idle=True):
    command(
        [
            plan["python"],
            plan["tools"]["counters"]["path"],
            "snapshot",
            "--endpoint",
            ep.removesuffix("/v1") + "/metrics",
            "--model",
            "gemma-4-31b",
            "--container-id",
            cid,
            "--attestation-sha256",
            args.plan_sha,
            "--output",
            str(path),
            "--capture",
        ],
        60,
    )
    result = read(path)
    if idle:
        for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            values = [
                float(row["value"])
                for row in result["series"]
                if row["metric"] == metric
            ]
            need(values and all(v == 0 for v in values), "server not idle")
    return {"path": str(path), "sha256": sha(path)}


def proofs(arm):
    op = overlay_metadata(arm)
    directory = Path(op["receipt_directory"])
    ref = arm["overlay_plan"]
    rows = [read(directory / f"target-rank-{rank}.json") for rank in range(2)]
    for row in rows:
        need(
            row["status"] == "verified_selected_words_all_tp_ranks"
            and row["storage_bytes_exact"]
            and row["plan"] == ref
            and row["logical_overlay_sha256"] == arm["logical_overlay_sha256"]
            and row["selected_coordinates"] == arm["selected_coordinates"],
            "all-rank selected words differ",
        )
    need(rows[0] == rows[1], "rank aggregate receipts differ")
    for rank in range(2):
        bypass = read(directory / f"draft_bypass-rank-{rank}.json")
        need(
            bypass["status"] == "pinned_draft_bypassed"
            and bypass["draft_tensor_edits"] == 0
            and bypass["plan"] == ref
            and bypass["draft_tree_sha256"] == plan["draft"]["tree"],
            "MTP bypass differs",
        )
    return {
        "logical_overlay_sha256": arm["logical_overlay_sha256"],
        "physical_source_tree": plan["parent"]["tree"],
        "selected_coordinates": arm["selected_coordinates"],
        "all_rank_exact": True,
        "empty_is_not_full_gpu_hash": arm["selected_coordinates"] == 0,
        "receipt_sha256": {p.name: sha(p) for p in sorted(directory.glob("*.json"))},
    }


def create(arm, slot, base):
    rec = {
        "label": arm["label"],
        "name": arm["container_name"],
        "slot": slot["id"],
        "status": "creating",
    }
    temporary.append(rec)
    publish()
    cmd = list(base["Config"]["Cmd"])
    need(cmd[0] == "/checkpoint", "server template model argument")
    cmd[0] = plan["parent"]["checkpoint"]
    for flag in (
        "--load-format",
        "--model-loader-extra-config",
        "--safetensors-load-strategy",
        "--data-parallel-size",
        "--data-parallel-rank",
    ):
        need(flag not in cmd, "unexpected loader/DP template override")
    need(cmd[cmd.index("--tensor-parallel-size") + 1] == "2", "template must be TP2")
    index = cmd.index("--speculative-config") + 1
    draft = json.loads(cmd[index])
    need(
        draft["num_speculative_tokens"] == 4
        and "num_speculative_tokens_per_batch_size" not in draft,
        "static MTP4 required",
    )
    draft["model"] = plan["draft"]["checkpoint"]
    cmd[index] = json.dumps(draft, separators=(",", ":"))
    ref = arm["overlay_plan"]
    cmd += [
        "--load-format",
        "kairyu_sparse_bf16_v1",
        "--model-loader-extra-config",
        json.dumps(
            {"overlay_plan": ref["path"], "overlay_plan_sha256": ref["sha256"]},
            separators=(",", ":"),
        ),
    ]
    cache = P / "cache" / arm["label"]
    cache.mkdir(parents=True)
    env = guard.environments(base["Config"]["Env"])
    env["PYTHONPATH"] = "/opt/kairyu-sparse-overlay" + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.update(
        XDG_CACHE_HOME=str(cache),
        XDG_CONFIG_HOME=str(cache / "config"),
        VLLM_CONFIG_ROOT=str(cache / "config/vllm"),
        HF_HOME=str(cache / "hf"),
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torch"),
        TRITON_CACHE_DIR=str(cache / "triton"),
        FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer-base"),
    )
    if env.get("VLLM_PLUGINS"):
        env["VLLM_PLUGINS"] = ",".join(
            dict.fromkeys([*env["VLLM_PLUGINS"].split(","), "kairyu_sparse_overlay"])
        )
    need(
        not any("\n" in k + v or "\r" in k + v for k, v in env.items()),
        "multiline environment unsupported",
    )
    ef = P / (".env-" + arm["label"])
    fd = os.open(ef, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("\n".join(k + "=" + v for k, v in env.items()) + "\n")
    argv = [
        "docker",
        "create",
        "--name",
        arm["container_name"],
        "--label",
        "kairyu.overlay.screen.owner=" + plan["owner_label"],
        "--runtime",
        "nvidia",
        "--gpus",
        '"device=' + ",".join(map(str, slot["gpus"])) + '"',
        "--user",
        "1003:1003",
        "--network",
        base["HostConfig"]["NetworkMode"],
        "--ipc",
        base["HostConfig"]["IpcMode"],
        "--cpuset-cpus",
        slot["cpuset"],
        "--shm-size",
        str(base["HostConfig"]["ShmSize"]),
        "--env-file",
        str(ef),
        "--no-healthcheck",
    ]
    if base["Config"].get("WorkingDir"):
        argv += ["--workdir", base["Config"]["WorkingDir"]]
    models = next(m for m in base["Mounts"] if m["Destination"] == "/models")
    need(not models["RW"], "models mount must be readonly")
    mounts = [
        (models["Source"], "/models", True),
        (plan["parent"]["checkpoint"], plan["parent"]["checkpoint"], True),
        (plan["draft"]["checkpoint"], plan["draft"]["checkpoint"], True),
        (str(P), str(P), True),
        (str(cache), str(cache), False),
    ]
    receipt = read(arm["overlay_plan"]["path"])["receipt_directory"]
    mounts.append((receipt, receipt, False))
    mounts += [(p, p, True) for p in plan["external_readonly_files"]]
    for source, target, readonly in mounts:
        argv += [
            "--mount",
            "type=bind,src="
            + source
            + ",dst="
            + target
            + (",readonly" if readonly else ""),
        ]
    try:
        rec["container_id"] = command([*argv, plan["image_id"], *cmd], 120)
    finally:
        ef.unlink(missing_ok=True)
    item = guard.inspect(rec["container_id"])
    need(
        item["Image"] == plan["image_id"]
        and item["Config"]["User"] == "1003:1003"
        and item["Config"]["Cmd"] == cmd
        and item["Config"]["Entrypoint"] == base["Config"]["Entrypoint"],
        "temporary image/command/user differs",
    )
    devices = item["HostConfig"].get("DeviceRequests") or []
    need(
        len(devices) == 1
        and set(devices[0].get("DeviceIDs") or []) == set(map(str, slot["gpus"])),
        "temporary GPU scope differs",
    )
    rec.update(
        status="loading",
        image_id=item["Image"],
        command_sha256=guard.digest(cmd),
        environment_sha256=guard.digest(env),
    )
    publish()
    return rec


def cleanup(record):
    if record.get("removed"):
        return
    query = subprocess.run(
        ["docker", "inspect", record.get("container_id", record["name"])],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if query.returncode == 0:
        item = json.loads(query.stdout)[0]
        need(
            item["Image"] == plan["image_id"]
            and (item["Config"].get("Labels") or {}).get("kairyu.overlay.screen.owner")
            == plan["owner_label"],
            "refuse foreign temporary cleanup",
        )
        command(["docker", "rm", "--force", item["Id"]], 120)
    record["removed"] = True


def startup_diagnostic(record):
    if record.get("status") != "loading" or not record.get("container_id"):
        return
    item = guard.inspect(record["container_id"])
    need(
        item["Image"] == plan["image_id"]
        and (item["Config"].get("Labels") or {}).get("kairyu.overlay.screen.owner")
        == plan["owner_label"],
        "diagnostic ownership differs",
    )
    result = subprocess.run(
        ["docker", "logs", "--tail", "2000", item["Id"]],
        capture_output=True,
        timeout=20,
    )
    raw = (result.stdout + b"\n" + result.stderr)[-1048576:]
    directory = P / "private-startup"
    directory.mkdir(exist_ok=True, mode=0o700)
    path = directory / (record["label"] + ".log")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
    text = raw.decode(errors="replace")
    classes = [
        "PermissionError",
        "FileNotFoundError",
        "ModuleNotFoundError",
        "ImportError",
        "ValueError",
        "RuntimeError",
        "TypeError",
        "AssertionError",
        "KeyError",
        "OSError",
        "OutOfMemoryError",
    ]
    record["startup_diagnostic"] = {
        "exit_code": item["State"].get("ExitCode"),
        "oom_killed": item["State"].get("OOMKilled"),
        "private_log_sha256": sha(path),
        "bytes": len(raw),
        "error_classes": [
            c for c in classes if re.search(r"(?<![A-Za-z])" + c + r"(?=[:\s])", text)
        ],
        "permission_denied": "Permission denied" in text,
        "no_module_named": "No module named" in text,
        "cuda_out_of_memory": "CUDA out of memory" in text,
        "raw_log_not_published": True,
    }


def launch(argv, log):
    with Path(log).open("xb") as stream:
        child = subprocess.Popen(
            argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
    children.append(child)
    return child


def wait_children(group, deadline):
    while any(child.poll() is None for child in group):
        need(time.monotonic() < deadline, "declared child time budget exhausted")
        failed = [child for child in group if child.poll() not in (None, 0)]
        need(not failed, "official runner process failed")
        time.sleep(1)
    need(
        all(child.returncode == 0 for child in group), "official runner process failed"
    )


def official_validate(check_fresh=False):
    official = plan["official"]
    need(
        official["source_head"] == "5e3f2df0c7f455013af3fa2d7abb3719858f808f",
        "immutable official source required",
    )
    need(
        official["protocol"]
        == {
            "tier": 30,
            "split": "dev",
            "sources": ["humaneval-plus", "mbpp-plus", "livecodebench", "scicode"],
            "chat_concurrency": 32,
            "logprob_concurrency": 8,
            "agent_concurrency": 1,
        },
        "official protocol differs",
    )
    need(
        len(official["archived_source_files_sha256"]) == 76
        and len(official["installed_source_files_sha256"]) == 82,
        "official source inventory missing",
    )
    source = Path(official["source_path"])
    for rel, digest in official["archived_source_files_sha256"].items():
        need(sha(source / rel) == digest, "official source changed")
    need(
        source.is_absolute()
        and source.resolve() == source
        and (source / "kairyu-bench").is_file(),
        "official source path invalid",
    )
    results = Path(official["results_dir"])
    need(
        results.is_absolute() and results.is_dir() and results.resolve() == results,
        "results directory invalid",
    )
    for item in (plan["parent"], plan["draft"]):
        checkpoint = Path(item["checkpoint"])
        need(
            not results.is_relative_to(checkpoint)
            and not checkpoint.is_relative_to(results),
            "official results overlap a checkpoint",
        )
    need(
        re.fullmatch("[a-z0-9][a-z0-9-]{0,90}", official["run_prefix"]),
        "safe fresh run prefix required",
    )
    need(
        0 < plan["official_seconds"] <= 7200,
        "bounded per-round official deadline required",
    )
    if not check_fresh:
        return
    for number, labels in enumerate(
        [plan["candidates"][i : i + 3] for i in range(0, len(plan["candidates"]), 3)], 1
    ):
        for arm in [plan["control"], *labels]:
            run_id = official_id(number, arm["label"])
            need(not (results / run_id).exists(), "official run ID already exists")


def official_id(number, label):
    return plan["official"]["run_prefix"] + "-r" + str(number) + "-" + label


def official_command(ep, arm, run_id, observer):
    official = plan["official"]
    protocol = official["protocol"]
    return [
        str(Path(official["source_path"]) / "kairyu-bench"),
        "wsp",
        "run",
        ep,
        "--results-dir",
        official["results_dir"],
        "--tier",
        str(protocol["tier"]),
        "--split",
        protocol["split"],
        "--only",
        ",".join(protocol["sources"]),
        "--label",
        arm["logical_overlay_sha256"],
        "--run-id",
        run_id,
        "--chat-concurrency",
        "32",
        "--logprob-concurrency",
        "8",
        "--agent-concurrency",
        "1",
        "--logprob-endpoint",
        ep,
        "--observe-config",
        str(observer),
    ]


def discover_owned_official(row, environment):
    """Recover uncaptured owned runners using only exact command/run-ID metadata."""
    expected = row["command"][1:]
    run_id = row["run_id"]
    need(
        expected.count("--run-id") == 1
        and expected[expected.index("--run-id") + 1] == run_id,
        "owned runner lacks exact unique run ID",
    )
    deadline = time.monotonic() + 20
    cp = subprocess.run(
        ["docker", "ps", "--all", "--no-trunc", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
        timeout=max(0.01, deadline - time.monotonic()),
        env=environment,
    )
    need(cp.returncode == 0, "official cleanup discovery failed")
    ids = cp.stdout.split()
    if not ids:
        return []
    cp = subprocess.run(
        ["docker", "inspect", *ids],
        capture_output=True,
        text=True,
        timeout=max(0.01, deadline - time.monotonic()),
        env=environment,
    )
    need(cp.returncode == 0, "official cleanup metadata unavailable")
    return [
        item["Id"]
        for item in json.loads(cp.stdout)
        if item["Id"] in ids and item["Config"].get("Cmd") == expected
    ]


def require_results_mount(container):
    """Require real host persistence, not merely identical mounts across arms."""
    requested = Path(plan["official"]["results_dir"])
    for mount in container["Mounts"]:
        destination = Path(mount["Destination"])
        if (
            mount["Type"] == "bind"
            and mount["RW"]
            and requested.is_relative_to(destination)
            and Path(mount["Source"]) / requested.relative_to(destination) == requested
        ):
            return
    raise OwnerCheckError("official result directory lacks its writable host bind")


def capture_official(rows, deadline):
    pending = {row["run_id"]: row for row in rows}
    image_parity = []
    environment = {**os.environ, **plan["official"].get("environment", {})}
    while pending and time.monotonic() < deadline:
        ids = subprocess.run(
            ["docker", "ps", "--no-trunc", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
            timeout=20,
            env=environment,
        )
        need(ids.returncode == 0, "official runner inventory failed")
        for cid in ids.stdout.split():
            cp = subprocess.run(
                ["docker", "inspect", cid],
                capture_output=True,
                text=True,
                timeout=20,
                env=environment,
            )
            if cp.returncode:
                continue
            item = json.loads(cp.stdout)[0]
            cmd = item["Config"].get("Cmd") or []
            matches = [key for key in pending if key in cmd]
            if not matches:
                continue
            need(len(matches) == 1, "ambiguous official runner")
            row = pending.pop(matches[0])
            need(cmd == row["command"][1:], "official runner command differs")
            row["runner_container_id"] = item["Id"]
            publish()
            require_results_mount(item)
            im = subprocess.run(
                ["docker", "image", "inspect", item["Image"]],
                capture_output=True,
                text=True,
                timeout=20,
                env=environment,
            )
            need(im.returncode == 0, "official image metadata unavailable")
            image = json.loads(im.stdout)[0]
            env = item["Config"].get("Env") or []
            keys = [v.split("=", 1)[0] for v in env]
            safe = [
                v
                for v in env
                if not any(
                    k in v.split("=", 1)[0].upper()
                    for k in (
                        "TOKEN",
                        "SECRET",
                        "KEY",
                        "PASS",
                        "AUTH",
                        "CREDENTIAL",
                        "COOKIE",
                    )
                )
            ]
            need(len(keys) == len(set(keys)), "duplicate official environment keys")
            replacements = {
                row["endpoint"]: "__ENDPOINT__",
                row["logical_overlay_sha256"]: "__LOGICAL_OVERLAY__",
                row["run_id"]: "__RUN_ID__",
                row["observer"]["path"]: "__OBSERVER_CONFIG__",
            }
            audit = {
                "run_id": row["run_id"],
                "container_id": item["Id"],
                "image_id": item["Image"],
                "rootfs": image["RootFS"],
                "image_config_sha256": guard.digest(image["Config"]),
                "host_config_sha256": guard.digest(item["HostConfig"]),
                "command": cmd,
                "normalized_command": [replacements.get(x, x) for x in cmd],
                "noncredential_environment_sha256": guard.digest(sorted(safe)),
                "duplicate_environment_keys": False,
                "user": item["Config"].get("User"),
                "working_dir": item["Config"].get("WorkingDir"),
                "runtime": item["HostConfig"].get("Runtime"),
                "network_mode": item["HostConfig"].get("NetworkMode"),
                "mounts": [
                    {k: m.get(k) for k in ("Type", "Source", "Destination", "RW")}
                    for m in item["Mounts"]
                ],
                "captured_at": now(),
                "credentials_persisted": False,
            }
            code = "import hashlib,importlib.util,importlib.metadata,json,sys;from pathlib import Path;p=Path(importlib.util.find_spec('kairyu_bench').origin).parent;names=json.loads(sys.argv[1]);print(json.dumps({'source_files':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names},'python':sys.version,'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}))"
            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    cid,
                    "python",
                    "-B",
                    "-c",
                    code,
                    json.dumps(
                        sorted(plan["official"]["installed_source_files_sha256"])
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=environment,
            )
            need(result.returncode == 0, "official installed source audit failed")
            installed = json.loads(result.stdout)
            need(
                installed["source_files"]
                == plan["official"]["installed_source_files_sha256"],
                "official installed source changed",
            )
            audit["installed_source_audit"] = installed
            save(Path(row["output"]) / "runner-oci.json", audit)
            row["runner_oci"] = {
                "path": str(Path(row["output"]) / "runner-oci.json"),
                "sha256": sha(Path(row["output"]) / "runner-oci.json"),
            }
            image_parity.append(
                {
                    k: audit[k]
                    for k in (
                        "image_id",
                        "rootfs",
                        "image_config_sha256",
                        "normalized_command",
                        "noncredential_environment_sha256",
                        "user",
                        "working_dir",
                        "runtime",
                        "network_mode",
                        "installed_source_audit",
                    )
                }
            )
            publish()
        if pending:
            time.sleep(1)
    need(not pending, "official runner capture incomplete; no rerun")
    need(
        all(x == image_parity[0] for x in image_parity),
        "official runner parity differs",
    )
    # Bind paths differ per arm; keep raw hashes, assert only normalized mount semantics.
    mounts = []
    for row in rows:
        audit = read(row["runner_oci"]["path"])
        observer_dir = str(Path(row["observer"]["path"]).parent)
        normalized = [
            {k: ("__OBSERVER_DIR__" if v == observer_dir else v) for k, v in m.items()}
            for m in audit["mounts"]
        ]
        mounts.append(sorted(normalized, key=lambda x: json.dumps(x, sort_keys=True)))
    need(all(x == mounts[0] for x in mounts), "normalized official mounts differ")


def observer_numeric(row):
    path = Path(row["observer"]["path"]).parent / "requests-client.jsonl"
    raw = path.read_bytes()
    need(not raw or raw.endswith(b"\n"), "incomplete observer record")
    records = [json.loads(line) for line in raw.splitlines()]
    records = [r for r in records if r.get("request_scope") == "request"]
    config = read(row["observer"]["path"])
    need(
        all(
            r["run_id"] == row["run_id"] and r["capture_id"] == config["capture_id"]
            for r in records
        ),
        "observer attribution differs",
    )
    return {
        "completed_http_attempts": len(records),
        "logical_calls": len({r["call_id"] for r in records}),
        "retry_attempts": sum(r["attempt_index"] > 0 for r in records),
        "errors": sum(bool(r.get("error")) for r in records),
        "input_tokens": sum(
            (r.get("usage") or {}).get("prompt_tokens", 0) for r in records
        ),
        "output_tokens": sum(
            (r.get("usage") or {}).get("completion_tokens", 0) for r in records
        ),
        "quota_exhausted": (path.parent / ".quota-exhausted").exists(),
        "agent_steps": None,
        "path": str(path),
        "sha256": sha(path),
        "scope": "completed item-traced HTTP attempts, not agent steps; startup request excluded",
    }


def official_round(number, arms, current, deadline):
    import uuid

    round_dir = P / "rounds" / str(number)
    round_dir.mkdir(parents=True)
    group = []
    round_record = {
        "round": number,
        "started_at": now(),
        "arms": [],
        "status": "official_running",
    }
    state["rounds"].append(round_record)
    for arm, record in zip(arms, current, strict=True):
        out = round_dir / arm["label"]
        out.mkdir()
        observer_dir = out / "observer"
        observer_dir.mkdir()
        observer = observer_dir / "capture.json"
        config = {
            "schema_version": 1,
            "capture_id": uuid.uuid4().hex,
            "output_dir": str(observer_dir),
            "limits": {"storage_bytes": 268435456},
        }
        with observer.open("x") as stream:
            json.dump(config, stream)
        run_id = official_id(number, arm["label"])
        need(
            not (Path(plan["official"]["results_dir"]) / run_id).exists(),
            "official run already exists",
        )
        argv = official_command(record["endpoint"], arm, run_id, observer)
        row = {
            "label": arm["label"],
            "container_id": record["container_id"],
            "endpoint": record["endpoint"],
            "logical_overlay_sha256": arm["logical_overlay_sha256"],
            "physical_source_tree": plan["parent"]["tree"],
            "physical_candidate_checkpoint": False,
            "load_proof": record["proof"],
            "output": str(out),
            "run_id": run_id,
            "command": argv,
            "observer": {"path": str(observer), "sha256": sha(observer)},
        }
        row["before_counters"] = counters(
            record["endpoint"], record["container_id"], out / "serving-before.json"
        )
        round_record["arms"].append(row)
        record["status"] = "official_running"
        publish()
        with (out / "official.log").open("xb") as stream:
            child = subprocess.Popen(
                argv,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, **plan["official"].get("environment", {})},
                # The immutable wrapper bind-mounts its caller directory.
                cwd=plan["official"]["results_dir"],
            )
        children.append(child)
        group.append(child)
        row["pid"] = child.pid
    state["status"] = "official_running"
    publish()
    capture_official(round_record["arms"], min(deadline, time.monotonic() + 180))
    wait_children(group, min(deadline, time.monotonic() + plan["official_seconds"]))
    manifests = []
    for row, record in zip(round_record["arms"], current, strict=True):
        out = Path(row["output"])
        run_dir = Path(plan["official"]["results_dir"]) / row["run_id"]
        metadata = read(run_dir / "run.json")
        report = read(run_dir / "report.json")
        need(
            metadata["status"] == report["run"]["status"] == "completed"
            and report["run"]["label"] == row["logical_overlay_sha256"],
            "official result identity differs",
        )
        need(
            metadata["tier"] == 30
            and metadata["split"] == "dev"
            and metadata["model_id"] == "gemma-4-31b",
            "official run protocol differs",
        )
        sources = report["sources"]
        need(
            {k: v["items"] for k, v in sources.items()}
            == {
                "humaneval-plus": 30,
                "mbpp-plus": 30,
                "livecodebench": 20,
                "scicode": 10,
            },
            "official cohort differs",
        )
        manifests.append(metadata["manifest_sha256"])
        row["official"] = {
            "run_path": str(run_dir),
            "run_sha256": sha(run_dir / "run.json"),
            "report_sha256": sha(run_dir / "report.json"),
            "code": report["areas"]["code"],
            "sources": sources,
            "unknown": len(report["unscored"]),
            "timing": report["timing"],
        }
        row["observer_numeric"] = observer_numeric(row)
        row["after_counters"] = counters(
            record["endpoint"], record["container_id"], out / "serving-after.json"
        )
        command(
            [
                plan["python"],
                plan["tools"]["counters"]["path"],
                "diff",
                str(out / "serving-before.json"),
                str(out / "serving-after.json"),
                "--output",
                str(out / "serving-diff.json"),
            ],
            60,
        )
        row["counter_diff"] = {
            "path": str(out / "serving-diff.json"),
            "sha256": sha(out / "serving-diff.json"),
        }
        state["official_calls"] += report["timing"]["chat_api_calls"]
        record["status"] = "ready"
        row["finished_at"] = now()
        publish()
    need(len(set(manifests)) == 1, "matched official item cohort differs")
    control_dir = round_record["arms"][0]["official"]["run_path"]
    for row in round_record["arms"][1:]:
        output = Path(row["output"]) / "versus-control-evidence.json"
        command(
            [
                plan["python"],
                plan["tools"]["evidence"]["path"],
                control_dir,
                row["official"]["run_path"],
                "--partial",
                "--output",
                str(output),
            ],
            120,
        )
        e = read(output)
        row["comparison"] = {
            "path": str(output),
            "sha256": sha(output),
            "official": e["official_comparison"],
            "changed_item_ids": e["changed_item_ids"],
        }
    round_record.update(
        status="completed",
        finished_at=now(),
        control_process_reused=number > 1,
        control_official_run_is_fresh=True,
        comparison_scope="exploratory logical overlays; not physical candidate checkpoints; physical finalist and fresh ordinary pair required before adoption",
    )
    publish()


def execute():
    global owned
    need(os.getuid() == 1003, "UID1003 required")
    validate(True)
    need(
        bool(os.environ.get("HF_TOKEN")),
        "private official auth required; never persisted",
    )
    official_validate(check_fresh=True)
    for image in (plan["image_id"],):
        need(
            command(["docker", "image", "inspect", "--format", "{{.Id}}", image], 30)
            == image,
            "pinned local image absent",
        )
    need(
        not (P / "execution-status.json").exists()
        and not (P / "completion.json").exists(),
        "owner already used",
    )
    for name in ("cache", "load-receipts", "rounds"):
        need(not (P / name).exists(), "fresh owner directories required")
    for arm in [plan["control"], *plan["candidates"]]:
        need(
            subprocess.run(
                ["docker", "inspect", arm["container_name"]],
                capture_output=True,
                timeout=20,
            ).returncode
            != 0,
            "temporary name already exists",
        )
    with (P / "execution-status.json").open("x") as stream:
        json.dump(state, stream)
    owned = True
    publish()
    base = guard.inspect(
        plan["serving_baseline"][plan["server_template_service"]]["id"]
    )
    for service, old in plan["serving_baseline"].items():
        counters(endpoint(old["id"]), old["id"], P / (service + "-before-stop.json"))
    for arm in [plan["control"], *plan["candidates"]]:
        Path(read(arm["overlay_plan"]["path"])["receipt_directory"]).mkdir(parents=True)
    validate(True)
    state["status"] = "stopping_exact_four"
    publish()
    for service, old in plan["serving_baseline"].items():
        stopped.append(service)
        command(["docker", "stop", "--time", "90", old["id"]])
        need(not guard.inspect(old["id"])["State"]["Running"], "original did not stop")
    deadline = time.monotonic() + plan["maximum_active_seconds"]
    control = None
    chunks = [
        plan["candidates"][i : i + 3] for i in range(0, len(plan["candidates"]), 3)
    ]
    for number, candidates in enumerate(chunks, 1):
        validate(False)
        need(time.monotonic() < deadline, "declared active window exhausted")
        state["status"] = "loading_round"
        state["active_round"] = number
        publish()
        fresh = []
        if control is None:
            control = create(plan["control"], plan["slots"][0], base)
            fresh.append(control)
        records = []
        for arm, slot in zip(candidates, plan["slots"][1:]):
            records.append(create(arm, slot, base))
        fresh += records
        # A single Docker start request starts all new servers; loading overlaps.
        command(["docker", "start", *[record["container_id"] for record in fresh]], 120)
        startup_deadline = min(deadline, time.monotonic() + plan["startup_seconds"])
        for record in fresh:
            record["endpoint"] = wait_ready(record, startup_deadline)
        arms = [plan["control"], *candidates]
        current = [control, *records]
        for arm, record in zip(arms, current, strict=True):
            record["proof"] = proofs(arm)
            record["status"] = "ready"
        validate(False)
        publish()
        official_round(number, arms, current, deadline)
        # Keep the same empty-overlay control process across rounds; remove only candidates.
        for record in records:
            cleanup(record)
        validate(False)
    state["screen_completed"] = True


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--plan-sha", required=True)
parser.add_argument("--authorization", type=Path)
parser.add_argument("--authorization-sha")
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()
need(sha(args.plan) == args.plan_sha, "plan SHA differs")
plan = read(args.plan)
P = args.plan.resolve().parent
if not args.apply:
    print(
        json.dumps(
            {
                "status": "prepared_no_operations",
                "binding_complete": plan.get("binding_complete", False),
                "candidate_capacity": 6,
                "planned_candidates": len(plan.get("candidates", [])),
                "maximum_rounds": 2,
                "planned_rounds": (len(plan.get("candidates", [])) + 2) // 3,
                "runtime_sha256": sha(__file__),
                "service_changes": 0,
                "model_calls": 0,
            }
        )
    )
    raise SystemExit(0)
need(args.authorization and args.authorization_sha, "root authorization required")
os.umask(0o077)
for tool in plan["tools"].values():
    need(sha(tool["path"]) == tool["sha256"], "tool pin differs")
sys.path.insert(0, str(Path(plan["repo"]) / "src"))
sp = importlib.util.spec_from_file_location(
    "overlay_screen_guard", plan["tools"]["redeploy"]["path"]
)
guard = importlib.util.module_from_spec(sp)
sp.loader.exec_module(guard)
lock = (P / "owner.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
owned = False
stopped = []
temporary = []
children = []
error = None
state = {
    "schema_version": 1,
    "status": "preflight",
    "started_at": now(),
    "started_epoch": time.time(),
    "plan_sha256": args.plan_sha,
    "temporary": temporary,
    "rounds": [],
    "model_scope": "Exploratory official code90 on explicitly labeled logical overlays; physical finalist confirmation required",
    "official_calls": 0,
    "checkpoint_writes": 0,
    "adoption_performed": False,
}


def interrupted(sig, frame):
    raise InterruptedError("owner interrupted")


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, interrupted)
try:
    execute()
except BaseException as exc:
    error = type(exc).__name__
    state.update(status="failed", error_type=error)
    if isinstance(exc, OwnerCheckError):
        state["owner_assertion_message"] = str(exc)
    for record in temporary:
        try:
            startup_diagnostic(record)
        except BaseException as diagnostic:
            record["startup_diagnostic_error_type"] = type(diagnostic).__name__
    try:
        publish()
    except BaseException:
        pass
finally:
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_IGN)
    for child in children:
        try:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=10)
        except BaseException as exc:
            state.setdefault("child_cleanup_errors", []).append(type(exc).__name__)
            error = error or type(exc).__name__
    # Child termination can precede capture: recover only full-Cmd/run-ID matches.
    for rd in state["rounds"]:
        for row in rd["arms"]:
            try:
                env = {**os.environ, **plan["official"].get("environment", {})}
                cid = row.get("runner_container_id")
                ids = [cid] if cid else discover_owned_official(row, env)
                if not cid:
                    row["cleanup_discovered_runner_ids"] = ids
                for cid in ids:
                    cp = subprocess.run(
                        ["docker", "inspect", cid],
                        capture_output=True,
                        text=True,
                        timeout=20,
                        env=env,
                    )
                    if cp.returncode == 0:
                        item = json.loads(cp.stdout)[0]
                        need(
                            item["Id"] == cid
                            and item["Config"].get("Cmd") == row["command"][1:],
                            "refuse foreign official runner cleanup",
                        )
                        cp = subprocess.run(
                            ["docker", "rm", "--force", cid],
                            capture_output=True,
                            timeout=120,
                            env=env,
                        )
                        need(cp.returncode == 0, "owned official runner cleanup failed")
            except BaseException as exc:
                state.setdefault("official_cleanup_errors", []).append(
                    {"run_id": row["run_id"], "type": type(exc).__name__}
                )
                error = error or type(exc).__name__
    # Failure of one temporary cleanup cannot skip any other cleanup or restoration.
    for record in temporary:
        try:
            cleanup(record)
        except BaseException as exc:
            state.setdefault("temporary_cleanup_errors", []).append(
                {"label": record["label"], "type": type(exc).__name__}
            )
            error = error or type(exc).__name__
    state["status"] = "restoring_same_four"
    state["restored"] = {}
    try:
        publish()
    except BaseException:
        pass
    restore_deadline = time.monotonic() + plan["restoration_seconds"]
    for service in stopped:
        try:
            command(
                ["docker", "start", plan["serving_baseline"][service]["id"]],
                max(1, min(120, restore_deadline - time.monotonic())),
            )
            state["restored"][service] = {"start_requested": True}
        except BaseException as exc:
            state["restored"][service] = {"error_type": type(exc).__name__}
            error = error or type(exc).__name__
    for service in stopped:
        try:
            while time.monotonic() < restore_deadline:
                if canonical(snapshot(service)) == canonical(
                    plan["serving_baseline"][service]
                ):
                    break
                time.sleep(2)
            need(
                canonical(snapshot(service))
                == canonical(plan["serving_baseline"][service]),
                "original restoration incomplete",
            )
            state["restored"][service] = {"same_cid_exact_healthy": True}
        except BaseException as exc:
            state["restored"][service] = {"error_type": type(exc).__name__}
            error = error or type(exc).__name__
    if owned:
        try:
            validate(True)
            state["final_source_serving_checks"] = True
        except BaseException as exc:
            state["final_check_error_type"] = type(exc).__name__
            error = error or type(exc).__name__
        state.update(
            status="completed"
            if not error and state.get("screen_completed")
            else "failed",
            finished_at=now(),
            finished_epoch=time.time(),
            functional_parity_claim=False,
            physical_save_and_fresh_official_score_required_before_adoption=True,
        )
        try:
            publish()
        finally:
            with (P / "completion.json").open("x") as stream:
                json.dump(state, stream, indent=2)
                stream.write("\n")
if not owned:
    print(
        json.dumps(
            {
                "status": "preflight_failed_no_ownership",
                "error_type": error,
                "existing_receipts_unchanged": True,
            }
        )
    )
raise SystemExit(0 if state["status"] == "completed" else 1)
