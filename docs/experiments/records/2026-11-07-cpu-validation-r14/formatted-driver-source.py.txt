"""Single CPU CI pass after body/KL and separated editing/serving training contracts; prior checks retained separately."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
B = R / "worktrees/kairyu-bench-weight-tools"
D = R / "analysis/fresh-purpose-20261107-cpu-validation-r14"
P = R / "infra/weight-tools-ci/bin"
D.mkdir(exist_ok=False)
TMP = R / "infra/tmp/r14"
TMP.mkdir(parents=True, exist_ok=True)
env = dict(
    os.environ,
    TMPDIR=str(TMP),
    PATH=str(P) + os.pathsep + os.environ["PATH"],
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
pins = {
    "scripts/experiments/gemma4-purpose/train_lora.py": "c9406e597d015ec4f3cf1d626b0208d26fd6de7d673c6efca580af7001756374",
    "scripts/experiments/gemma4-purpose/train_lora_body_kl.py": "290365237443e0138079e106452c88dcf64779e0dc6834447f29e92666c90ef8",
    "docs/experiments/scripts/fresh-av-native-gpu4-20261107.py": "8d48cd69f9fe4109dec8fe190dde665483b24c6edb92e59dde4118859de6b2da",
    "docs/experiments/scripts/fresh-body-kl-plan-20261107.py": "27557f013728bcfe3af577bed8e1dde32f8d05bd1e71fe576872aef56ee4ab84",
    "docs/experiments/scripts/fresh-n30-complete-audit.py": "e471c64712fdd70f37439f80bb86137889287c823b99b6ff945ee90eec1e75b1",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (D / name).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


record_sources = sorted(
    p
    for p in (B / "docs/experiments/records").rglob("*")
    if p.is_file() and p.name.endswith((".py", ".py.txt"))
)


def inventory():
    paths = []
    for root in ("src", "tests", "scripts", "docs/experiments/scripts"):
        paths.extend(
            p for p in (B / root).rglob("*.py") if "__pycache__" not in p.parts
        )
    paths.extend((B / "pyproject.toml", B / ".github/workflows/ci.yml"))
    paths.extend(B / x for x in shell_paths)
    paths.extend(record_sources)
    return {str(p.relative_to(B)): sha(p) for p in sorted(set(paths))}


tracked = (
    subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=B)
    .decode()
    .split("\0")
)
shell_paths = sorted(p for p in tracked if p.endswith(".sh") and (B / p).is_file())
assert len(shell_paths) > 0
assert all(sha(B / path) == value for path, value in pins.items())
before = inventory()
save("sources-before.json", before)
save(
    "freeze-inputs.json",
    {
        "pins": pins,
        "driver_sha256": sha(Path(__file__)),
        "shell_paths": shell_paths,
        "source_scope": "Python source/tests/experiment scripts, pyproject, CI workflow and tracked shell; existing immutable record .py/.py.txt sources also pinned at start, without executing them. Newly packaged records and Markdown excluded",
        "record_source_count": len(record_sources),
        "TMPDIR": str(TMP),
        "PATH_prefix": str(P),
    },
)
commands = [
    (
        "unittest",
        [str(P / "python"), "-m", "unittest", "discover", "-s", "tests", "-v"],
    ),
    ("ruff", [str(P / "ruff"), "check", "."]),
    ("mypy", [str(P / "mypy"), "--ignore-missing-imports", "src"]),
    ("data", [str(P / "python"), "scripts/update_model_comparison.py", "--check"]),
]
started = time.time()
events = []
for label, command in commands:
    t = time.time()
    with (
        (D / (label + ".stdout")).open("x") as out,
        (D / (label + ".stderr")).open("x") as err,
    ):
        result = subprocess.run(
            ["nice", "-n", "15", *command], cwd=B, env=env, stdout=out, stderr=err
        )
    event = {
        "name": label,
        "command": ["nice", "-n", "15", *command],
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - t,
        "stdout_sha256": sha(D / (label + ".stdout")),
        "stderr_sha256": sha(D / (label + ".stderr")),
    }
    events.append(event)
    (D / "progress.json").write_text(json.dumps(events, indent=2) + "\n")
    print(json.dumps(event), flush=True)
shell_rows = []
t = time.time()
for path in shell_paths:
    result = subprocess.run(
        ["nice", "-n", "15", "sh", "-n", path],
        cwd=B,
        env=env,
        text=True,
        capture_output=True,
    )
    shell_rows.append(
        {
            "path": path,
            "sha256": sha(B / path),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )
shell = {
    "status": "passed" if all(x["returncode"] == 0 for x in shell_rows) else "failed",
    "count": len(shell_rows),
    "rows": shell_rows,
    "elapsed_seconds": time.time() - t,
    "scope": "All tracked repository shell files; runtime gitignored cache traversal intentionally not part of source syntax check",
}
save("repository-shell-check.json", shell)
events.append(
    {
        "name": "tracked_shell",
        "returncode": 0 if shell["status"] == "passed" else 1,
        "elapsed_seconds": shell["elapsed_seconds"],
        "count": len(shell_rows),
    }
)
after = inventory()
save("sources-after.json", after)
changes = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
receipt = {
    "started_epoch": started,
    "elapsed_seconds": time.time() - started,
    "results": events,
    "source_count_before": len(before),
    "source_count_after": len(after),
    "sources_unchanged": not changes,
    "source_changes": changes,
    "explicit_pins_unchanged": all(
        sha(B / path) == value for path, value in pins.items()
    ),
    "success": all(x["returncode"] == 0 for x in events) and not changes,
    "full_suite_repetitions": 1,
    "model_GPU_service_operations": 0,
    "TMPDIR": str(TMP),
    "PATH_prefix": str(P),
    "runtime_cache_find_not_repeated": True,
}
save("completion.json", receipt)
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if receipt["success"] else 1)
