"""Stage pure-Python overlay wheels as the calling non-root user; no Docker calls."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import zipfile

SUPPORT = (
    "src/kairyu_bench/__init__.py",
    "src/kairyu_bench/wsp/__init__.py",
    "src/kairyu_bench/wsp/weights/__init__.py",
    "src/kairyu_bench/wsp/weights/checkpoint.py",
    "src/kairyu_bench/wsp/weights/editor.py",
)
PLUGIN = Path("scripts/experiments/vllm_sparse_overlay")
SUPPORT_PROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "kairyu-sparse-overlay-support"
version = "0.1.0"
requires-python = ">=3.11"
[tool.setuptools.packages.find]
where = ["src"]
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sources(root: Path) -> dict[str, str]:
    names = [*SUPPORT, str(PLUGIN / "pyproject.toml"), str(PLUGIN / "Dockerfile"),
             str(PLUGIN / "stage_image.py"), "tests/test_vllm_sparse_overlay.py"]
    names += [str(p.relative_to(root)) for p in sorted((root / PLUGIN / "src").rglob("*.py"))]
    for name in names:
        path = root / name
        if not path.is_file() or path.resolve() != path:
            raise ValueError("source must be a canonical regular file")
    return {name: sha(root / name) for name in sorted(names)}


def command(argv: list[str], output: Path, label: str, env: dict[str, str]) -> None:
    with (output / (label + ".log")).open("xb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        if code:
            raise RuntimeError(f"{label} failed, see owned log")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root, output = args.root.absolute(), args.output.absolute()
    owned = False
    try:
        if os.geteuid() == 0:
            raise ValueError("run as the ordinary task user, never root")
        if root != root.resolve() or output != output.resolve() or output.is_relative_to(root):
            raise ValueError("canonical output outside the source repository required")
        if output.exists():
            raise ValueError("fresh output required; failed staging is not retried")
        pins = sources(root)
        input_sha = hashlib.sha256(encoded(pins)).hexdigest()
        build_files = importlib.metadata.files("setuptools") or []
        if not any(str(p) == "setuptools/command/bdist_wheel.py" for p in build_files):
            raise ValueError("existing setuptools must provide its bundled bdist_wheel command")
        result = {"schema_version": 1, "status": "preview", "input_sha256": pins,
                  "input_bundle_sha256": input_sha, "output": str(output),
                  "uid": os.geteuid(), "python": sys.executable,
                  "build_packages": {name: importlib.metadata.version(name)
                                     for name in ("pip", "setuptools")},
                  "wheel_provider": "setuptools.command.bdist_wheel",
                  "docker_actions": 0, "model_calls": 0, "gpu_calls": 0,
                  "support_scope": "exact reader/editor plus package initializers; no benchmark datasets"}
        if not args.apply:
            print(json.dumps(result, indent=2))
            return 0
        if args.expected_input_sha256 != input_sha:
            raise ValueError("apply requires the exact preview input bundle SHA")
        output.mkdir(parents=True, exist_ok=False)
        owned = True
        inputs, context = output / "inputs", output / "context"
        inputs.mkdir()
        context.mkdir()
        support, plugin = inputs / "support", inputs / "plugin"
        support.mkdir()
        (support / "pyproject.toml").write_text(SUPPORT_PROJECT)
        for name in SUPPORT:
            target = support / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        plugin.mkdir()
        shutil.copyfile(root / PLUGIN / "pyproject.toml", plugin / "pyproject.toml")
        for name in pins:
            if name.startswith(str(PLUGIN / "src") + "/"):
                target = plugin / Path(name).relative_to(PLUGIN)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
        wheels, tmp = output / "wheels", output / "tmp"
        wheels.mkdir()
        tmp.mkdir()
        env = {**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp)}
        command([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
                 "--no-build-isolation", "--no-cache-dir", "--wheel-dir", str(wheels),
                 str(support), str(plugin)], output, "wheel", env)
        artifacts = sorted(wheels.glob("*.whl"))
        if len(artifacts) != 2:
            raise ValueError("exactly two locally built wheels required")
        for path in artifacts:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                metadata = [n for n in names if n.endswith(".dist-info/WHEEL")]
                if (not path.name.endswith("-py3-none-any.whl") or len(metadata) != 1
                        or b"Root-Is-Purelib: true" not in archive.read(metadata[0])
                        or any(n.endswith((".so", ".pyd", ".dll")) for n in names)):
                    raise ValueError("only universal pure-Python wheels are allowed")
        command([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                 "--no-compile", "--no-cache-dir", "--target", str(context / "python"),
                 *(str(p) for p in artifacts)], output, "install", env)
        for name in [*SUPPORT, *(n for n in pins if n.startswith(str(PLUGIN / "src") + "/"))]:
            relative = Path(name).relative_to("src") if name in SUPPORT else Path(name).relative_to(PLUGIN / "src")
            if sha(context / "python" / relative) != pins[name]:
                raise ValueError("installed Python bytes differ from reviewed source")
        shutil.copyfile(root / PLUGIN / "Dockerfile", context / "Dockerfile")
        (context / "smoke").mkdir()
        shutil.copyfile(root / "tests/test_vllm_sparse_overlay.py",
                        context / "smoke/test_vllm_sparse_overlay.py")
        if sources(root) != pins:
            raise ValueError("source changed during staging")
        result.update(status="staged_not_built", wheels={p.name: sha(p) for p in artifacts},
                      context_sha256={str(p.relative_to(context)): sha(p)
                                      for p in sorted(context.rglob("*")) if p.is_file()})
        (output / "stage.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "message": str(error)}
        if owned:
            (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
