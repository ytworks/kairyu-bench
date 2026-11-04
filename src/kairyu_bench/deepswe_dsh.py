"""DeepSeek Harness (sdk-minimal) as a Pier agent for DeepSWE.

The SDK and its native runtime are installed from the hash-locked wheels into
each task image. The uploaded runner drives one official instruction and writes
``dsh-result.json``; the task's own collect hook grades the agent's commits.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from pier.agents.installed.antigravity_sdk import _UV_INSTALLER_SHA256, _UV_VERSION
from pier.agents.installed.base import BaseInstalledAgent, with_prompt_template
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

from kairyu_bench.deepswe import DEEPSEEK_HARNESS, DSH_LOCK, DSH_PATCH, read_object
from kairyu_bench.deepswe_runtime import TransientAgentError, _model_id, api_allowlist

AGENT_ROOT = "/installed-agent"
VENV = f"{AGENT_ROOT}/venv"
PYTHON = f"{VENV}/bin/python"
RUNNER = f"{AGENT_ROOT}/kairyu_dsh_runner.py"
PATCH = f"{AGENT_ROOT}/kairyu-dsh-minimal.patch.json"
RUNTIME_CWD = f"{AGENT_ROOT}/runtime-cwd"
LOGS = "/logs/agent"
RESULT = "dsh-result.json"
# dsh retries these itself; once its own retries are exhausted Pier may retry
# the trial, and an exhausted trial stays unscored.
TRANSIENT_STATUSES = {
    f"provider:{code}"
    for code in ("RATE_LIMIT", "SERVER", "TRANSPORT", "TIMEOUT", "EMPTY_RESPONSE")
}


def agent_config(model_id: str, conditions: dict[str, Any]) -> dict[str, Any]:
    if conditions["agent_version"] != read_object(DSH_LOCK)["version"]:
        raise ValueError("DeepSWE conditions were resolved for another dsh lock")
    kwargs: dict[str, Any] = {
        "context_window": conditions["context_window"],
        "max_tokens": conditions["max_tokens"],
    }
    if conditions["reasoning_effort"] != "agent_default":
        kwargs["reasoning_effort"] = conditions["reasoning_effort"]
    return {
        "import_path": "kairyu_bench.deepswe_dsh:KairyuDeepSeekHarnessAgent",
        "model_name": f"{conditions['agent_provider']}/{_model_id(model_id)}",
        "kwargs": kwargs,
    }


def install_script(lock: dict[str, Any]) -> str:
    site = f"{VENV}/lib/python{lock['python']}/site-packages"
    executables = " ".join(f"{site}/{path}" for path in lock["runtime_executables"])
    requirements = "\n".join(lock["requirements"])
    return f"""
set -euo pipefail
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y ca-certificates curl
elif command -v yum >/dev/null 2>&1; then
  yum install -y ca-certificates curl
else
  command -v curl >/dev/null 2>&1
fi
command -v sha256sum >/dev/null 2>&1
UV_INSTALLER=/tmp/uv-{_UV_VERSION}-installer.sh
curl -LsSf https://astral.sh/uv/{_UV_VERSION}/install.sh -o "$UV_INSTALLER"
echo "{_UV_INSTALLER_SHA256}  $UV_INSTALLER" | sha256sum -c -
env UV_INSTALL_DIR=/usr/local/bin sh "$UV_INSTALLER"
rm -f "$UV_INSTALLER"
mkdir -p {AGENT_ROOT} {RUNTIME_CWD}
export UV_PYTHON_INSTALL_DIR={AGENT_ROOT}/python
uv venv --python {lock["python"]} {VENV}
cat > {AGENT_ROOT}/requirements.txt <<'KAIRYU_DSH_REQUIREMENTS'
{requirements}
KAIRYU_DSH_REQUIREMENTS
uv pip install --python {PYTHON} --require-hashes --no-deps --only-binary :all: \\
  -r {AGENT_ROOT}/requirements.txt
chmod -R a+rX {AGENT_ROOT}
chmod a+x {executables}
""".strip()


class KairyuDeepSeekHarnessAgent(BaseInstalledAgent):
    def __init__(
        self,
        *args,
        context_window: int,
        max_tokens: int,
        reasoning_effort: str | None = None,
        **kwargs,
    ):
        self._lock = read_object(DSH_LOCK)
        if reasoning_effort is not None and reasoning_effort not in self._lock[
            "reasoning_efforts"
        ]:
            raise ValueError(f"unsupported dsh reasoning effort: {reasoning_effort}")
        self._context_window = context_window
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        super().__init__(*args, version=self._lock["version"], **kwargs)

    @staticmethod
    def name() -> str:
        return DEEPSEEK_HARNESS

    def get_version_command(self) -> str | None:
        return (
            f'{PYTHON} -c "from importlib.metadata import version; '
            "print(version('deepseek-harness-sdk'))\""
        )

    def install_spec(self) -> AgentInstallSpec:
        # Launching the runtime proves glibc, the executable bits and startup.
        verify = (
            'home=$(mktemp -d) && DSH_HOME="$home" '
            f'{VENV}/bin/dsh --version && rm -rf "$home" && {self.get_version_command()}'
        )
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._lock["version"],
            steps=[
                InstallStep(
                    user="root",
                    env={"DEBIAN_FRONTEND": "noninteractive"},
                    run=install_script(self._lock),
                ),
                InstallStep(user="agent", run=verify),
            ],
            verification_command=self.get_version_command(),
            metadata={
                "python": self._lock["python"],
                "uv": _UV_VERSION,
                "deepseek-harness-sdk": self._lock["version"],
                "revision": self._lock["revision"],
            },
        )

    def network_allowlist(self) -> NetworkAllowlist:
        return api_allowlist(self._endpoint())

    def _endpoint(self) -> str:
        return self._get_env("OPENAI_BASE_URL") or self._get_env("OPENAI_API_BASE") or ""

    async def setup(self, environment) -> None:
        await super().setup(environment)
        await environment.upload_file(
            Path(__file__).with_name("deepswe_dsh_runner.py"), RUNNER
        )
        await environment.upload_file(DSH_PATCH, PATCH)
        await self.exec_as_root(environment, f"chmod a+r {RUNNER} {PATCH}")

    def populate_context_post_run(self, context) -> None:
        status = "MissingAgentResult"
        usage: dict[str, Any] = {}
        calls = None
        path = self.logs_dir / RESULT
        if path.is_file():
            try:
                result = read_object(path)
            except ValueError:
                status = "InvalidAgentResult"
            else:
                status = result.get("status") or "MissingAgentExitStatus"
                usage = result.get("usage") or {}
                calls = result.get("model_calls")
        context.metadata = {**(context.metadata or {}), "dsh_exit_status": status}
        context.n_input_tokens = usage.get("input_tokens")
        context.n_output_tokens = usage.get("output_tokens")
        context.n_cache_tokens = usage.get("cache_read_tokens")
        context.n_agent_steps = calls
        context.cost_usd = None  # No valid pricing is available for a local API.

    @with_prompt_template
    async def run(self, instruction, environment, context) -> None:
        env = {
            "DEEPSEEK_BASE_URL": self._endpoint(),
            "DEEPSEEK_API_KEY": self._get_env("OPENAI_API_KEY") or "not-required",
            "DSH_CONTEXT_WINDOW": str(self._context_window),
            "DSH_TELEMETRY_MODE": "DISABLED",
            "KAIRYU_DSH_MODEL": self._parsed_model_name or "",
            "KAIRYU_DSH_MAX_TOKENS": str(self._max_tokens),
        }
        if self._reasoning_effort is not None:
            env["KAIRYU_DSH_REASONING_EFFORT"] = self._reasoning_effort
        command = (
            f"{PYTHON} {RUNNER} --instruction {shlex.quote(instruction)} "
            f"--logs-dir {LOGS} --runtime-cwd {RUNTIME_CWD} --patch {PATCH} "
            f"2>&1 </dev/null | tee {LOGS}/deepseek-harness.txt"
        )
        await self.exec_as_agent(environment, command=command, env=env)
        self.populate_context_post_run(context)
        status = context.metadata["dsh_exit_status"]
        if status in TRANSIENT_STATUSES:
            raise TransientAgentError(status)
