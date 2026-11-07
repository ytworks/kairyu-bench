"""Optional Harbor agent extension using its native LLM call kwargs.

Only imported by the pinned Harbor runtime. Task identity is read from the
trial config that Harbor writes before setup, never inferred from prompts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2.terminus_2 import Terminus2

from kairyu_bench.wsp.observe.tracing import RequestRecorder


class ObservedTerminus2(Terminus2):
    def __init__(self, *args: Any, llm_call_kwargs: dict[str, Any] | None = None, **kwargs: Any):
        call_kwargs = dict(llm_call_kwargs or {})
        self.observation_headers = dict(call_kwargs.get("extra_headers") or {})
        if any(key.lower() == "traceparent" for key in self.observation_headers):
            raise ValueError("observation requires ownership of the trial traceparent")
        # Terminus2 copies the outer kwargs dict. Keep this nested header dict
        # until its official setup hook can read authoritative trial identity.
        call_kwargs["extra_headers"] = self.observation_headers
        self.observation_registered = False
        super().__init__(*args, llm_call_kwargs=call_kwargs, **kwargs)

    async def setup(self, environment: Any) -> None:
        if not self.observation_registered:
            config_path = Path(os.environ["KAIRYU_BENCH_WSP_OBSERVE_CONFIG"])
            trial_dir = Path(self.logs_dir).parent
            trial = json.loads((trial_dir / "config.json").read_text(encoding="utf-8"))
            if trial.get("trial_name") != trial_dir.name:
                raise ValueError("observation trial path and official identity differ")
            name = str(trial["task"]["name"]).removeprefix("terminal-bench/")
            selected = os.environ.get("KAIRYU_BENCH_TERMINAL_BENCH_TASKS", "").split(",")
            if name not in selected:
                raise ValueError("observation received an unselected Terminal-Bench task")
            recorder = RequestRecorder(
                config_path, run_id=os.environ["KAIRYU_BENCH_WSP_OBSERVE_RUN_ID"],
            )
            # Harbor assigns a new context_id UUID to every trial instance,
            # including a retry. Its public setup hook has no retry ordinal.
            self.observation_headers.update(recorder.register_trial(
                f"terminal-bench:{name}", str(self.context_id), attempt_index=None,
            ))
            self.observation_registered = True
        await super().setup(environment)
