#!/usr/bin/env python3
"""Fixed ARC10 response recording; run the unchanged official CLI via a proxy.

Usage: python arc-token-logprob-diagnostic.py NEW_OUTPUT.json -- run ...
No API calls on import/help. Output is bounded to 2 MiB and flushed after the run.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import threading

MAX_BYTES = 2 * 1024**2
MAX_CALLS = 40
MAX_TOKENS = 1024
RECORD_BYTES = (MAX_BYTES - 16384) // MAX_CALLS
TARGETS = tuple("arc-challenge:" + x for x in (
    "Mercury_7163818", "Mercury_SC_401278", "LEAP_2007_8_10417", "Mercury_SC_400401",
    "MDSA_2010_4_20", "Mercury_7218400", "Mercury_7206938", "Mercury_7074848",
    "Mercury_402348", "Mercury_SC_413624",
))
EXPECTED_SOURCE = {
    "client.py": "3a99ddd6c19697bee533ba3e87d3d13170ba5299571b523b773cd72e60cc7b2c",
    "sources/logprob.py": "fafc8d0c6b1623621301f2515b1c393948ffbed3fbe5ecfa86571bac1ef6fe89",
    "sources/__init__.py": "92a8db31d765d06a8f3c111bf42618fa8f49b8db0b7d2915d0eea0ef6b8ccc4a",
    "cli.py": "406b07cb9bed06bd3b814292329cd67d0acf89fcaf5fddca8858cc865bbed090",
    "run.py": "9272730f132a72002cc38f7c3c47690893f8288652364c1b644e0cf27bf04ce2",
}
MANIFEST_SHA256 = "a3a0b60e0872a38c6b1abf7e97d7fe729fabf9e25cd3f09a3da9544b9ca39b5b"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Recorder:
    def __init__(self):
        self.records = []
        self.errors = Counter()
        self.calls = 0
        self.client_calls = Counter()
        self.lock = threading.Lock()

    def error(self, code):
        with self.lock:
            self.errors[code if code in self.errors or len(self.errors) < 8 else "other"] += 1

    def note_call(self, method):
        with self.lock:
            self.client_calls[method] += 1

    def add(self, record):
        # Recording faults never change the official outcome or response.
        try:
            payload = encoded(record) + b"\n"
            with self.lock:
                self.calls += 1
                limit = max(1024, RECORD_BYTES) if "capture_error" in record else RECORD_BYTES
                if (len(self.records) >= MAX_CALLS or len(payload) > limit
                        or sum(map(len, self.records)) + len(payload) > MAX_BYTES - 16384):
                    self.errors["record_quota"] += 1
                else:
                    self.records.append(payload)
        except Exception:
            self.error("record_encoding")

    def flush(self, output, returncode, error_type=None):
        rows = [json.loads(row) for row in self.records]
        complete = (returncode == 0 and error_type is None and not self.errors
                    and len(rows) == MAX_CALLS and self.calls == MAX_CALLS
                    and Counter(r["item_id"] for r in rows) == Counter({x: 4 for x in TARGETS})
                    and all(r.get("official_sum_matches") is True for r in rows))
        summary = {"record_type": "diagnostic_summary", "schema_version": 1,
                   "scope": "fixed ARC10 observer-OFF response diagnostic, not a full evaluation",
                   "capture_complete": complete, "official_returncode": returncode,
                   "official_exception_type": error_type, "expected_calls": MAX_CALLS,
                   "recorded_calls": len(rows), "finished_call_records": self.calls,
                   "capture_errors": dict(self.errors), "max_bytes": MAX_BYTES,
                   "logical_client_calls": dict(self.client_calls),
                   "transport_retries_not_inferred": True,
                   "max_tokens_per_call": MAX_TOKENS, "source_sha256": EXPECTED_SOURCE,
                   "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "token_hash_encoding": "SHA256 of compact JSON integer token array",
                   "missing_logprob": "null, never zero", "raw_response_modified": False,
                   "scorer_modified": False, "environment_or_auth_saved": False}
        header = encoded(summary)[:-1] + b',"records":['
        if len(header) + sum(map(len, self.records)) + 3 > MAX_BYTES:
            raise ValueError("diagnostic output exceeds its fixed quota")
        with output.open("xb") as stream:
            stream.write(header)
            for index, row in enumerate(self.records):
                if index:
                    stream.write(b",")
                stream.write(row.rstrip(b"\n"))
            stream.write(b"]}\n")
        return summary


class RecordingClient:
    """Per-item proxy; original methods, arguments, results and errors pass through."""
    def __init__(self, delegate, recorder, item_id):
        self.delegate, self.recorder, self.item_id = delegate, recorder, item_id
        self.tokenizations = []
        self.pending = []
        self.choice_index = 0

    def tokenize(self, text, *, add_special_tokens):
        self.recorder.note_call("tokenize")
        result = self.delegate.tokenize(text, add_special_tokens=add_special_tokens)
        try:
            if len(self.tokenizations) < 2:
                self.tokenizations.append((len(result), hashlib.sha256(text.encode()).hexdigest()))
        except Exception:
            self.recorder.error("capture_tokenization")
        return result

    def prompt_logprobs(self, token_ids, *, top_k):
        tokenizations, self.tokenizations = self.tokenizations, []
        index, self.choice_index = self.choice_index, self.choice_index + 1
        self.recorder.note_call("prompt_logprobs")
        base = {"record_type": "choice_token_logprobs", "item_id": self.item_id,
                "choice_index": index, "prompt_tokens": len(token_ids),
                "tokenization_lengths": [entry[0] for entry in tokenizations]}
        try:
            result = self.delegate.prompt_logprobs(token_ids, top_k=top_k)
        except BaseException as error:
            self.pending.append({**base, "capture_error": {"category": "official_request",
                                                          "type": type(error).__name__}})
            self.recorder.error("official_request")
            raise
        category = "scope"
        try:
            lengths = [entry[0] for entry in tokenizations]
            if index >= 4 or len(token_ids) > MAX_TOKENS or len(lengths) != 2:
                raise ValueError("record scope")
            category = "boundary"
            start = len(token_ids) - lengths[0] + lengths[1]
            base["continuation_start"] = start
            base["token_ids_sha256"] = hashlib.sha256(encoded(token_ids)).hexdigest()
            if not 1 <= start <= len(token_ids) or len(result) != len(token_ids):
                raise ValueError("record boundary")
            category = "continuation_logprob"
            values = []
            for position in range(start, len(token_ids)):
                token, entry = token_ids[position], result[position]
                value = None if entry is None or token not in entry else entry[token][0]
                values.append(value if value is None or math.isfinite(value) else None)
            total = 0.0
            for value in values:
                if value is None:
                    raise ValueError("missing continuation logprob")
                total += value
            record = {"record_type": "choice_token_logprobs", "item_id": self.item_id,
                      "choice_index": index, "token_ids": list(token_ids),
                      "token_ids_sha256": hashlib.sha256(encoded(token_ids)).hexdigest(),
                      "prompt_tokens": len(token_ids), "continuation_start": start,
                      "continuation_stop": len(token_ids), "continuation_logprobs": values,
                      "text_sha256": {"whole": tokenizations[0][1], "head": tokenizations[1][1]},
                      "tokenization_lengths": {"whole": lengths[0], "head": lengths[1]},
                      "bos_tokens": len(token_ids) - lengths[0], "continuation_sum": total}
            category = "record_quota"
            if len(encoded(record)) > RECORD_BYTES - 512:
                raise ValueError("record quota")
            self.pending.append(record)
        except Exception as error:
            self.pending.append({**base, "capture_error": {"category": category,
                                                          "type": type(error).__name__}})
            self.recorder.error("capture_" + category)
        return result

    def finish(self, outcome):
        for record in self.pending:
            try:
                value = outcome["logliks"][record["choice_index"]] if outcome is not None else None
                finite = isinstance(value, (int, float)) and math.isfinite(value)
                record["official_sum"] = value if finite else None
                record["official_sum_matches"] = finite and value == record.get("continuation_sum")
                self.recorder.add(record)
            except Exception:
                self.recorder.error("capture_finalization")


@contextmanager
def recording_adapter(recorder):
    from kairyu_bench.wsp.sources import ADAPTERS
    from kairyu_bench.wsp.sources import logprob  # noqa: F401 - register canonical adapters

    original = ADAPTERS["arc-challenge"]

    def execute(context, item, row):
        if context.logprob is None:
            recorder.error("missing_official_client")
            return original.execute(context, item, row)
        proxy = RecordingClient(context.logprob, recorder, item.id)
        try:
            result = original.execute(replace(context, logprob=proxy), item, row)
        except BaseException:
            proxy.finish(None)
            raise
        proxy.finish(result)
        return result

    ADAPTERS["arc-challenge"] = replace(original, execute=execute)
    try:
        yield
    finally:
        ADAPTERS["arc-challenge"] = original


def validate_run(arguments):
    from kairyu_bench.wsp import cli

    source = Path(cli.__file__).parent
    if {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in EXPECTED_SOURCE} != EXPECTED_SOURCE:
        raise ValueError("official source differs from the inspected immutable runner")
    args = cli.build_parser().parse_args(arguments)
    if (args.command != "run" or args.tier != 10 or args.split != "dev"
            or args.only != "arc-challenge" or args.manifest is not None
            or args.incomplete_manifest or args.observe_config is not None
            or args.logprob_endpoint != [args.endpoint] or not args.run_id
            or (args.chat_concurrency, args.logprob_concurrency, args.agent_concurrency) != (256, 32, 4)):
        raise ValueError("wrapper requires the registered ARC10 observer-OFF conditions")
    manifest = cli.load_manifest(None, require_all_areas=True)
    selected = [item.id for item in manifest.select(10, "dev")
                if item.source == "arc-challenge" and item.id not in args.exclude_item]
    if manifest.sha256 != MANIFEST_SHA256 or tuple(selected) != TARGETS:
        raise ValueError("wrapper requires the exact fixed-order ten and canonical manifest")
    return cli


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments == ["--help"]:
        print(__doc__)
        return 0
    output = Path(arguments.pop(0))
    if arguments[:1] != ["--"]:
        raise ValueError("separate the original wsp run arguments with --")
    arguments.pop(0)
    root = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/restart-direct-20261108")
    if (not output.is_absolute() or output.resolve() != output or not output.is_relative_to(root)
            or not output.parent.is_dir() or output.exists()):
        raise ValueError("output must be a fresh registered file under the NVMe restart root")
    cli = validate_run(arguments)
    recorder = Recorder()
    code, error_type = None, None
    try:
        with recording_adapter(recorder):
            code = cli.main(arguments)
        return code
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        recorder.flush(output, code, error_type)


if __name__ == "__main__":
    raise SystemExit(main())
