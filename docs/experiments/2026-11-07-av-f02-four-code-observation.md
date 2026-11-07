# AV / F02 four-task observation inspection, 2026-11-07

This CPU-only diagnostic retains the three unchanged failures plus the declared
passing control `humaneval-plus:HumanEval/30`. It does not rerun a benchmark,
change its selection, or replace the full 26-code-item audit or 778 official
outcomes. It reads each existing canonical merged JSON once, preserving the
selected task objects and all original run/site/capture metadata byte-for-byte.
The omitted tasks remain in the original inputs, whose full SHA256 values are
recorded. This is a display selection, not an alternate score.

The existing viewer supports separate AV and F02 views. Strict activation
subtraction is unavailable: `pair_key` and `runtime.vllm_version` are absent.
F02 `closed=false`, unlinked requests and partial status remain visible. No
pairing metadata or writer completion is synthesized. Generated prefixes may
differ at the same decode ordinal. The existing heatmap excludes rejected MTP
records even when accepted-only is unchecked; those records remain in the
selected JSON and all-code audit. Inspect accepted counts before reading a cell.

The following one-off extraction code is executed from this Markdown on the
remote host with the existing CPU Python. It changes no model, raw trace,
canonical merged JSON, or official result. It validates the known pretty-JSON
task boundaries, the exact 778 full items, unique IDs, unchanged input stats,
and all four requested IDs; malformed or unexpectedly sized data fails.

```python
import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.nice(15)
R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
REPO = R / "worktrees/kairyu-bench-weight-tools"
OUT = R / "analysis/fresh-purpose-20261107-f02-live10-v1/selected-code-view-v1"
IDS = {
    "livecodebench:arc191_c", "mbpp-plus:Mbpp/410",
    "mbpp-plus:Mbpp/576", "humaneval-plus:HumanEval/30",
}
INPUTS = {
    "av": R / "analysis/fresh-purpose-20261107-av-live10-v1/merged.json",
    "f02": R / "analysis/fresh-purpose-20261107-f02-live10-v1/merged.json",
}
OUT.mkdir(mode=0o700, exist_ok=False)
started = time.time()
receipt = {
    "purpose": "Four declared diagnostic tasks, not a reduced WSP evaluation",
    "selection": sorted(IDS), "expected_full_item_count": 778,
    "source": "existing canonical merged files; no raw capture rescan",
    "inputs": {}, "strict_activation_delta_supported": False,
    "limitations": [
        "pair_key and runtime.vllm_version are absent and remain absent",
        "F02 worker closed=false remains unchanged; no close acknowledgement",
        "same decode ordinal need not have same generated prefix",
        "existing viewer hides accepted=false records from its heatmap",
        "all 26 code numeric audits and all 778 official outcomes remain intact",
    ],
}
for label, source in INPUTS.items():
    before = source.stat()
    digest = hashlib.sha256()
    state = "header"
    header, footer, kept = [], [], []
    current, task_id, total, all_ids = [], None, 0, set()
    with source.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if state == "header":
                header.append(line)
                if line == b'      "tasks": [\n':
                    state = "tasks"
                continue
            if state == "footer":
                footer.append(line)
                continue
            if line == b'      ]\n':
                assert not current and task_id is None
                state = "footer"
                footer.append(line)
            elif line == b'        {\n':
                assert not current and task_id is None
                current = [line]
            elif line.startswith(b'          "id": '):
                assert len(current) == 1 and task_id is None
                task_id = json.loads(line.split(b": ", 1)[1].rstrip(b",\r\n"))
                assert task_id not in all_ids
                all_ids.add(task_id)
                if task_id in IDS:
                    current.append(line)
            elif line in (b'        },\n', b'        }\n'):
                assert task_id is not None
                total += 1
                if task_id in IDS:
                    current.append(b'        }\n')
                    kept.append(b"".join(current))
                current, task_id = [], None
            elif task_id in IDS:
                current.append(line)
    after = source.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert state == "footer" and total == 778 and len(kept) == len(IDS) and IDS <= all_ids
    sha = digest.hexdigest()
    if label == "f02":
        assert sha == "a56a03136c7da8b1bcd0b25720885a82ac030802397477c4637c8ae2e83ebdf4"
    body = b"".join(header) + b",\n".join(x.rstrip(b"\n") for x in kept) + b"\n" + b"".join(footer)
    assert len(body) < 64 * 1024**2
    output = OUT / f"{label}-selected.json"
    with output.open("xb") as handle:
        handle.write(body)
    selected = json.loads(body)
    assert selected["schema_version"] == 1 and len(selected["runs"]) == 1
    run = selected["runs"][0]
    assert {t["id"] for t in run["tasks"]} == IDS
    summaries = []
    for task in run["tasks"]:
        rows = task["observations"]
        summaries.append({
            **{k: task.get(k) for k in ("id", "source", "area", "score", "status")},
            "observation_cells": len(rows),
            "pair_key_present": sum("pair_key" in x for x in rows),
            "accepted_true": sum(x.get("accepted") is True for x in rows),
            "accepted_false": sum(x.get("accepted") is False for x in rows),
            "accepted_unknown": sum(x.get("accepted") is None for x in rows),
            "ranks": sorted({x["rank"] for x in rows}),
            "call_ids": sorted({x["call_id"] for x in rows}),
        })
    receipt["inputs"][label] = {
        "path": str(source), "bytes": before.st_size, "sha256": sha,
        "stat_unchanged": True, "full_task_count": total,
        "selected_path": str(output), "selected_bytes": len(body),
        "selected_sha256": hashlib.sha256(body).hexdigest(),
        "run_id": run["id"], "checkpoint": run["checkpoint"],
        "runtime": run["runtime"], "capture_status": run["capture"]["status"],
        "workers_closed": [x.get("closed") for x in run["capture"]["workers"]],
        "sites": len(run["sites"]), "tasks": summaries,
    }
    print(json.dumps({"label": label, "full_sha256": sha, "full_tasks": total,
                      "selected_bytes": len(body), "tasks": summaries}), flush=True)
    del body, selected, run, kept
    gc.collect()
receipt["elapsed_seconds"] = time.time() - started
(OUT / "extraction-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"elapsed_seconds": receipt["elapsed_seconds"], "output": str(OUT)}))
```

## Retained extraction failure and corrected scope

The first display extraction read AV merged and failed its pre-output `total ==
778` assertion. No selected JSON or viewer was written. The canonical merge
also retains two unlinked requests as `unattributed/...` task objects; official
778-item completeness and merged-task count are different quantities. This is
a display-extraction failure, not a new benchmark failure. Its original code
above and `selected-code-view-v1/failure.json` remain. The corrected one-off code
below validates the exact official item-ID set against immutable `items.jsonl`,
separately requires the two additional unattributed tasks, and retains the
original full capture metadata and full-input hashes. It reads each merged
input once more; there is no raw-capture rescan.

```python
import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.nice(15)
R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
REPO = R / "worktrees/kairyu-bench-weight-tools"
OUT = R / "analysis/fresh-purpose-20261107-f02-live10-v1/selected-code-view-v2"
IDS = {
    "livecodebench:arc191_c", "mbpp-plus:Mbpp/410",
    "mbpp-plus:Mbpp/576", "humaneval-plus:HumanEval/30",
}
INPUTS = {
    "av": R / "analysis/fresh-purpose-20261107-av-live10-v1/merged.json",
    "f02": R / "analysis/fresh-purpose-20261107-f02-live10-v1/merged.json",
}
OUT.mkdir(mode=0o700, exist_ok=False)
started = time.time()
receipt = {
    "purpose": "Four declared diagnostic tasks, not a reduced WSP evaluation",
    "selection": sorted(IDS), "expected_full_item_count": 778,
    "source": "existing canonical merged files; no raw capture rescan",
    "inputs": {}, "strict_activation_delta_supported": False,
    "limitations": [
        "pair_key and runtime.vllm_version are absent and remain absent",
        "F02 worker closed=false remains unchanged; no close acknowledgement",
        "same decode ordinal need not have same generated prefix",
        "existing viewer hides accepted=false records from its heatmap",
        "all 26 code numeric audits and all 778 official outcomes remain intact",
    ],
}
for label, source in INPUTS.items():
    result_path = R / f"results/gemma4-fresh-{label}-observed-dev10-20261107-v1/items.jsonl"
    result_bytes = result_path.read_bytes()
    expected_ids = {json.loads(line)["id"] for line in result_bytes.splitlines()}
    assert len(expected_ids) == 778
    before = source.stat()
    digest = hashlib.sha256()
    state = "header"
    header, footer, kept = [], [], []
    current, task_id, total, all_ids = [], None, 0, set()
    with source.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if state == "header":
                header.append(line)
                if line == b'      "tasks": [\n':
                    state = "tasks"
                continue
            if state == "footer":
                footer.append(line)
                continue
            if line == b'      ]\n':
                assert not current and task_id is None
                state = "footer"
                footer.append(line)
            elif line == b'        {\n':
                assert not current and task_id is None
                current = [line]
            elif line.startswith(b'          "id": '):
                assert len(current) == 1 and task_id is None
                task_id = json.loads(line.split(b": ", 1)[1].rstrip(b",\r\n"))
                assert task_id not in all_ids
                all_ids.add(task_id)
                if task_id in IDS:
                    current.append(line)
            elif line in (b'        },\n', b'        }\n'):
                assert task_id is not None
                total += 1
                if task_id in IDS:
                    current.append(b'        }\n')
                    kept.append(b"".join(current))
                current, task_id = [], None
            elif task_id in IDS:
                current.append(line)
    after = source.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    nonofficial = all_ids - expected_ids
    assert state == "footer" and expected_ids <= all_ids and len(kept) == len(IDS) and IDS <= all_ids
    assert len(nonofficial) == 2 and all(x.startswith("unattributed/") for x in nonofficial)
    sha = digest.hexdigest()
    if label == "f02":
        assert sha == "a56a03136c7da8b1bcd0b25720885a82ac030802397477c4637c8ae2e83ebdf4"
    body = b"".join(header) + b",\n".join(x.rstrip(b"\n") for x in kept) + b"\n" + b"".join(footer)
    assert len(body) < 64 * 1024**2
    output = OUT / f"{label}-selected.json"
    with output.open("xb") as handle:
        handle.write(body)
    selected = json.loads(body)
    assert selected["schema_version"] == 1 and len(selected["runs"]) == 1
    run = selected["runs"][0]
    assert {t["id"] for t in run["tasks"]} == IDS
    summaries = []
    for task in run["tasks"]:
        rows = task["observations"]
        summaries.append({
            **{k: task.get(k) for k in ("id", "source", "area", "score", "status")},
            "observation_cells": len(rows),
            "pair_key_present": sum("pair_key" in x for x in rows),
            "accepted_true": sum(x.get("accepted") is True for x in rows),
            "accepted_false": sum(x.get("accepted") is False for x in rows),
            "accepted_unknown": sum(x.get("accepted") is None for x in rows),
            "ranks": sorted({x["rank"] for x in rows}),
            "call_ids": sorted({x["call_id"] for x in rows}),
        })
    receipt["inputs"][label] = {
        "path": str(source), "bytes": before.st_size, "sha256": sha,
        "stat_unchanged": True, "full_merged_task_count": total,
        "official_item_count": len(expected_ids), "nonofficial_task_ids": sorted(nonofficial),
        "official_items_path": str(result_path), "official_items_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "selected_path": str(output), "selected_bytes": len(body),
        "selected_sha256": hashlib.sha256(body).hexdigest(),
        "run_id": run["id"], "checkpoint": run["checkpoint"],
        "runtime": run["runtime"], "capture_status": run["capture"]["status"],
        "workers_closed": [x.get("closed") for x in run["capture"]["workers"]],
        "sites": len(run["sites"]), "tasks": summaries,
    }
    print(json.dumps({"label": label, "full_sha256": sha, "full_tasks": total,
                      "selected_bytes": len(body), "tasks": summaries}), flush=True)
    del body, selected, run, kept
    gc.collect()
receipt["elapsed_seconds"] = time.time() - started
(OUT / "extraction-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"elapsed_seconds": receipt["elapsed_seconds"], "output": str(OUT)}))
```

## Existing compare/view execution and bounded validation

The next recorded block calls the existing `observe` CLI functions with the
same arguments as `wsp observe compare` and `wsp observe view`, using the existing
CPU Python and repository source. It does not build a runner or contact any
service. Static weights are initially omitted while the independent full diff
runs; the selected edit-manifest norm is not passed off as a full-model diff.
The actual eight compressed task payloads are decoded and checked for exact
round-trip equality, then the existing offline DOM test harness checks all
four tasks × two runs × two ranks and confirms strict delta remains blocked.
This verifies data and interactions; it is not a browser visual-layout test.

```python
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104")
REPO = R / "worktrees/kairyu-bench-weight-tools"
OUT = R / "analysis/fresh-purpose-20261107-f02-live10-v1/selected-code-view-v2"
os.nice(15)
sys.path.insert(0, str(REPO / "src"))
from kairyu_bench.wsp.observe.cli import main
from kairyu_bench.wsp.observe.report import _JS

started = time.time()
commands = [
    ["compare", str(OUT / "av-selected.json"), str(OUT / "f02-selected.json"),
     "--output", str(OUT / "comparison.json")],
    ["view", str(OUT / "comparison.json"), "--output", str(OUT / "comparison.html")],
]
for args in commands:
    assert main(args) == 0
comparison = json.loads((OUT / "comparison.json").read_text())
assert len(comparison["runs"]) == 2
for run in comparison["runs"]:
    assert "vllm_version" not in run["runtime"]
    assert all("pair_key" not in row for task in run["tasks"] for row in task["observations"])
assert all(x["closed"] is False for x in comparison["runs"][1]["capture"]["workers"])
spec = importlib.util.spec_from_file_location("existing_report_tests", REPO / "tests/test_wsp_observe_report.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser = module._Parser()
html = (OUT / "comparison.html").read_text()
assert len(html.encode()) < 8 * 1024**2
parser.feed(html)
assert not parser.resources
packed = json.loads(parser.data)
assert len(parser.payloads) == 8
for run, original in zip(packed["runs"], comparison["runs"], strict=True):
    for task, expected in zip(run["tasks"], original["tasks"], strict=True):
        metadata = task["observations_payload"]
        raw = gzip.decompress(base64.b64decode(parser.payloads[metadata["element_id"]]))
        assert len(raw) == metadata["raw_bytes"]
        assert json.loads(raw) == expected["observations"]
script = module._NODE[:module._NODE.index('const data=input.data')]
script = script.replace('const context={document,window:{},console};',
    'for(const [id,text] of Object.entries(input.payloads))elements[id].textContent=text;'
    'const context={document,window:{},console,DecompressionStream,Blob,TextDecoder,Uint8Array,atob};')
script += r"""
(async()=>{
 let views=0,blocked=0;
 for(const task of input.data.runs[0].tasks){
  elements.task.value=JSON.stringify([task.source,task.id]);
  for(const side of ["baseline","candidate"]){
   elements.view.value=side;elements.phase.value="prefill";
   for(const rank of ["0","1"]){
    elements.rank.value=rank;await api.render();
    assert.equal(elements["load-state"].dataset.state,"ready");
    assert(cellButtons().length>0);views++;
   }
  }
  elements.view.value="diff";await api.render();
  assert.equal(cellButtons().length,0);
  assert(elements.pairing.textContent.includes("明示pair_keyなし"));blocked++;
 }
 console.log(JSON.stringify({prefill_side_rank_views:views,strict_delta_blocked_tasks:blocked,cache:api.cacheInfo()}));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
result = subprocess.run(["node", "-e", script], input=json.dumps({
    "data": packed, "payloads": parser.payloads, "ids": parser.ids, "js": _JS,
}), capture_output=True, text=True, timeout=60, check=False)
(OUT / "viewer-dom-validation.json").write_text(json.dumps({
    "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
    "method": "existing report test offline DOM harness with actual eight task payloads; not browser layout/screenshot validation",
}, indent=2) + "\n")
assert result.returncode == 0, result.stdout + result.stderr
readout = {"scope": "declared four tasks, layer 25 only; side-by-side values, no strict delta", "tasks": []}
audits = {
    "av": json.loads((R / "analysis/fresh-purpose-20261107-av-live10-v1/code-capture-audit-v2.json").read_text()),
    "f02": json.loads((R / "analysis/fresh-purpose-20261107-f02-live10-v1/code-capture-audit.json").read_text()),
}
for index, task in enumerate(comparison["runs"][0]["tasks"]):
    item = {"id": task["id"], "source": task["source"]}
    for label, run in zip(["av", "f02"], comparison["runs"], strict=True):
        t = run["tasks"][index]
        assert t["id"] == item["id"]
        audit_item = next(x for x in audits[label]["code_items"] if x["id"] == t["id"])
        assert len(audit_item["calls"]) == 1
        call = next(iter(audit_item["calls"].values()))
        item[label] = {
            "score": t["score"], "call_id": call["call_id"], "request_id": call["request_id"],
            "input_sha256": call["input_sha256"], "seed": call["seed"], "seconds": call["seconds"],
            "prompt_tokens": call["usage"]["prompt_tokens"], "completion_tokens": call["usage"]["completion_tokens"],
            "ranks": {},
        }
        for rank in [0, 1]:
            audit_rank = call["ranks"][str(rank)]
            prefill = [x for x in t["observations"] if x["phase"] == "prefill" and x["rank"] == rank and x["layer"] == 25]
            assert len(prefill) == 3 and all(x["sample_count"] == 8 and x["accepted"] is True for x in prefill)
            item[label]["ranks"][str(rank)] = {
                "prefill_exact_positions": audit_rank["prefill_positions"],
                "prefill_layer25_bins": prefill,
                "decode_offset64_layer25": [x for x in audit_rank["profile_offset_64"] if x["layer"] == 25],
            }
    assert item["av"]["input_sha256"] == item["f02"]["input_sha256"]
    assert item["av"]["seed"] == item["f02"]["seed"]
    for rank in ["0", "1"]:
        assert item["av"]["ranks"][rank]["prefill_exact_positions"] == item["f02"]["ranks"][rank]["prefill_exact_positions"]
    readout["tasks"].append(item)
(OUT / "four-task-readout.json").write_text(json.dumps(readout, indent=2) + "\n")
def digest(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
receipt = {
    "commands": commands, "weights_attached": False,
    "weight_reason": "full diff still running; selected edit norm is not a full-checkpoint denominator",
    "source_files": [digest(REPO / rel) for rel in [
        "src/kairyu_bench/wsp/observe/cli.py", "src/kairyu_bench/wsp/observe/core.py",
        "src/kairyu_bench/wsp/observe/report.py", "tests/test_wsp_observe_report.py"]],
    "outputs": [digest(OUT / name) for name in [
        "av-selected.json", "f02-selected.json", "comparison.json", "comparison.html",
        "extraction-receipt.json", "viewer-dom-validation.json", "four-task-readout.json"]],
    "validation": "8 task gzip payloads exact-equal source rows, no external resources, actual-data offline DOM interactions passed",
    "elapsed_seconds": time.time() - started,
}
(OUT / "viewer-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"html": digest(OUT / "comparison.html"), "dom": result.stdout, "elapsed_seconds": receipt["elapsed_seconds"]}))
```

## Result and inspection limits

[Offline interactive viewer](artifacts/av-f02-four-code-observation-20261107.html) (2,718,906 bytes), [exact numeric readout](artifacts/av-f02-four-code-readout-20261107.json), and [input/output hashes, counts, failed extraction and validation receipt](artifacts/av-f02-four-code-observation-receipt-20261107.json) are included in this PR. Download/open the HTML in a browser with `DecompressionStream`; it is self-contained and makes no external requests. The actual-data offline DOM checks exercised 16 prefill side/rank views and all four blocked strict-delta views. Browser layout was not visually inspected.

Each input contains 780 merged task objects: 778 official items and two separate unattributed requests. Exactly four declared tasks were selected, with 11,160 merged observation cells per run, including 1,800 rejected-state cells. The selected JSON files are 17,663,301 (AV) and 17,662,655 (F02) bytes. The successful two-input extraction took 56.38 seconds; compare/view, payload round trips, DOM checks and numeric readout took 3.88 seconds. This adds zero benchmark or model requests.

Full canonical merged input SHA256 values:

- AV: `e9510108836aad9d3e3dbeca07959dca4f5464b1e95c2bf08fd28609d0840e3e`
- F02: `a56a03136c7da8b1bcd0b25720885a82ac030802397477c4637c8ae2e83ebdf4`

All four full input hashes and seeds match across runs; each task has one observed call and both TP ranks. Prefill samples occupy the same eight exact positions in each paired task: 0–7 for `arc191_c`, and 32–39 for the other three. The tables below show layer 25 only, not whole-layer or whole-token averages. The prefill RMS is the existing eight-sample merged RMS (square-root of the mean of squared per-position RMS); the JSON retains all bin metadata. Every prefill row is accepted=true. The two MBPP tasks have distinct full input hashes despite identical recorded prefill values, illustrating why these few positions cannot identify a problem-specific capability.

| Task | Rank | AV attention / MLP / residual RMS | F02 attention / MLP / residual RMS |
| --- | ---: | --- | --- |
| humaneval-plus:HumanEval/30 | 0 | 1.1776266 / 0.2346393 / 4.0466881 | 1.1804242 / 0.2346371 / 4.0467896 |
| humaneval-plus:HumanEval/30 | 1 | 1.1776266 / 0.2346393 / 4.0466881 | 1.1804242 / 0.2346371 / 4.0467896 |
| livecodebench:arc191_c | 0 | 1.6233045 / 2.7864862 / 3.1793138 | 1.6238720 / 2.7837607 / 3.1819512 |
| livecodebench:arc191_c | 1 | 1.6233045 / 2.7864862 / 3.1793138 | 1.6238720 / 2.7837607 / 3.1819512 |
| mbpp-plus:Mbpp/410 | 0 | 1.1782795 / 0.2350888 / 4.0425128 | 1.1796540 / 0.2351689 / 4.0432306 |
| mbpp-plus:Mbpp/410 | 1 | 1.1782795 / 0.2350888 / 4.0425128 | 1.1796540 / 0.2351689 / 4.0432306 |
| mbpp-plus:Mbpp/576 | 0 | 1.1782795 / 0.2350888 / 4.0425128 | 1.1796540 / 0.2351689 / 4.0432306 |
| mbpp-plus:Mbpp/576 | 1 | 1.1782795 / 0.2350888 / 4.0425128 | 1.1796540 / 0.2351689 / 4.0432306 |

At exact prompt-relative decode offset 64, generated-prefix equality is unknown. These are side-by-side execution values, not matched-prefix deltas. For `arc191_c` the recorded execution was rejected in both runs, so its row is intentionally absent from the heatmap but remains below and in the numeric artifact.

| Task | Rank | Accepted AV / F02 | AV attention / MLP / residual RMS | F02 attention / MLP / residual RMS |
| --- | ---: | --- | --- | --- |
| humaneval-plus:HumanEval/30 | 0 | true / true | 1.0460653 / 0.1905250 / 4.4930630 | 1.0570539 / 0.2014072 / 4.5656343 |
| humaneval-plus:HumanEval/30 | 1 | true / true | 1.0460653 / 0.1905250 / 4.4930630 | 1.0570539 / 0.2014072 / 4.5656343 |
| livecodebench:arc191_c | 0 | false / false | 0.7461417 / 0.1537662 / 4.5466490 | 0.7443407 / 0.1535384 / 4.5455685 |
| livecodebench:arc191_c | 1 | false / false | 0.7461417 / 0.1537662 / 4.5466490 | 0.7443407 / 0.1535384 / 4.5455685 |
| mbpp-plus:Mbpp/410 | 0 | true / true | 1.0307635 / 0.1910870 / 5.0224252 | 0.7827431 / 0.1744092 / 4.6482496 |
| mbpp-plus:Mbpp/410 | 1 | true / true | 1.0307635 / 0.1910870 / 5.0224252 | 0.7827431 / 0.1744092 / 4.6482496 |
| mbpp-plus:Mbpp/576 | 0 | true / true | 0.9021181 / 0.1694616 / 4.5182757 | 0.9009171 / 0.1688469 / 4.5177398 |
| mbpp-plus:Mbpp/576 | 1 | true / true | 0.9021181 / 0.1694616 / 4.5182757 | 0.9009171 / 0.1688469 / 4.5177398 |

The same three tasks fail and the declared HumanEval control passes in both complete official runs. MBPP/410 has visibly different offset-64 RMS while its official failure is unchanged; the passing control also has different RMS. These observations do not establish a beneficial direction, causal skill location, or matched continuation. No cross-call, cross-rank or cross-phase pooling was introduced. The full 26-code audit, all 778 scores, failed proxy gate and F02 closed=false remain authoritative context.

## Recorded invocation

All commands ran through the authorized SSH connection. The one-off blocks were executed with the existing CPU Python, selected from this Markdown (v1 first, corrected v2 second, viewer third). At the time of each successful invocation its block was the final Python block; for reproduction the immutable block hashes in the receipt identify the exact code. Fresh output directories are required; existing outputs are not replaced.

```sh
/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/infra/weight-tools-ci/bin/python - <<'PY'
from pathlib import Path
p = Path("/mnt/nvme/kairyu/wsp/gemma4-coding-20261104/worktrees/kairyu-bench-weight-tools/docs/experiments/2026-11-07-av-f02-four-code-observation.md")
code = p.read_text().rsplit("```python\n", 1)[1].split("```", 1)[0]
exec(compile(code, str(p), "exec"), {"__name__": "__main__"})
PY
```
